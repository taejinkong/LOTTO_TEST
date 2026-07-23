"""Scenario-based Lotto candidate scoring."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

from lotto_analyzer import (
    Draw,
    feature_set,
    matching_rules,
    mine_rules,
    odd_even,
    target_features,
    target_family,
)


RARE_OE_PATTERNS = {"OE=5:1", "OE=1:5", "OE=6:0", "OE=0:6"}
MODEL_NAME = "integrated_rules_v2"
LEGACY_MODEL_NAME = "pair_cycle_v1"
SCORE_WEIGHTS = {
    "pattern": 0.03,
    "transition": 0.02,
    "conditional": 0.04,
    "number": 0.03,
    "pair": 0.56,
    "cycle": 0.32,
}
LEGACY_SCORE_WEIGHTS = {"pair": 0.14, "cycle": 0.08}
RARE_AFTER_RARE_PENALTY = 0.35


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    min_low: int | None = None
    max_low: int | None = None
    min_high: int | None = None
    max_high: int | None = None
    min_sum: int | None = None
    max_sum: int | None = None
    min_first: int | None = None
    max_first: int | None = None
    min_consecutive: int | None = None
    max_consecutive: int | None = None


SCENARIOS = [
    Scenario("balanced", "균형형: 합계 100~159, 연속번호 1쌍 이하", min_sum=100, max_sum=159, max_consecutive=1),
    Scenario("high_sum", "고합계형: 합계 160 이상", min_sum=160),
    Scenario("high_start", "고첫수형: 첫 수 21 이상", min_first=21),
    Scenario("consecutive", "연속형: 연속번호 2쌍 이상", min_consecutive=2),
    Scenario("high_band", "고번호형: 31~45 번호 3개 이상", min_high=3),
]


def candidate_target_features(
    numbers: tuple[int, ...],
    previous_numbers: tuple[int, ...] | None = None,
) -> list[str]:
    """Return the same full feature set used by the rule miner."""
    candidate = Draw(0, "", numbers, 0)
    previous = Draw(-1, "", previous_numbers, 0) if previous_numbers else None
    return target_features(candidate, previous)


def split_candidate_features(
    numbers: tuple[int, ...],
    previous_numbers: tuple[int, ...] | None = None,
) -> tuple[list[str], list[str]]:
    features = candidate_target_features(numbers, previous_numbers)
    transition = [
        feature
        for feature in features
        if feature.startswith("CARRY=") or feature.startswith("NEIGHBOR=")
    ]
    static = [feature for feature in features if feature not in transition]
    return static, transition


def low_value(numbers: tuple[int, ...]) -> int:
    return sum(1 <= number <= 10 for number in numbers)


def high_value(numbers: tuple[int, ...]) -> int:
    return sum(31 <= number <= 45 for number in numbers)


def consecutive_value(numbers: tuple[int, ...]) -> int:
    return sum(1 for left, right in zip(numbers, numbers[1:]) if right - left == 1)


def matches_scenario(numbers: tuple[int, ...], scenario: Scenario) -> bool:
    low = low_value(numbers)
    high = high_value(numbers)
    total = sum(numbers)
    first = numbers[0]
    consecutive = consecutive_value(numbers)

    checks = [
        scenario.min_low is None or low >= scenario.min_low,
        scenario.max_low is None or low <= scenario.max_low,
        scenario.min_high is None or high >= scenario.min_high,
        scenario.max_high is None or high <= scenario.max_high,
        scenario.min_sum is None or total >= scenario.min_sum,
        scenario.max_sum is None or total <= scenario.max_sum,
        scenario.min_first is None or first >= scenario.min_first,
        scenario.max_first is None or first <= scenario.max_first,
        scenario.min_consecutive is None or consecutive >= scenario.min_consecutive,
        scenario.max_consecutive is None or consecutive <= scenario.max_consecutive,
    ]
    return all(checks)


def build_raw_target_scores(
    training_draws: list[Draw],
    max_conditions: int,
    min_support: int,
    min_confidence: float,
    min_lift: float,
) -> dict[str, float]:
    rules = mine_rules(
        draws=training_draws,
        max_conditions=max_conditions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    latest_features = set(feature_set(training_draws, len(training_draws) - 1))
    matched = matching_rules(rules, latest_features)

    scores: dict[str, float] = defaultdict(float)
    for rule in matched:
        scores[rule.target] += rule.lift * rule.confidence * min(rule.total, 30)
    return dict(scores)


def normalize_target_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    max_by_family: dict[str, float] = defaultdict(float)
    for target, score in raw_scores.items():
        max_by_family[target_family(target)] = max(max_by_family[target_family(target)], score)

    normalized = {}
    for target, score in raw_scores.items():
        family_max = max_by_family[target_family(target)]
        normalized[target] = score / family_max if family_max else 0
    return normalized


def build_distribution_scores(training_draws: list[Draw]) -> dict[str, float]:
    """Score historical pattern prevalence, normalized within each family."""
    counts: Counter[str] = Counter()
    max_by_family: dict[str, int] = defaultdict(int)
    for index, draw in enumerate(training_draws):
        previous = training_draws[index - 1] if index > 0 else None
        for feature in target_features(draw, previous):
            counts[feature] += 1
    for feature, count in counts.items():
        family = target_family(feature)
        max_by_family[family] = max(max_by_family[family], count)
    return {
        feature: count / max_by_family[target_family(feature)]
        for feature, count in counts.items()
        if max_by_family[target_family(feature)]
    }


def build_number_scores(training_draws: list[Draw], recent_window: int) -> dict[int, float]:
    all_counts = Counter(number for draw in training_draws for number in draw.numbers)
    recent_draws = training_draws[-recent_window:]
    recent_counts = Counter(number for draw in recent_draws for number in draw.numbers)
    last_seen = {}
    for index, draw in enumerate(training_draws):
        for number in draw.numbers:
            last_seen[number] = index

    max_all = max(all_counts.values())
    max_recent = max(recent_counts.values()) or 1
    latest_index = len(training_draws) - 1
    max_gap = max(latest_index - last_seen.get(number, -1) for number in range(1, 46))

    scores = {}
    for number in range(1, 46):
        frequency_score = all_counts[number] / max_all
        recent_score = recent_counts[number] / max_recent
        overdue_score = (latest_index - last_seen.get(number, -1)) / max_gap if max_gap else 0
        scores[number] = frequency_score * 0.35 + recent_score * 0.45 + overdue_score * 0.20
    return scores


def build_pair_scores(training_draws: list[Draw], recent_window: int) -> dict[tuple[int, int], float]:
    all_counts = Counter()
    recent_counts = Counter()
    for draw in training_draws:
        all_counts.update(combinations(draw.numbers, 2))
    for draw in training_draws[-recent_window:]:
        recent_counts.update(combinations(draw.numbers, 2))

    max_all = max(all_counts.values()) if all_counts else 1
    max_recent = max(recent_counts.values()) if recent_counts else 1
    scores = {}
    for pair in combinations(range(1, 46), 2):
        scores[pair] = (all_counts[pair] / max_all) * 0.45 + (recent_counts[pair] / max_recent) * 0.55
    return scores


def build_cycle_scores(training_draws: list[Draw]) -> dict[int, float]:
    seen_indexes: dict[int, list[int]] = defaultdict(list)
    for index, draw in enumerate(training_draws):
        for number in draw.numbers:
            seen_indexes[number].append(index)

    latest_index = len(training_draws) - 1
    raw_scores = {}
    for number in range(1, 46):
        indexes = seen_indexes[number]
        if len(indexes) < 2:
            raw_scores[number] = 0.0
            continue
        intervals = [right - left for left, right in zip(indexes, indexes[1:])]
        average_interval = sum(intervals) / len(intervals)
        current_gap = latest_index - indexes[-1]
        raw_scores[number] = min(current_gap / average_interval, 2.0) if average_interval else 0.0

    max_score = max(raw_scores.values()) or 1.0
    return {number: score / max_score for number, score in raw_scores.items()}


def score_candidate(
    numbers: tuple[int, ...],
    target_scores: dict[str, float],
    number_scores: dict[int, float],
    pair_scores: dict[tuple[int, int], float] | None = None,
    cycle_scores: dict[int, float] | None = None,
    previous_oe: str | None = None,
    rare_after_rare_penalty: float = RARE_AFTER_RARE_PENALTY,
    distribution_scores: dict[str, float] | None = None,
    previous_numbers: tuple[int, ...] | None = None,
) -> float:
    parts = canonical_score_parts(
        numbers,
        pair_scores or {},
        cycle_scores or {},
        previous_oe,
        rare_after_rare_penalty,
        distribution_scores=distribution_scores,
        conditional_scores=target_scores,
        number_scores=number_scores,
        previous_numbers=previous_numbers,
    )
    return parts["total"]


def canonical_score_parts(
    numbers: tuple[int, ...],
    pair_scores: dict[tuple[int, int], float],
    cycle_scores: dict[int, float],
    previous_oe: str | None = None,
    rare_after_rare_penalty: float = RARE_AFTER_RARE_PENALTY,
    *,
    distribution_scores: dict[str, float] | None = None,
    conditional_scores: dict[str, float] | None = None,
    number_scores: dict[int, float] | None = None,
    previous_numbers: tuple[int, ...] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return the exact integrated score components shared by CLI and web."""
    active_weights = weights or SCORE_WEIGHTS
    static_features, transition_features = split_candidate_features(numbers, previous_numbers)
    all_features = [*static_features, *transition_features]

    def average(feature_scores: dict[str, float] | None, features: list[str]) -> float:
        if not feature_scores or not features:
            return 0.0
        return sum(feature_scores.get(feature, 0.0) for feature in features) / len(features)

    pattern = average(distribution_scores, static_features)
    transition = average(distribution_scores, transition_features)
    conditional = average(conditional_scores, all_features)
    number = (
        sum(number_scores.get(item, 0.0) for item in numbers) / 6
        if number_scores
        else 0.0
    )
    pair = (
        sum(pair_scores.get(item, 0.0) for item in combinations(numbers, 2)) / 15
        if pair_scores
        else 0.0
    )
    cycle = (
        sum(cycle_scores.get(item, 0.0) for item in numbers) / 6
        if cycle_scores
        else 0.0
    )
    penalty = (
        rare_after_rare_penalty
        if previous_oe in RARE_OE_PATTERNS and odd_even(numbers) in RARE_OE_PATTERNS
        else 0.0
    )
    total = (
        pattern * active_weights.get("pattern", 0.0)
        + transition * active_weights.get("transition", 0.0)
        + conditional * active_weights.get("conditional", 0.0)
        + number * active_weights.get("number", 0.0)
        + pair * active_weights.get("pair", 0.0)
        + cycle * active_weights.get("cycle", 0.0)
        - penalty
    )
    return {
        "pattern": pattern,
        "transition": transition,
        "conditional": conditional,
        "number": number,
        "pair": pair,
        "cycle": cycle,
        "penalty": penalty,
        "total": total,
    }


def top_candidates_by_scenario(
    target_scores: dict[str, float],
    number_scores: dict[int, float],
    per_scenario: int,
    pair_scores: dict[tuple[int, int], float] | None = None,
    cycle_scores: dict[int, float] | None = None,
    previous_oe: str | None = None,
    rare_after_rare_penalty: float = RARE_AFTER_RARE_PENALTY,
    distribution_scores: dict[str, float] | None = None,
    previous_numbers: tuple[int, ...] | None = None,
) -> dict[str, list[tuple[float, tuple[int, ...]]]]:
    heaps: dict[str, list[tuple[float, tuple[int, ...]]]] = {scenario.name: [] for scenario in SCENARIOS}

    for numbers in combinations(range(1, 46), 6):
        score = score_candidate(
            numbers,
            target_scores,
            number_scores,
            pair_scores=pair_scores,
            cycle_scores=cycle_scores,
            previous_oe=previous_oe,
            rare_after_rare_penalty=rare_after_rare_penalty,
            distribution_scores=distribution_scores,
            previous_numbers=previous_numbers,
        )
        for scenario in SCENARIOS:
            if not matches_scenario(numbers, scenario):
                continue
            heap = heaps[scenario.name]
            if len(heap) < per_scenario:
                heapq.heappush(heap, (score, numbers))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, numbers))

    return {name: sorted(heap, reverse=True) for name, heap in heaps.items()}


def build_prediction_inputs(
    training_draws: list[Draw],
    max_conditions: int = 3,
    min_support: int = 15,
    min_confidence: float = 0.45,
    min_lift: float = 1.25,
    recent_window: int = 50,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[int, float],
    dict[tuple[int, int], float],
    dict[int, float],
]:
    raw_scores = build_raw_target_scores(
        training_draws=training_draws,
        max_conditions=max_conditions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    return (
        build_distribution_scores(training_draws),
        normalize_target_scores(raw_scores),
        build_number_scores(training_draws, recent_window),
        build_pair_scores(training_draws, recent_window),
        build_cycle_scores(training_draws),
    )
