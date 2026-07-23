#!/usr/bin/env python3
"""Lotto statistics and conditional rule analyzer."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


NUMBER_COLUMNS = [f"번호{i}" for i in range(1, 7)]
COLOR_GROUPS = (
    ("G1", 1, 10),
    ("G2", 11, 20),
    ("G3", 21, 30),
    ("G4", 31, 40),
    ("G5", 41, 45),
)
RARE_OE_PATTERNS = {"OE=5:1", "OE=1:5", "OE=6:0", "OE=0:6"}


@dataclass(frozen=True)
class Draw:
    round_no: int
    date: str
    numbers: tuple[int, ...]
    bonus: int


@dataclass(frozen=True)
class Rule:
    antecedent: tuple[str, ...]
    target: str
    hits: int
    total: int
    confidence: float
    base_rate: float
    lift: float


def load_draws(csv_path: Path) -> list[Draw]:
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        draws = []
        for row in reader:
            numbers = tuple(sorted(int(row[column]) for column in NUMBER_COLUMNS))
            draws.append(
                Draw(
                    round_no=int(row["회차"].replace("회", "")),
                    date=row["추첨일"],
                    numbers=numbers,
                    bonus=int(row["보너스"]),
                )
            )
    return sorted(draws, key=lambda draw: draw.round_no)


def odd_even(numbers: tuple[int, ...]) -> str:
    odd = sum(number % 2 for number in numbers)
    return f"OE={odd}:{6 - odd}"


def color_pattern(numbers: tuple[int, ...]) -> str:
    counts = []
    for _, start, end in COLOR_GROUPS:
        counts.append(sum(start <= number <= end for number in numbers))
    return "COLOR=" + "-".join(str(count) for count in counts)


def color_presence(numbers: tuple[int, ...]) -> list[str]:
    features = []
    for label, start, end in COLOR_GROUPS:
        count = sum(start <= number <= end for number in numbers)
        features.append(f"{label}_COUNT={count}")
        if count:
            features.append(f"{label}_HAS=Y")
    return features


def tail_pattern(numbers: tuple[int, ...]) -> str:
    tails = Counter(number % 10 for number in numbers)
    duplicate_count = sum(1 for count in tails.values() if count >= 2)
    max_duplicate = max(tails.values())
    return f"TAIL_DUP={duplicate_count},MAX={max_duplicate}"


def sum_bin(numbers: tuple[int, ...]) -> str:
    total = sum(numbers)
    if total < 100:
        value = "<100"
    elif total < 120:
        value = "100-119"
    elif total < 140:
        value = "120-139"
    elif total < 160:
        value = "140-159"
    else:
        value = "160+"
    return f"SUM={value}"


def low_count(numbers: tuple[int, ...]) -> str:
    return f"LOW_1_10={sum(1 <= number <= 10 for number in numbers)}"


def high_count(numbers: tuple[int, ...]) -> str:
    return f"HIGH_31_45={sum(31 <= number <= 45 for number in numbers)}"


def consecutive_pattern(numbers: tuple[int, ...]) -> str:
    pairs = sum(1 for left, right in zip(numbers, numbers[1:]) if right - left == 1)
    return f"CONSEC={pairs if pairs < 3 else '3+'}"


def ac_value(numbers: tuple[int, ...]) -> int:
    differences = {right - left for left, right in combinations(numbers, 2)}
    return len(differences) - 5


def ac_bin(numbers: tuple[int, ...]) -> str:
    value = ac_value(numbers)
    if value <= 5:
        label = "LOW"
    elif value <= 8:
        label = "MID"
    else:
        label = "HIGH"
    return f"AC={label}"


def gap_pattern(numbers: tuple[int, ...]) -> str:
    gaps = [right - left for left, right in zip(numbers, numbers[1:])]
    if not gaps:
        return "GAP=NA"
    avg = sum(gaps) / len(gaps)
    if avg < 5:
        label = "TIGHT"
    elif avg < 8:
        label = "MID"
    else:
        label = "WIDE"
    return f"GAP={label}"


def first_number_bin(numbers: tuple[int, ...]) -> str:
    first = numbers[0]
    if first <= 5:
        label = "1-5"
    elif first <= 10:
        label = "6-10"
    elif first <= 20:
        label = "11-20"
    else:
        label = "21+"
    return f"FIRST={label}"


def draw_features(draw: Draw) -> list[str]:
    numbers = draw.numbers
    features = [
        odd_even(numbers),
        color_pattern(numbers),
        tail_pattern(numbers),
        sum_bin(numbers),
        low_count(numbers),
        high_count(numbers),
        consecutive_pattern(numbers),
        ac_bin(numbers),
        gap_pattern(numbers),
        first_number_bin(numbers),
    ]
    features.extend(color_presence(numbers))
    return features


def transition_features(previous: Draw, current: Draw) -> list[str]:
    previous_numbers = set(previous.numbers)
    current_numbers = set(current.numbers)
    carry = len(previous_numbers & current_numbers)
    neighbor = 0
    for number in current_numbers:
        if number - 1 in previous_numbers or number + 1 in previous_numbers:
            neighbor += 1
    return [f"CARRY={carry}", f"NEIGHBOR={neighbor if neighbor < 4 else '4+'}"]


def feature_set(draws: list[Draw], index: int) -> list[str]:
    features = draw_features(draws[index])
    if index > 0:
        features.extend(transition_features(draws[index - 1], draws[index]))
        features.extend(f"PREV_{feature}" for feature in draw_features(draws[index - 1]))
    return features


def target_features(draw: Draw, previous: Draw | None = None) -> list[str]:
    """Return every analyzable feature that a candidate draw can satisfy.

    Older versions only exposed nine coarse pattern families as prediction
    targets.  The integrated model also scores tail duplication, per-band
    counts/presence, carry-over numbers, and neighbours of the previous draw.
    """
    features = draw_features(draw)
    if previous is not None:
        features.extend(transition_features(previous, draw))
    return features


def target_family(feature: str) -> str:
    return feature.split("=", 1)[0]


def mine_rules(
    draws: list[Draw],
    max_conditions: int,
    min_support: int,
    min_confidence: float,
    min_lift: float,
) -> list[Rule]:
    base_counts = Counter()
    family_totals = Counter()
    for index in range(2, len(draws)):
        for target in target_features(draws[index], draws[index - 1]):
            base_counts[target] += 1
            family_totals[target_family(target)] += 1

    conditional_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    support_counts: Counter[tuple[str, ...]] = Counter()

    for index in range(1, len(draws) - 1):
        source_features = feature_set(draws, index)
        next_targets = target_features(draws[index + 1], draws[index])
        for condition_count in range(1, max_conditions + 1):
            for antecedent in combinations(source_features, condition_count):
                support_counts[antecedent] += 1
                for target in next_targets:
                    conditional_counts[antecedent][target] += 1

    rules = []
    for antecedent, counts in conditional_counts.items():
        total = support_counts[antecedent]
        if total < min_support:
            continue
        for target, hits in counts.items():
            confidence = hits / total
            family_total = family_totals[target_family(target)]
            base_rate = base_counts[target] / family_total if family_total else 0
            lift = confidence / base_rate if base_rate else 0
            if confidence >= min_confidence and lift >= min_lift:
                rules.append(
                    Rule(
                        antecedent=antecedent,
                        target=target,
                        hits=hits,
                        total=total,
                        confidence=confidence,
                        base_rate=base_rate,
                        lift=lift,
                    )
                )

    return sorted(
        rules,
        key=lambda rule: (rule.lift, rule.confidence, rule.hits, -len(rule.antecedent)),
        reverse=True,
    )


def matching_rules(rules: list[Rule], latest_features: set[str]) -> list[Rule]:
    return [
        rule
        for rule in rules
        if all(condition in latest_features for condition in rule.antecedent)
    ]


def summarize_distribution(draws: list[Draw]) -> str:
    lines = ["기본 홀짝 분포"]
    counts = Counter(odd_even(draw.numbers) for draw in draws)
    total = sum(counts.values())
    for label, count in counts.most_common():
        lines.append(f"- {label.replace('OE=', '')}: {count}회 ({count / total:.1%})")
    return "\n".join(lines)


def count_g1(numbers: tuple[int, ...]) -> int:
    return sum(1 <= number <= 10 for number in numbers)


def g1_absence_rate(draws: list[Draw], indexes: list[int]) -> tuple[int, int, float]:
    total = len(indexes)
    hits = sum(1 for index in indexes if count_g1(draws[index].numbers) == 0)
    return hits, total, hits / total if total else 0


def summarize_g1_absence(draws: list[Draw]) -> str:
    total = len(draws)
    distribution = Counter(count_g1(draw.numbers) for draw in draws)
    absent = distribution[0]

    lines = [
        "",
        "1~10 그룹 미출현 분석",
        f"- 전체 기준: {absent}/{total}회 ({absent / total:.1%})",
        "- 1~10 포함 개수 분포: "
        + ", ".join(
            f"{count}개={occurrences}회({occurrences / total:.1%})"
            for count, occurrences in sorted(distribution.items())
        ),
        "- 직전 회차의 1~10 포함 개수별 다음 회차 미출현율:",
    ]

    indexes_by_previous_g1: dict[int, list[int]] = defaultdict(list)
    for index in range(1, len(draws)):
        previous_count = count_g1(draws[index - 1].numbers)
        indexes_by_previous_g1[previous_count].append(index)
    for previous_count, indexes in sorted(indexes_by_previous_g1.items()):
        hits, cases, rate = g1_absence_rate(draws, indexes)
        lines.append(f"  - 직전 {previous_count}개: {hits}/{cases}회 ({rate:.1%})")

    lines.append("- 1~10 미출현 연속 이후 다음 회차 미출현율:")
    for streak_length in (1, 2, 3):
        indexes = []
        for index in range(streak_length, len(draws)):
            streak_indexes = range(index - streak_length, index)
            if all(count_g1(draws[streak_index].numbers) == 0 for streak_index in streak_indexes):
                indexes.append(index)
        hits, cases, rate = g1_absence_rate(draws, indexes)
        lines.append(f"  - {streak_length}회 연속 후: {hits}/{cases}회 ({rate:.1%})")

    latest_g1 = count_g1(draws[-1].numbers)
    latest_indexes = indexes_by_previous_g1.get(latest_g1, [])
    hits, cases, rate = g1_absence_rate(draws, latest_indexes)
    lines.append(
        f"- 최신 회차 기준 직전 1~10 포함 {latest_g1}개 조건: "
        f"다음 미출현 {hits}/{cases}회 ({rate:.1%})"
    )
    return "\n".join(lines)


def summarize_rare_oe_transition(draws: list[Draw]) -> str:
    transitions = Counter()
    cases = 0
    rare_after_rare = 0
    for previous, current in zip(draws, draws[1:]):
        previous_oe = odd_even(previous.numbers)
        current_oe = odd_even(current.numbers)
        if previous_oe not in RARE_OE_PATTERNS:
            continue
        cases += 1
        transitions[current_oe] += 1
        rare_after_rare += int(current_oe in RARE_OE_PATTERNS)

    lines = [
        "",
        "희귀 홀짝 전이 분석",
        "- 희귀 홀짝: 5:1, 1:5, 6:0, 0:6",
    ]
    if not cases:
        lines.append("- 희귀 홀짝 직후 사례가 없습니다.")
        return "\n".join(lines)

    lines.append(f"- 희귀 홀짝 직후 다음 회차도 희귀 홀짝: {rare_after_rare}/{cases}회 ({rare_after_rare / cases:.1%})")
    lines.append("- 희귀 홀짝 직후 다음 회차 홀짝 분포:")
    for label, count in transitions.most_common():
        lines.append(f"  - {label.replace('OE=', '')}: {count}회 ({count / cases:.1%})")
    return "\n".join(lines)


def format_rule(rule: Rule) -> str:
    conditions = " AND ".join(rule.antecedent)
    target = rule.target
    return (
        f"IF {conditions} THEN NEXT {target} "
        f"= {rule.hits}/{rule.total}, "
        f"conf {rule.confidence:.1%}, base {rule.base_rate:.1%}, lift {rule.lift:.2f}"
    )


def summarize_latest(draws: list[Draw], rules: list[Rule], limit: int) -> str:
    latest = draws[-1]
    latest_features = set(feature_set(draws, len(draws) - 1))
    matched = matching_rules(rules, latest_features)

    score_by_target: dict[str, float] = defaultdict(float)
    evidence_by_target: dict[str, list[Rule]] = defaultdict(list)
    for rule in matched:
        score_by_target[rule.target] += rule.lift * rule.confidence * min(rule.total, 30)
        evidence_by_target[rule.target].append(rule)

    ranked_targets = sorted(score_by_target.items(), key=lambda item: item[1], reverse=True)

    lines = [
        "",
        f"최신 회차: {latest.round_no}회 ({latest.date}) {list(latest.numbers)}",
        "최신 회차 조건과 맞는 다음 회차 후보 패턴",
    ]
    for target, score in ranked_targets[:limit]:
        best_rule = evidence_by_target[target][0]
        lines.append(f"- {target}: score {score:.2f}, 근거 {best_rule.hits}/{best_rule.total}, lift {best_rule.lift:.2f}")
    if not ranked_targets:
        lines.append("- 조건에 맞는 강한 규칙이 없습니다. 필터 기준을 낮춰보세요.")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> str:
    draws = load_draws(Path(args.csv))
    rules = mine_rules(
        draws=draws,
        max_conditions=args.max_conditions,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
    )

    lines = [
        f"분석 대상: {draws[0].round_no}회~{draws[-1].round_no}회, 총 {len(draws)}회",
        summarize_distribution(draws),
        summarize_g1_absence(draws),
        summarize_rare_oe_transition(draws),
        "",
        "조건부 연계 규칙 TOP",
    ]
    for rule in rules[: args.limit]:
        lines.append(f"- {format_rule(rule)}")
    lines.append(summarize_latest(draws, rules, args.latest_limit))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Lotto conditional rules.")
    parser.add_argument("--csv", default="lotto_winners_2020_2026.csv", help="Input CSV path.")
    parser.add_argument("--limit", type=int, default=30, help="Number of top rules to print.")
    parser.add_argument("--latest-limit", type=int, default=12, help="Number of latest matching targets to print.")
    parser.add_argument("--max-conditions", type=int, default=3, help="Maximum number of IF conditions.")
    parser.add_argument("--min-support", type=int, default=10, help="Minimum condition occurrence count.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum rule confidence.")
    parser.add_argument("--min-lift", type=float, default=1.25, help="Minimum rule lift.")
    return parser.parse_args()


def main() -> None:
    print(build_report(parse_args()))


if __name__ == "__main__":
    main()
