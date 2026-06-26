#!/usr/bin/env python3
"""Backtest Lotto prediction with a single-pass candidate evaluator."""

from __future__ import annotations

import argparse
import heapq
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from lotto_analyzer import Draw, load_draws, odd_even, target_features
from lotto_prediction import (
    RARE_OE_PATTERNS,
    SCENARIOS,
    build_cycle_scores,
    build_number_scores,
    build_pair_scores,
    build_raw_target_scores,
    candidate_target_features,
    high_value,
    low_value,
    normalize_target_scores,
    score_candidate,
)


DEFAULT_TOP_K = (50, 300, 1000, 5000, 10000, 100000, 1000000)
DEFAULT_PORTFOLIO_QUOTAS = {
    "normal": 20,
    "high_sum": 8,
    "high_band": 7,
    "no_low": 5,
    "high_start": 5,
    "consecutive": 5,
}


@dataclass(frozen=True)
class CandidateInfo:
    numbers: tuple[int, ...]
    features: tuple[str, ...]
    low: int
    high: int
    total: int
    first: int
    consecutive: int


@dataclass
class BacktestResult:
    total_candidates: int = 0
    actual_basic_score: float = 0.0
    actual_portfolio_score: float = 0.0
    actual_basic_rank: int = 1
    actual_portfolio_rank: int = 1
    basic_top: list[tuple[float, tuple[int, ...]]] = field(default_factory=list)
    portfolio_top: list[tuple[float, tuple[int, ...]]] = field(default_factory=list)
    scenario_top: dict[str, list[tuple[float, tuple[int, ...]]]] = field(default_factory=dict)
    scenario_contains_actual: dict[str, bool] = field(default_factory=dict)
    final_portfolio: list[tuple[float, tuple[int, ...], str]] = field(default_factory=list)
    final_portfolio_contains_actual: bool = False
    final_portfolio_prizes: Counter[str] = field(default_factory=Counter)


def consecutive_value(numbers: tuple[int, ...]) -> int:
    return sum(1 for left, right in zip(numbers, numbers[1:]) if right - left == 1)


def build_candidate(numbers: tuple[int, ...]) -> CandidateInfo:
    return CandidateInfo(
        numbers=numbers,
        features=tuple(candidate_target_features(numbers)),
        low=low_value(numbers),
        high=high_value(numbers),
        total=sum(numbers),
        first=numbers[0],
        consecutive=consecutive_value(numbers),
    )


def scenario_names(candidate: CandidateInfo) -> list[str]:
    names = []
    if 1 <= candidate.low <= 2 and 100 <= candidate.total <= 159 and candidate.consecutive <= 1:
        names.append("normal")
    if candidate.low == 0:
        names.append("no_low")
    if candidate.total >= 160:
        names.append("high_sum")
    if candidate.first >= 21:
        names.append("high_start")
    if candidate.consecutive >= 2:
        names.append("consecutive")
    if candidate.high >= 3:
        names.append("high_band")
    return names


def raw_pattern_score(candidate: CandidateInfo, raw_scores: dict[str, float]) -> float:
    return sum(raw_scores.get(feature, 0.0) for feature in candidate.features)


def portfolio_score(
    candidate: CandidateInfo,
    normalized_scores: dict[str, float],
    number_scores: dict[int, float],
    pair_scores: dict[tuple[int, int], float],
    cycle_scores: dict[int, float],
    previous_oe: str | None = None,
) -> float:
    return score_candidate(
        candidate.numbers,
        normalized_scores,
        number_scores,
        pair_scores=pair_scores,
        cycle_scores=cycle_scores,
        previous_oe=previous_oe,
    )


def push_top(heap: list[tuple[float, tuple[int, ...]]], item: tuple[float, tuple[int, ...]], limit: int) -> None:
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def build_final_portfolio(
    scenario_top: dict[str, list[tuple[float, tuple[int, ...]]]],
    quotas: dict[str, int],
) -> list[tuple[float, tuple[int, ...], str]]:
    selected: list[tuple[float, tuple[int, ...], str]] = []
    seen: set[tuple[int, ...]] = set()

    for scenario_name, quota in quotas.items():
        added = 0
        for score, numbers in scenario_top.get(scenario_name, []):
            if numbers in seen:
                continue
            selected.append((score, numbers, scenario_name))
            seen.add(numbers)
            added += 1
            if added >= quota:
                break

    if len(selected) < sum(quotas.values()):
        leftovers = []
        for scenario_name, candidates in scenario_top.items():
            for score, numbers in candidates:
                if numbers not in seen:
                    leftovers.append((score, numbers, scenario_name))
        leftovers.sort(reverse=True)
        for score, numbers, scenario_name in leftovers:
            selected.append((score, numbers, scenario_name))
            seen.add(numbers)
            if len(selected) >= sum(quotas.values()):
                break

    return selected


def prize_rank(numbers: tuple[int, ...], actual: Draw) -> str | None:
    matched = len(set(numbers) & set(actual.numbers))
    bonus_matched = actual.bonus in numbers
    if matched == 6:
        return "1등"
    if matched == 5 and bonus_matched:
        return "2등"
    if matched == 5:
        return "3등"
    return None


def count_prizes(portfolio: list[tuple[float, tuple[int, ...], str]], actual: Draw) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, numbers, _ in portfolio:
        rank = prize_rank(numbers, actual)
        if rank:
            counts[rank] += 1
    return counts


class BacktestEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.draws = load_draws(Path(args.csv))
        self.training_draws = [draw for draw in self.draws if draw.round_no < args.target_round]
        self.actual = next(draw for draw in self.draws if draw.round_no == args.target_round)
        self.previous = self.training_draws[-1]
        self.previous_oe = odd_even(self.previous.numbers)
        self.raw_scores = build_raw_target_scores(
            training_draws=self.training_draws,
            max_conditions=args.max_conditions,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift,
        )
        self.normalized_scores = normalize_target_scores(self.raw_scores)
        self.number_scores = build_number_scores(self.training_draws, args.recent_window)
        self.pair_scores = build_pair_scores(self.training_draws, args.recent_window)
        self.cycle_scores = build_cycle_scores(self.training_draws)
        self.actual_candidate = build_candidate(self.actual.numbers)

    def candidates(self):
        if self.args.mode == "sample":
            seen = set()
            while len(seen) < self.args.sample_size:
                numbers = tuple(sorted(random.sample(range(1, 46), 6)))
                if numbers in seen:
                    continue
                seen.add(numbers)
                yield numbers
            if self.actual.numbers not in seen:
                yield self.actual.numbers
            return

        yield from combinations(range(1, 46), 6)

    def run(self) -> BacktestResult:
        scenario_limit = self.args.per_scenario if self.args.mode in {"full", "sample"} else 0
        result = BacktestResult(
            actual_basic_score=raw_pattern_score(self.actual_candidate, self.raw_scores),
            actual_portfolio_score=portfolio_score(
                self.actual_candidate,
                self.normalized_scores,
                self.number_scores,
                self.pair_scores,
                self.cycle_scores,
                previous_oe=self.previous_oe,
            ),
            scenario_top={scenario.name: [] for scenario in SCENARIOS},
            scenario_contains_actual={scenario.name: False for scenario in SCENARIOS},
        )

        for numbers in self.candidates():
            candidate = build_candidate(numbers)
            basic_score = raw_pattern_score(candidate, self.raw_scores)
            improved_score = portfolio_score(
                candidate,
                self.normalized_scores,
                self.number_scores,
                self.pair_scores,
                self.cycle_scores,
                self.previous_oe,
            )
            result.total_candidates += 1

            if basic_score > result.actual_basic_score:
                result.actual_basic_rank += 1
            if improved_score > result.actual_portfolio_score:
                result.actual_portfolio_rank += 1

            push_top(result.basic_top, (basic_score, numbers), self.args.top)
            push_top(result.portfolio_top, (improved_score, numbers), self.args.top)

            if scenario_limit:
                for name in scenario_names(candidate):
                    if numbers == self.actual.numbers:
                        result.scenario_contains_actual[name] = True
                    push_top(result.scenario_top[name], (improved_score, numbers), scenario_limit)

        result.basic_top.sort(reverse=True)
        result.portfolio_top.sort(reverse=True)
        for name in result.scenario_top:
            result.scenario_top[name].sort(reverse=True)
            result.scenario_contains_actual[name] = any(
                numbers == self.actual.numbers for _, numbers in result.scenario_top[name]
            )
        result.final_portfolio = build_final_portfolio(result.scenario_top, DEFAULT_PORTFOLIO_QUOTAS)
        result.final_portfolio_contains_actual = any(
            numbers == self.actual.numbers for _, numbers, _ in result.final_portfolio
        )
        result.final_portfolio_prizes = count_prizes(result.final_portfolio, self.actual)
        return result

    def pattern_comparison(self) -> list[str]:
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for target, score in self.raw_scores.items():
            grouped[target.split("=", 1)[0]].append((target, score))

        actual_features = set(target_features(self.actual))
        lines = ["패턴 예측 비교"]
        matches = 0
        total = 0
        for family in sorted(grouped):
            ranked = sorted(grouped[family], key=lambda item: item[1], reverse=True)
            predicted = ranked[0][0]
            actual_target = next((feature for feature in actual_features if feature.startswith(f"{family}=")), "")
            result = "MATCH" if predicted == actual_target else "MISS"
            matches += int(result == "MATCH")
            total += 1
            lines.append(f"- {family}: 예측 {predicted}, 실제 {actual_target}, {result}")
        lines.append(f"- 패턴 적중: {matches}/{total}")
        return lines


def topk_lines(rank: int, top_k_values: list[int]) -> list[str]:
    return [f"- Top {top_k:,}: {'YES' if rank <= top_k else 'NO'}" for top_k in top_k_values]


def contribution_lines(engine: BacktestEngine) -> list[str]:
    candidate = engine.actual_candidate
    lines = ["실제 조합 항목별 점수 기여"]
    for feature in candidate.features:
        raw = engine.raw_scores.get(feature, 0.0)
        normalized = engine.normalized_scores.get(feature, 0.0)
        lines.append(f"- {feature}: raw={raw:.2f}, normalized={normalized:.3f}")
    number_score = sum(engine.number_scores[number] for number in candidate.numbers) / 6
    pair_score = sum(engine.pair_scores[pair] for pair in combinations(candidate.numbers, 2)) / 15
    cycle_score = sum(engine.cycle_scores[number] for number in candidate.numbers) / 6
    lines.append(f"- 번호 개별 평균 점수: {number_score:.3f}")
    lines.append(f"- 번호쌍 평균 점수: {pair_score:.3f}")
    lines.append(f"- 출현주기 평균 점수: {cycle_score:.3f}")
    actual_oe = odd_even(candidate.numbers)
    if engine.previous_oe in RARE_OE_PATTERNS and actual_oe in RARE_OE_PATTERNS:
        lines.append("- 희귀 홀짝 연속 패널티: 적용")
    else:
        lines.append("- 희귀 홀짝 연속 패널티: 미적용")
    return lines


def build_report(args: argparse.Namespace) -> str:
    engine = BacktestEngine(args)
    result = engine.run()
    top_k_values = sorted(set(args.top_k))

    lines = [
        f"백테스트: {engine.training_draws[0].round_no}회~{engine.previous.round_no}회 데이터로 {engine.actual.round_no}회 예측",
        f"모드: {args.mode}, 평가 조합 수: {result.total_candidates:,}",
        f"직전 회차 {engine.previous.round_no}회: {list(engine.previous.numbers)}",
        f"실제 {engine.actual.round_no}회: {list(engine.actual.numbers)} + 보너스 {engine.actual.bonus}",
        "",
        *engine.pattern_comparison(),
        "",
        "기존 단일 점수 방식",
        f"- 실제 조합 점수: {result.actual_basic_score:.2f}",
        f"- 실제 조합 순위: {result.actual_basic_rank:,}/{result.total_candidates:,}",
        *topk_lines(result.actual_basic_rank, top_k_values),
        "",
        "개선 포트폴리오 점수 방식",
        f"- 실제 조합 점수: {result.actual_portfolio_score:.3f}",
        f"- 실제 조합 순위: {result.actual_portfolio_rank:,}/{result.total_candidates:,}",
        *topk_lines(result.actual_portfolio_rank, top_k_values),
    ]

    if args.mode in {"full", "sample"}:
        total_scenario_candidates = sum(len(items) for items in result.scenario_top.values())
        found_any = any(result.scenario_contains_actual.values())
        lines.extend(
            [
                "",
                f"시나리오별 Top {args.per_scenario} 실제 1등 조합 포함 여부",
            ]
        )
        for scenario in SCENARIOS:
            lines.append(
                f"- {scenario.name}: {'YES' if result.scenario_contains_actual[scenario.name] else 'NO'} "
                f"({scenario.description})"
            )
        lines.append(
            f"- 전체 시나리오 후보 {total_scenario_candidates}개 안 실제 1등 조합 포함 "
            f"{'YES' if found_any else 'NO'}"
        )

        lines.extend(
            [
                "",
                "최종 Top 50 포트폴리오",
                "- 쿼터: "
                + ", ".join(f"{name}={quota}" for name, quota in DEFAULT_PORTFOLIO_QUOTAS.items()),
                "- 당첨 등수 집계: "
                + ", ".join(
                    f"{rank}={result.final_portfolio_prizes.get(rank, 0)}개"
                    for rank in ("1등", "2등", "3등")
                ),
            ]
        )
        for rank, (score, numbers, scenario_name) in enumerate(result.final_portfolio[: args.show], start=1):
            lines.append(f"- {rank}위 [{scenario_name}] {list(numbers)} score={score:.3f}")

        lines.extend(["", "개선 방식 시나리오별 1위"])
        for scenario in SCENARIOS:
            if not result.scenario_top[scenario.name]:
                lines.append(f"- {scenario.name}: 후보 없음")
                continue
            score, numbers = result.scenario_top[scenario.name][0]
            lines.append(f"- {scenario.name}: {list(numbers)} score={score:.3f}")

    if args.show_contribution:
        lines.extend(["", *contribution_lines(engine)])

    lines.extend(
        [
            "",
            "시뮬레이션 결론",
            "- 현재 구조는 Top 50 포트폴리오 안에서 1~3등 조합 개수를 기준으로 평가합니다.",
            "- 패턴 적중이 있어도 실제 등수 조합이 나오지 않으면 해당 가중치 조합은 실전성이 낮은 것으로 봅니다.",
            "- 번호쌍/출현주기 점수는 추가됐지만, 등수 집계가 개선되지 않으면 가중치 재조정 대상입니다.",
            "- 다음 개선 판단은 단일 회차가 아니라 여러 회차의 Top 50 내 1~3등 발생 횟수로 해야 합니다.",
        ]
    )

    return "\n".join(lines)


def parse_top_k(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest prediction for a Lotto round.")
    parser.add_argument("--csv", default="lotto_winners_2020_2026.csv", help="Input CSV path.")
    parser.add_argument("--target-round", type=int, default=1228, help="Round to predict.")
    parser.add_argument("--mode", choices=["fast", "full", "sample"], default="full", help="Backtest mode.")
    parser.add_argument("--sample-size", type=int, default=200000, help="Random sample size for sample mode.")
    parser.add_argument("--top", type=int, default=100, help="Number of top candidates to retain.")
    parser.add_argument("--show", type=int, default=10, help="Number of final portfolio candidates to print.")
    parser.add_argument("--top-k", type=parse_top_k, default=list(DEFAULT_TOP_K), help="Comma-separated Top-K thresholds.")
    parser.add_argument("--per-scenario", type=int, default=50, help="Scenario portfolio size.")
    parser.add_argument("--recent-window", type=int, default=50, help="Recent draw window for number scores.")
    parser.add_argument("--max-conditions", type=int, default=3, help="Maximum IF conditions.")
    parser.add_argument("--min-support", type=int, default=15, help="Minimum rule support.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum rule confidence.")
    parser.add_argument("--min-lift", type=float, default=1.25, help="Minimum rule lift.")
    parser.add_argument("--show-contribution", action="store_true", help="Show actual-combination score contribution.")
    return parser.parse_args()


def main() -> None:
    print(build_report(parse_args()))


if __name__ == "__main__":
    main()
