#!/usr/bin/env python3
"""Generate Lotto candidates with the same pair-cycle model as the web app."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from lotto_analyzer import load_draws, odd_even
from lotto_prediction import (
    MODEL_NAME,
    SCENARIOS,
    build_cycle_scores,
    build_pair_scores,
    canonical_score_parts,
    matches_scenario,
)


def parse_number_set(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    try:
        return {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise SystemExit(f"번호 형식 오류: {raw}") from exc


def has_consecutive(numbers: tuple[int, ...]) -> bool:
    return any(right - left == 1 for left, right in zip(numbers, numbers[1:]))


def validate(fixed: set[int], excluded: set[int], count: int, pool_size: int) -> None:
    if fixed & excluded:
        raise SystemExit(f"고정수와 제외수가 겹칩니다: {sorted(fixed & excluded)}")
    if len(fixed) > 6:
        raise SystemExit("고정수는 최대 6개입니다.")
    invalid = {number for number in fixed | excluded if not 1 <= number <= 45}
    if invalid:
        raise SystemExit(f"1~45 범위를 벗어난 번호: {sorted(invalid)}")
    if count < 1 or pool_size < count:
        raise SystemExit("--count는 1 이상이고 --pool-size보다 클 수 없습니다.")
    available = 45 - len(fixed | excluded)
    if 6 - len(fixed) > available:
        raise SystemExit("제외수가 너무 많아 6개 조합을 만들 수 없습니다.")


def generate_candidates(
    *,
    count: int,
    pool_size: int,
    fixed: set[int],
    excluded: set[int],
    scenario_name: str,
    no_consecutive: bool,
    seed: int,
    history: set[tuple[int, ...]],
    pair_scores: dict[tuple[int, int], float],
    cycle_scores: dict[int, float],
    previous_oe: str,
) -> list[tuple[dict[str, float], tuple[int, ...]]]:
    rng = random.Random(seed)
    selectable = [number for number in range(1, 46) if number not in fixed | excluded]
    needed = 6 - len(fixed)
    scenario = next((item for item in SCENARIOS if item.name == scenario_name), None)
    found: dict[tuple[int, ...], dict[str, float]] = {}
    max_attempts = max(pool_size * 30, 20_000)
    for _ in range(max_attempts):
        if len(found) >= pool_size:
            break
        picks = rng.sample(selectable, needed) if needed else []
        numbers = tuple(sorted([*fixed, *picks]))
        if numbers in found or numbers in history:
            continue
        if no_consecutive and has_consecutive(numbers):
            continue
        if scenario is not None and not matches_scenario(numbers, scenario):
            continue
        found[numbers] = canonical_score_parts(numbers, pair_scores, cycle_scores, previous_oe)
    if not found:
        raise RuntimeError("조건을 만족하는 조합을 찾지 못했습니다. 제약을 완화하세요.")
    ranked = sorted(((parts, numbers) for numbers, parts in found.items()), key=lambda item: item[0]["total"], reverse=True)
    return ranked[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="페어·주기 점수 기반 로또 후보 생성기")
    parser.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    parser.add_argument("--count", type=int, default=6, help="출력 조합 수")
    parser.add_argument("--pool-size", type=int, default=30_000, help="점수화할 무작위 후보 수")
    parser.add_argument("--scenario", choices=["all", *(item.name for item in SCENARIOS)], default="all")
    parser.add_argument("--fix", default="", help="고정수, 쉼표 구분")
    parser.add_argument("--exclude", default="", help="제외수, 쉼표 구분")
    parser.add_argument("--no-consecutive", action="store_true", help="연속번호가 있는 조합 제외")
    parser.add_argument("--recent-window", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed = parse_number_set(args.fix)
    excluded = parse_number_set(args.exclude)
    validate(fixed, excluded, args.count, args.pool_size)
    draws = load_draws(Path(args.csv))
    pair_scores = build_pair_scores(draws, args.recent_window)
    cycle_scores = build_cycle_scores(draws)
    ranked = generate_candidates(
        count=args.count,
        pool_size=args.pool_size,
        fixed=fixed,
        excluded=excluded,
        scenario_name=args.scenario,
        no_consecutive=args.no_consecutive,
        seed=args.seed,
        history={draw.numbers for draw in draws},
        pair_scores=pair_scores,
        cycle_scores=cycle_scores,
        previous_oe=odd_even(draws[-1].numbers),
    )
    print(f"기준: {draws[0].round_no}~{draws[-1].round_no}회 | 모델: {MODEL_NAME} | 대상: {draws[-1].round_no + 1}회")
    print(f"후보 풀: {args.pool_size:,}개 | 시나리오: {args.scenario} | 1등 확률: 각 1/8,145,060")
    for rank, (parts, numbers) in enumerate(ranked, 1):
        print(
            f"{rank:>2}. {list(numbers)} score={parts['total']:.6f} "
            f"pair={parts['pair']:.4f} cycle={parts['cycle']:.4f} penalty={parts['penalty']:.2f}"
        )
    print("주의: 백테스트 순위 점수이며 당첨을 보장하거나 개별 조합 확률을 높이지 않습니다.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
