#!/usr/bin/env python3
"""Generate scenario-diversified Lotto candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from lotto_analyzer import load_draws, odd_even
from lotto_prediction import SCENARIOS, build_prediction_inputs, top_candidates_by_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scenario-based Lotto candidate numbers.")
    parser.add_argument("--csv", default="lotto_winners_2020_2026.csv", help="Input CSV path.")
    parser.add_argument("--per-scenario", type=int, default=5, help="Candidates to print per scenario.")
    parser.add_argument("--recent-window", type=int, default=50, help="Recent draw window for number scores.")
    parser.add_argument("--max-conditions", type=int, default=2, help="Maximum IF conditions.")
    parser.add_argument("--min-support", type=int, default=15, help="Minimum rule support.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum rule confidence.")
    parser.add_argument("--min-lift", type=float, default=1.25, help="Minimum rule lift.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draws = load_draws(Path(args.csv))
    distribution_scores, target_scores, number_scores, pair_scores, cycle_scores = build_prediction_inputs(
        training_draws=draws,
        max_conditions=args.max_conditions,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        recent_window=args.recent_window,
    )
    candidates = top_candidates_by_scenario(
        target_scores,
        number_scores,
        args.per_scenario,
        pair_scores=pair_scores,
        cycle_scores=cycle_scores,
        previous_oe=odd_even(draws[-1].numbers),
        distribution_scores=distribution_scores,
        previous_numbers=draws[-1].numbers,
    )

    latest = draws[-1]
    print(f"기준 데이터: {draws[0].round_no}회~{latest.round_no}회, 총 {len(draws)}회")
    print(f"최신 회차: {latest.round_no}회 {list(latest.numbers)} + 보너스 {latest.bonus}")
    print()
    print("시나리오별 후보")
    for scenario in SCENARIOS:
        print(f"\n[{scenario.name}] {scenario.description}")
        for rank, (score, numbers) in enumerate(candidates[scenario.name][: args.per_scenario], start=1):
            print(f"- {rank}위 {list(numbers)} score={score:.3f}")


if __name__ == "__main__":
    main()
