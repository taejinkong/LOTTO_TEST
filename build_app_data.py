#!/usr/bin/env python3
"""Build the static data payload used by the LOTTO PWA."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean

from lotto_analyzer import Draw, odd_even
from lotto_prediction import (
    MODEL_NAME,
    RARE_AFTER_RARE_PENALTY,
    RARE_OE_PATTERNS,
    SCORE_WEIGHTS,
    SCENARIOS,
    build_cycle_scores,
    build_pair_scores,
)


CSV_PATH = Path("lotto_winners_2020_2026.csv")
OUT_PATH = Path("app/data.js")
RECENT_WINDOW = 50


@dataclass(frozen=True)
class Row:
    round_no: int
    date: str
    numbers: tuple[int, ...]
    bonus: int
    winners: int
    amount: int

    def as_draw(self) -> Draw:
        return Draw(self.round_no, self.date, self.numbers, self.bonus)


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for record in csv.DictReader(file):
            rows.append(
                Row(
                    round_no=int(record["회차"].replace("회", "")),
                    date=record["추첨일"].strip(),
                    numbers=tuple(sorted(int(record[f"번호{i}"]) for i in range(1, 7))),
                    bonus=int(record["보너스"]),
                    winners=int(record["1등 당첨자수(명)"]),
                    amount=int(record["1등 당첨금액(원)"]),
                )
            )
    return sorted(rows, key=lambda row: row.round_no)


def build_stats(rows: list[Row]) -> dict:
    frequency = {number: 0 for number in range(1, 46)}
    odd_even_counts: dict[str, int] = {}
    sum_bins: dict[str, int] = {}
    for row in rows:
        for number in row.numbers:
            frequency[number] += 1
        oe = odd_even(row.numbers).removeprefix("OE=")
        odd_even_counts[oe] = odd_even_counts.get(oe, 0) + 1
        lower = (sum(row.numbers) // 20) * 20
        label = f"{lower}-{lower + 19}"
        sum_bins[label] = sum_bins.get(label, 0) + 1
    return {
        "numberFreq": frequency,
        "oddEven": dict(sorted(odd_even_counts.items())),
        "sumBins": [
            {"label": label, "count": sum_bins[label]}
            for label in sorted(sum_bins, key=lambda item: int(item.split("-")[0]))
        ],
        "averageSum": round(mean(sum(row.numbers) for row in rows), 1),
    }


def build_prediction(rows: list[Row]) -> dict:
    draws = [row.as_draw() for row in rows]
    pair_scores = build_pair_scores(draws, RECENT_WINDOW)
    cycle_scores = build_cycle_scores(draws)
    return {
        "name": MODEL_NAME,
        "nextRound": rows[-1].round_no + 1,
        "recentWindow": RECENT_WINDOW,
        "weights": SCORE_WEIGHTS,
        "rareAfterRarePenalty": RARE_AFTER_RARE_PENALTY,
        "rareOePatterns": sorted(RARE_OE_PATTERNS),
        "previousOe": odd_even(rows[-1].numbers),
        "pairScores": {
            f"{left}-{right}": round(pair_scores[(left, right)], 8)
            for left, right in combinations(range(1, 46), 2)
        },
        "cycleScores": {str(number): round(cycle_scores[number], 8) for number in range(1, 46)},
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
    }


def validation_summary() -> dict:
    return {
        "rounds": "1199~1228회",
        "count": 30,
        "metric": "실제 당첨 조합 백분위(낮을수록 우수)",
        "variants": [
            {"label": "기존 전체 모델", "averagePercentile": 50.0, "medianPercentile": 46.5},
            {"label": "조건부 패턴 제거", "averagePercentile": 40.8, "medianPercentile": 38.2},
            {"label": "현재 페어+주기", "averagePercentile": 39.1, "medianPercentile": 36.4},
        ],
        "top100kHits": 0,
        "source": "backtest_ablation_30.txt, backtest_stage2_30.txt",
    }


def insights() -> list[dict]:
    return [
        {
            "title": "현재 모델은 페어와 출현주기만 사용",
            "tone": "good",
            "body": (
                "30회 워크포워드 백테스트에서 조건부 패턴과 개별 번호 빈도는 순위를 악화시켰다. "
                "그래서 현재 생성기는 번호쌍의 전체·최근 동반 빈도와 번호별 출현주기 점수만 합산한다."
            ),
        },
        {
            "title": "Python과 웹의 점수식이 동일",
            "tone": "info",
            "body": (
                "두 생성기 모두 페어 평균×0.14 + 주기 평균×0.08을 사용하고, 직전과 후보가 모두 "
                "희귀 홀짝 패턴이면 0.35를 감점한다. 고정·제외·시나리오는 점수가 아닌 후보공간 제약이다."
            ),
        },
        {
            "title": "백테스트 개선은 당첨 예측 증명이 아님",
            "tone": "warn",
            "body": (
                "현재 모델의 평균 백분위는 39.1%였지만 30회 중 Top 100,000 진입은 0회였다. "
                "모든 단일 조합의 1등 확률은 여전히 1/8,145,060이며 생성 결과는 보장값이 아니다."
            ),
        },
    ]


def main() -> None:
    rows = load_rows(CSV_PATH)
    latest = rows[-1]
    payload = {
        "meta": {
            "firstRound": rows[0].round_no,
            "lastRound": latest.round_no,
            "count": len(rows),
            "latest": {
                "round": latest.round_no,
                "date": latest.date,
                "nums": list(latest.numbers),
                "bonus": latest.bonus,
                "winners": latest.winners,
                "amountEok": round(latest.amount / 1e8, 1),
            },
            "builtFrom": CSV_PATH.name,
        },
        "draws": [
            [row.round_no, row.date, *row.numbers, row.bonus, row.winners, round(row.amount / 1e8, 1)]
            for row in rows
        ],
        "stats": build_stats(rows),
        "prediction": build_prediction(rows),
        "validation": validation_summary(),
        "insights": insights(),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    javascript = "// Generated by build_app_data.py. Do not edit directly.\n"
    javascript += "window.LOTTO = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    OUT_PATH.write_text(javascript, encoding="utf-8")
    print(f"생성: {OUT_PATH}")
    print(f"  회차: {rows[0].round_no}~{latest.round_no} ({len(rows)}회)")
    print(f"  모델: {MODEL_NAME}, 다음 대상 {latest.round_no + 1}회")
    print(f"  최신: {latest.round_no}회 {list(latest.numbers)}+{latest.bonus}, 당첨 {latest.winners}명")


if __name__ == "__main__":
    main()
