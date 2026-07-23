#!/usr/bin/env python3
"""Build the static data payload used by the LOTTO PWA."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean

from lotto_analyzer import Draw, odd_even
from lotto_prediction import (
    LEGACY_MODEL_NAME,
    LEGACY_SCORE_WEIGHTS,
    MODEL_NAME,
    RARE_AFTER_RARE_PENALTY,
    RARE_OE_PATTERNS,
    SCORE_WEIGHTS,
    SCENARIOS,
    build_cycle_scores,
    build_distribution_scores,
    build_number_scores,
    build_pair_scores,
    build_raw_target_scores,
    normalize_target_scores,
)


CSV_PATH = Path("lotto_winners_2020_2026.csv")
OUT_PATH = Path("app/data.js")
RECENT_WINDOW = 50
RULE_CONFIG = {
    "maxConditions": 2,
    "minSupport": 15,
    "minConfidence": 0.45,
    "minLift": 1.25,
}
RULE_SAMPLE_SIZE = 200_000
RULE_SAMPLE_SEED = 20260723


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
    raw_conditional_scores = build_raw_target_scores(
        draws,
        max_conditions=RULE_CONFIG["maxConditions"],
        min_support=RULE_CONFIG["minSupport"],
        min_confidence=RULE_CONFIG["minConfidence"],
        min_lift=RULE_CONFIG["minLift"],
    )
    distribution_scores = build_distribution_scores(draws)
    conditional_scores = normalize_target_scores(raw_conditional_scores)
    number_scores = build_number_scores(draws, RECENT_WINDOW)
    pair_scores = build_pair_scores(draws, RECENT_WINDOW)
    cycle_scores = build_cycle_scores(draws)
    return {
        "name": MODEL_NAME,
        "legacyName": LEGACY_MODEL_NAME,
        "nextRound": rows[-1].round_no + 1,
        "recentWindow": RECENT_WINDOW,
        "weights": SCORE_WEIGHTS,
        "legacyWeights": LEGACY_SCORE_WEIGHTS,
        "rareAfterRarePenalty": RARE_AFTER_RARE_PENALTY,
        "rareOePatterns": sorted(RARE_OE_PATTERNS),
        "previousOe": odd_even(rows[-1].numbers),
        "previousNumbers": list(rows[-1].numbers),
        "ruleConfig": RULE_CONFIG,
        "activeConditionalTargets": len(conditional_scores),
        "distributionScores": {
            feature: round(score, 8)
            for feature, score in sorted(distribution_scores.items())
        },
        "conditionalScores": {
            feature: round(score, 8)
            for feature, score in sorted(conditional_scores.items())
        },
        "numberScores": {
            str(number): round(number_scores[number], 8)
            for number in range(1, 46)
        },
        "pairScores": {
            f"{left}-{right}": round(pair_scores[(left, right)], 8)
            for left, right in combinations(range(1, 46), 2)
        },
        "cycleScores": {str(number): round(cycle_scores[number], 8) for number in range(1, 46)},
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "featureGroups": [
            {"key": "pattern", "label": "기본 패턴", "rules": "홀짝·번호대·끝수·합계·저/고번호·연속·AC·간격·첫 수"},
            {"key": "transition", "label": "전이 패턴", "rules": "직전 회차 이월수·이웃수"},
            {"key": "conditional", "label": "조건부 규칙", "rules": "IF 이전 패턴 THEN 다음 패턴"},
            {"key": "number", "label": "개별 번호", "rules": "전체·최근 빈도와 현재 미출현"},
            {"key": "pair", "label": "번호쌍", "rules": "전체·최근 동반 출현"},
            {"key": "cycle", "label": "출현주기", "rules": "평균 간격 대비 현재 미출현"},
        ],
    }


def max_consecutive_run(numbers: tuple[int, ...]) -> int:
    longest = current = 1
    for previous, number in zip(numbers, numbers[1:]):
        current = current + 1 if number == previous + 1 else 1
        longest = max(longest, current)
    return longest


def experimental_rule_checks(numbers: tuple[int, ...], previous: tuple[int, ...]) -> dict[str, bool]:
    odd_count = sum(number % 2 for number in numbers)
    bands = {(number - 1) // 10 for number in numbers}
    previous_set = set(previous)
    return {
        "sum_100_175": 100 <= sum(numbers) <= 175,
        "mixed_odd_even": 0 < odd_count < 6,
        "tail_sum_13_38": 13 <= sum(number % 10 for number in numbers) <= 38,
        "no_three_run": max_consecutive_run(numbers) < 3,
        "three_bands": len(bands) >= 3,
        "carry_max_two": sum(number in previous_set for number in numbers) <= 2,
    }


def build_rule_analysis(rows: list[Row]) -> dict:
    rules = [
        {
            "id": "sum_100_175",
            "label": "합계 100~175",
            "description": "번호 6개의 합계가 100~175인 조합만 통과",
        },
        {
            "id": "mixed_odd_even",
            "label": "홀짝 극단 제외",
            "description": "전부 홀수 또는 전부 짝수인 조합 제외",
        },
        {
            "id": "tail_sum_13_38",
            "label": "끝수 합계 13~38",
            "description": "각 번호 일의 자리 합계가 13~38인 조합만 통과",
        },
        {
            "id": "no_three_run",
            "label": "3연속 번호 제외",
            "description": "예: 7·8·9처럼 세 번호가 이어지는 조합 제외",
        },
        {
            "id": "three_bands",
            "label": "3개 번호대 이상",
            "description": "1~10·11~20·21~30·31~40·41~45 중 최소 3개 번호대 사용",
        },
        {
            "id": "carry_max_two",
            "label": "직전 번호 2개 이하",
            "description": "직전 회차 당첨번호와 겹치는 번호를 최대 2개로 제한",
        },
    ]
    counts = {rule["id"]: 0 for rule in rules}
    combined_count = 0
    historical_rows = rows[1:]
    for previous, row in zip(rows, historical_rows):
        checks = experimental_rule_checks(row.numbers, previous.numbers)
        for rule_id, passed in checks.items():
            counts[rule_id] += int(passed)
        combined_count += int(all(checks.values()))

    rng = random.Random(RULE_SAMPLE_SEED)
    sample_counts = {rule["id"]: 0 for rule in rules}
    sample_combined_count = 0
    latest_numbers = rows[-1].numbers
    population = list(range(1, 46))
    for _ in range(RULE_SAMPLE_SIZE):
        numbers = tuple(sorted(rng.sample(population, 6)))
        checks = experimental_rule_checks(numbers, latest_numbers)
        for rule_id, passed in checks.items():
            sample_counts[rule_id] += int(passed)
        sample_combined_count += int(all(checks.values()))

    historical_total = len(historical_rows)
    for rule in rules:
        rule_id = rule["id"]
        rule["historicalPassRate"] = round(counts[rule_id] / historical_total * 100, 1)
        rule["candidatePassRate"] = round(sample_counts[rule_id] / RULE_SAMPLE_SIZE * 100, 1)
    return {
        "rules": rules,
        "historicalRounds": f"{historical_rows[0].round_no}~{historical_rows[-1].round_no}회",
        "historicalCount": historical_total,
        "candidateSampleSize": RULE_SAMPLE_SIZE,
        "candidateSampleSeed": RULE_SAMPLE_SEED,
        "combinedHistoricalPassRate": round(combined_count / historical_total * 100, 1),
        "combinedCandidatePassRate": round(sample_combined_count / RULE_SAMPLE_SIZE * 100, 1),
        "warning": (
            "통과율은 규칙의 예측력이 아니라 후보를 얼마나 남기거나 버리는지를 보여준다. "
            "과거 당첨조합을 제외한 규칙이 미래 당첨조합도 제외할 수 있다."
        ),
    }


def validation_summary() -> dict:
    return {
        "rounds": "1204~1233회",
        "count": 30,
        "sampleSize": 20_000,
        "metric": "실제 당첨 조합의 표본 추정 백분위(낮을수록 우수)",
        "variants": [
            {"label": "페어+주기 v1", "averagePercentile": 40.8, "medianPercentile": 37.7},
            {"label": "전체 규칙 통합 v2", "averagePercentile": 42.2, "medianPercentile": 38.2},
            {
                "label": "순수 무작위 기대 기준",
                "averagePercentile": 50.0,
                "medianPercentile": 50.0,
                "baseline": True,
            },
        ],
        "randomMean95Range": [39.7, 60.3],
        "top100kHits": 0,
        "source": "backtest_integrated_v2_30.txt",
        "currentModelStatus": (
            "동일한 최근 30회·회차당 20,000개 표본에서 v2가 v1보다 우수하지 않았습니다. "
            "전체 규칙 사용은 분석 범위 확장이지 성능 개선을 뜻하지 않습니다."
        ),
    }


def insights() -> list[dict]:
    return [
        {
            "title": "전체 분석 규칙을 조합에 반영",
            "tone": "good",
            "body": (
                "통합 규칙 v2는 기본 분포, 이월·이웃 전이, 조건부 연계, 개별 번호, 번호쌍, "
                "출현주기를 모두 0~1 범위로 정규화해 합산한다."
            ),
        },
        {
            "title": "검증형 모델도 선택 가능",
            "tone": "info",
            "body": (
                "번호 생성 화면에서 전체 규칙 통합 v2와 기존 페어+주기 v1을 바꿔 비교할 수 있다. "
                "Python 생성기의 기본값도 통합 규칙 v2이며 두 구현은 같은 점수식을 사용한다."
            ),
        },
        {
            "title": "규칙 추가는 당첨 확률 증가를 뜻하지 않음",
            "tone": "warn",
            "body": (
                "최근 30회 표본 검증에서 통합 v2 평균 백분위는 42.2%, 페어+주기 v1은 40.8%였고 "
                "둘 다 Top 100,000 진입은 0회였다. 모든 단일 조합의 1등 확률은 1/8,145,060으로 같다."
            ),
        },
        {
            "title": "실험 필터는 기본값으로 강제하지 않음",
            "tone": "warn",
            "body": (
                "합계·홀짝·끝수 합계·연속번호·번호대·직전 번호 규칙은 선택형 실험 필터다. "
                "각 규칙의 과거 당첨조합 생존율을 확인한 뒤 사용하며, 핫번호를 모든 조합에 강제하지 않는다."
            ),
        },
        {
            "title": "예측은 잠그고 추첨 뒤 정확히 채점",
            "tone": "good",
            "body": (
                "번호 생성 결과를 시드·모델·조건과 함께 로컬 예측 원장에 잠글 수 있다. "
                "새 회차 데이터가 들어오면 보너스 번호를 포함한 실제 1~5등 기준으로 자동 채점한다."
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
        "ruleAnalysis": build_rule_analysis(rows),
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
