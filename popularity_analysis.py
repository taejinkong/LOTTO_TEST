#!/usr/bin/env python3
"""당첨자 수 ↔ 번호 특성 상관분석 — 인간 구매 심리 편향 탐지.

추첨은 무작위(7회 검증 완료)지만 '그 조합을 몇 명이 샀는가'는 무작위가 아니다.
사람들은 생일·패턴·인기번호로 편향되게 산다. 이 도구는 당첨번호의 특성으로
1등 당첨자 수를 얼마나 설명할 수 있는지 측정한다.

목표: 1등 확률(고정)이 아니라 '당첨 시 당첨자 수'를 줄이는 = 기대 수령액을
높이는 조합 특성을 찾는다.

타겟:
  winners      : 1등 당첨자 수 (원자료, 판매량 변동 포함)
  winners_adj  : 판매량 보정 당첨자 수 (풀=당첨자수×1인당금액으로 정규화)

특성(인간 선호 가설):
  birthday   1~31 개수 (날짜 편향)
  month      1~12 개수 (월 편향)
  big        32~45 개수 (생일로 못 만드는 비인기 영역)
  consec     연속쌍 수 (시각적 패턴 선호)
  max_run    최대 연속 길이
  same_tail  끝수 중복 개수
  sum_mid    합계가 중앙(120~160)이면 1
  spread     번호 범위(max-min)
  has_lucky7 7 포함 여부

예시:
  python3 popularity_analysis.py
  python3 popularity_analysis.py --recent 200
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
from statistics import mean, pstdev


SEP = "=" * 70


@dataclass
class Row:
    round_no: int
    numbers: tuple[int, ...]
    bonus: int
    winners: int
    amount: int  # 1인당 당첨금액


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            nums = tuple(sorted(int(r[f"번호{k}"]) for k in range(1, 7)))
            rows.append(
                Row(
                    round_no=int(r["회차"].replace("회", "")),
                    numbers=nums,
                    bonus=int(r["보너스"]),
                    winners=int(r["1등 당첨자수(명)"]),
                    amount=int(r["1등 당첨금액(원)"]),
                )
            )
    return sorted(rows, key=lambda x: x.round_no)


# ── 특성 ──────────────────────────────────────────────────────────────────────

def features(nums: tuple[int, ...]) -> dict[str, float]:
    consec = sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1)
    # 최대 연속 길이
    max_run = run = 1
    for a, b in zip(nums, nums[1:]):
        run = run + 1 if b - a == 1 else 1
        max_run = max(max_run, run)
    tails = [n % 10 for n in nums]
    same_tail = len(tails) - len(set(tails))
    total = sum(nums)
    return {
        "birthday":   sum(1 <= n <= 31 for n in nums),
        "month":      sum(1 <= n <= 12 for n in nums),
        "big":        sum(32 <= n <= 45 for n in nums),
        "consec":     consec,
        "max_run":    max_run,
        "same_tail":  same_tail,
        "sum_mid":    1.0 if 120 <= total <= 160 else 0.0,
        "spread":     nums[-1] - nums[0],
        "has_lucky7": 1.0 if 7 in nums else 0.0,
        "all_le31":   1.0 if nums[-1] <= 31 else 0.0,  # 전부 생일권
    }


FEATURE_DESC = {
    "birthday":   "1~31 개수(날짜편향, 클수록 인기예상)",
    "month":      "1~12 개수(월편향)",
    "big":        "32~45 개수(비인기영역, 클수록 당첨자少 예상)",
    "consec":     "연속쌍 수(패턴선호)",
    "max_run":    "최대 연속길이",
    "same_tail":  "끝수 중복 수",
    "sum_mid":    "합계 중앙(120~160)",
    "spread":     "번호 범위(max-min)",
    "has_lucky7": "행운수 7 포함",
    "all_le31":   "전부 31이하(생일권 조합)",
}


# ── 통계 ──────────────────────────────────────────────────────────────────────

def pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """피어슨 상관계수와 근사 t값."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    mx, my = mean(xs), mean(ys)
    sx, sy = pstdev(xs), pstdev(ys)
    if sx == 0 or sy == 0:
        return 0.0, 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    r = cov / (sx * sy)
    r = max(-0.999, min(0.999, r))
    t = r * sqrt((n - 2) / (1 - r * r))
    return r, t


def group_compare(feat_vals: list[float], target: list[float]) -> tuple[float, float, float]:
    """특성 상위/하위 그룹의 타겟 평균 비교. (저평균, 고평균, 차이%)."""
    paired = sorted(zip(feat_vals, target))
    n = len(paired)
    third = max(1, n // 3)
    low = [t for _, t in paired[:third]]
    high = [t for _, t in paired[-third:]]
    lm, hm = mean(low), mean(high)
    diff_pct = (hm - lm) / lm * 100 if lm else 0.0
    return lm, hm, diff_pct


def analyze(rows: list[Row]) -> list[str]:
    # 판매량 보정: 풀 = winners × amount ∝ 판매량. 풀 평균으로 정규화.
    pools = [r.winners * r.amount for r in rows]
    pool_mean = mean(pools)
    winners = [float(r.winners) for r in rows]
    winners_adj = [r.winners * pool_mean / (r.winners * r.amount) for r in rows]  # = pool_mean/amount
    log_winners = [log(w) for w in winners]

    feats: dict[str, list[float]] = {k: [] for k in FEATURE_DESC}
    for r in rows:
        f = features(r.numbers)
        for k in FEATURE_DESC:
            feats[k].append(f[k])

    lines = [
        SEP,
        "당첨자 수 ↔ 번호 특성 상관분석",
        SEP,
        f"회차 수: {len(rows)}  당첨자수 평균: {mean(winners):.1f} (범위 {int(min(winners))}~{int(max(winners))})",
        f"판매량 보정: 풀(당첨자수×1인당금액) 평균 {pool_mean/1e8:.0f}억으로 정규화",
        "",
        "타겟 = log(당첨자수).  r>0: 그 특성↑일수록 당첨자 多(인기).  r<0: 당첨자 少(비인기)",
        "|t|>=2 유의(*), |t|>=2.6 다중보정후 유의(**)",
        "",
        f"  {'특성':<11} {'r':>7} {'t':>7}   {'하위⅓당첨자':>10} {'상위⅓당첨자':>10} {'격차%':>7}",
        "  " + "─" * 66,
    ]

    n_tests = len(FEATURE_DESC)
    crit = 2.6  # ≈ Bonferroni 0.05/10 양측
    results = []
    for k in FEATURE_DESC:
        r, t = pearson(feats[k], log_winners)
        lm, hm, diff = group_compare(feats[k], winners)
        results.append((k, r, t, lm, hm, diff))

    # |r| 큰 순 정렬
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    for k, r, t, lm, hm, diff in results:
        mark = "**" if abs(t) >= crit else ("*" if abs(t) >= 2 else "")
        lines.append(
            f"  {k:<11} {r:>+7.3f} {t:>+7.2f}   {lm:>10.1f} {hm:>10.1f} {diff:>+7.1f}{mark}"
        )

    lines += [
        "",
        "특성 설명:",
    ]
    for k, _, _, _, _, _ in results:
        lines.append(f"  - {k}: {FEATURE_DESC[k]}")

    # 유의 특성 요약
    sig = [(k, r, t) for k, r, t, *_ in results if abs(t) >= 2]
    lines += ["", "─" * 70]
    if sig:
        lines.append("유의한 신호(|t|>=2):")
        for k, r, t in sig:
            direction = "당첨자 많아짐(인기)" if r > 0 else "당첨자 적어짐(비인기)"
            strong = " ★다중보정후도 유의" if abs(t) >= crit else ""
            lines.append(f"  - {k}: r={r:+.3f} → 이 특성 클수록 {direction}{strong}")
        lines += [
            "",
            "→ r<0 인 특성을 만족하는 조합 = 당첨 시 당첨자 적음 = 기대 수령액 ↑",
            "→ 이 특성들을 '비인기 점수'로 묶어 조합 생성에 반영 가능",
        ]
    else:
        lines.append("유의한 신호 없음 — 번호 특성으로 당첨자 수 설명 불가")
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="당첨자 수 ↔ 번호 특성 상관분석.")
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    p.add_argument("--recent", type=int, default=0, help="최근 N회차만 분석 (0=전체)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.csv))
    if args.recent > 0:
        rows = rows[-args.recent:]
    print(f"데이터: {rows[0].round_no}~{rows[-1].round_no}회 ({len(rows)}회)\n")
    print("\n".join(analyze(rows)))


if __name__ == "__main__":
    main()
