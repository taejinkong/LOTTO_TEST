#!/usr/bin/env python3
"""구간(분절점 사이) 내부에서 규칙 항목별 예측 정확도 측정.

가설:
  1~10 미출현(LOW_1_10=0) 회차를 분절점으로 보고, 분절점과 분절점 사이를
  하나의 구간으로 묶으면, 같은 구간 안에서는 규칙성(연속성)이 유지되어
  항목별 예측 정확도가 전체/구간경계보다 높을 것이다.

측정:
  각 전환(회차 i -> i+1)을 분류
    intra : i와 i+1이 같은 구간 (구간 내부 전환)
    cross : i+1이 분절점 (구간 경계를 넘는 전환)
  각 규칙 항목(OE/SUM/AC/...)에 대해 '직전 값이 다음에도 유지되는 비율'을
  intra/cross/전체로 측정하고, 무작위 기대일치율 대비 lift로 비교한다.

예시:
  python3 segment_rule_accuracy.py
  python3 segment_rule_accuracy.py --csv lotto_winners_2020_2026.csv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

from lotto_analyzer import (
    Draw,
    ac_bin,
    color_pattern,
    consecutive_pattern,
    count_g1,
    first_number_bin,
    gap_pattern,
    high_count,
    load_draws,
    low_count,
    odd_even,
    sum_bin,
)


SEP = "=" * 70

# 규칙 항목(=family) 정의. LOW/COLOR는 분절점 정의와 직접 엮여 cross가 왜곡됨(주석).
FAMILIES = {
    "OE":     ("홀짝",     odd_even),
    "SUM":    ("합계구간", sum_bin),
    "AC":     ("AC값",     ac_bin),
    "HIGH":   ("31-45개수", high_count),
    "CONSEC": ("연속쌍",   consecutive_pattern),
    "GAP":    ("간격",     gap_pattern),
    "FIRST":  ("첫수",     first_number_bin),
    "LOW":    ("1-10개수", low_count),     # ※ 분절점 정의와 직접 연관
    "COLOR":  ("색상분포", color_pattern), # ※ G1 포함, 분절점과 연관 + 카테고리 과다
}

# cross 비교가 분절점 정의에 오염되는 항목
# FIRST: 첫수>=11 <=> LOW=0 <=> 분절점 (623회 데이터에서 136/136 완전 동치 확인)
CONTAMINATED = {"LOW", "COLOR", "FIRST"}


def segment_ids(draws: list[Draw]) -> list[int]:
    """분절점(LOW=0, 단 첫 회차 제외)을 새 구간 시작으로 구간 번호 부여."""
    ids: list[int] = []
    cur = 0
    for i, d in enumerate(draws):
        if i > 0 and count_g1(d.numbers) == 0:
            cur += 1
        ids.append(cur)
    return ids


def expected_match_rate(values: list[str]) -> float:
    """해당 항목 값 분포에서 무작위로 두 회차가 같은 값일 기대확률 = sum(p_k^2)."""
    counts = Counter(values)
    total = len(values)
    return sum((c / total) ** 2 for c in counts.values()) if total else 0.0


def prop_ztest(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """두 비율 차이의 z (intra vs cross)."""
    if n_a == 0 or n_b == 0:
        return 0.0
    pa, pb = hits_a / n_a, hits_b / n_b
    pooled = (hits_a + hits_b) / (n_a + n_b)
    se = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    return (pa - pb) / se if se else 0.0


def analyze(draws: list[Draw]) -> list[str]:
    seg = segment_ids(draws)
    n_segments = seg[-1] + 1

    # 항목별 전환 집계
    # bucket: 'intra' / 'cross' / 'all'
    hits: dict[str, dict[str, int]] = {f: defaultdict(int) for f in FAMILIES}
    counts: dict[str, dict[str, int]] = {f: defaultdict(int) for f in FAMILIES}
    all_values: dict[str, list[str]] = {f: [] for f in FAMILIES}

    for f, (_, fn) in FAMILIES.items():
        all_values[f] = [fn(d.numbers) for d in draws]

    for i in range(len(draws) - 1):
        bucket = "intra" if seg[i] == seg[i + 1] else "cross"
        for f, (_, fn) in FAMILIES.items():
            v_now = fn(draws[i].numbers)
            v_next = fn(draws[i + 1].numbers)
            match = int(v_now == v_next)
            for b in (bucket, "all"):
                counts[f][b] += 1
                hits[f][b] += match

    n_intra = counts["OE"]["intra"]
    n_cross = counts["OE"]["cross"]

    lines = [
        SEP,
        "구간 내부 규칙 항목 예측 정확도 (직전 값 유지 = persistence)",
        SEP,
        f"전체 회차: {len(draws)}  구간 수: {n_segments}  "
        f"평균 구간 길이: {len(draws)/n_segments:.1f}",
        f"전환 분류:  intra(구간내부) {n_intra}회   cross(구간경계) {n_cross}회",
        "",
        "각 항목: 실제 유지율 / 무작위 기대유지율 = lift (1.0=무작위, >1=지속성 있음)",
        "intra-z = intra와 cross 유지율 차이의 z값 (|z|>=2 유의)",
        "",
        f"  {'항목':<11} {'intra유지':>9} {'lift':>5}   {'cross유지':>9} {'lift':>5}   "
        f"{'전체':>7} {'lift':>5}  {'z':>5}",
        "  " + "─" * 74,
    ]

    rows = []
    for f, (label, _) in FAMILIES.items():
        exp = expected_match_rate(all_values[f])
        if exp == 0:
            continue
        r_intra = hits[f]["intra"] / counts[f]["intra"] if counts[f]["intra"] else 0
        r_cross = hits[f]["cross"] / counts[f]["cross"] if counts[f]["cross"] else 0
        r_all = hits[f]["all"] / counts[f]["all"] if counts[f]["all"] else 0
        l_intra, l_cross, l_all = r_intra / exp, r_cross / exp, r_all / exp
        z = prop_ztest(hits[f]["intra"], counts[f]["intra"], hits[f]["cross"], counts[f]["cross"])
        rows.append((f, label, r_intra, l_intra, r_cross, l_cross, r_all, l_all, z))

    # 다중비교 보정: 검정한 항목 수만큼 Bonferroni. 임계 z ≈ 2.7 (alpha 0.05/9)
    n_tests = len(rows)
    crit_z = 2.69 if n_tests >= 9 else 2.39  # 0.05/9 양측 ≈ 2.69, 0.05/6 ≈ 2.39

    # intra lift 높은 순으로 정렬
    rows.sort(key=lambda r: r[3], reverse=True)
    for f, label, r_intra, l_intra, r_cross, l_cross, r_all, l_all, z in rows:
        # 보정 후에도 유의한 것만 ** / 보정 전 유의는 * / 오염 항목은 z 신뢰 불가
        if f in CONTAMINATED:
            mark = ""
        elif abs(z) >= crit_z:
            mark = "**"
        elif abs(z) >= 2:
            mark = "*"
        else:
            mark = ""
        warn = " ⚠오염" if f in CONTAMINATED else ""
        lines.append(
            f"  {label:<10} {r_intra:>8.1%} {l_intra:>5.2f}   "
            f"{r_cross:>8.1%} {l_cross:>5.2f}   "
            f"{r_all:>6.1%} {l_all:>5.2f}  {z:>5.1f}{mark}{warn}"
        )

    lines += [
        "",
        "⚠오염 = LOW/COLOR/FIRST는 분절점 정의(1-10=0)와 동치/연관되어 z가 자기참조 인공물",
        f"  (첫수>=11 <=> 1-10미출현 <=> 분절점, 데이터에서 완전 동치)",
        f"* = 보정전 유의(|z|>=2),  ** = 다중비교 보정후에도 유의(|z|>={crit_z:.1f}, {n_tests}개 검정)",
        "",
        "해석 가이드:",
        "  - intra-lift 가 1.0 근처면: 구간 내부에서도 항목이 무작위처럼 행동 → 예측 못 씀",
        "  - 오염 아닌 항목이 ** 를 받아야 비로소 '구간 내 지속성'이 진짜 신호",
        "  - * 만 받은 항목은 다중비교(9개 동시검정) 고려 시 우연일 가능성 높음",
    ]
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="구간 내 규칙 항목 예측 정확도 측정.")
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    draws = load_draws(Path(args.csv))
    print(f"데이터: {draws[0].round_no}~{draws[-1].round_no}회 ({len(draws)}회)\n")
    print("\n".join(analyze(draws)))


if __name__ == "__main__":
    main()
