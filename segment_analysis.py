#!/usr/bin/env python3
"""분절점(1~10 미출현 회차) 기준 구간/전후 규칙성 분석.

가설:
  1~10 번호가 하나도 안 나온 회차(LOW_1_10=0)를 분절점으로 보고,
  그 전후로 통계적 규칙성이 달라지는지 검증한다.

두 모드:
  segment : 분절점으로 데이터를 구간으로 나눠 구간별 통계 비교
  event   : 모든 분절점을 정렬해 직전 N회 vs 직후 N회 통계 비교 (핵심)

예시:
  python3 segment_analysis.py --mode event --window 3
  python3 segment_analysis.py --mode segment --min-len 4
  python3 segment_analysis.py --mode both
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev

from lotto_analyzer import Draw, load_draws, count_g1


SEP = "=" * 64


# ── 지표 ──────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "odd":      "홀수 개수",
    "sum":      "번호 합계",
    "ac":       "AC값",
    "g1_low":   "1~10 개수",
    "g5_high":  "41~45 개수",
    "high":     "31~45 개수",
    "consec":   "연속번호 쌍",
    "mean_num": "번호 평균값",
}


def ac_value(numbers: tuple[int, ...]) -> int:
    diffs = {b - a for a, b in combinations(numbers, 2)}
    return len(diffs) - 5


def metrics(numbers: tuple[int, ...]) -> dict[str, float]:
    return {
        "odd":      sum(n % 2 for n in numbers),
        "sum":      sum(numbers),
        "ac":       ac_value(numbers),
        "g1_low":   sum(1 <= n <= 10 for n in numbers),
        "g5_high":  sum(41 <= n <= 45 for n in numbers),
        "high":     sum(31 <= n <= 45 for n in numbers),
        "consec":   sum(1 for a, b in zip(numbers, numbers[1:]) if b - a == 1),
        "mean_num": sum(numbers) / 6,
    }


def band_distribution(draws_subset: list[Draw]) -> dict[str, float]:
    """번호대별 평균 출현 개수 (G1~G5)."""
    bands = {"G1": (1, 10), "G2": (11, 20), "G3": (21, 30), "G4": (31, 40), "G5": (41, 45)}
    acc = {label: [] for label in bands}
    for d in draws_subset:
        for label, (lo, hi) in bands.items():
            acc[label].append(sum(lo <= n <= hi for n in d.numbers))
    return {label: mean(vals) if vals else 0.0 for label, vals in acc.items()}


# ── 통계 헬퍼 ─────────────────────────────────────────────────────────────────

def summarize(values: list[float]) -> tuple[float, float, int]:
    """평균, 표준편차, 개수."""
    if not values:
        return 0.0, 0.0, 0
    if len(values) == 1:
        return values[0], 0.0, 1
    return mean(values), pstdev(values), len(values)


def significance(
    a_vals: list[float], b_vals: list[float]
) -> tuple[float, str]:
    """두 집단 평균차의 표준화 효과크기와 유의 표시.

    Welch 표준오차 기반. |차이| / SE 가 2 이상이면 유의(*),
    3 이상이면 강한 유의(**).
    """
    if len(a_vals) < 2 or len(b_vals) < 2:
        return 0.0, ""
    ma, sa, na = summarize(a_vals)
    mb, sb, nb = summarize(b_vals)
    se = sqrt(sa * sa / na + sb * sb / nb)
    if se == 0:
        return 0.0, ""
    z = (mb - ma) / se
    mark = "**" if abs(z) >= 3 else ("*" if abs(z) >= 2 else "")
    return z, mark


# ── 분절점 ────────────────────────────────────────────────────────────────────

def breakpoint_indexes(draws: list[Draw]) -> list[int]:
    return [i for i, d in enumerate(draws) if count_g1(d.numbers) == 0]


# ── EVENT STUDY ───────────────────────────────────────────────────────────────

def run_event_study(draws: list[Draw], window: int) -> list[str]:
    bp = breakpoint_indexes(draws)

    before: dict[str, list[float]] = defaultdict(list)
    after: dict[str, list[float]] = defaultdict(list)
    next1: dict[str, list[float]] = defaultdict(list)

    before_draws: list[Draw] = []
    after_draws: list[Draw] = []
    next1_draws: list[Draw] = []

    for i in bp:
        for j in range(max(0, i - window), i):
            for k, v in metrics(draws[j].numbers).items():
                before[k].append(v)
            before_draws.append(draws[j])
        for j in range(i + 1, min(len(draws), i + 1 + window)):
            for k, v in metrics(draws[j].numbers).items():
                after[k].append(v)
            after_draws.append(draws[j])
        if i + 1 < len(draws):
            for k, v in metrics(draws[i + 1].numbers).items():
                next1[k].append(v)
            next1_draws.append(draws[i + 1])

    baseline: dict[str, list[float]] = defaultdict(list)
    for d in draws:
        for k, v in metrics(d.numbers).items():
            baseline[k].append(v)

    lines = [
        SEP,
        f"EVENT STUDY — 분절점 전후 ±{window}회 규칙성 비교",
        SEP,
        f"분절점(1~10 미출현) 수: {len(bp)}회",
        f"직전 윈도우 표본: {len(before_draws)}회 / 직후: {len(after_draws)}회 / 직후 1회: {len(next1_draws)}회",
        "",
        "지표별 평균 (baseline=전체평균, before=분절점 직전, after=분절점 직후, next=직후 첫 회차)",
        "z = (after - before)/표준오차,  * |z|≥2,  ** |z|≥3",
        "",
        f"  {'지표':<12} {'baseline':>9} {'before':>8} {'after':>8} {'next':>8}   {'after-before':>12} {'z':>6}",
        "  " + "─" * 70,
    ]

    for key, label in METRIC_LABELS.items():
        b_mean = mean(baseline[key])
        bf_mean = mean(before[key]) if before[key] else 0.0
        af_mean = mean(after[key]) if after[key] else 0.0
        n1_mean = mean(next1[key]) if next1[key] else 0.0
        z, mark = significance(before[key], after[key])
        diff = af_mean - bf_mean
        lines.append(
            f"  {label:<11} {b_mean:>9.2f} {bf_mean:>8.2f} {af_mean:>8.2f} {n1_mean:>8.2f}   "
            f"{diff:>+12.2f} {z:>6.1f}{mark}"
        )

    # 번호대 분포: 분절점 직후 첫 회차 vs 전체
    lines += [
        "",
        "번호대별 평균 출현 개수 — 분절점 '직후 첫 회차' vs 전체",
        "  (분절점은 정의상 G1=0 → 직후에 1~10이 반등하는지가 핵심)",
        "",
        f"  {'번호대':<6} {'전체':>8} {'직후1회':>8} {'차이':>8}",
        "  " + "─" * 34,
    ]
    base_band = band_distribution(draws)
    next_band = band_distribution(next1_draws)
    band_names = {"G1": "1-10", "G2": "11-20", "G3": "21-30", "G4": "31-40", "G5": "41-45"}
    for label, name in band_names.items():
        diff = next_band[label] - base_band[label]
        lines.append(f"  {name:<6} {base_band[label]:>8.2f} {next_band[label]:>8.2f} {diff:>+8.2f}")

    lines += [
        "",
        "해석:",
        "  - z 절댓값이 2 미만이면 분절점 전후 차이는 표본 변동 수준 (= 규칙성 변화 없음)",
        "  - '직후1회' G1 차이가 양수면 분절점 다음에 1~10이 반등하는 경향",
        "  - z 에 * 표시가 거의 없으면, 분절점은 통계적 전환점이 아니라 우연한 사건",
    ]
    return lines


# ── SEGMENT 분석 ──────────────────────────────────────────────────────────────

def run_segment_analysis(draws: list[Draw], min_len: int, show: int) -> list[str]:
    bp = breakpoint_indexes(draws)

    # 분절점을 경계로 구간 분할 (분절점 회차는 각 구간의 시작에 포함)
    segments: list[list[Draw]] = []
    for a, b in zip(bp, bp[1:]):
        segments.append(draws[a:b])
    if bp:
        segments.append(draws[bp[-1]:])

    lengths = [len(s) for s in segments]
    usable = [s for s in segments if len(s) >= min_len]

    lines = [
        SEP,
        f"SEGMENT 분석 — 분절점으로 나눈 구간별 통계",
        SEP,
        f"전체 구간 수: {len(segments)}",
        f"구간 길이: 최소 {min(lengths)}, 최대 {max(lengths)}, 평균 {mean(lengths):.1f}",
        f"분석 가능 구간(길이≥{min_len}): {len(usable)}개 "
        f"({len(usable)/len(segments):.0%}) — 나머지는 표본 부족으로 통계 무의미",
        "",
    ]

    if not usable:
        lines.append(f"길이 {min_len} 이상 구간이 없습니다. --min-len 을 낮춰보세요.")
        return lines

    # 구간별 핵심 지표 평균 (긴 구간 우선 표시)
    lines += [
        f"구간별 지표 평균 (긴 구간 상위 {show}개)",
        "",
        f"  {'구간(회차)':<16} {'길이':>4} {'홀수':>5} {'합계':>6} {'AC':>5} "
        f"{'1-10':>5} {'31-45':>6} {'연속':>5}",
        "  " + "─" * 60,
    ]

    def seg_metric_means(seg: list[Draw]) -> dict[str, float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for d in seg:
            for k, v in metrics(d.numbers).items():
                acc[k].append(v)
        return {k: mean(v) for k, v in acc.items()}

    ranked = sorted(usable, key=len, reverse=True)[:show]
    for seg in ranked:
        m = seg_metric_means(seg)
        span = f"{seg[0].round_no}-{seg[-1].round_no}"
        lines.append(
            f"  {span:<16} {len(seg):>4} {m['odd']:>5.1f} {m['sum']:>6.1f} {m['ac']:>5.1f} "
            f"{m['g1_low']:>5.1f} {m['high']:>6.1f} {m['consec']:>5.1f}"
        )

    # 구간 간 변동성 — 각 지표가 구간마다 얼마나 흔들리는가 vs 회차단위 변동
    lines += [
        "",
        "구간 간 변동 vs 전체 변동 (분석 가능 구간 기준)",
        "  비율<1 이면 구간 평균이 전체보다 안정적 = 구간이 의미있는 묶음일 가능성",
        "  비율≈1 이면 구간 구분이 무의미 (구간이 그냥 무작위 토막)",
        "",
        f"  {'지표':<12} {'구간평균 표준편차':>16} {'회차단위 표준편차':>16} {'비율':>6}",
        "  " + "─" * 56,
    ]
    seg_means_by_key: dict[str, list[float]] = defaultdict(list)
    for seg in usable:
        m = seg_metric_means(seg)
        for k, v in m.items():
            seg_means_by_key[k].append(v)
    draw_vals_by_key: dict[str, list[float]] = defaultdict(list)
    for d in draws:
        for k, v in metrics(d.numbers).items():
            draw_vals_by_key[k].append(v)

    for key, label in METRIC_LABELS.items():
        seg_sd = pstdev(seg_means_by_key[key]) if len(seg_means_by_key[key]) > 1 else 0.0
        draw_sd = pstdev(draw_vals_by_key[key])
        ratio = seg_sd / draw_sd if draw_sd else 0.0
        lines.append(f"  {label:<11} {seg_sd:>16.2f} {draw_sd:>16.2f} {ratio:>6.2f}")

    lines += [
        "",
        "해석:",
        "  - '비율'이 1보다 충분히 작아야 구간이 통계적으로 의미있는 묶음",
        "  - 구간 길이가 평균 4~5회로 짧아 구간 내 통계 신뢰도는 낮음",
    ]
    return lines


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="분절점(1~10 미출현) 기준 구간/전후 규칙성 분석.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    p.add_argument("--mode", choices=["event", "segment", "both"], default="both")
    p.add_argument("--window", type=int, default=3, help="event 모드: 분절점 전후 윈도우 크기")
    p.add_argument("--min-len", type=int, default=4, help="segment 모드: 분석할 최소 구간 길이")
    p.add_argument("--show", type=int, default=15, help="segment 모드: 표시할 구간 수")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    draws = load_draws(Path(args.csv))
    out: list[str] = [
        f"데이터: {draws[0].round_no}~{draws[-1].round_no}회 ({len(draws)}회)",
        "",
    ]
    if args.mode in {"event", "both"}:
        out += run_event_study(draws, args.window)
        out.append("")
    if args.mode in {"segment", "both"}:
        out += run_segment_analysis(draws, args.min_len, args.show)
    print("\n".join(out))


if __name__ == "__main__":
    main()
