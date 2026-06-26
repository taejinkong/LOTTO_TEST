#!/usr/bin/env python3
"""1~12 신호의 시간 안정성 검증.

검증된 신호(popularity_regression.py): 당첨번호의 1~12 개수↑ → 당첨자↑.
이 신호가 시기에 따라 일관되는지(부호·강도), 특히 최근 회차에서도
유지되는지 확인한다. 최근에 약해졌다면 계수를 갱신해야 한다.

분석:
  1. 시기 K분할 — 각 구간에서 n_1_12 회귀계수·t, 0개 vs 3+개 당첨자 격차
  2. 롤링 윈도우 — 윈도우를 이동하며 계수 추이
  3. 최근 구간 집중 — 최근 N회만으로 재추정

타겟: log(당첨자수). 구간별 회귀라 시대별 판매량 차이는 절편에 흡수.

예시:
  python3 signal_stability.py
  python3 signal_stability.py --segments 5 --window 250 --step 50
"""

from __future__ import annotations

import argparse
import csv
from math import log, sqrt
from pathlib import Path
from statistics import mean

import numpy as np


SEP = "=" * 70


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            nums = tuple(sorted(int(r[f"번호{k}"]) for k in range(1, 7)))
            rows.append((int(r["회차"].replace("회", "")), nums, int(r["1등 당첨자수(명)"])))
    return sorted(rows, key=lambda x: x[0])


def n_low12(nums) -> int:
    return sum(1 <= n <= 12 for n in nums)


def regress(rows) -> dict:
    """log(winners) ~ n_1_12 단순회귀. 계수, t, 그룹 평균."""
    x = np.array([n_low12(nums) for _, nums, _ in rows], float)
    y = np.array([log(w) for _, _, w in rows], float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    with np.errstate(all="ignore"):
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        resid = y - X @ beta
        rss = float(resid @ resid)
        sigma2 = rss / (n - 2) if n > 2 else 0
        se = sqrt(sigma2 * XtX_inv[1, 1]) if sigma2 > 0 else 0
    slope = beta[1]
    t = slope / se if se else 0.0
    g0 = [w for _, nums, w in rows if n_low12(nums) == 0]
    g3 = [w for _, nums, w in rows if n_low12(nums) >= 3]
    return {
        "n": n, "slope": slope, "t": t,
        "g0_mean": mean(g0) if g0 else float("nan"),
        "g3_mean": mean(g3) if g3 else float("nan"),
        "g0_n": len(g0), "g3_n": len(g3),
    }


def bar(t: float) -> str:
    """t값 부호/크기 막대."""
    mag = min(int(abs(t) * 2), 12)
    ch = "█" if abs(t) >= 2 else "▪"
    return (ch * mag).rjust(12) if t >= 0 else ("░" * mag).rjust(12)


def segment_analysis(rows, k: int) -> list[str]:
    size = len(rows) // k
    lines = [
        SEP,
        f"1) 시기 {k}분할 — 구간별 1~12 신호",
        SEP,
        f"  {'구간(회차)':<16} {'회차수':>5} {'계수':>8} {'t':>7} "
        f"{'0개당첨자':>8} {'3+개당첨자':>9} {'격차%':>6}",
        "  " + "─" * 64,
    ]
    for i in range(k):
        seg = rows[i*size:] if i == k-1 else rows[i*size:(i+1)*size]
        r = regress(seg)
        span = f"{seg[0][0]}-{seg[-1][0]}"
        gap = (r["g3_mean"] - r["g0_mean"]) / r["g3_mean"] * 100 if r["g3_mean"] else 0
        mark = "**" if abs(r["t"]) >= 2.6 else ("*" if abs(r["t"]) >= 2 else "")
        lines.append(
            f"  {span:<16} {r['n']:>5} {r['slope']:>+8.4f} {r['t']:>+7.2f}{mark:<2} "
            f"{r['g0_mean']:>8.1f} {r['g3_mean']:>9.1f} {gap:>+6.0f}"
        )
    return lines


def rolling_analysis(rows, window: int, step: int) -> list[str]:
    lines = [
        "",
        SEP,
        f"2) 롤링 윈도우 (창 {window}회, 보폭 {step}회) — 계수 추이",
        SEP,
        f"  {'끝회차':>6} {'계수':>8} {'t':>7}  {'부호/강도(█=유의)':<14}",
        "  " + "─" * 48,
    ]
    i = 0
    pos_count = sig_count = total = 0
    while i + window <= len(rows):
        seg = rows[i:i+window]
        r = regress(seg)
        end_round = seg[-1][0]
        lines.append(f"  {end_round:>6} {r['slope']:>+8.4f} {r['t']:>+7.2f}  {bar(r['t'])}")
        total += 1
        pos_count += r["slope"] > 0
        sig_count += r["t"] >= 2
        i += step
    # 마지막 윈도우 포함 보장
    if (len(rows) - window) % step != 0:
        seg = rows[-window:]
        r = regress(seg)
        lines.append(f"  {seg[-1][0]:>6} {r['slope']:>+8.4f} {r['t']:>+7.2f}  {bar(r['t'])}  ←최신")
        total += 1
        pos_count += r["slope"] > 0
        sig_count += r["t"] >= 2
    lines += [
        "",
        f"  양(+)의 계수 비율: {pos_count}/{total} ({pos_count/total*100:.0f}%)",
        f"  유의(t>=2) 비율:   {sig_count}/{total} ({sig_count/total*100:.0f}%)",
    ]
    return lines


def recent_focus(rows, recents: list[int]) -> list[str]:
    lines = [
        "",
        SEP,
        "3) 최근 구간 집중 — 신호가 지금도 살아있는가",
        SEP,
        f"  {'기간':<14} {'회차수':>5} {'계수':>8} {'t':>7} {'0개당첨자':>8} {'3+개당첨자':>9}",
        "  " + "─" * 58,
    ]
    full = regress(rows)
    lines.append(
        f"  {'전체':<14} {full['n']:>5} {full['slope']:>+8.4f} {full['t']:>+7.2f}"
        f"{'':<2} {full['g0_mean']:>8.1f} {full['g3_mean']:>9.1f}"
    )
    for nrec in recents:
        if nrec >= len(rows):
            continue
        seg = rows[-nrec:]
        r = regress(seg)
        lines.append(
            f"  {'최근 '+str(nrec)+'회':<14} {r['n']:>5} {r['slope']:>+8.4f} {r['t']:>+7.2f}"
            f"{'':<2} {r['g0_mean']:>8.1f} {r['g3_mean']:>9.1f}"
        )
    return lines


def verdict(rows, k: int) -> list[str]:
    size = len(rows) // k
    seg_results = []
    for i in range(k):
        seg = rows[i*size:] if i == k-1 else rows[i*size:(i+1)*size]
        seg_results.append(regress(seg))
    slopes = [r["slope"] for r in seg_results]
    last_seg = seg_results[-1]          # 최근 시기 구간
    r100 = regress(rows[-100:]) if len(rows) > 100 else regress(rows)
    r200 = regress(rows[-200:]) if len(rows) > 200 else regress(rows)
    r300 = regress(rows[-300:]) if len(rows) > 300 else regress(rows)

    all_pos = all(s > 0 for s in slopes)
    # 약화 추세 감지: 최근 시기 구간이 비유의 + 최근 100회 신호 소멸
    decaying = last_seg["t"] < 1.5 and r100["t"] < 1.0

    lines = ["", SEP, "종합 판정", SEP]
    if not all_pos:
        neg = sum(1 for s in slopes if s <= 0)
        lines += [
            f"  ✗ 불안정: {neg}개 시기 구간에서 계수 부호 반전 → 시기 의존적",
        ]
    elif decaying:
        lines += [
            "  △ 방향 일관하나 '약화 추세': 부호는 전 구간 양(+)이지만",
            f"     최근 시기 구간 t={last_seg['t']:.2f}, 최근 100회 t={r100['t']:.2f} → 신호 소멸",
            "     (강했던 과거 916~1070 구간이 전체/롤링 평균을 끌어올린 것)",
            "",
            "  해석:",
            "  - 신호 자체는 과거에 실재했으나 최근으로 올수록 희석(판매량↑/QP↑ 추정)",
            f"  - 효과 크기 추이: 전체 {regress(rows)['slope']:.3f} → 최근300 {r300['slope']:.3f}"
            f" → 최근200 {r200['slope']:.3f} → 최근100 {r100['slope']:.3f}",
            "",
            "  권장:",
            "  - lotto_combo_generator.py 계수(0.058)는 과거 평균 → 최근 실정엔 과대",
            "  - 최근 300회 기준(보수적)으로 갱신하거나, 효과를 약하게 고지",
            "  - '당첨 시 +20%'는 과거치이며 최근엔 더 작을 수 있음을 명시",
        ]
    else:
        lines += [
            "  ✓ 안정적: 전 구간 계수 양(+) + 최근 구간도 신호 유지",
            "  → 계수(0.058)를 신뢰하고 사용 가능",
        ]
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1~12 신호 시간 안정성 검증.")
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--recents", default="100,200,300")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.csv))
    recents = [int(x) for x in args.recents.split(",") if x.strip()]
    out = [f"데이터: {rows[0][0]}~{rows[-1][0]}회 ({len(rows)}회)", ""]
    out += segment_analysis(rows, args.segments)
    out += rolling_analysis(rows, args.window, args.step)
    out += recent_focus(rows, recents)
    out += verdict(rows, args.segments)
    print("\n".join(out))


if __name__ == "__main__":
    main()
