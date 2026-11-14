#!/usr/bin/env python3
"""'1~10이 하나도 없는 회차'가 언제 오는지 예측할 수 있는지 검증한다.

예측할 수 있다면 그 회차에만 11~45 조합을 사면 1등 확률이 오른다
(해당 회차라는 걸 알면 내 11~45 조합의 1등 확률은 1/C(35,6) = 기본의 5.02배).
예측할 수 없다면 11~45 조합의 1등 확률은 언제 사도 1/8,145,060으로 같다.

1. 해저드: 마지막 해당 후 g회가 지났을 때 이번이 해당일 확률 (주기가 있으면 g에 따라 변함)
2. 최근 W회 동안 나온 1~10 번호 개수에 따른 해당률 ('밀렸다/몰렸다' 가설)
3. 워크포워드 타이밍 규칙: 과거 데이터로만 '가장 잘 맞던 공백 구간'을 고르고 다음 회차에 적용
   → 규칙이 고른 회차의 실제 해당률이 기본 19.93%보다 높은가. 같은 절차를 순서 섞은 이력에도 돌려 비교.

예: python3 no_low_timing.py --sims 2000
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np


def load(path: Path, low: int):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            nums = [int(row[f"번호{i}"]) for i in range(1, 7)]
            rows.append((int(row["회차"].replace("회", "")), nums))
    rows.sort()
    x = np.array([min(n) > low for _, n in rows], dtype=int)
    lowcnt = np.array([sum(v <= low for v in n) for _, n in rows])
    return np.array([r for r, _ in rows]), x, lowcnt


def gaps_since(x: np.ndarray) -> np.ndarray:
    """g[t] = 회차 t 직전까지 마지막 해당 이후 지난 회차 수 (직전이 해당이면 1)."""
    g = np.zeros(len(x), dtype=int)
    last = -1
    for t in range(len(x)):
        g[t] = t - last if last >= 0 else 0
        if x[t]:
            last = t
    return g


BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 99)]


def walkforward(x: np.ndarray, start: int, min_n: int = 30) -> tuple[int, int]:
    """매 회차, 그 이전 데이터만으로 해당률이 가장 높았던 공백 버킷을 고르고 이번 회차가 그 버킷이면 '산다'."""
    g = gaps_since(x)
    bidx = np.full(len(x), -1)
    for i, (a, b) in enumerate(BUCKETS):
        bidx[(g >= a) & (g <= b)] = i
    nb = len(BUCKETS)
    hits = np.zeros(nb)
    cnt = np.zeros(nb)
    for t in range(start):
        if bidx[t] >= 0:
            cnt[bidx[t]] += 1
            hits[bidx[t]] += x[t]
    bought = won = 0
    for t in range(start, len(x)):
        rate = np.where(cnt >= min_n, hits / np.maximum(cnt, 1), -1)
        best = int(rate.argmax())
        if bidx[t] == best:
            bought += 1
            won += x[t]
        if bidx[t] >= 0:
            cnt[bidx[t]] += 1
            hits[bidx[t]] += x[t]
    return bought, won


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("lotto_winners_2020_2026.csv"))
    ap.add_argument("--low", type=int, default=10)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rounds, x, lowcnt = load(args.csv, args.low)
    R = len(x)
    p0 = comb(45 - args.low, 6) / comb(45, 6)
    base = x.mean()
    print(f"데이터 {rounds[0]}~{rounds[-1]}회 · 1~{args.low} 없는 회차 {x.sum()}회 · 실측 {base * 100:.1f}% · 이론 {p0 * 100:.2f}%")

    # 1. 해저드
    g = gaps_since(x)
    print("\n[1] 마지막 해당 후 경과 회차별 이번 회차 해당률 (주기가 있다면 특정 구간이 튄다)")
    print(f"{'경과':>8}{'회차 수':>8}{'해당':>6}{'해당률':>9}{'95% 범위(이론 기준)':>22}")
    for a, b in BUCKETS:
        m = (g >= a) & (g <= b)
        n, h = m.sum(), x[m].sum()
        sd = np.sqrt(p0 * (1 - p0) / n)
        lab = f"{a}" if a == b else f"{a}~{b}" if b < 99 else f"{a}+"
        flag = " *" if abs(h / n - p0) > 1.96 * sd else ""
        print(f"{lab:>8}{n:>8}{h:>6}{h / n * 100:>8.1f}%{f'{(p0 - 1.96 * sd) * 100:.1f}~{(p0 + 1.96 * sd) * 100:.1f}%':>22}{flag}")

    # 2. 최근 W회 1~10 개수
    print("\n[2] 최근 W회 동안 나온 1~10 번호 개수에 따른 이번 회차 해당률")
    for W in (3, 5, 10):
        cs = np.concatenate([[0], np.cumsum(lowcnt)])
        recent = np.array([cs[t] - cs[t - W] for t in range(W, R)])
        y = x[W:]
        exp = W * 6 * 10 / 45
        lo_m, hi_m = recent < np.percentile(recent, 33), recent > np.percentile(recent, 67)
        r = np.corrcoef(recent, y)[0, 1]
        se = 1 / np.sqrt(len(y))
        print(f"  W={W:>2} (기대 {exp:.1f}개): 적게 나온 1/3 구간 {y[lo_m].mean() * 100:.1f}% · "
              f"많이 나온 1/3 구간 {y[hi_m].mean() * 100:.1f}% · 상관 {r:+.3f} (무작위 95% ±{1.96 * se:.3f})")

    # 3. 워크포워드 규칙
    start = 200
    bought, won = walkforward(x, start)
    rate = won / bought if bought else 0
    sims = []
    for _ in range(args.sims):
        b, w = walkforward(rng.permutation(x), start)
        sims.append(w / b - x[start:].mean() if b else 0)
    sims = np.array(sims)
    edge = rate - x[start:].mean()
    pv = (sims >= edge).mean()
    lo, hi = np.percentile(sims, [2.5, 97.5])
    print(f"\n[3] 워크포워드 타이밍 규칙 ({rounds[start]}~{rounds[-1]}회 적용)")
    print(f"  규칙이 '이번엔 11~45'라고 고른 회차 {bought}회 중 실제 해당 {won}회 = {rate * 100:.1f}% "
          f"(같은 기간 전체 {x[start:].mean() * 100:.1f}%, 차이 {edge * 100:+.1f}%p)")
    print(f"  순서를 섞은 이력 {args.sims}개에서 같은 절차의 차이 95% 범위 {lo * 100:+.1f} ~ {hi * 100:+.1f}%p · 단측 p {pv:.3f}")

    # 현재 상태
    gnow = len(x) - np.flatnonzero(x)[-1]
    print(f"\n현재: 마지막 해당 {rounds[np.flatnonzero(x)[-1]]}회, 다음 회차({rounds[-1] + 1}회)는 경과 {gnow}회 시점")
    print(f"  무작위라면 다음 회차가 해당일 확률 {p0 * 100:.2f}% (경과와 무관)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
