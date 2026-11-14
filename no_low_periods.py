#!/usr/bin/env python3
"""회차 기간(구간)별로 '1~10 없는 1등 조합'의 규칙이 있는지 본다.

기간: 기본 250회 단위(1~250, 251~500, …, 1001~1249) + 100회 단위.
  1. 기간별 '1~10 없는 회차' 비율이 기간마다 다른가
  2. 기간별 번호 빈도 치우침이 다음 기간으로 이어지는가 (이어진다면 '그 시기의 흐름'이 있다는 뜻)
  3. 후보 요소(연번·AC·합계 등)의 기간별 배수 — 모든 기간에서 1을 넘는가
  4. 워크포워드: 직전 기간에 많이 나온 번호 6개/요소를 다음 기간에 쓰면 이론보다 나은가
p는 이론(11~45 조합 전수) 무작위 이력 또는 순서 섞기와 비교.

예: python3 no_low_periods.py --block 250 --sims 2000
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

LOW = 10


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["회차"].replace("회", "")), sorted(int(row[f"번호{i}"]) for i in range(1, 7))))
    rows.sort()
    return rows


def cons(C):
    return (np.diff(C, axis=1) == 1).sum(1)


def ac(C):
    d = np.sort(np.stack([C[:, j] - C[:, i] for i, j in combinations(range(6), 2)], 1), 1)
    return 1 + (np.diff(d, axis=1) != 0).sum(1) - 5


def no_tail_dup(C):
    t = C % 10
    return np.all(np.stack([(t == d).sum(1) <= 1 for d in range(10)]), 0)


CANDS = {
    "연번 0쌍 또는 3쌍+": lambda C: (cons(C) == 0) | (cons(C) >= 3),
    "연번 3쌍 이상": lambda C: cons(C) >= 3,
    "AC 4~5": lambda C: np.isin(ac(C), [4, 5]),
    "합계 140 미만": lambda C: C.sum(1) < 140,
    "31~45 3개": lambda C: (C >= 31).sum(1) == 3,
    "끝수 중복 없음": no_tail_dup,
    "마지막 번호 44~45": lambda C: C[:, 5] >= 44,
    "홀짝 3:3": lambda C: (C % 2).sum(1) == 3,
}


def freq_dev(C: np.ndarray) -> np.ndarray:
    """번호 11~45 빈도의 표준화 편차."""
    cnt = np.bincount(C.ravel() - 11, minlength=35)
    e = len(C) * 6 / 35
    return (cnt - e) / np.sqrt(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("lotto_winners_2020_2026.csv"))
    ap.add_argument("--block", type=int, default=250)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load(args.csv)
    R = len(rows)
    rn = np.array([r for r, _ in rows])
    flag = np.array([n[0] > LOW for _, n in rows])
    p0 = comb(35, 6) / comb(45, 6)
    allc = np.array(list(combinations(range(11, 46), 6)), dtype=np.int16)

    # 1. 기간별 해당 비율 (100회, 250회)
    for B in (100, args.block):
        blk = (rn - 1) // B
        print(f"[1] {B}회 단위 기간별 '1~10 없는 회차' 비율 (이론 {p0 * 100:.1f}%)")
        parts, chis = [], 0.0
        for b in np.unique(blk):
            m = blk == b
            n, h = m.sum(), flag[m].sum()
            e = n * p0
            chis += (h - e) ** 2 / (e * (1 - p0))
            parts.append(f"{rn[m][0]}~{rn[m][-1]}: {h}/{n} {h / n * 100:.0f}%")
        sims = []
        for _ in range(args.sims):
            f = rng.random(R) < p0
            c = 0.0
            for b in np.unique(blk):
                m = blk == b
                e = m.sum() * p0
                c += (f[m].sum() - e) ** 2 / (e * (1 - p0))
            sims.append(c)
        for i in range(0, len(parts), 5):
            print("   " + " · ".join(parts[i:i + 5]))
        print(f"   기간 간 차이 카이제곱 {chis:.1f} (기간 {len(np.unique(blk))}개) · p {(np.array(sims) >= chis).mean():.3f}\n")

    idx = np.flatnonzero(flag)
    C = np.array([rows[i][1] for i in idx])
    cr = rn[idx]
    blk = (cr - 1) // args.block
    ub = np.unique(blk)
    sizes = [int((blk == b).sum()) for b in ub]
    labels = [f"{b * args.block + 1}~{min((b + 1) * args.block, rn[-1])}" for b in ub]

    # 2. 번호 빈도 치우침의 기간 간 지속
    devs = [freq_dev(C[blk == b]) for b in ub]
    r_real = np.mean([np.corrcoef(devs[i], devs[i + 1])[0, 1] for i in range(len(ub) - 1)])
    sims = []
    for _ in range(args.sims):
        d = [freq_dev(allc[rng.integers(0, len(allc), s)]) for s in sizes]
        sims.append(np.mean([np.corrcoef(d[i], d[i + 1])[0, 1] for i in range(len(d) - 1)]))
    sims = np.array(sims)
    print(f"[2] {args.block}회 기간별 번호 빈도 치우침이 다음 기간으로 이어지는가")
    print(f"   기간 {len(ub)}개 ({', '.join(f'{l}회 {s}개' for l, s in zip(labels, sizes))})")
    print(f"   인접 기간 상관 평균 {r_real:+.3f} · 무작위 95% {np.percentile(sims, 2.5):+.3f} ~ {np.percentile(sims, 97.5):+.3f} · 양측 p {min(1, 2 * min((sims >= r_real).mean(), (sims <= r_real).mean())):.3f}")
    vals = np.arange(11, 46)
    for l, d in zip(labels, devs):
        o = np.argsort(-d)
        print(f"   {l:<10} 많이: {', '.join(str(vals[i]) for i in o[:5])}  · 적게: {', '.join(str(vals[i]) for i in o[-5:])}")

    # 3. 후보 요소의 기간별 배수
    print(f"\n[3] 우세 후보의 {args.block}회 기간별 배수 (실측 비율 ÷ 이론 비율, 1보다 크면 우세)")
    print(f"   {'후보':<18}{'이론':>6} " + "".join(f"{l:>11}" for l in labels) + f"{'1 초과':>8}{'우연이면':>8}")
    for name, f in CANDS.items():
        p = f(allc).mean()
        m = f(C)
        mult = [m[blk == b].mean() / p for b in ub]
        above = sum(x > 1 for x in mult)
        # 무작위로 기간마다 1을 넘을 확률
        ps = []
        for s in sizes:
            from math import floor
            k = np.arange(s + 1)
            # 이항 P(X > s*p)
            from math import comb as cb
            ps.append(sum(cb(s, int(x)) * p ** x * (1 - p) ** (s - x) for x in k if x > s * p))
        exp_above = sum(ps)
        print(f"   {name:<18}{p * 100:>5.1f}% " + "".join(f"{x:>10.2f}x" for x in mult) + f"{above:>5}/{len(ub)}{exp_above:>8.1f}")

    # 4. 워크포워드: 직전 기간 상위 6개 번호를 다음 기간에
    print(f"\n[4] 워크포워드 — 직전 {args.block}회 기간에 많이 나온 번호 6개를 다음 기간에 쓰면")
    hits = exp = 0.0
    for i in range(1, len(ub)):
        top = np.argsort(-devs[i - 1])[:6] + 11
        nxt = C[blk == ub[i]]
        h = np.isin(nxt, top).sum()
        e = len(nxt) * 6 * 6 / 35
        hits += h
        exp += e
        print(f"   {labels[i - 1]} 상위 {','.join(map(str, sorted(top)))} → {labels[i]}: 적중 {h} / 기대 {e:.1f}")
    print(f"   합계 {hits:.0f} / 기대 {exp:.1f} ({(hits / exp - 1) * 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
