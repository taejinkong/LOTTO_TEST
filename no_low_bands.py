#!/usr/bin/env python3
"""1~10이 없는 1등 조합의 번호대(구간)별 규칙을 찾는다.

구간: 11~20, 21~30, 31~40, 41~45. 이론은 11~45 6개 조합 1,623,160개 전수.
검정
  A. 구간 간 조건부 규칙: 'A구간이 a개면 B구간은 b개' 모든 조합의 z, 최대 |z|를 무작위 이력과 비교
  B. 구간 안 번호별 빈도 (같은 구간 안에서 특정 번호가 더 나오나)
  C. 구간 안 연번 쌍 수 (앞선 '연번' 치우침이 어느 구간에서 생기나)
  D. 자리별(작은 순 1~6번째) 번호 분포
  E. 직전 회차(1~10 포함 전체) 구간 개수 → 이번 해당 회차 구간 개수 상관
  F. 직전 '해당 회차' 구간 개수 → 다음 해당 회차 구간 개수 상관
  각 검정의 p는 이론 분포에서 뽑은 같은 크기 무작위 이력(또는 순서 섞기)과 비교한 경험적 값.
  기간 안정성: 1~624회 / 625~1249회로 나눠 같은 방향인지 함께 표시.

예: python3 no_low_bands.py --sims 2000
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

LOW = 10
BANDS = [(11, 20), (21, 30), (31, 40), (41, 45)]
NAMES = ["11~20", "21~30", "31~40", "41~45"]


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["회차"].replace("회", "")), sorted(int(row[f"번호{i}"]) for i in range(1, 7))))
    rows.sort()
    return rows


def band_counts(C: np.ndarray, bands=BANDS) -> np.ndarray:
    return np.stack([((C >= a) & (C <= b)).sum(1) for a, b in bands], 1)


def band_consec(C: np.ndarray) -> np.ndarray:
    """구간 안에서만 이어진 연번 쌍 수 (구간 경계 20-21 같은 건 '경계' 열로 따로)."""
    d = np.diff(C, axis=1) == 1
    left = C[:, :-1]
    out = []
    for a, b in BANDS:
        out.append((d & (left >= a) & (left < b)).sum(1))
    out.append((d & np.isin(left, [20, 30, 40])).sum(1))
    return np.stack(out, 1)


def pair_table(bc: np.ndarray) -> np.ndarray:
    """(i, j, a, b) 개수 — i<j 구간, a,b=0..6."""
    T = np.zeros((4, 4, 7, 7))
    for i in range(4):
        for j in range(i + 1, 4):
            np.add.at(T[i, j], (bc[:, i], bc[:, j]), 1)
    return T


def max_abs_z(T: np.ndarray, P: np.ndarray, n: int) -> tuple[float, tuple]:
    E = P * n
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(E >= 5, (T - E) / np.sqrt(E * (1 - P)), 0)
    k = np.unravel_index(np.abs(Z).argmax(), Z.shape)
    return float(np.abs(Z).max()), k, Z


def corr_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = (X - X.mean(0)) / X.std(0)
    Y = (Y - Y.mean(0)) / Y.std(0)
    return X.T @ Y / len(X)


def pval(sims: np.ndarray, v: float) -> float:
    return float((np.asarray(sims) >= v).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("lotto_winners_2020_2026.csv"))
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load(args.csv)
    idx = [i for i, (_, n) in enumerate(rows) if n[0] > LOW]
    real = np.array([rows[i][1] for i in idx])
    rnd = np.array([rows[i][0] for i in idx])
    N = len(real)
    early = rnd <= 624
    allc = np.array(list(combinations(range(LOW + 1, 46), 6)), dtype=np.int16)
    bc_all = band_counts(allc)
    bc = band_counts(real)
    print(f"1~10 없는 1등 조합 {N}개 · 이론 11~45 조합 {len(allc):,}개 전수 · 무작위 이력 {args.sims}개 비교\n")

    def sample(n):
        return rng.integers(0, len(allc), n)

    # A. 구간 간 조건부 규칙
    P = pair_table(bc_all) / len(allc)
    T = pair_table(bc)
    mz, k, Z = max_abs_z(T, P, N)
    sims = [max_abs_z(pair_table(bc_all[sample(N)]), P, N)[0] for _ in range(args.sims)]
    i, j, a, b = k
    print("[A] 구간 간 조건부 규칙 ('A구간 a개 & B구간 b개' 동시 출현, 기대 5회 이상 칸 전체)")
    print(f"  가장 튀는 칸: {NAMES[i]} {a}개 & {NAMES[j]} {b}개 — 실측 {int(T[k])}회 / 기대 {P[k] * N:.1f}회 (z {Z[k]:+.2f})")
    print(f"  최대 |z| {mz:.2f} · 무작위 이력 최대 |z| 95% 범위 ~{np.percentile(sims, 95):.2f} · p {pval(sims, mz):.3f}")
    # 상위 3칸과 기간 안정성
    flat = np.argsort(-np.abs(Z), None)[:4]
    for f in flat:
        i, j, a, b = np.unravel_index(f, Z.shape)
        m = (bc[:, i] == a) & (bc[:, j] == b)
        e1, e2 = P[i, j, a, b] * early.sum(), P[i, j, a, b] * (~early).sum()
        print(f"    {NAMES[i]} {a}개 & {NAMES[j]} {b}개: z {Z[i, j, a, b]:+.2f} · 1~624회 {m[early].sum()}/{e1:.1f} · 625~회 {m[~early].sum()}/{e2:.1f}")

    # B. 구간 안 번호 빈도
    print("\n[B] 구간 안 번호별 빈도 (같은 구간 번호끼리는 이론상 동일 확률)")
    tot_chi, tot_df = 0.0, 0
    chis_real = []
    for (a, b), nm in zip(BANDS, NAMES):
        cnt = np.array([(real == x).sum() for x in range(a, b + 1)])
        e = cnt.sum() / len(cnt)
        chi = ((cnt - e) ** 2 / e).sum()
        chis_real.append(chi)
        top = range(a, b + 1)[int(cnt.argmax())]
        bot = range(a, b + 1)[int(cnt.argmin())]
        print(f"  {nm}: 카이제곱 {chi:5.1f} (자유도 {len(cnt) - 1}, 무작위 평균 {len(cnt) - 1}) · 최다 {top}({cnt.max()}) 최소 {bot}({cnt.min()}) · 기대 {e:.1f}")
        tot_chi += chi
        tot_df += len(cnt) - 1
    sims = []
    for _ in range(args.sims):
        s = allc[sample(N)]
        c = 0.0
        for a, b in BANDS:
            cnt = np.array([(s == x).sum() for x in range(a, b + 1)])
            e = cnt.sum() / len(cnt)
            c += ((cnt - e) ** 2 / e).sum()
        sims.append(c)
    print(f"  합계 카이제곱 {tot_chi:.1f} (자유도 {tot_df}) · p {pval(sims, tot_chi):.3f}")

    # C. 구간 안 연번
    cs_all = band_consec(allc)
    cs = band_consec(real)
    print("\n[C] 구간별 연번 쌍 (앞선 '연번 치우침'이 어디서 생기나)")
    labels = NAMES + ["경계(20-21,30-31,40-41)"]
    for q, nm in enumerate(labels):
        mean_e = cs_all[:, q].mean()
        dist_e = np.bincount(cs_all[:, q], minlength=6)[:4] / len(allc)
        dist_o = np.bincount(cs[:, q], minlength=6)[:4] / N
        sm = np.array([cs_all[sample(N), q].mean() for _ in range(args.sims)])
        v = cs[:, q].mean()
        p = min(1, 2 * min((sm >= v).mean(), (sm <= v).mean()))
        e1 = cs[early, q].mean() - mean_e
        e2 = cs[~early, q].mean() - mean_e
        print(f"  {nm:<22} 평균 {v:.3f} / 이론 {mean_e:.3f} · 0쌍 {dist_o[0] * 100:.0f}%({dist_e[0] * 100:.0f}) 2쌍+ {(1 - dist_o[0] - dist_o[1]) * 100:.1f}%({(1 - dist_e[0] - dist_e[1]) * 100:.1f})"
              f" · p {p:.3f} · 전반 {e1:+.3f} 후반 {e2:+.3f}")

    # D. 자리별 분포
    print("\n[D] 자리별(작은 순) 번호 분포 — 평균과 카이제곱")
    tot = 0.0
    pos_real = []
    for pidx in range(6):
        vals = np.arange(11, 46)
        pe = np.array([(allc[:, pidx] == v).mean() for v in vals])
        po = np.array([(real[:, pidx] == v).sum() for v in vals])
        m = pe * N >= 5
        o = np.append(po[m], po[~m].sum())
        e = np.append(pe[m] * N, pe[~m].sum() * N)
        e = np.where(e > 0, e, 1e-9)
        chi = ((o - e) ** 2 / e).sum()
        pos_real.append((pe, m))
        tot += chi
        print(f"  {pidx + 1}번째: 평균 {real[:, pidx].mean():.2f} / 이론 {(pe * vals).sum():.2f} · 카이제곱 {chi:.1f} (칸 {len(o)})")
    sims = []
    for _ in range(min(args.sims, 1000)):
        s = allc[sample(N)]
        c = 0.0
        for pidx, (pe, m) in enumerate(pos_real):
            po = np.bincount(s[:, pidx] - 11, minlength=35)
            o = np.append(po[m], po[~m].sum())
            e = np.append(pe[m] * N, pe[~m].sum() * N)
            e = np.where(e > 0, e, 1e-9)
            c += ((o - e) ** 2 / e).sum()
        sims.append(c)
    print(f"  6자리 합계 카이제곱 {tot:.1f} · p {pval(sims, tot):.3f}")

    # E. 직전 회차 → 이번 해당 회차
    prev = np.array([rows[i - 1][1] for i in idx if i > 0])
    cur = bc[[n for n, i in enumerate(idx) if i > 0]]
    pb = band_counts(prev, [(1, 10)] + BANDS)
    R = corr_matrix(pb, cur)
    stat = np.abs(R).max()
    sims = [np.abs(corr_matrix(pb, cur[rng.permutation(len(cur))])).max() for _ in range(args.sims)]
    a, b = np.unravel_index(np.abs(R).argmax(), R.shape)
    pn = ["1~10"] + NAMES
    print("\n[E] 직전 회차(어떤 회차든) 구간 개수 → 이번 해당 회차 구간 개수")
    print(f"  가장 강한 연결: 직전 {pn[a]} 개수 → 이번 {NAMES[b]} 개수 상관 {R[a, b]:+.3f}")
    print(f"  최대 |상관| {stat:.3f} · 순서 섞기 95% ~{np.percentile(sims, 95):.3f} · p {pval(sims, stat):.3f}")
    print(f"  같은 구간끼리(직전 X → 이번 X): " + ", ".join(f"{NAMES[q]} {R[q + 1, q]:+.3f}" for q in range(4)))

    # F. 직전 해당 회차 → 다음 해당 회차
    R2 = corr_matrix(bc[:-1], bc[1:])
    stat2 = np.abs(R2).max()
    sims = []
    for _ in range(args.sims):
        o = bc[rng.permutation(N)]
        sims.append(np.abs(corr_matrix(o[:-1], o[1:])).max())
    a, b = np.unravel_index(np.abs(R2).argmax(), R2.shape)
    print("\n[F] 직전 '1~10 없는 회차' 구간 개수 → 다음 '1~10 없는 회차' 구간 개수")
    print(f"  가장 강한 연결: {NAMES[a]} → {NAMES[b]} 상관 {R2[a, b]:+.3f} · 최대 |상관| 순서 섞기 95% ~{np.percentile(sims, 95):.3f} · p {pval(sims, stat2):.3f}")
    print(f"  같은 구간끼리: " + ", ".join(f"{NAMES[q]} {R2[q, q]:+.3f}" for q in range(4)))
    same = (bc[:-1] == bc[1:]).all(1).mean()
    pat_p = {}
    for row in map(tuple, bc_all):
        pat_p[row] = pat_p.get(row, 0) + 1
    same_e = sum((v / len(allc)) ** 2 for v in pat_p.values())
    print(f"  구간 패턴이 바로 다음 해당 회차와 똑같을 확률: 실측 {same * 100:.1f}% / 무작위 {same_e * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
