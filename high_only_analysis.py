#!/usr/bin/env python3
"""1~10을 빼고 11~45 번호만으로 회차 간 1등 번호 규칙성을 찾는다.

각 회차 본번호 6개 중 11 이상인 것만 남긴 집합 H_t(평균 4.67개)를 본다.
보너스는 쓰지 않는다. 모든 통계량은 같은 방식으로 만든 무작위 추첨 이력
(회차 수 동일, 6/45 무작위)과 비교해 경험적 p값을 낸다.
조건부로 보면 11~45 안의 35개 번호는 무작위라면 서로 교환 가능하므로
'1~10 제외'는 원래 무작위성 검정을 35개 번호로 좁힌 것과 같다.

검정 묶음
  A. 번호별 빈도 균등성(카이제곱)
  B. 직전 회차와 겹치는 개수(시차 1~5)
  C. '이 번호 다음 회차엔 저 번호' 전이표(35x35) 치우침 + 전반/후반 재현성
  D. 같은 회차 동반 출현 쌍(595쌍) 치우침 + 전반/후반 재현성
  E. 미출현 기간(갭)에 따라 다음 출현 확률이 달라지는가
  F. 최근 W회 많이 나온(핫)/안 나온(콜드) 6개를 고르면 더 맞는가(워크포워드)
  G. 집합 모양: 합계·홀수 개수·연번 쌍·번호대(11~20/21~30/31~40/41~45), 회차 간 합계 자기상관

검정이 여러 개라 p<0.05 몇 개는 우연히 나온다(검정 수 × 0.05개 정도).

예:
  python3 high_only_analysis.py
  python3 high_only_analysis.py --sims 2000 --low 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            r = int(row["회차"].replace("회", ""))
            rows.append((r, [int(row[f"번호{i}"]) for i in range(1, 7)]))
    rows.sort()
    return np.array([r for r, _ in rows]), np.array([n for _, n in rows])


def to_binary(draws: np.ndarray, low: int) -> np.ndarray:
    """(..., R, 6) 번호 → (..., R, 45-low) 0/1 행렬 (열 j = 번호 low+1+j)."""
    m = 45 - low
    shape = draws.shape[:-1] + (m,)
    b = np.zeros(shape, dtype=np.int16)
    idx = draws - (low + 1)
    valid = idx >= 0
    for c in range(6):
        sel = valid[..., c]
        col = np.where(sel, idx[..., c], 0)
        onehot = np.zeros(shape, dtype=np.int16)
        np.put_along_axis(onehot, col[..., None], 1, axis=-1)
        b += onehot * sel[..., None]
    return b


def stats_batch(B: np.ndarray, nums: np.ndarray, low: int, tie: np.ndarray) -> dict[str, np.ndarray]:
    """B: (S, R, M). 각 통계량을 (S,) 배열로."""
    S, R, M = B.shape
    Bf = B.astype(np.float64)
    k = Bf.sum(-1)                      # (S,R) 회차별 11~45 개수
    out: dict[str, np.ndarray] = {}

    # A. 빈도 균등성
    col = Bf.sum(1)
    e = col.sum(-1, keepdims=True) / M
    out["A 빈도 카이제곱"] = ((col - e) ** 2 / e).sum(-1)
    out["A 최다-최소 빈도차"] = col.max(-1) - col.min(-1)

    # B. 직전 회차와 겹침
    for lag in (1, 2, 3, 5):
        ov = (Bf[:, lag:] * Bf[:, :-lag]).sum(-1)
        out[f"B 시차{lag} 평균 겹침"] = ov.mean(-1)

    # C. 전이표 (a가 t-1에 → b가 t에)
    def trans(x):
        T = np.einsum("sra,srb->sab", x[:, :-1], x[:, 1:])
        E = np.einsum("sa,sb->sab", x[:, :-1].sum(1), x[:, 1:].sum(1)) / (x.shape[1] - 1)
        return T, E
    T, E = trans(Bf)
    out["C 전이표 카이제곱"] = ((T - E) ** 2 / E).sum((-1, -2))
    out["C 전이표 최대 z"] = ((T - E) / np.sqrt(E)).max((-1, -2))
    h = R // 2
    T1, E1 = trans(Bf[:, :h])
    T2, E2 = trans(Bf[:, h:])
    z1 = ((T1 - E1) / np.sqrt(E1)).reshape(S, -1)
    z2 = ((T2 - E2) / np.sqrt(E2)).reshape(S, -1)
    out["C 전이 전반-후반 상관"] = corr_rows(z1, z2)

    # D. 동반 출현 쌍
    iu = np.triu_indices(M, 1)
    def pairs(x):
        P = np.einsum("sra,srb->sab", x, x)[:, iu[0], iu[1]]
        kk = x.sum(-1)
        ep = (kk * (kk - 1) / 2).sum(-1, keepdims=True) / len(iu[0])
        return P, ep
    P, ep = pairs(Bf)
    out["D 쌍 카이제곱"] = ((P - ep) ** 2 / ep).sum(-1)
    out["D 최다 쌍 횟수"] = P.max(-1)
    P1, ep1 = pairs(Bf[:, :h])
    P2, ep2 = pairs(Bf[:, h:])
    out["D 쌍 전반-후반 상관"] = corr_rows((P1 - ep1) / np.sqrt(ep1), (P2 - ep2) / np.sqrt(ep2))

    # E. 갭별 출현률: 버킷 [1], [2-3], [4-7], [8-15], [16+]
    edges = np.array([1, 2, 4, 8, 16, 10**9])
    hits = np.zeros((S, 5))
    expv = np.zeros((S, 5))
    last = np.full((S, M), -1)
    for t in range(R):
        if t >= 20:
            gap = t - last                              # 마지막 출현 후 지난 회차 수
            bucket = np.searchsorted(edges, gap, side="right") - 1
            p = k[:, t:t + 1] / M                        # 이번 회차 조건부 기대
            for b_ in range(5):
                m_ = bucket == b_
                hits[:, b_] += (B[:, t] * m_).sum(-1)
                expv[:, b_] += (p * m_).sum(-1)
        last = np.where(B[:, t] > 0, t, last)
    out["E 갭 버킷 카이제곱"] = ((hits - expv) ** 2 / np.maximum(expv, 1e-9)).sum(-1)
    out["E 직전 출현(갭1) 초과율%"] = (hits[:, 0] / expv[:, 0] - 1) * 100
    out["E 장기 미출현(16+) 초과율%"] = (hits[:, 4] / expv[:, 4] - 1) * 100

    # F. 핫/콜드 6개 워크포워드 (t >= 100)
    cs = np.concatenate([np.zeros((S, 1, M)), np.cumsum(Bf, 1)], 1)
    start = 100
    for W in (10, 30, 100):
        score = cs[:, start:R] - cs[:, start - W:R - W]  # (S, R-start, M) 직전 W회 빈도
        order = np.lexsort((np.broadcast_to(tie, score.shape), score), axis=-1)
        cold = order[..., :6]
        hot = order[..., -6:]
        tgt = Bf[:, start:R]
        exp6 = (6 * k[:, start:R] / M).sum(-1)
        hh = np.take_along_axis(tgt, hot, -1).sum((-1, -2))
        cc = np.take_along_axis(tgt, cold, -1).sum((-1, -2))
        out[f"F 핫6(W={W}) 초과 적중%"] = (hh / exp6 - 1) * 100
        out[f"F 콜드6(W={W}) 초과 적중%"] = (cc / exp6 - 1) * 100

    # G. 집합 모양 (11~45 부분집합)
    vals = np.arange(low + 1, 46)
    hsum = (Bf * vals).sum(-1)
    out["G 합계 평균(11~45만)"] = hsum.mean(-1)
    out["G 홀수 개수 평균"] = (Bf * (vals % 2)).sum(-1).mean(-1)
    out["G 연번 쌍 평균"] = (Bf[..., 1:] * Bf[..., :-1]).sum(-1).mean(-1)
    dec = [(11, 20), (21, 30), (31, 40), (41, 45)]
    dec_counts = np.stack([(Bf * ((vals >= a) & (vals <= b))).sum((-1, -2)) for a, b in dec], -1)
    dec_e = dec_counts.sum(-1, keepdims=True) * np.array([(b - a + 1) for a, b in dec]) / M
    out["G 번호대 카이제곱"] = ((dec_counts - dec_e) ** 2 / dec_e).sum(-1)
    resid = hsum - k * vals.mean()                       # 개수 효과 제거한 합계
    out["G 합계 시차1 자기상관"] = corr_rows(resid[:, 1:], resid[:, :-1])
    out["G 개수 시차1 자기상관"] = corr_rows(k[:, 1:], k[:, :-1])
    return out


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


def random_draws(rng, S: int, R: int) -> np.ndarray:
    return np.argsort(rng.random((S, R, 45)), -1)[..., :6] + 1


def main() -> int:
    ap = argparse.ArgumentParser(description="1~low 제외 번호 규칙성 검정")
    ap.add_argument("--csv", type=Path, default=Path("lotto_winners_2020_2026.csv"))
    ap.add_argument("--low", type=int, default=10, help="제외할 상한 (기본 10 → 11~45만)")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rounds, draws = load(args.csv)
    R, M = len(draws), 45 - args.low
    rng = np.random.default_rng(args.seed)
    tie = rng.random(M)                                  # 동점 처리 고정 난수(실측·무작위 공통)

    B_real = to_binary(draws[None], args.low)
    real = {k: v[0] for k, v in stats_batch(B_real, draws[None], args.low, tie).items()}

    sims: dict[str, list] = {k: [] for k in real}
    chunk = 100
    for s0 in range(0, args.sims, chunk):
        d = random_draws(rng, min(chunk, args.sims - s0), R)
        for k, v in stats_batch(to_binary(d, args.low), d, args.low, tie).items():
            sims[k].append(v)
    sims = {k: np.concatenate(v) for k, v in sims.items()}

    k_mean = B_real[0].sum(-1).mean()
    print(f"데이터 {rounds[0]}~{rounds[-1]}회 {R}회차 · {args.low + 1}~45 ({M}개 번호) · "
          f"회차당 평균 {k_mean:.2f}개 (무작위 기대 {6 * M / 45:.2f}) · 무작위 이력 {args.sims}개와 비교\n")
    print(f"{'통계량':<26}{'실측':>10}{'무작위 중앙':>12}{'무작위 95% 범위':>22}{'양측 p':>9}")
    flagged = 0
    for k, v in real.items():
        s = sims[k]
        lo, med, hi = np.percentile(s, [2.5, 50, 97.5])
        p = min(1.0, 2 * min((s >= v).mean(), (s <= v).mean()))
        mark = " *" if p < 0.05 else ""
        flagged += p < 0.05
        print(f"{k:<26}{v:>10.3f}{med:>12.3f}{f'{lo:.3f} ~ {hi:.3f}':>22}{p:>9.3f}{mark}")
    n = len(real)
    print(f"\n검정 {n}개 중 p<0.05: {flagged}개 (모두 무작위여도 평균 {n * 0.05:.1f}개는 나온다)")

    # 참고: 가장 많이/적게 나온 번호와 가장 강한 전이·쌍 (실측)
    B0 = B_real[0].astype(float)
    col = B0.sum(0)
    vals = np.arange(args.low + 1, 46)
    o = np.argsort(col)
    print("\n참고(실측) 최다 출현:", ", ".join(f"{vals[i]}({int(col[i])})" for i in o[::-1][:5]),
          "/ 최소:", ", ".join(f"{vals[i]}({int(col[i])})" for i in o[:5]))
    T = np.einsum("ra,rb->ab", B0[:-1], B0[1:])
    E = np.outer(B0[:-1].sum(0), B0[1:].sum(0)) / (R - 1)
    z = (T - E) / np.sqrt(E)
    top = np.dstack(np.unravel_index(np.argsort(z, None)[::-1][:5], z.shape))[0]
    print("참고(실측) 가장 강한 전이 a→다음회 b:",
          ", ".join(f"{vals[a]}→{vals[b]} (z {z[a, b]:.1f})" for a, b in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
