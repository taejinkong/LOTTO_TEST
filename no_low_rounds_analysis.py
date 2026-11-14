#!/usr/bin/env python3
"""1등 본번호 6개가 모두 11 이상(1~10이 하나도 없음)인 회차만 골라 규칙성을 찾는다.

무작위라면 이런 회차의 확률은 C(35,6)/C(45,6) = 19.93%.
비교 기준
  - 회차 배치(언제 나오나): 실제 0/1 순서를 섞은 순열 3000개 (개수 보존)
  - 조합 내부(무엇이 나오나): 11~45에서 6개를 무작위로 뽑은 같은 수의 조합 3000세트
  - 당첨자 수: 판매량으로 나눈 '무작위 기대 대비 당첨자 배수'를 회차 순열로 비교

예:
  python3 no_low_rounds_analysis.py
  python3 no_low_rounds_analysis.py --low 10 --sims 3000
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np

TOTAL = comb(45, 6)


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((
                int(row["회차"].replace("회", "")),
                row["추첨일"],
                [int(row[f"번호{i}"]) for i in range(1, 7)],
                int(row["1등 당첨자수(명)"] or 0),
                int(row["총판매금액(원)"] or 0),
            ))
    rows.sort()
    return rows


def runs_stats(x: np.ndarray) -> dict[str, float]:
    """x: (S, R) 0/1. 연속 회차 지표."""
    out = {}
    both = (x[:, 1:] * x[:, :-1]).sum(-1)
    prev = x[:, :-1].sum(-1)
    out["직전도 해당일 때 이번 해당률%"] = both / np.maximum(prev, 1) * 100
    # 최장 연속
    longest = np.zeros(x.shape[0])
    cur = np.zeros(x.shape[0])
    gaps_max = np.zeros(x.shape[0])
    gcur = np.zeros(x.shape[0])
    for t in range(x.shape[1]):
        cur = np.where(x[:, t] > 0, cur + 1, 0)
        gcur = np.where(x[:, t] > 0, 0, gcur + 1)
        longest = np.maximum(longest, cur)
        gaps_max = np.maximum(gaps_max, gcur)
    out["최장 연속 해당(회)"] = longest
    out["최장 공백(회)"] = gaps_max
    # 간격의 변동계수 (주기성 있으면 작아짐)
    cv = []
    for row in x:
        idx = np.flatnonzero(row)
        g = np.diff(idx)
        cv.append(g.std() / g.mean() if len(g) > 1 else np.nan)
    out["간격 변동계수"] = np.array(cv)
    # 전반/후반 비율 차
    h = x.shape[1] // 2
    out["후반-전반 해당률차(%p)"] = (x[:, h:].mean(-1) - x[:, :h].mean(-1)) * 100
    return out


def combo_stats(C: np.ndarray, low: int) -> dict[str, np.ndarray]:
    """C: (S, N, 6) 조합(11~45). 시간 순서대로."""
    S, N, _ = C.shape
    M = 45 - low
    B = np.zeros((S, N, M))
    for c in range(6):
        oh = np.zeros((S, N, M))
        np.put_along_axis(oh, (C[..., c] - low - 1)[..., None], 1, -1)
        B += oh
    vals = np.arange(low + 1, 46)
    out = {}
    col = B.sum(1)
    e = N * 6 / M
    out["번호 빈도 카이제곱"] = ((col - e) ** 2 / e).sum(-1)
    iu = np.triu_indices(M, 1)
    P = np.einsum("sna,snb->sab", B, B)[:, iu[0], iu[1]]
    ep = N * 15 / len(iu[0])
    out["동반 쌍 카이제곱"] = ((P - ep) ** 2 / ep).sum(-1)
    out["최다 동반 쌍 횟수"] = P.max(-1)
    s = (B * vals).sum(-1)
    out["합계 평균"] = s.mean(-1)
    out["합계 표준편차"] = s.std(-1)
    out["홀수 개수 평균"] = (B * (vals % 2)).sum(-1).mean(-1)
    out["연번 쌍 평균"] = (B[..., 1:] * B[..., :-1]).sum(-1).mean(-1)
    out["최솟값 평균"] = C.min(-1).mean(-1)
    out["최댓값 평균"] = C.max(-1).mean(-1)
    dec = [(11, 20), (21, 30), (31, 40), (41, 45)]
    dc = np.stack([(B * ((vals >= a) & (vals <= b))).sum((-1, -2)) for a, b in dec], -1)
    de = N * 6 * np.array([b - a + 1 for a, b in dec]) / M
    out["번호대 카이제곱"] = ((dc - de) ** 2 / de).sum(-1)
    ov = (B[:, 1:] * B[:, :-1]).sum(-1)
    out["연이은 해당 회차끼리 겹침"] = ov.mean(-1)
    out["해당 회차 합계 자기상관"] = corr(s[:, 1:], s[:, :-1])
    return out


def corr(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


def report(title, real, sims):
    print(f"\n[{title}]")
    print(f"{'통계량':<24}{'실측':>10}{'무작위 중앙':>12}{'무작위 95% 범위':>22}{'양측 p':>9}")
    flagged = 0
    for k, v in real.items():
        s = sims[k][~np.isnan(sims[k])]
        lo, med, hi = np.percentile(s, [2.5, 50, 97.5])
        p = min(1.0, 2 * min((s >= v).mean(), (s <= v).mean()))
        flagged += p < 0.05
        print(f"{k:<24}{v:>10.3f}{med:>12.3f}{f'{lo:.3f} ~ {hi:.3f}':>22}{p:>9.3f}{' *' if p < 0.05 else ''}")
    return len(real), flagged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("lotto_winners_2020_2026.csv"))
    ap.add_argument("--low", type=int, default=10)
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load(args.csv)
    R = len(rows)
    x = np.array([min(n) > args.low for _, _, n, _, _ in rows], dtype=float)
    p0 = comb(45 - args.low, 6) / TOTAL
    hit = [r for r, flag in zip(rows, x) if flag]
    N = len(hit)
    sd = np.sqrt(R * p0 * (1 - p0))
    print(f"데이터 {rows[0][0]}~{rows[-1][0]}회 {R}회차 · 1~{args.low} 없는 회차 {N}회 ({N / R * 100:.1f}%)"
          f" · 무작위 기대 {R * p0:.1f}회 ({p0 * 100:.2f}%) · z {(N - R * p0) / sd:+.2f}")
    print("최근 해당 회차:", ", ".join(f"{r}회 {'-'.join(map(str, n))}" for r, _, n, _, _ in hit[-8:]))
    last = rows[-1][0] - hit[-1][0]
    print(f"마지막 해당 이후 {last}회 경과 (무작위라면 {last}회 이상 연속 미해당 확률 {(1 - p0) ** last * 100:.0f}%)")

    # 1) 언제 나오나
    real = {k: v[0] for k, v in runs_stats(x[None]).items()}
    perm = np.array([rng.permutation(x) for _ in range(args.sims)])
    sims = runs_stats(perm)
    n1, f1 = report("언제 나오나 — 회차 순서 섞기와 비교", real, sims)

    # 2) 무엇이 나오나
    C = np.array([sorted(n) for _, _, n, _, _ in hit])[None]
    real = {k: v[0] for k, v in combo_stats(C, args.low).items()}
    pool = np.arange(args.low + 1, 46)
    sims = {}
    for s0 in range(0, args.sims, 250):
        S = min(250, args.sims - s0)
        Cs = np.sort(pool[np.argsort(rng.random((S, N, len(pool))), -1)[..., :6]], -1)
        for k, v in combo_stats(Cs, args.low).items():
            sims.setdefault(k, []).append(v)
    sims = {k: np.concatenate(v) for k, v in sims.items()}
    n2, f2 = report(f"무엇이 나오나 — 11~45 무작위 6개 조합 {N}개와 비교", real, sims)

    # 3) 당첨자 수 (판매량 대비)
    sales = np.array([r[4] for r in rows], dtype=float)
    ok = sales > 0                                      # 판매액이 비어 있는 회차는 뺀다
    w = np.array([r[3] for r in rows], dtype=float)[ok]
    ratio = w / (sales[ok] / 1000 / TOTAL)
    mask = (x > 0)[ok]
    def gap(m):
        return np.log(ratio[m] + 0.5).mean() - np.log(ratio[~m] + 0.5).mean()
    g = gap(mask)
    gs = np.array([gap(rng.permutation(mask)) for _ in range(args.sims)])
    pv = min(1, 2 * min((gs >= g).mean(), (gs <= g).mean()))
    print(f"\n[당첨자 수 — 판매액 있는 {ok.sum()}회차, 판매량 기준 무작위 기대 대비 배수]")
    print(f"해당 회차 {mask.sum()}회 중앙값 {np.median(ratio[mask]):.2f}배 · 나머지 {np.median(ratio[~mask]):.2f}배"
          f" · 평균 당첨자 {w[mask].mean():.1f}명 vs {w[~mask].mean():.1f}명 · 순열 p {pv:.4f}")

    n, f = n1 + n2, f1 + f2
    print(f"\n검정 {n}개 중 p<0.05: {f}개 (모두 무작위여도 평균 {n * 0.05:.1f}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
