#!/usr/bin/env python3
"""1~10이 없는 1등 조합끼리만 모아 기존 분석 요소별 출현 확률을 본다.

기준은 11~45에서 만들 수 있는 6개 조합 C(35,6)=1,623,160개를 전부 세어 얻은 '정확한 이론 확률'.
요소(lotto_analyzer.py와 같은 정의, 1~10이 없으니 번호대·첫번호·합계 구간은 11~45에 맞춰 나눔)
  홀짝, 합계 구간, 번호대 패턴(11-20/21-30/31-40/41-45 개수), 끝수 중복, 31~45 개수, 연번 쌍,
  AC값, 평균 간격, 첫 번호, 마지막 번호, 직전 회차 이월 개수, 직전 회차 이웃수 개수
검정
  1) 요소별 카이제곱(실측 vs 이론) — p값은 이론 분포에서 같은 수(N)를 뽑는 몬테카를로
  2) 전반/후반 재현성: 앞 절반에서 이론보다 많던 값이 뒤 절반에서도 많은가
  3) 워크포워드: 앞선 해당 회차들의 실측 빈도로 다음 해당 회차를 예측하면 이론 확률보다 나은가
     (로그우도 차이, 0보다 크면 실측 빈도에 쓸모 있는 정보가 있다는 뜻)

예: python3 no_low_features.py --sims 2000
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from itertools import combinations
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


def feats(C: np.ndarray) -> dict[str, np.ndarray]:
    """C: (N,6) 정렬된 조합 → 요소별 범주 라벨(문자열 배열)."""
    out = {}
    odd = (C % 2).sum(1)
    out["홀짝(홀:짝)"] = np.array([f"{o}:{6 - o}" for o in odd])
    s = C.sum(1)
    out["합계"] = np.select([s < 140, s < 160, s < 180, s < 200], ["<140", "140-159", "160-179", "180-199"], "200+")
    bands = [((C >= a) & (C <= b)).sum(1) for a, b in ((11, 20), (21, 30), (31, 40), (41, 45))]
    out["번호대 패턴"] = np.array(["-".join(map(str, t)) for t in zip(*bands)])
    for name, cnt in zip(("11~20 개수", "21~30 개수", "31~40 개수", "41~45 개수"), bands):
        out[name] = cnt.astype(str)
    tails = C % 10
    dup = np.zeros(len(C), int)
    mx = np.zeros(len(C), int)
    for d in range(10):
        c = (tails == d).sum(1)
        dup += c >= 2
        mx = np.maximum(mx, c)
    out["끝수 중복"] = np.array([f"{a},max{b}" for a, b in zip(dup, mx)])
    out["31~45 개수"] = (C >= 31).sum(1).astype(str)
    cons = (np.diff(C, axis=1) == 1).sum(1)
    out["연번 쌍"] = np.where(cons >= 3, "3+", cons.astype(str))
    diffs = np.stack([C[:, j] - C[:, i] for i, j in combinations(range(6), 2)], 1)
    dsort = np.sort(diffs, 1)
    distinct = 1 + (np.diff(dsort, axis=1) != 0).sum(1)
    out["AC값"] = (distinct - 5).astype(str)
    gap = (C[:, 5] - C[:, 0]) / 5
    out["평균 간격"] = np.select([gap < 4, gap < 5, gap < 6], ["<4", "4-5", "5-6"], "6+")
    f0 = C[:, 0]
    out["첫 번호"] = np.select([f0 <= 12, f0 <= 15, f0 <= 20], ["11-12", "13-15", "16-20"], "21+")
    l0 = C[:, 5]
    out["마지막 번호"] = np.select([l0 <= 38, l0 <= 41, l0 <= 43], ["~38", "39-41", "42-43"], "44-45")
    return out


def chi(obs: Counter, probs: dict[str, float], n: int) -> float:
    """기대 5 미만 범주는 하나로 합쳐 카이제곱."""
    small = [k for k, p in probs.items() if p * n < 5]
    o = {k: obs.get(k, 0) for k in probs if k not in small}
    e = {k: probs[k] * n for k in probs if k not in small}
    if small:
        o["_etc"] = sum(obs.get(k, 0) for k in small)
        e["_etc"] = sum(probs[k] for k in small) * n
    return sum((o[k] - e[k]) ** 2 / e[k] for k in e if e[k] > 0)


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
    prevs = [set(rows[i - 1][1]) if i > 0 else set() for i in idx]
    N = len(real)

    allc = np.array(list(combinations(range(LOW + 1, 46), 6)), dtype=np.int16)
    F_all = feats(allc)
    F_real = feats(real)

    # 직전 회차 대비 이월/이웃: 이론은 각 회차의 실제 직전 번호에 대해 11~45 조합 전체로 계산
    allbits = np.zeros((len(allc), 46), bool)
    np.put_along_axis(allbits, allc.astype(int), True, 1)
    carry_p, nb_p = Counter(), Counter()
    carry_obs, nb_obs = [], []
    trans_samples_c, trans_samples_n = [], []
    for nums, prev in zip(real, prevs):
        pv = np.zeros(46, bool)
        pv[list(prev)] = True
        nbmask = np.zeros(46, bool)
        for p in prev:
            for q in (p - 1, p + 1):
                if 1 <= q <= 45:
                    nbmask[q] = True
        c = (allbits & pv).sum(1)
        nb = (allbits & nbmask).sum(1)
        nb = np.minimum(nb, 4)
        for v, k in zip(*np.unique(c, return_counts=True)):
            carry_p[str(v)] += k / len(allc) / N
        for v, k in zip(*np.unique(nb, return_counts=True)):
            nb_p["4+" if v == 4 else str(v)] += k / len(allc) / N
        carry_obs.append(str(len(prev & set(nums))))
        v = sum(1 for x in nums if x - 1 in prev or x + 1 in prev)
        nb_obs.append("4+" if v >= 4 else str(v))
        pick = rng.integers(0, len(allc), args.sims)
        trans_samples_c.append(c[pick].astype(str))
        trans_samples_n.append(np.where(nb[pick] >= 4, "4+", nb[pick].astype(str)))
    F_real["직전 이월 개수"] = np.array(carry_obs)
    F_real["직전 이웃수 개수"] = np.array(nb_obs)

    print(f"1~10 없는 1등 조합 {N}개 (전체 {len(rows)}회 중) · 이론 기준 11~45 조합 {len(allc):,}개 전수\n")

    probs_all = {}
    for name in F_real:
        if name in F_all:
            k, cnt = np.unique(F_all[name], return_counts=True)
            probs_all[name] = dict(zip(k, cnt / len(allc)))
    probs_all["직전 이월 개수"] = dict(carry_p)
    probs_all["직전 이웃수 개수"] = dict(nb_p)

    summary = []
    for name, probs in probs_all.items():
        obs = Counter(F_real[name])
        stat = chi(obs, probs, N)
        # 몬테카를로 p
        if name in F_all:
            sims = [chi(Counter(F_all[name][rng.integers(0, len(allc), N)]), probs, N) for _ in range(args.sims)]
        else:
            src = trans_samples_c if "이월" in name else trans_samples_n
            M = np.stack(src)  # (N, sims)
            sims = [chi(Counter(M[:, j]), probs, N) for j in range(args.sims)]
        p = (np.array(sims) >= stat).mean()
        # 전반/후반 재현성: (관측-기대) 부호 일치 상관
        h = N // 2
        o1, o2 = Counter(F_real[name][:h]), Counter(F_real[name][h:])
        keys = [k for k in probs if probs[k] * h >= 5]
        r = np.nan
        if len(keys) >= 3:
            d1 = [(o1.get(k, 0) - probs[k] * h) / np.sqrt(probs[k] * h) for k in keys]
            d2 = [(o2.get(k, 0) - probs[k] * (N - h)) / np.sqrt(probs[k] * (N - h)) for k in keys]
            r = np.corrcoef(d1, d2)[0, 1]
        summary.append((name, stat, p, r))

        print(f"■ {name}  (카이제곱 {stat:.1f}, 무작위 대비 p {p:.3f})")
        top = sorted(probs.items(), key=lambda kv: -kv[1])
        shown = [kv for kv in top if kv[1] >= 0.01][:8]
        line = []
        for k, pr in shown:
            o = obs.get(k, 0)
            z = (o - pr * N) / np.sqrt(N * pr * (1 - pr))
            line.append(f"{k}: 실측 {o / N * 100:.1f}% / 이론 {pr * 100:.1f}%{' ▲' if z > 1.96 else ' ▼' if z < -1.96 else ''}")
        for i in range(0, len(line), 3):
            print("   " + "   |   ".join(line[i:i + 3]))

    # 3) 워크포워드 로그우도: 실측 빈도(직전까지, 이론 사전 50회분으로 평활) vs 이론
    def wf_gain(lab: np.ndarray, probs: dict[str, float], prior: float = 50, start: int = 50) -> float:
        cnt = Counter(lab[:start])
        g = 0.0
        for t in range(start, len(lab)):
            k = lab[t]
            emp = (cnt.get(k, 0) + prior * probs.get(k, 1e-9)) / (t + prior)
            g += np.log(emp / max(probs.get(k, 1e-9), 1e-12))
            cnt[k] += 1
        return g

    static = [n for n in probs_all if n in F_all]
    gain = sum(wf_gain(F_real[n], probs_all[n]) for n in static)
    null = []
    for _ in range(min(args.sims, 500)):
        pick = rng.integers(0, len(allc), N)
        null.append(sum(wf_gain(F_all[n][pick], probs_all[n]) for n in static))
    null = np.array(null)
    lo, hi = np.percentile(null, [2.5, 97.5])
    print(f"\n[워크포워드] 앞선 해당 회차의 실측 빈도로 다음 해당 회차 요소를 예측한 로그우도 이득 합: {gain:+.2f}")
    print(f"  (이론 분포에서 뽑은 무작위 이력이면 95% 범위 {lo:+.2f} ~ {hi:+.2f}, 이득이 이 위로 넘어야 실측 빈도가 쓸모 있음)")

    # 번호별 출현 확률
    cnt = Counter(real.ravel().tolist())
    exp = N * 6 / 35
    order = sorted(range(LOW + 1, 46), key=lambda x: -cnt[x])
    print(f"\n[번호별 출현] 해당 조합 {N}개 안에서 각 번호 이론 출현 확률 {6 / 35 * 100:.1f}% (기대 {exp:.1f}회)")
    print("  많이: " + ", ".join(f"{x}({cnt[x]}회 {cnt[x] / N * 100:.1f}%)" for x in order[:6]))
    print("  적게: " + ", ".join(f"{x}({cnt[x]}회 {cnt[x] / N * 100:.1f}%)" for x in order[-6:]))
    sd = np.sqrt(N * (6 / 35) * (29 / 35))
    print(f"  무작위 95% 범위: 번호당 약 {exp - 1.96 * sd:.0f}~{exp + 1.96 * sd:.0f}회 (35개 중 1~2개는 벗어나는 게 정상)")

    flagged = [s for s in summary if s[2] < 0.05]
    print(f"\n요소 {len(summary)}개 중 p<0.05: {len(flagged)}개 {[s[0] for s in flagged]} (모두 무작위여도 평균 {len(summary) * 0.05:.1f}개)")
    rs = [s[3] for s in summary if not np.isnan(s[3])]
    print(f"전반/후반 치우침 방향 상관 평균 {np.mean(rs):+.2f} (재현되는 치우침이면 +, 우연이면 0 근처)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
