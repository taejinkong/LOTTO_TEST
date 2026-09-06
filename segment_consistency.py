#!/usr/bin/env python3
"""임의 구간에서 특성들이 '일정한 규칙'을 띠는지 검정한다.

구간을 여러 개 훑으면 우연히 튀는 구간이 반드시 나온다. 그래서 개별 구간의
z 값을 그대로 읽으면 안 된다. 이 스크립트는 그 함정을 셔플 몬테카를로로 막는다.

방법
----
1. 각 회차마다 특성값을 계산한다(1~12 개수, 홀수 개수, 합계, 최대 연속 길이 …).
2. 여러 길이(기본 60·120·250·400회)의 **모든 구간**을 훑어, 구간 평균이 전체
   평균에서 얼마나 떨어졌는지 z 로 잰다. 그중 가장 극단적인 값 max|z| 를 기록한다.
3. 회차 **순서만 섞어서**(값의 집합은 그대로) 같은 스캔을 수백 번 반복한다.
   섞으면 시간 구조가 사라지므로, 이때 나오는 max|z| 분포가 곧 '아무 규칙이
   없을 때 스캔이 만들어내는 착시의 크기'다.
4. 실제 max|z| 가 그 분포의 상위 5% 안에 들면 그 특성은 시간 구조가 있다고 본다.

이 방식은 구간을 몇 개 훑든 다중비교가 자동으로 보정된다. 개별 구간의 z 를
따로 보정할 필요가 없다.

사용법
------
    python3 segment_consistency.py
    python3 segment_consistency.py --shuffles 2000 --lengths 60,120,250,400
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CSV_PATH = Path("lotto_winners_2020_2026.csv")


def load_draws(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rounds, nums = [], []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("회차"):
                continue
            rounds.append(int(row["회차"].replace("회", "")))
            nums.append(sorted(int(row[f"번호{i}"]) for i in range(1, 7)))
    order = np.argsort(rounds)
    return np.array(rounds)[order], np.array(nums)[order]


# ── 특성들: 지금까지 이 프로젝트에서 살펴본 것들 ────────────────────
def features(nums: np.ndarray) -> dict[str, np.ndarray]:
    n = len(nums)
    diffs = np.diff(nums, axis=1)

    def max_run(row):
        best = run = 1
        for d in row:
            run = run + 1 if d == 1 else 1
            best = max(best, run)
        return best

    ac = np.empty(n)
    for i, row in enumerate(nums):
        ac[i] = len({int(b) - int(a) for j, a in enumerate(row) for b in row[j + 1:]}) - 5

    feats = {
        "1~12 개수": (nums <= 12).sum(axis=1).astype(float),
        "1~10 개수": (nums <= 10).sum(axis=1).astype(float),
        "홀수 개수": (nums % 2 == 1).sum(axis=1).astype(float),
        "번호 합계": nums.sum(axis=1).astype(float),
        "최대 연속 길이": np.array([max_run(r) for r in diffs], float),
        "연속 쌍 개수": (diffs == 1).sum(axis=1).astype(float),
        "끝수 종류 수": np.array([len(set(int(v) % 10 for v in r)) for r in nums], float),
        "번호대 종류 수": np.array([len({min((int(v) - 1) // 10, 4) for v in r}) for r in nums], float),
        "AC 값": ac,
        "첫 번호": nums[:, 0].astype(float),
        "평균 간격": diffs.mean(axis=1).astype(float),
    }
    carry = np.zeros(n)
    for i in range(1, n):
        carry[i] = len(set(nums[i]) & set(nums[i - 1]))
    feats["직전 회차 이월수"] = carry
    return feats


def scan_max_z(values: np.ndarray, lengths: list[int]) -> float:
    """모든 구간을 훑어 가장 극단적인 |z| 를 돌려준다(빠른 누적합 방식)."""
    n = len(values)
    mu, sd = values.mean(), values.std(ddof=0)
    if sd == 0:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(values)])
    best = 0.0
    for L in lengths:
        if L >= n:
            continue
        sums = cum[L:] - cum[:-L]                 # 길이 L 인 모든 구간의 합
        z = (sums / L - mu) / (sd / np.sqrt(L))   # 구간 평균의 표준화값
        best = max(best, float(np.abs(z).max()))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="구간별 규칙성 검정(셔플 몬테카를로)")
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--lengths", default="60,120,250,400")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lengths = [int(v) for v in args.lengths.split(",")]
    rounds, nums = load_draws(Path(args.csv))
    feats = features(nums)
    rng = np.random.default_rng(args.seed)

    print(f"데이터: {rounds[0]}~{rounds[-1]}회, {len(rounds)}회")
    print(f"구간 길이: {lengths} → 특성마다 훑는 구간 "
          f"{sum(max(0, len(rounds) - L + 1) for L in lengths):,}개")
    print(f"셔플: {args.shuffles}회 (회차 순서만 섞음, 값의 집합은 그대로)\n")
    print(f"{'특성':16}{'실제 max|z|':>12}{'셔플 95%':>11}{'셔플 최대':>11}{'p':>8}   판정")
    print("-" * 76)

    flagged = []
    for name, values in feats.items():
        observed = scan_max_z(values, lengths)
        null = np.empty(args.shuffles)
        work = values.copy()
        for i in range(args.shuffles):
            rng.shuffle(work)
            null[i] = scan_max_z(work, lengths)
        p = float((null >= observed).mean())
        cut95 = float(np.quantile(null, 0.95))
        verdict = "구조 있음" if p < 0.05 else "무작위 범위"
        if p < 0.05:
            flagged.append((name, observed, p))
        print(f"{name:16}{observed:>12.2f}{cut95:>11.2f}{null.max():>11.2f}{p:>8.3f}   {verdict}")

    print()
    if flagged:
        print("다중비교 보정 후에도 남은 특성:")
        for name, z, p in flagged:
            print(f"  - {name}: max|z|={z:.2f}, p={p:.3f}")
        print("\n주의: 특성 여러 개를 동시에 봤으므로 특성 수만큼 한 번 더 보정해야 한다"
              f"(본페로니 기준 p < {0.05 / len(feats):.4f}).")
    else:
        print("모든 특성이 무작위 범위 안이다. 구간을 아무리 잘라도 시간 구조가 없다.")
    print("\n※ 개별 구간의 z 를 그대로 읽으면 안 된다. 구간을 수천 개 훑으면 |z|>3 은")
    print("   무작위 데이터에서도 흔히 나온다(위 '셔플 최대' 열이 그 크기다).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
