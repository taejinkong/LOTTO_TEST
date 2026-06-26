#!/usr/bin/env python3
"""구간(분절점 사이) 내 번호 상태별 다음 회차 출현 확률 측정.

가설:
  분절점(1~10 미출현=LOW_1_10=0) 이후 구간이 진행될수록,
  '이미 나온 번호 / 아직 안 나온 번호 / 중복될 번호'의 다음 회차
  출현 확률이 무작위 기대치(6/45=13.3%)와 달라질 것이다.

측정:
  각 구간 안에서 회차를 순서대로 진행하며, 매 회차 직전 시점의
  '구간 내 누적 출현 횟수(prior_count)'별로 그 번호가 이번 회차에
  나왔는지를 집계. prior=0(미출현), 1, 2, 3+ 로 구분.
  무작위라면 모든 prior에서 출현율 = 6/45.

  selection 오염 회피: 분절점이 1~10 정의와 엮이므로 전체(1~45) 외에
  11~45만 따로도 측정한다.

검증:
  - 이항 z검정 (관측 출현율 vs 6/45)
  - 회차 순서 셔플 몬테카를로로 null 분포와 비교

예시:
  python3 segment_number_prob.py
  python3 segment_number_prob.py --montecarlo 500
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from math import sqrt
from pathlib import Path

from lotto_analyzer import Draw, count_g1, load_draws


SEP = "=" * 70
P0 = 6 / 45  # 한 회차에서 특정 번호가 나올 무작위 확률


def segment_ids(draws: list[Draw]) -> list[int]:
    ids: list[int] = []
    cur = 0
    for i, d in enumerate(draws):
        if i > 0 and count_g1(d.numbers) == 0:
            cur += 1
        ids.append(cur)
    return ids


def measure(
    draws: list[Draw],
    number_range: range,
) -> dict[int, tuple[int, int]]:
    """prior_count 버킷 -> (trials, hits). 버킷: 0,1,2,3(=3+)."""
    seg = segment_ids(draws)
    seg_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(seg):
        seg_to_indices[s].append(i)

    trials: dict[int, int] = defaultdict(int)
    hits: dict[int, int] = defaultdict(int)

    for indices in seg_to_indices.values():
        counts = {n: 0 for n in number_range}
        for pos, idx in enumerate(indices):
            nums = set(draws[idx].numbers)
            if pos > 0:  # 첫 회차는 prior 정보 없음 → 예측 대상 제외
                for n in number_range:
                    bucket = min(counts[n], 3)
                    trials[bucket] += 1
                    hits[bucket] += 1 if n in nums else 0
            for n in nums:
                if n in counts:
                    counts[n] += 1
    return {b: (trials[b], hits[b]) for b in sorted(trials)}


def ztest(hits: int, trials: int, p0: float = P0) -> tuple[float, float]:
    """관측 출현율 vs p0 의 이항 z. (rate, z)."""
    if trials == 0:
        return 0.0, 0.0
    rate = hits / trials
    se = sqrt(p0 * (1 - p0) / trials)
    return rate, (rate - p0) / se if se else 0.0


def format_table(title: str, result: dict[int, tuple[int, int]]) -> list[str]:
    labels = {0: "미출현(0회)", 1: "1회 나옴", 2: "2회 나옴", 3: "3회+ 나옴"}
    lines = [
        title,
        f"  {'상태':<14} {'시도':>8} {'출현':>7} {'출현율':>8} {'기대':>7} {'편차%p':>7} {'z':>7}",
        "  " + "─" * 62,
    ]
    # 미출현(0) vs 중복(>=1) 종합도 계산
    not_yet = result.get(0, (0, 0))
    repeat_t = sum(result.get(b, (0, 0))[0] for b in (1, 2, 3))
    repeat_h = sum(result.get(b, (0, 0))[1] for b in (1, 2, 3))

    for b in (0, 1, 2, 3):
        if b not in result:
            continue
        t, h = result[b]
        rate, z = ztest(h, t)
        mark = "**" if abs(z) >= 3 else ("*" if abs(z) >= 2 else "")
        lines.append(
            f"  {labels[b]:<13} {t:>8,} {h:>7,} {rate:>7.2%} {P0:>7.2%} "
            f"{(rate - P0) * 100:>+7.2f} {z:>7.2f}{mark}"
        )

    lines.append("  " + "─" * 62)
    r0, z0 = ztest(*reversed(not_yet)) if not_yet[0] else (0, 0)
    r0, z0 = ztest(not_yet[1], not_yet[0])
    rr, zr = ztest(repeat_h, repeat_t)
    lines.append(
        f"  {'[안나온 번호]':<13} {not_yet[0]:>8,} {not_yet[1]:>7,} {r0:>7.2%} "
        f"{P0:>7.2%} {(r0 - P0) * 100:>+7.2f} {z0:>7.2f}"
    )
    lines.append(
        f"  {'[이미나온 번호]':<12} {repeat_t:>8,} {repeat_h:>7,} {rr:>7.2%} "
        f"{P0:>7.2%} {(rr - P0) * 100:>+7.2f} {zr:>7.2f}"
    )
    return lines


def montecarlo_null(
    draws: list[Draw], number_range: range, n_iter: int, seed: int
) -> list[str]:
    """회차 순서를 셔플해 구간 구조를 무작위화한 null 분포와 실제 비교.

    '안 나온 번호' 출현율의 실제값이 셔플 분포에서 얼마나 벗어나는지(z)로
    구간 내 위치 효과가 진짜인지 검증.
    """
    actual = measure(draws, number_range)
    not_yet = actual.get(0, (0, 0))
    actual_rate = not_yet[1] / not_yet[0] if not_yet[0] else 0.0

    rng = random.Random(seed)
    null_rates: list[float] = []
    shuffled = list(draws)
    for _ in range(n_iter):
        rng.shuffle(shuffled)
        # 셔플 후엔 분절점 위치가 바뀌므로 구간 재계산됨
        res = measure(shuffled, number_range)
        ny = res.get(0, (0, 0))
        if ny[0]:
            null_rates.append(ny[1] / ny[0])

    if not null_rates:
        return ["  몬테카를로: 표본 부족"]

    mean_null = sum(null_rates) / len(null_rates)
    var = sum((r - mean_null) ** 2 for r in null_rates) / len(null_rates)
    sd = sqrt(var)
    z = (actual_rate - mean_null) / sd if sd else 0.0
    mark = "**" if abs(z) >= 3 else ("*" if abs(z) >= 2 else "")
    return [
        "",
        f"몬테카를로 검증 — '안 나온 번호' 출현율 (셔플 {n_iter}회)",
        f"  실제값:        {actual_rate:.3%}",
        f"  셔플 평균:     {mean_null:.3%}",
        f"  셔플 표준편차: {sd:.3%}",
        f"  z = {z:.2f}{mark}   ({'유의 — 구간 위치 효과 실재' if abs(z) >= 2 else '무의미 — 무작위와 동일'})",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="구간 내 번호 상태별 출현 확률 측정.")
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    p.add_argument("--montecarlo", type=int, default=300, help="셔플 반복 수 (0=생략)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    draws = load_draws(Path(args.csv))
    seg = segment_ids(draws)

    out = [
        f"데이터: {draws[0].round_no}~{draws[-1].round_no}회 ({len(draws)}회)",
        f"구간 수: {seg[-1] + 1}  무작위 기대 출현율: {P0:.2%} (6/45)",
        "",
        SEP,
        "구간 내 번호 상태별 다음 회차 출현 확률",
        SEP,
        "",
    ]
    out += format_table("[ 전체 번호 1~45 ]", measure(draws, range(1, 46)))
    out += ["", ""]
    out += format_table(
        "[ 11~45 만 — 분절점(1~10) 오염 제거 ]", measure(draws, range(11, 46))
    )

    if args.montecarlo > 0:
        out += montecarlo_null(draws, range(1, 46), args.montecarlo, args.seed)

    out += [
        "",
        "해석:",
        "  - 모든 z가 |2| 미만이면: 구간 내 이력이 다음 출현에 영향 없음 = 예측 항목 불가",
        "  - '안나온/이미나온' 출현율이 6/45에서 유의하게 벗어나야 신호",
        "  - 몬테카를로 z가 유의해야 '구간 위치' 효과가 진짜 (단순 분포편향 아님)",
    ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
