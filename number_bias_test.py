#!/usr/bin/env python3
"""번호별 출현 빈도 카이제곱 적합도 검정 — 추첨기 물리 편향 탐지.

지금까지의 검증은 모두 '회차 간 관계'(시점/구간/이력)를 봤고 전부 무작위였다.
이 검정은 시점과 무관하게 '45개 번호가 균등하게 나오는가'만 본다.
특정 번호가 물리적으로 더/덜 나온다면(공·기계 편향) selection bias 없이
실재할 수 있는 유일한 신호다.

검정:
  1. 본번호(6개) 카이제곱 적합도: H0 = 모든 번호 균등(각 기대빈도 동일)
  2. 본번호+보너스(7개) 카이제곱
  3. 끝수(0~9), 홀짝, 번호대 구간 균등성 보조 검정
  4. 순열 몬테카를로로 카이제곱 통계량의 경험적 p값 교차검증

자유도 44 카이제곱 임계값: 0.05→60.48, 0.01→69.96

예시:
  python3 number_bias_test.py
  python3 number_bias_test.py --montecarlo 2000
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from lotto_analyzer import Draw, load_draws


SEP = "=" * 68

# 카이제곱 임계값 (자유도: 상위확률)
CHI2_CRIT = {
    44: {0.10: 56.37, 0.05: 60.48, 0.01: 69.96},
    9:  {0.10: 14.68, 0.05: 16.92, 0.01: 21.67},
    1:  {0.10: 2.71,  0.05: 3.84,  0.01: 6.63},
    4:  {0.10: 7.78,  0.05: 9.49,  0.01: 13.28},
}


def chi_square(observed: list[int], expected: list[float]) -> float:
    return sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)


def verdict(stat: float, df: int) -> str:
    crit = CHI2_CRIT.get(df, {})
    if not crit:
        return "임계값표 없음"
    if stat >= crit[0.01]:
        return f"H0 기각(p<0.01) — 편향 강함"
    if stat >= crit[0.05]:
        return f"H0 기각(p<0.05) — 편향 의심"
    if stat >= crit[0.10]:
        return f"경계(p<0.10)"
    return "H0 유지 — 균등(편향 없음)"


def number_counts(draws: list[Draw], include_bonus: bool) -> Counter:
    c: Counter = Counter()
    for d in draws:
        c.update(d.numbers)
        if include_bonus:
            c[d.bonus] += 1
    return c


def test_numbers(draws: list[Draw], include_bonus: bool, title: str) -> tuple[list[str], float]:
    counts = number_counts(draws, include_bonus)
    balls_per_draw = 7 if include_bonus else 6
    total = len(draws) * balls_per_draw
    expected = total / 45
    observed = [counts.get(n, 0) for n in range(1, 46)]
    stat = chi_square(observed, [expected] * 45)

    ranked = sorted(((counts.get(n, 0), n) for n in range(1, 46)), reverse=True)
    hot = ranked[:5]
    cold = ranked[-5:]

    lines = [
        title,
        f"  총 추출: {total}회  번호당 기대빈도: {expected:.1f}회",
        f"  카이제곱(df=44): {stat:.2f}   → {verdict(stat, 44)}",
        f"  최다출현: " + ", ".join(f"{n}번({c}회)" for c, n in hot),
        f"  최소출현: " + ", ".join(f"{n}번({c}회)" for c, n in reversed(cold)),
        f"  최다-최소 격차: {hot[0][0] - cold[-1][0]}회 "
        f"(기대 표준편차 ±{(expected * (1 - 1/45)) ** 0.5:.1f}회)",
    ]
    return lines, stat


def test_tails(draws: list[Draw]) -> list[str]:
    counts: Counter = Counter()
    for d in draws:
        counts.update(n % 10 for n in d.numbers)
    total = sum(counts.values())
    expected = [total * (5 if t in (0,) else 5) / 45 for t in range(10)]
    # 끝수 분포: 1~45에서 끝수 0은 {10,20,30,40}=4개, 1~9는 각 5개(1,11,21,31,41 등),
    # 끝수 5는 {5,15,25,35,45}=5개. 정확히:
    tail_pool = Counter(n % 10 for n in range(1, 46))
    exp = [total * tail_pool[t] / 45 for t in range(10)]
    obs = [counts.get(t, 0) for t in range(10)]
    stat = chi_square(obs, exp)
    return [
        "[ 끝수(0~9) 균등성 ]",
        f"  카이제곱(df=9): {stat:.2f}   → {verdict(stat, 9)}",
        "  끝수별 출현: " + ", ".join(f"{t}:{counts.get(t,0)}" for t in range(10)),
    ]


def test_oddeven(draws: list[Draw]) -> list[str]:
    odd = sum(1 for d in draws for n in d.numbers if n % 2)
    total = len(draws) * 6
    even = total - odd
    # 1~45: 홀수 23개, 짝수 22개
    exp_odd = total * 23 / 45
    exp_even = total * 22 / 45
    stat = chi_square([odd, even], [exp_odd, exp_even])
    return [
        "[ 홀짝 균등성 ]",
        f"  카이제곱(df=1): {stat:.2f}   → {verdict(stat, 1)}",
        f"  홀수 {odd}회(기대 {exp_odd:.0f}) / 짝수 {even}회(기대 {exp_even:.0f})",
    ]


def test_bands(draws: list[Draw]) -> list[str]:
    bands = [(1, 9), (10, 18), (19, 27), (28, 36), (37, 45)]  # 9개씩 균등 5구간
    counts = [sum(1 for d in draws for n in d.numbers if lo <= n <= hi) for lo, hi in bands]
    total = len(draws) * 6
    exp = [total / 5] * 5
    stat = chi_square(counts, exp)
    return [
        "[ 번호대 5구간(9개씩) 균등성 ]",
        f"  카이제곱(df=4): {stat:.2f}   → {verdict(stat, 4)}",
        "  구간별: " + ", ".join(f"{lo}-{hi}:{c}" for (lo, hi), c in zip(bands, counts)),
    ]


def montecarlo_pvalue(draws: list[Draw], actual_stat: float, n_iter: int, seed: int) -> list[str]:
    """순열 몬테카를로: 무작위 추첨을 n_iter번 생성해 카이제곱 통계량 null 분포 구축.

    실제 통계량보다 큰 무작위 통계량 비율 = 경험적 p값.
    """
    rng = random.Random(seed)
    n_draws = len(draws)
    ge = 0
    null_stats = []
    expected = n_draws * 6 / 45
    for _ in range(n_iter):
        c: Counter = Counter()
        for _ in range(n_draws):
            c.update(rng.sample(range(1, 46), 6))
        obs = [c.get(n, 0) for n in range(1, 46)]
        s = chi_square(obs, [expected] * 45)
        null_stats.append(s)
        if s >= actual_stat:
            ge += 1
    null_stats.sort()
    p = ge / n_iter
    mean_null = sum(null_stats) / len(null_stats)
    return [
        "",
        f"순열 몬테카를로 교차검증 (무작위 추첨 {n_iter}회 시뮬레이션)",
        f"  실제 카이제곱:     {actual_stat:.2f}",
        f"  무작위 평균:       {mean_null:.2f}  (이론 기대 ≈ 44)",
        f"  무작위 95% 지점:   {null_stats[int(len(null_stats)*0.95)]:.2f}",
        f"  경험적 p값:        {p:.3f}   "
        f"({'편향 신호 있음' if p < 0.05 else '무작위와 구분 안 됨'})",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="번호 편향 카이제곱 검정.")
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    p.add_argument("--montecarlo", type=int, default=2000, help="순열 시뮬 횟수 (0=생략)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    draws = load_draws(Path(args.csv))

    out = [
        f"데이터: {draws[0].round_no}~{draws[-1].round_no}회 ({len(draws)}회)",
        "",
        SEP,
        "번호 편향 카이제곱 적합도 검정",
        SEP,
        "H0(귀무가설): 45개 번호가 모두 균등하게 추첨된다 (편향 없음)",
        "",
    ]

    main_lines, main_stat = test_numbers(draws, include_bonus=False, title="[ 본번호 6개 ]")
    bonus_lines, _ = test_numbers(draws, include_bonus=True, title="[ 본번호+보너스 7개 ]")
    out += main_lines + [""] + bonus_lines + [""]
    out += test_tails(draws) + [""]
    out += test_oddeven(draws) + [""]
    out += test_bands(draws)

    if args.montecarlo > 0:
        out += montecarlo_pvalue(draws, main_stat, args.montecarlo, args.seed)

    out += [
        "",
        "해석:",
        "  - 본번호 카이제곱 < 60.48 이면 번호 편향 없음 = 핫넘버/콜드넘버는 착시",
        "  - 몬테카를로 p값 >= 0.05 면 이론 검정과 일치(편향 없음 확정)",
        "  - 편향이 없다는 건 '많이 나온 번호'에 베팅할 근거가 통계적으로 없다는 뜻",
    ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
