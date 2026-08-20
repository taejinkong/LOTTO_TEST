#!/usr/bin/env python3
"""비인기 조합 전략 — 백테스트 검증 + 조합 생성.

검증된 견고 신호(popularity_regression.py): 당첨번호에 1~12가 많을수록
당첨자 수가 많다(인기). 역으로 1~12를 피하면 당첨 시 당첨자가 적어
기대 수령액이 높아진다. 1등 확률(1/8,145,060)은 바뀌지 않는다.

backtest: 과거 회차를 1~12 개수로 그룹화해 '실제' 평균 당첨자 수 격차 측정
          (판매량 보정 = 풀(당첨자수×1인당금액) 정규화 동시 제시)
generate: 1~12를 배제한 비인기 조합 생성, 예측 당첨자 수와 함께 출력

절편과 계수는 하드코딩하지 않고 실행할 때마다 CSV에서 직접 적합한다
(log(당첨자수) ~ n_1_12, signal_stability.py 와 동일한 정의). 데이터가
갱신되면 값도 따라 움직이므로 별도로 고쳐 넣을 필요가 없다.

기본 적합 창은 최근 200회다. 신호가 시간이 갈수록 약해지고 있어 전체 표본으로
적합하면 현재보다 강한 효과와 낮은 절편(과거의 적은 판매량)을 쓰게 된다.
2026-08-20 기준 최근 200회에서는 |t|<2 로 유의하지 않으며, 그때는 출력에
경고가 함께 찍힌다. 창을 바꾸려면 --fit-window 를 쓴다(0=전체).

예시:
  python3 unpopular_strategy.py --mode backtest
  python3 unpopular_strategy.py --mode generate --count 10
  python3 unpopular_strategy.py --mode both
  python3 unpopular_strategy.py --mode generate --fit-window 300
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from math import exp, log, sqrt
from pathlib import Path
from statistics import mean, pstdev


SEP = "=" * 68


@dataclass
class Row:
    round_no: int
    numbers: tuple[int, ...]
    winners: int
    amount: int


def load_rows(path: Path) -> list[Row]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            nums = tuple(sorted(int(r[f"번호{k}"]) for k in range(1, 7)))
            rows.append(Row(
                round_no=int(r["회차"].replace("회", "")),
                numbers=nums,
                winners=int(r["1등 당첨자수(명)"]),
                amount=int(r["1등 당첨금액(원)"]),
            ))
    return sorted(rows, key=lambda x: x.round_no)


def n_low12(nums: tuple[int, ...]) -> int:
    return sum(1 <= n <= 12 for n in nums)


def welch_t(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    va, vb = pstdev(a) ** 2, pstdev(b) ** 2
    se = sqrt(va / len(a) + vb / len(b))
    return (ma - mb) / se if se else 0.0


# ── 회귀 적합 ─────────────────────────────────────────────────────────────────

@dataclass
class Fit:
    """log(당첨자수) ~ n_1_12 단순회귀 결과."""
    intercept: float
    coef: float
    t: float
    n: int
    first_round: int
    last_round: int
    mean_low12: float   # 적합 구간의 평균 1~12 개수 = '전형적인' 조합

    @property
    def significant(self) -> bool:
        return abs(self.t) >= 2.0

    def predicted(self, n_low: float) -> float:
        """1~12 개수가 n_low 일 때의 예측 당첨자 수."""
        return exp(self.intercept + self.coef * n_low)

    @property
    def typical_winners(self) -> float:
        """같은 모델·같은 구간에서 '전형적인' 조합의 예측 당첨자 수.

        비교 기준을 전체 기간 실제 평균으로 잡으면 안 된다. 절편이 최근 구간에
        맞춰져 있는데 평균은 판매량이 적던 과거를 포함해서, 1~12가 0개인
        비인기 조합조차 '평균보다 당첨자가 많다'고 뒤집혀 보인다.
        """
        return self.predicted(self.mean_low12)


def fit_low12(rows: list[Row]) -> Fit:
    """signal_stability.py 와 동일한 정의로 OLS 적합(외부 의존성 없이)."""
    usable = [r for r in rows if r.winners > 0]
    if len(usable) < 3:
        raise SystemExit("회귀에 필요한 회차 수가 부족합니다.")

    xs = [float(n_low12(r.numbers)) for r in usable]
    ys = [log(r.winners) for r in usable]
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        raise SystemExit("1~12 개수가 모두 같아 회귀할 수 없습니다.")

    coef = sxy / sxx
    intercept = my - coef * mx
    sse = sum((y - (intercept + coef * x)) ** 2 for x, y in zip(xs, ys))
    se = sqrt(sse / (n - 2) / sxx)
    return Fit(
        intercept=intercept,
        coef=coef,
        t=coef / se if se else 0.0,
        n=n,
        first_round=usable[0].round_no,
        last_round=usable[-1].round_no,
        mean_low12=mx,
    )


# ── 백테스트 ──────────────────────────────────────────────────────────────────

def backtest(rows: list[Row]) -> list[str]:
    pool_mean = mean(r.winners * r.amount for r in rows)

    groups: dict[int, list[Row]] = {0: [], 1: [], 2: [], 3: []}
    for r in rows:
        groups[min(n_low12(r.numbers), 3)].append(r)

    lines = [
        SEP,
        "백테스트: 1~12 개수별 '실제' 당첨자 수",
        SEP,
        f"전체 {len(rows)}회  전체 평균 당첨자 {mean(r.winners for r in rows):.1f}명",
        "판매량 보정 = 풀(당첨자수×1인당금액) 평균으로 정규화한 당첨자 수",
        "",
        f"  {'1~12개수':<9} {'회차수':>5} {'실제당첨자':>9} {'보정당첨자':>9} {'1인당금액(억)':>12}",
        "  " + "─" * 52,
    ]
    labels = {0: "0개", 1: "1개", 2: "2개", 3: "3개+"}
    raw_by_group: dict[int, list[float]] = {}
    for g in (0, 1, 2, 3):
        grp = groups[g]
        if not grp:
            continue
        raw = [float(r.winners) for r in grp]
        adj = [r.winners * pool_mean / (r.winners * r.amount) for r in grp]
        amt = [r.amount / 1e8 for r in grp]
        raw_by_group[g] = raw
        lines.append(
            f"  {labels[g]:<9} {len(grp):>5} {mean(raw):>9.1f} {mean(adj):>9.1f} {mean(amt):>12.1f}"
        )

    # 0개(비인기) vs 3+개(인기) 비교
    if 0 in raw_by_group and 3 in raw_by_group:
        lo, hi = raw_by_group[0], raw_by_group[3]
        t = welch_t(hi, lo)
        reduction = (mean(hi) - mean(lo)) / mean(hi) * 100
        lines += [
            "",
            f"  1~12 '3개+' 회차 평균 당첨자: {mean(hi):.1f}명",
            f"  1~12 '0개'  회차 평균 당첨자: {mean(lo):.1f}명",
            f"  → 0개 조합이 {reduction:.0f}% 적은 당첨자  (Welch t={t:.2f}, "
            f"{'유의' if abs(t) >= 2 else '경계/비유의'})",
        ]

    # 연속 검증: n_low12 ↔ winners 단조성
    lines += [
        "",
        "단조성 확인: 1~12 개수가 늘수록 당첨자도 느는가",
        "  " + " → ".join(
            f"{labels[g]}:{mean(raw_by_group[g]):.1f}명" for g in (0, 1, 2, 3) if g in raw_by_group
        ),
    ]
    return lines


# ── 조합 생성 ─────────────────────────────────────────────────────────────────

def predict_log_winners(nums: tuple[int, ...], fit: Fit) -> float:
    """검증된 견고 신호(n_1_12)만으로 예측. 절편+계수는 적합 결과를 그대로 쓴다."""
    return fit.intercept + fit.coef * n_low12(nums)


def generate(
    rows: list[Row], count: int, max_low12: int, seed: int, fit: Fit
) -> list[str]:
    rng = random.Random(seed)
    overall_avg = mean(r.winners for r in rows)

    pool = [n for n in range(13, 46)]  # 1~12 배제 풀
    low_pool = list(range(1, 13))

    combos: set[tuple[int, ...]] = set()
    attempts = 0
    while len(combos) < count and attempts < count * 2000:
        attempts += 1
        k_low = rng.randint(0, max_low12)
        picks = rng.sample(low_pool, k_low) + rng.sample(pool, 6 - k_low)
        combo = tuple(sorted(picks))
        combos.add(combo)

    lines = [
        SEP,
        "비인기 조합 생성 (1등 확률 동일, 당첨 시 당첨자 최소화 지향)",
        SEP,
        f"제약: 1~12 최대 {max_low12}개  |  전체 기간 실제 평균 당첨자 {overall_avg:.1f}명",
        f"예측 당첨자 = exp({fit.intercept:.4f} + {fit.coef:+.4f}×[1~12개수])",
        f"  적합 구간 {fit.first_round}~{fit.last_round}회 ({fit.n}회), t={fit.t:+.2f}"
        f" ({'유의' if fit.significant else '비유의'})  ※ 견고 신호만 사용",
        f"  전형比 = 같은 구간 평균 1~12개수({fit.mean_low12:.2f}개) 조합의 예측값"
        f" {fit.typical_winners:.1f}명 대비",
        "",
        f"  {'조합':<30} {'1~12':>4} {'예측당첨자':>9} {'전형比':>7}",
        "  " + "─" * 54,
    ]
    typical = fit.typical_winners
    ranked = sorted(combos, key=lambda c: predict_log_winners(c, fit))
    for combo in ranked:
        pred = exp(predict_log_winners(combo, fit))
        vs = (pred - typical) / typical * 100
        lines.append(
            f"  {str(list(combo)):<30} {n_low12(combo):>4} {pred:>9.1f} {vs:>+6.0f}%"
        )
    lines += [
        "",
        "주의:",
        "  - 1등 당첨 확률은 1/8,145,060 으로 모든 조합이 동일하다.",
        "  - 이 전략은 '당첨됐을 때' 당첨자가 적어 1인당 수령액이 커지는 것만 노린다.",
        "  - 효과 크기는 작다(모델 R²≈2%). 보장이 아니라 기대값 최적화다.",
        "  - 1~12 개수가 같으면 예측값도 같다. 그 안의 순서는 무작위다.",
    ]
    if not fit.significant:
        lines += [
            "  - ⚠ 이 적합 구간에서 신호는 통계적으로 유의하지 않다(|t|<2).",
            "      방향만 참고하고 예측 당첨자 수의 절대값은 신뢰하지 말 것.",
        ]
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="비인기 조합 전략 백테스트/생성.")
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    p.add_argument("--mode", choices=["backtest", "generate", "both"], default="both")
    p.add_argument("--count", type=int, default=10, help="생성할 조합 수")
    p.add_argument("--max-low12", type=int, default=1, help="1~12 허용 최대 개수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fit-window", type=int, default=200,
        help="회귀 적합에 쓸 최근 회차 수 (0=전체). 신호가 약화 중이라 최근 창을 쓴다",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.csv))
    window = rows if args.fit_window <= 0 else rows[-args.fit_window:]
    fit = fit_low12(window)

    out = [
        f"데이터: {rows[0].round_no}~{rows[-1].round_no}회 ({len(rows)}회)",
        f"회귀 적합: {fit.first_round}~{fit.last_round}회 ({fit.n}회) "
        f"절편 {fit.intercept:.4f}  계수 {fit.coef:+.4f}  t={fit.t:+.2f}"
        f" ({'유의' if fit.significant else '비유의'})",
        "",
    ]
    if args.mode in {"backtest", "both"}:
        out += backtest(rows) + [""]
    if args.mode in {"generate", "both"}:
        out += generate(rows, args.count, args.max_low12, args.seed, fit)
    print("\n".join(out))


if __name__ == "__main__":
    main()
