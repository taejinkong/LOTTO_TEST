#!/usr/bin/env python3
"""당첨자 수 다중회귀 — 번호 특성의 독립적 기여 + 통합 설명력.

1단계(popularity_analysis.py)는 특성을 하나씩 봤다. 여기선 다중회귀로
여러 특성을 동시에 넣어 (a) 각 특성의 독립 기여 (b) 다중공선성 (c) 모델
전체 설명력(R²)을 측정한다.

공선성 회피: 번호 구간을 배타적으로 설계.
  n_1_12  (1~12 개수)   ─ 월/날짜 선호 핵심
  n_13_31 (13~31 개수)  ─ 날짜권 나머지
  n_32_45 = 6 - 위 둘   ─ reference(생략), 계수는 '32~45 대비' 해석
추가(구간 독립): consec, same_tail, has_lucky7

타겟: log(당첨자수)

검증: 순열검정(타겟 셔플)으로 R²의 경험적 p값.

예시:
  python3 popularity_regression.py
  python3 popularity_regression.py --permute 5000
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from math import log, sqrt
from pathlib import Path

import numpy as np


SEP = "=" * 70

# 회귀에 넣을 특성 (n_32_45는 reference로 생략)
# A단계: 시각 패턴 특성(max_run, max_decade) 보강, 약했던 consec/has_lucky7 정리
FEATURE_NAMES = ["n_1_12", "n_13_31", "max_run", "max_decade", "same_tail"]
FEATURE_DESC = {
    "n_1_12":     "1~12 개수 (월/날짜 선호)",
    "n_13_31":    "13~31 개수 (날짜권)",
    "max_run":    "최대 연속런 길이 (연속 패턴 선호)",
    "max_decade": "한 십의자리 구간 최대 집중 개수 (예:30번대 몰림)",
    "same_tail":  "끝수 중복 수",
}


def load_data(path: Path, recent: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]], np.ndarray]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            nums = tuple(sorted(int(r[f"번호{k}"]) for k in range(1, 7)))
            rows.append((int(r["회차"].replace("회", "")), nums, int(r["1등 당첨자수(명)"])))
    rows.sort(key=lambda x: x[0])
    if recent > 0:
        rows = rows[-recent:]

    X_list, y_list, combos, win_list = [], [], [], []
    for _, nums, winners in rows:
        X_list.append(feature_vector(nums))
        y_list.append(log(winners))
        win_list.append(winners)
        combos.append(nums)
    return np.array(X_list, float), np.array(y_list, float), combos, np.array(win_list, float)


def feature_vector(nums: tuple[int, ...]) -> list[float]:
    # 최대 연속런 길이
    max_run = run = 1
    for a, b in zip(nums, nums[1:]):
        run = run + 1 if b - a == 1 else 1
        max_run = max(max_run, run)
    # 한 십의자리 구간 최대 집중 (0:1-9, 1:10-19, 2:20-29, 3:30-39, 4:40-45)
    max_decade = max(Counter(n // 10 for n in nums).values())
    tails = [n % 10 for n in nums]
    same_tail = len(tails) - len(set(tails))
    return [
        sum(1 <= n <= 12 for n in nums),
        sum(13 <= n <= 31 for n in nums),
        float(max_run),
        float(max_decade),
        float(same_tail),
    ]


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """절편 포함 OLS. 계수, 표준오차, t, R², adj-R², F."""
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])  # 절편
    p = Xd.shape[1]
    XtX = Xd.T @ Xd
    XtX_inv = np.linalg.pinv(XtX)  # 수치 안정성을 위해 의사역행렬
    beta = XtX_inv @ (Xd.T @ y)
    resid = y - Xd @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    adj_r2 = 1 - (rss / (n - p)) / (tss / (n - 1))
    sigma2 = rss / (n - p)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = beta / se
    f_stat = (r2 / (p - 1)) / ((1 - r2) / (n - p))
    return {"beta": beta, "se": se, "t": t, "r2": r2, "adj_r2": adj_r2, "f": f_stat, "n": n, "p": p}


def vif(X: np.ndarray) -> list[float]:
    """각 특성의 분산팽창계수 (다중공선성 진단). >5면 주의, >10이면 심각."""
    n, k = X.shape
    vifs = []
    for j in range(k):
        others = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        target = X[:, j]
        beta = np.linalg.lstsq(others, target, rcond=None)[0]
        resid = target - others @ beta
        rss = float(resid @ resid)
        tss = float(((target - target.mean()) ** 2).sum())
        r2 = 1 - rss / tss if tss else 0
        vifs.append(1 / (1 - r2) if r2 < 1 else float("inf"))
    return vifs


def permutation_test(X: np.ndarray, y: np.ndarray, actual_r2: float, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    ge = 0
    yp = y.copy()
    for _ in range(n_perm):
        rng.shuffle(yp)
        ge += ols(X, yp)["r2"] >= actual_r2
    return ge / n_perm


def crit_t(n: int) -> float:
    return 2.6  # 다중보정 근사 임계


def analyze(path: Path, recent: int, n_perm: int, seed: int) -> list[str]:
    X, y, combos, winners = load_data(path, recent)
    res = ols(X, y)
    vifs = vif(X)

    lines = [
        SEP,
        "당첨자 수 다중회귀 (타겟 = log 당첨자수)",
        SEP,
        f"회차 수: {res['n']}  특성 수: {res['p']-1} (+절편)  reference 구간: 32~45",
        "",
        f"  {'특성':<12} {'계수':>9} {'표준오차':>8} {'t':>7} {'VIF':>6}",
        "  " + "─" * 50,
    ]
    names = ["(절편)"] + FEATURE_NAMES
    for i, name in enumerate(names):
        v = f"{vifs[i-1]:>6.2f}" if i > 0 else "     -"
        mark = "**" if abs(res["t"][i]) >= crit_t(res["n"]) else ("*" if abs(res["t"][i]) >= 2 else "")
        lines.append(
            f"  {name:<12} {res['beta'][i]:>+9.4f} {res['se'][i]:>8.4f} "
            f"{res['t'][i]:>+7.2f}{mark} {v}"
        )

    lines += [
        "",
        f"모델 설명력: R² = {res['r2']:.4f}  (조정 R² = {res['adj_r2']:.4f})",
        f"  → 번호 특성이 당첨자수(log) 변동의 {res['r2']*100:.1f}%를 설명",
        f"F-통계량: {res['f']:.2f}",
    ]

    if n_perm > 0:
        p = permutation_test(X, y, res["r2"], n_perm, seed)
        lines.append(
            f"순열검정 p값: {p:.4f} ({n_perm}회)  "
            f"→ {'모델 전체 유의 (우연 아님)' if p < 0.05 else '우연 범위'}"
        )

    lines += ["", "특성 해석 (계수>0: 당첨자↑=인기 / <0: 당첨자↓=비인기):"]
    for i, name in enumerate(FEATURE_NAMES, start=1):
        coef = res["beta"][i]
        t = res["t"][i]
        sig = "유의**" if abs(t) >= crit_t(res["n"]) else ("유의*" if abs(t) >= 2 else "비유의")
        direction = "인기(당첨자多)" if coef > 0 else "비인기(당첨자少)"
        lines.append(f"  - {name} ({FEATURE_DESC[name]}): {direction}, {sig}")

    # VIF 경고
    high_vif = [(FEATURE_NAMES[j], vifs[j]) for j in range(len(FEATURE_NAMES)) if vifs[j] > 5]
    if high_vif:
        lines += ["", "⚠ 다중공선성 주의 (VIF>5): " + ", ".join(f"{n}={v:.1f}" for n, v in high_vif)]
    else:
        lines += ["", "✓ 다중공선성 양호 (모든 VIF<5) — 각 계수 독립 해석 가능"]

    # 비인기 점수 적용 예시: 모델로 당첨자수 예측이 가장 낮은/높은 조합
    Xd = np.column_stack([np.ones(res["n"]), X])
    pred = Xd @ res["beta"]
    order = np.argsort(pred)
    lines += [
        "",
        "─" * 70,
        "모델 예측 기준 (참고): 같은 데이터 내 비인기/인기 조합",
        "  ※ 예측 당첨자수 낮은=비인기 조합의 실제 당첨자수도 낮은지 확인",
        "",
        f"  {'유형':<8} {'번호':<26} {'예측log':>8} {'실제당첨자':>9}",
    ]
    for label, idx in [("비인기", order[:3]), ("인기", order[-3:])]:
        for i in idx:
            lines.append(
                f"  {label:<8} {str(list(combos[i])):<26} {pred[i]:>8.2f} {int(winners[i]):>9}"
            )

    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="당첨자 수 다중회귀 분석.")
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    p.add_argument("--recent", type=int, default=0, help="최근 N회차만 (0=전체)")
    p.add_argument("--permute", type=int, default=3000, help="순열검정 횟수 (0=생략)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with np.errstate(all="ignore"):
        report = analyze(Path(args.csv), args.recent, args.permute, args.seed)
    print("\n".join(report))


if __name__ == "__main__":
    main()
