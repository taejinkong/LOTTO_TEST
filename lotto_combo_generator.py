#!/usr/bin/env python3
"""로또 조합 생성기 — 검증된 비인기 신호 + 사용자 커스터마이즈.

전제(9단계 분석 결론):
  - 1등 6개 적중 확률은 어떤 조합이든 1/8,145,060 으로 동일. 못 올린다.
  - 데이터가 지지하는 유일한 실재 신호: 당첨번호에 1~12가 많을수록 당첨자
    수가 많다(인기). 1~12를 피하면 '당첨 시' 1인당 수령액이 커진다(검증됨).

이 툴은 1등 확률을 속이지 않는다. 같은 1/814만 안에서, 당첨됐을 때
당첨자가 적을 가능성이 높은(=기대 수령액이 큰) 비인기 조합을 생성한다.
사용자 제약(고정수·제외수·과거조합 회피·번호대 분산)을 함께 반영한다.

예시:
  python3 lotto_combo_generator.py --count 10
  python3 lotto_combo_generator.py --fix 30,42 --exclude 4,8,22 --count 5
  python3 lotto_combo_generator.py --max-low12 0 --avoid-past 5
  python3 lotto_combo_generator.py --spread --count 8
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from math import exp
from pathlib import Path
from statistics import mean


SEP = "=" * 66

# 회귀 계수 — 최근 200회(1030~1229) 기준 (signal_stability.py 보수 갱신)
# 과거 전체 기준은 0.058이나 신호 약화로 최근치(0.045)를 채택. 절편도 최근 수준.
# 데이터 갱신 시: python3 signal_stability.py 로 최근 추이 재확인 후 갱신.
INTERCEPT = 2.514
COEF_LOW12 = 0.0446
COEF_BASIS = "최근 200회(1030~1229) 기준"


def load_history(path: Path, recent: int = 200) -> tuple[list[frozenset[int]], float, float]:
    """과거 1등 조합(전체), 평균 당첨자수·평균 풀(원).

    당첨자수/풀 평균은 계수 기준과 일관되게 최근 `recent`회로 산출.
    """
    records = []  # (round_no, frozenset, winners, pool)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            nums = frozenset(int(r[f"번호{k}"]) for k in range(1, 7))
            w = int(r["1등 당첨자수(명)"])
            a = int(r["1등 당첨금액(원)"])
            records.append((int(r["회차"].replace("회", "")), nums, w, w * a))
    records.sort(key=lambda x: x[0])  # 회차 오름차순 (CSV는 내림차순 저장)
    past = [rec[1] for rec in records]
    recent_rec = records[-recent:] if recent and recent < len(records) else records
    recent_w = [rec[2] for rec in recent_rec]
    recent_p = [rec[3] for rec in recent_rec]
    return past, mean(recent_w), mean(recent_p)


def n_low12(nums: tuple[int, ...]) -> int:
    return sum(1 <= n <= 12 for n in nums)


def predict_winners(nums: tuple[int, ...]) -> float:
    return exp(INTERCEPT + COEF_LOW12 * n_low12(nums))


def max_decade_concentration(nums: tuple[int, ...]) -> int:
    from collections import Counter
    return max(Counter(n // 10 for n in nums).values())


def has_low10(nums: tuple[int, ...]) -> bool:
    return any(1 <= n <= 10 for n in nums)


def is_rare_oe(nums: tuple[int, ...]) -> bool:
    """홀짝 희귀 패턴: 0:6, 1:5, 5:1, 6:0."""
    return sum(n % 2 for n in nums) in (0, 1, 5, 6)


# ── 제약 검증 ─────────────────────────────────────────────────────────────────

def passes(
    combo: tuple[int, ...],
    fix: set[int],
    exclude: set[int],
    min_low12: int,
    max_low12: int,
    past: list[frozenset[int]],
    avoid_past: int,
    spread: bool,
    no_low10: bool,
    no_rare_oe: bool,
) -> bool:
    s = set(combo)
    if exclude & s:
        return False
    if not fix <= s:
        return False
    if not min_low12 <= n_low12(combo) <= max_low12:
        return False
    if no_low10 and has_low10(combo):
        return False
    if no_rare_oe and is_rare_oe(combo):
        return False
    # 과거 1등 조합 회피: 완전일치는 항상, avoid_past 지정 시 그 수 이상 겹침도
    overlap_limit = avoid_past if avoid_past > 0 else 6
    for p in past:
        if len(s & p) >= overlap_limit:
            return False
    # 번호대 분산: 한 십의자리에 4개 이상 몰리지 않게 (옵션)
    if spread and max_decade_concentration(combo) >= 4:
        return False
    return True


def generate(
    count: int,
    fix: set[int],
    exclude: set[int],
    min_low12: int,
    max_low12: int,
    past: list[frozenset[int]],
    avoid_past: int,
    spread: bool,
    no_low10: bool,
    no_rare_oe: bool,
    seed: int,
) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    available = [n for n in range(1, 46) if n not in exclude and n not in fix]
    if no_low10:  # 1~10 배제 시 풀에서 미리 제거(탐색 효율)
        available = [n for n in available if n > 10]
    need = 6 - len(fix)

    found: set[tuple[int, ...]] = set()
    max_attempts = count * 40000
    for _ in range(max_attempts):
        if len(found) >= count * 4:  # 충분히 모으면 중단(이후 점수 정렬)
            break
        if need < 0 or need > len(available):
            break
        picks = rng.sample(available, need) if need > 0 else []
        combo = tuple(sorted(list(fix) + picks))
        if combo in found:
            continue
        if passes(combo, fix, exclude, min_low12, max_low12, past, avoid_past,
                  spread, no_low10, no_rare_oe):
            found.add(combo)

    # 비인기(예측 당첨자수 낮은) 순으로 정렬해 상위 count개
    ranked = sorted(found, key=lambda c: (predict_winners(c), c))
    return ranked[:count]


# ── 출력 ──────────────────────────────────────────────────────────────────────

def build_report(combos, avg_winners, avg_pool, args) -> list[str]:
    lines = [
        SEP,
        "로또 조합 생성 — 비인기(기대 수령액 최대화) 전략",
        SEP,
        f"1등 확률: 1/8,145,060 (모든 조합 동일, 불변)",
        f"최근 평균 당첨자: {avg_winners:.1f}명  |  최근 평균 1등 풀: {avg_pool/1e8:.0f}억원",
        f"신호 기준: {COEF_BASIS}  (계수 {COEF_LOW12})",
        "",
        "적용 제약: "
        + ", ".join(filter(None, [
            f"고정수={sorted(args._fix)}" if args._fix else "",
            f"제외수={sorted(args._exclude)}" if args._exclude else "",
            (f"1~12 정확히 {args.min_low12}개" if args.min_low12 == args.max_low12
             else f"1~12 {args.min_low12}~{args.max_low12}개"),
            "1~10 완전배제" if args.no_low10 else "",
            "홀짝 희귀배제" if args.no_rare_oe else "",
            f"과거조합 {args.avoid_past}개+겹침 회피" if args.avoid_past else "과거 완전일치 회피",
            "번호대 분산" if args.spread else "",
        ])),
        "",
        f"  {'조합':<28} {'1~12':>4} {'예측당첨자':>9} {'예측1인당(억)':>12}",
        "  " + "─" * 58,
    ]
    if not combos:
        lines.append("  ⚠ 제약을 만족하는 조합을 찾지 못했습니다. 제약을 완화하세요.")
        return lines
    for c in combos:
        pred = predict_winners(c)
        per_person = avg_pool / pred / 1e8
        lines.append(
            f"  {str(list(c)):<28} {n_low12(c):>4} {pred:>9.1f} {per_person:>12.1f}"
        )

    best = combos[0]
    pred_best = predict_winners(best)
    lines += [
        "",
        f"최적(가장 비인기): {list(best)}",
        f"  예측 당첨자 {pred_best:.1f}명 (평균 {avg_winners:.1f}명 대비 {(pred_best-avg_winners)/avg_winners*100:+.0f}%)",
        f"  예측 1인당 수령액 {avg_pool/pred_best/1e8:.1f}억 "
        f"(평균 조합 대비 약 {(avg_winners/pred_best-1)*100:+.0f}%)",
    ]

    # 구매 장수별 1등 확률 (확률을 올리는 유일한 정직한 방법)
    n_combos = len(combos)
    lines += [
        "",
        f"이 {n_combos}장 기준 1등 적중 확률: {n_combos}/8,145,060 = 1/{8_145_060//n_combos:,}",
        f"  (10배 = {n_combos*10}장, 100배 = {n_combos*100}장 — 장수만이 확률을 올림)",
    ]

    if args.no_low10 or args.no_rare_oe:
        lines += [
            "",
            "  ※ 제약공간 필터 안내:",
            "  - 1~10/홀짝 필터는 후보공간을 줄이지만 '무조건부' 적중률은 그대로입니다.",
            "  - 당첨번호가 우연히 필터를 만족한 회차(약 17%)에서만 조건부로 유리하며,",
            "    그 회차를 미리 알 수 없어 여러 회차 평균은 무제약과 동일합니다.",
        ]

    lines += [
        "",
        "정직한 고지:",
        "  - 1등 당첨 확률은 이 조합도 1/8,145,060 으로 동일합니다.",
        "  - '예측 당첨자'는 당첨됐을 경우의 기대치이며, 효과는 작습니다(모델 R²≈2%).",
        "  - 이 툴은 당첨을 보장하지 않습니다. 당첨 시 수령액 기댓값만 높입니다.",
        "",
        "  ⚠ 신호 약화 주의:",
        "  - 1~12 회피 효과는 과거(916~1070회)엔 강했으나 최근 약화 중입니다.",
        "  - 최근 100회 기준으로는 통계적 효과가 사실상 0에 가깝습니다.",
        "  - 계수는 최근 200회 기준으로 보수적으로 낮췄으나, 방향만 참고하세요.",
    ]
    return lines


def parse_int_list(value: str) -> set[int]:
    if not value:
        return set()
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def validate(args) -> None:
    conflict = args._fix & args._exclude
    if conflict:
        sys.exit(f"오류: 고정수와 제외수가 겹칩니다: {sorted(conflict)}")
    if len(args._fix) > 6:
        sys.exit(f"오류: 고정수는 최대 6개입니다 (현재 {len(args._fix)}개).")
    bad = {n for n in (args._fix | args._exclude) if not 1 <= n <= 45}
    if bad:
        sys.exit(f"오류: 1~45 범위를 벗어난 번호: {sorted(bad)}")
    if args.min_low12 > args.max_low12:
        sys.exit(f"오류: --min-low12({args.min_low12})가 --max-low12({args.max_low12})보다 큽니다.")
    fix_low = n_low12(tuple(args._fix))
    if fix_low > args.max_low12:
        sys.exit(f"오류: 고정수의 1~12 개수({fix_low})가 --max-low12({args.max_low12})를 초과합니다.")
    if args.no_low10:
        fix_low10 = {n for n in args._fix if 1 <= n <= 10}
        if fix_low10:
            sys.exit(f"오류: --no-low10 과 고정수 1~10({sorted(fix_low10)})이 충돌합니다.")
    avail = 45 - len(args._exclude) - len(args._fix)
    if 6 - len(args._fix) > avail:
        sys.exit("오류: 제외수가 너무 많아 6개를 채울 수 없습니다.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="비인기 로또 조합 생성기 (기대 수령액 최대화).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", default="lotto_winners_2020_2026.csv")
    p.add_argument("--count", type=int, default=10, help="생성할 조합 수")
    p.add_argument("--fix", default="", help="반드시 포함할 고정수 (쉼표구분)")
    p.add_argument("--exclude", default="", help="반드시 제외할 번호 (쉼표구분)")
    p.add_argument("--min-low12", type=int, default=0, help="1~12 허용 최소 개수 (기본 0)")
    p.add_argument("--max-low12", type=int, default=1, help="1~12 허용 최대 개수 (기본 1)")
    p.add_argument("--avoid-past", type=int, default=0,
                   help="과거 1등과 N개 이상 겹치면 회피 (0=완전일치만 회피)")
    p.add_argument("--spread", action="store_true", help="한 십의자리 4개+ 몰림 회피")
    p.add_argument("--no-low10", action="store_true", help="1~10 번호 완전 배제")
    p.add_argument("--no-rare-oe", action="store_true", help="홀짝 희귀(0:6/1:5/5:1/6:0) 배제")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args._fix = parse_int_list(args.fix)
    args._exclude = parse_int_list(args.exclude)
    validate(args)

    past, avg_winners, avg_pool = load_history(Path(args.csv))
    combos = generate(
        count=args.count,
        fix=args._fix,
        exclude=args._exclude,
        min_low12=args.min_low12,
        max_low12=args.max_low12,
        past=past,
        avoid_past=args.avoid_past,
        spread=args.spread,
        no_low10=args.no_low10,
        no_rare_oe=args.no_rare_oe,
        seed=args.seed,
    )
    print("\n".join(build_report(combos, avg_winners, avg_pool, args)))


if __name__ == "__main__":
    main()
