#!/usr/bin/env python3
"""앱 '구매 기록'에서 내보낸 JSON을 당첨 CSV와 대조해 누적 성적을 출력한다.

예:
  python3 purchase_report.py lotto-purchases-2026-10-31.json
  python3 purchase_report.py a.json b.json        # 여러 기기 파일을 합쳐 분석(중복 제거)

웹앱 구매 기록 화면과 같은 계산을 한다.
- 1등 금액은 CSV의 1인당 당첨금(원 단위), 4·5등은 고정 5만/5천 원.
- 2·3등은 CSV에 금액이 없어 앱에서 입력한 prizeOverride가 없으면 '미입력'으로 표시한다.
- 같은 회차 조합끼리는 같은 추첨을 공유하므로 z는 근사치다.

구매 JSON에는 개인 구매 내역이 들어 있다. 공개 저장소에 커밋하지 않는다(.gitignore 처리).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

TOTAL = 8_145_060
GRADE_COMBOS = {1: 1, 2: 6, 3: 228, 4: 11_115, 5: 182_780}
FIXED_PRIZE = {4: 50_000, 5: 5_000}
ANY_P = sum(GRADE_COMBOS.values()) / TOTAL
MATCH_MEAN = 0.8
MATCH_VAR = 6 * (6 / 45) * (39 / 45) * (39 / 44)
PRICE = 1_000
MODES = {"app": "앱 생성", "auto": "자동", "manual": "수동", "semi": "반자동", "other": "기타"}


def load_draws(path: Path) -> dict[int, dict]:
    draws = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            r = int(row["회차"].replace("회", ""))
            draws[r] = {
                "nums": {int(row[f"번호{i}"]) for i in range(1, 7)},
                "bonus": int(row["보너스"]),
                "first": int(row["1등 당첨금액(원)"] or 0),
            }
    return draws


def load_purchases(paths: list[Path]) -> list[dict]:
    seen_ids, seen_sig, out = set(), set(), []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rec in payload["records"] if isinstance(payload, dict) else payload:
            sig = (rec["round"], tuple(sorted(",".join(map(str, l["numbers"])) for l in rec["lines"])))
            if rec.get("id") in seen_ids or sig in seen_sig:
                continue
            seen_ids.add(rec.get("id"))
            seen_sig.add(sig)
            out.append(rec)
    return out


def grade(numbers: list[int], draw: dict) -> tuple[int, int | None]:
    m = len(draw["nums"].intersection(numbers))
    if m == 6:
        return m, 1
    if m == 5:
        return m, 2 if draw["bonus"] in numbers else 3
    return m, {4: 4, 3: 5}.get(m)


def stats(entries: list[tuple[dict, list[tuple[int, int | None]], int, bool]]) -> dict:
    results = [res for _, rs, _, _ in entries for res in rs]
    n = len(results)
    grades = Counter(g for _, g in results if g)
    anyp = sum(grades.values())
    mean = sum(m for m, _ in results) / n if n else 0.0
    return {
        "rounds": len({rec["round"] for rec, *_ in entries}),
        "n": n,
        "spent": n * PRICE,
        "prize": sum(p for *_, p, _ in entries),
        "unknown": any(u for *_, u in entries),
        "grades": grades,
        "any": anyp,
        "exp_any": n * ANY_P,
        "mean": mean,
        "z_mean": (mean - MATCH_MEAN) / math.sqrt(MATCH_VAR / n) if n else float("nan"),
        "z_any": (anyp - n * ANY_P) / math.sqrt(n * ANY_P * (1 - ANY_P)) if n else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="구매 기록 누적 분석")
    p.add_argument("json", nargs="+", type=Path, help="앱에서 내보낸 lotto-purchases-*.json")
    p.add_argument("--csv", type=Path, default=Path("lotto_winners_2020_2026.csv"))
    args = p.parse_args()

    draws = load_draws(args.csv)
    records = load_purchases(args.json)
    scored, pending = [], 0
    for rec in records:
        draw = draws.get(rec["round"])
        if not draw:
            pending += len(rec["lines"])
            continue
        rs = [grade(l["numbers"], draw) for l in rec["lines"]]
        prizes = [draw["first"] if g == 1 else FIXED_PRIZE.get(g, 0 if g is None else None) for _, g in rs]
        override = rec.get("prizeOverride")
        unknown = override is None and any(x is None for x in prizes)
        prize = override if override is not None else sum(x or 0 for x in prizes)
        scored.append((rec, rs, prize, unknown))

    print(f"구매 기록 {len(records)}건 · 채점 {sum(len(r[1]) for r in scored)}장 · 추첨 대기 {pending}장")
    if not scored:
        return 0
    s = stats(scored)
    fixed_exp = s["n"] * (FIXED_PRIZE[4] * GRADE_COMBOS[4] + FIXED_PRIZE[5] * GRADE_COMBOS[5]) / TOTAL
    print(f"\n누적 지출 {s['spent']:,}원 · 당첨금 {s['prize']:,}원{' (2·3등 미입력)' if s['unknown'] else ''}"
          f" · 회수율 {s['prize'] / s['spent'] * 100:.1f}%")
    print(f"4·5등 고정 당첨금 무작위 기대 {fixed_exp:,.0f}원 ({fixed_exp / s['spent'] * 100:.1f}%)")
    print("\n등수  내 결과  무작위 기대")
    for g in range(1, 6):
        print(f" {g}등  {s['grades'][g]:>5}장  {s['n'] * GRADE_COMBOS[g] / TOTAL:>10.4f}장")
    print(f"\n평균 일치 {s['mean']:.3f}개 (기대 0.800, z {s['z_mean']:+.2f}) · 5등 이상 {s['any']} / 기대 {s['exp_any']:.2f} (z {s['z_any']:+.2f})")

    by_mode = defaultdict(list)
    for entry in scored:
        by_mode[entry[0].get("mode", "other")].append(entry)
    print("\n방식      표본          평균 일치     5등+/기대     회수율")
    for mode, items in by_mode.items():
        m = stats(items)
        print(f"{MODES.get(mode, mode):<6} {m['rounds']:>3}회 {m['n']:>5}장   {m['mean']:.3f} (z {m['z_mean']:+.2f})"
              f"   {m['any']:>3}/{m['exp_any']:6.2f}   {m['prize'] / m['spent'] * 100:6.1f}%")

    lines = [l["numbers"] for rec in records for l in rec["lines"]]
    freq = Counter(x for ns in lines for x in ns)
    exp = len(lines) * 6 / 45
    chi = sum((freq.get(x, 0) - exp) ** 2 / exp for x in range(1, 46))
    print(f"\n번호 습관 (전체 {len(lines)}장): 상위 {', '.join(f'{x}({c})' for x, c in freq.most_common(5))}"
          f" · 미사용 {45 - len(freq)}개 · 카이제곱 {chi:.1f} (자유도 44, 무작위면 평균 44)")
    low12 = sum(sum(1 for x in ns if x <= 12) for ns in lines) / len(lines)
    print(f"1~12 개수 평균 {low12:.2f} (무작위 {6 * 12 / 45:.2f})")
    print("\n|z|<1.96이면 무작위로 산 것과 구분되지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
