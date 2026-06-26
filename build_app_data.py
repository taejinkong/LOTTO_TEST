#!/usr/bin/env python3
"""LOTTO PWA 데이터 빌드.

CSV를 읽어 앱이 오프라인으로 쓸 수 있는 app/data.js 를 생성한다.
프론트엔드(fetch 없이 동작)를 위해 window.LOTTO 전역에 모든 데이터를 담는다.

생성 내용:
- meta / draws(압축 배열)
- 번호 빈도, 홀짝, 합계, 1~12 개수 분포
- 비인기 전략 백테스트(1~12 그룹별 실제/보정 당첨자)
- 신호 안정성(4분할 OLS 계수)
- 조합 생성 회귀계수(최근 200회 log당첨자 ~ 1~12개수 OLS)
- 인사이트(우리 대화의 검증 결론)

사용:
  python3 build_app_data.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import exp, log, sqrt
from pathlib import Path
from statistics import mean, pstdev

CSV_PATH = Path("lotto_winners_2020_2026.csv")
OUT_PATH = Path("app/data.js")


@dataclass
class Row:
    round_no: int
    date: str
    numbers: tuple[int, ...]
    bonus: int
    winners: int
    amount: int


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            nums = tuple(sorted(int(r[f"번호{k}"]) for k in range(1, 7)))
            rows.append(
                Row(
                    round_no=int(r["회차"].replace("회", "")),
                    date=r["추첨일"].strip(),
                    numbers=nums,
                    bonus=int(r["보너스"]),
                    winners=int(r["1등 당첨자수(명)"]),
                    amount=int(r["1등 당첨금액(원)"]),
                )
            )
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


def ols_single(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """y = a + b x. (intercept, slope, t_slope) 반환."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return my, 0.0, 0.0
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sse = sum(e * e for e in resid)
    dof = n - 2
    se_b = sqrt((sse / dof) / sxx) if dof > 0 and sxx > 0 else 0.0
    t = b / se_b if se_b else 0.0
    return a, b, t


def build_stats(rows: list[Row]) -> dict:
    freq = {n: 0 for n in range(1, 46)}
    for r in rows:
        for n in r.numbers:
            freq[n] += 1

    odd_even: dict[str, int] = {}
    for r in rows:
        odd = sum(n % 2 for n in r.numbers)
        key = f"{odd}:{6 - odd}"
        odd_even[key] = odd_even.get(key, 0) + 1

    # 합계 분포(20 단위 구간)
    sum_bins: dict[str, int] = {}
    for r in rows:
        s = sum(r.numbers)
        lo = (s // 20) * 20
        key = f"{lo}-{lo + 19}"
        sum_bins[key] = sum_bins.get(key, 0) + 1
    sum_bins_sorted = [
        {"label": k, "count": sum_bins[k]}
        for k in sorted(sum_bins, key=lambda x: int(x.split("-")[0]))
    ]

    return {
        "numberFreq": freq,
        "oddEven": dict(sorted(odd_even.items(), key=lambda kv: kv[0])),
        "sumBins": sum_bins_sorted,
    }


def build_backtest(rows: list[Row]) -> dict:
    pool_mean = mean(r.winners * r.amount for r in rows)
    groups: dict[int, list[Row]] = {0: [], 1: [], 2: [], 3: []}
    for r in rows:
        groups[min(n_low12(r.numbers), 3)].append(r)

    labels = {0: "0개", 1: "1개", 2: "2개", 3: "3개+"}
    table = []
    raw_by_group: dict[int, list[float]] = {}
    for g in (0, 1, 2, 3):
        grp = groups[g]
        if not grp:
            continue
        raw = [float(r.winners) for r in grp]
        adj = [r.winners * pool_mean / (r.winners * r.amount) for r in grp]
        amt = [r.amount / 1e8 for r in grp]
        raw_by_group[g] = raw
        table.append(
            {
                "label": labels[g],
                "rounds": len(grp),
                "avgWinners": round(mean(raw), 1),
                "adjWinners": round(mean(adj), 1),
                "avgAmountEok": round(mean(amt), 1),
            }
        )

    compare = {}
    if 0 in raw_by_group and 3 in raw_by_group:
        lo, hi = raw_by_group[0], raw_by_group[3]
        t = welch_t(hi, lo)
        compare = {
            "hiMean": round(mean(hi), 1),
            "loMean": round(mean(lo), 1),
            "reductionPct": round((mean(hi) - mean(lo)) / mean(hi) * 100),
            "welchT": round(t, 2),
            "significant": abs(t) >= 2,
        }

    # 신호 안정성: 4분할 OLS(log당첨자 ~ 1~12개수)
    stability = []
    q = len(rows) // 4
    for i in range(4):
        seg = rows[i * q : (i + 1) * q] if i < 3 else rows[i * q :]
        xs = [float(n_low12(r.numbers)) for r in seg]
        ys = [log(max(r.winners, 1)) for r in seg]
        _, slope, t = ols_single(xs, ys)
        stability.append(
            {
                "label": f"{seg[0].round_no}~{seg[-1].round_no}회",
                "coef": round(slope, 3),
                "t": round(t, 2),
            }
        )

    return {
        "groups": table,
        "compare": compare,
        "stability": stability,
    }


def build_model(rows: list[Row]) -> dict:
    """조합 생성용: 최근 200회 log당첨자 ~ 1~12개수 OLS."""
    recent = rows[-200:]
    xs = [float(n_low12(r.numbers)) for r in recent]
    ys = [log(max(r.winners, 1)) for r in recent]
    intercept, coef, t = ols_single(xs, ys)
    mean_low12 = mean(xs)  # 전형적 조합의 1~12 개수 (모델 기준)
    baseline = exp(intercept + coef * mean_low12)  # 전형적 조합 예측 당첨자
    return {
        "intercept": round(intercept, 4),
        "coef": round(coef, 4),
        "tCoef": round(t, 2),
        "basis": f"최근 200회 ({recent[0].round_no}~{recent[-1].round_no})",
        "meanLow12": round(mean_low12, 2),
        "baselineWinners": round(baseline, 1),
        "overallAvgWinners": round(mean(r.winners for r in rows), 1),
    }


def build_low12_dist(rows: list[Row]) -> list[dict]:
    dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for r in rows:
        dist[n_low12(r.numbers)] += 1
    return [{"k": k, "count": v} for k, v in dist.items()]


def insights() -> list[dict]:
    return [
        {
            "title": "당첨번호 예측은 불가능하다",
            "tone": "warn",
            "body": (
                "623~624회 데이터로 7번 검증했다. 우리 예측 모델의 평균 백분위는 "
                "정확히 50% — 실제 당첨번호를 814만 조합의 한가운데에 놓는다. 사실상 무작위다. "
                "조건부 IF-THEN 패턴 점수는 노이즈를 넘어 '해로웠다'(제거하니 50%→40.8% 개선). "
                "추첨기 물리 편향도 카이제곱으로 기각(핫/콜드 번호 격차는 자연변동·착시)."
            ),
        },
        {
            "title": "단 하나의 실재 신호: 1~12 회피",
            "tone": "good",
            "body": (
                "추첨은 무작위지만 '사람들의 번호 선택'은 편향돼 있다. 당첨번호에 1~12(월/작은 수)가 "
                "많을수록 그 회차 당첨자 수가 많다 — 다중회귀에서 유일하게 유의한 변수. "
                "역으로 1~12를 피하면 당첨 시 당첨자가 적어 1인당 수령액이 커진다."
            ),
        },
        {
            "title": "효과: 당첨 시 기대 수령액 ~25% ↑",
            "tone": "good",
            "body": (
                "백테스트(검증완료): 1~12가 '0개'인 회차 vs '3개+'인 회차 — 당첨자 약 14% 적고 "
                "1인당 금액 약 25% 높았다(당첨 시 평균 5억 더, Welch t≈2.0). "
                "단 효과 크기는 작고(모델 R²≈2%) 경계선 유의다. 보장이 아니라 기대값 최적화."
            ),
        },
        {
            "title": "1등 확률은 못 바꾼다",
            "tone": "warn",
            "body": (
                "어떤 조합이든 1등 확률은 1/8,145,060으로 동일하다. 이 전략은 '당첨됐을 때' "
                "덜 나눠 갖는 것만 노린다. 적중률을 올리는 유일한 방법은 더 많은 장을 사는 것뿐."
            ),
        },
        {
            "title": "신호는 약해지는 중",
            "tone": "warn",
            "body": (
                "구간별로 보면 916~1070회에서 가장 강했고(t≈3.8) 최근 구간에서 희석됐다. "
                "판매량 증가·자동선택(QP) 비율 상승으로 1~12 회피의 이점이 과거의 절반 이하로 줄었다. "
                "방향은 유효하나 과대평가 금물."
            ),
        },
    ]


def main() -> None:
    rows = load_rows(CSV_PATH)
    latest = rows[-1]

    payload = {
        "meta": {
            "firstRound": rows[0].round_no,
            "lastRound": latest.round_no,
            "count": len(rows),
            "latest": {
                "round": latest.round_no,
                "date": latest.date,
                "nums": list(latest.numbers),
                "bonus": latest.bonus,
                "winners": latest.winners,
                "amountEok": round(latest.amount / 1e8, 1),
            },
            "builtFrom": CSV_PATH.name,
        },
        # 압축 배열: [회차, 날짜, n1..n6, 보너스, 당첨자수, 금액(억)]
        "draws": [
            [r.round_no, r.date, *r.numbers, r.bonus, r.winners, round(r.amount / 1e8, 1)]
            for r in rows
        ],
        "stats": build_stats(rows),
        "low12Dist": build_low12_dist(rows),
        "backtest": build_backtest(rows),
        "model": build_model(rows),
        "insights": insights(),
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    js = "// 자동 생성됨 — build_app_data.py 실행 결과. 직접 수정하지 마세요.\n"
    js += "window.LOTTO = " + json.dumps(payload, ensure_ascii=False) + ";\n"
    OUT_PATH.write_text(js, encoding="utf-8")

    m = payload["model"]
    print(f"생성: {OUT_PATH}")
    print(f"  회차: {payload['meta']['firstRound']}~{payload['meta']['lastRound']} ({payload['meta']['count']}회)")
    print(f"  모델: intercept={m['intercept']} coef={m['coef']} (t={m['tCoef']}, {m['basis']})")
    print(f"  최신: {latest.round_no}회 {list(latest.numbers)}+{latest.bonus}, 당첨 {latest.winners}명")


if __name__ == "__main__":
    main()
