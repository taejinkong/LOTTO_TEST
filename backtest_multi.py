#!/usr/bin/env python3
"""Multi-round backtest engine for Lotto prediction quality analysis.

회차별 표본 기반 순위 추정과 Ablation 비교를 지원한다.

예시:
  python3 backtest_multi.py --start-round 1100 --end-round 1228
  python3 backtest_multi.py --start-round 1150 --end-round 1228 --ablation all
  python3 backtest_multi.py --start-round 1200 --end-round 1228 --sample-size 200000
  python3 backtest_multi.py --start-round 1100 --end-round 1228 --ablation all --limit-rounds 30
"""

from __future__ import annotations

import argparse
import heapq
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from lotto_analyzer import Draw, load_draws, odd_even
from lotto_prediction import (
    build_cycle_scores,
    build_number_scores,
    build_pair_scores,
    build_raw_target_scores,
    candidate_target_features,
    high_value,
    low_value,
    normalize_target_scores,
)


TOTAL_COMBINATIONS = 8_145_060
DEFAULT_CSV = "lotto_winners_2020_2026.csv"
DEFAULT_TOP_K = (50, 300, 1_000, 10_000, 100_000)
DEFAULT_PORTFOLIO_QUOTAS: dict[str, int] = {
    "balanced": 25,
    "high_sum": 8,
    "high_band": 7,
    "high_start": 5,
    "consecutive": 5,
}
RARE_OE = frozenset({"OE=5:1", "OE=1:5", "OE=6:0", "OE=0:6"})
SEP = "=" * 62


# ── Ablation ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_pattern: bool = True
    use_number: bool = True
    use_pair: bool = True
    use_cycle: bool = True


ABLATION_PRESETS: dict[str, AblationConfig] = {
    # ── stage1: 패턴 포함 전체에서 하나씩 제거 ──
    "baseline":     AblationConfig("baseline"),
    "no_cycle":     AblationConfig("no_cycle",     use_cycle=False),
    "no_pair":      AblationConfig("no_pair",      use_pair=False),
    "no_number":    AblationConfig("no_number",    use_number=False),
    "no_pattern":   AblationConfig("no_pattern",   use_pattern=False),
    "only_pattern": AblationConfig("only_pattern", use_number=False, use_pair=False, use_cycle=False),
    # ── stage2: 패턴 제거 후(number+pair+cycle) 내부 분해 ──
    "np_all":     AblationConfig("np_all",     use_pattern=False),                                       # = no_pattern (기준)
    "np_no_num":  AblationConfig("np_no_num",  use_pattern=False, use_number=False),                     # pair+cycle
    "np_no_pair": AblationConfig("np_no_pair", use_pattern=False, use_pair=False),                       # number+cycle
    "np_no_cyc":  AblationConfig("np_no_cyc",  use_pattern=False, use_cycle=False),                      # number+pair
    "only_num":   AblationConfig("only_num",   use_pattern=False, use_pair=False, use_cycle=False),      # number만
    "only_pair":  AblationConfig("only_pair",  use_pattern=False, use_number=False, use_cycle=False),    # pair만
    "only_cyc":   AblationConfig("only_cyc",   use_pattern=False, use_number=False, use_pair=False),     # cycle만
}

# 묶음 실행용 그룹
ABLATION_GROUPS: dict[str, list[str]] = {
    "all":    ["baseline", "no_cycle", "no_pair", "no_number", "no_pattern", "only_pattern"],
    "stage2": ["np_all", "np_no_num", "np_no_pair", "np_no_cyc", "only_num", "only_pair", "only_cyc"],
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    round_no: int
    actual: tuple[int, ...]
    bonus: int
    est_rank: int       # 전체 8,145,060 중 추정 순위
    sample_size: int    # 실제 표본 크기 (actual 포함)
    pattern_hits: int
    pattern_total: int
    prizes: Counter = field(default_factory=Counter)

    @property
    def percentile(self) -> float:
        return self.est_rank / TOTAL_COMBINATIONS * 100


@dataclass
class MultiRoundSummary:
    config_name: str
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rounds)

    def avg_percentile(self) -> float:
        return mean(r.percentile for r in self.rounds) if self.rounds else 0.0

    def median_percentile(self) -> float:
        return median(r.percentile for r in self.rounds) if self.rounds else 0.0

    def topk_hits(self, k: int) -> int:
        return sum(1 for r in self.rounds if r.est_rank <= k)

    def avg_pattern_accuracy(self) -> float:
        valid = [r for r in self.rounds if r.pattern_total > 0]
        return mean(r.pattern_hits / r.pattern_total for r in valid) * 100 if valid else 0.0

    def prize_total(self) -> Counter:
        total: Counter = Counter()
        for r in self.rounds:
            total.update(r.prizes)
        return total

    def best_round(self) -> RoundResult | None:
        return min(self.rounds, key=lambda r: r.est_rank) if self.rounds else None

    def worst_round(self) -> RoundResult | None:
        return max(self.rounds, key=lambda r: r.est_rank) if self.rounds else None


# ── Scoring ───────────────────────────────────────────────────────────────────

def consecutive_count(numbers: tuple[int, ...]) -> int:
    return sum(1 for a, b in zip(numbers, numbers[1:]) if b - a == 1)


def score_combo(
    numbers: tuple[int, ...],
    config: AblationConfig,
    normalized_scores: dict[str, float],
    number_scores: dict[int, float],
    pair_scores: dict[tuple[int, int], float],
    cycle_scores: dict[int, float],
    previous_oe: str | None,
) -> float:
    features = candidate_target_features(numbers)
    pattern = sum(normalized_scores.get(f, 0.0) for f in features)         if config.use_pattern else 0.0
    number  = sum(number_scores[n] for n in numbers) / 6                   if config.use_number  else 0.0
    pair    = sum(pair_scores[p] for p in combinations(numbers, 2)) / 15   if config.use_pair    else 0.0
    cycle   = sum(cycle_scores[n] for n in numbers) / 6                    if config.use_cycle   else 0.0

    score = pattern * 0.62 + number * 0.16 + pair * 0.14 + cycle * 0.08
    if previous_oe in RARE_OE and odd_even(numbers) in RARE_OE:
        score -= 0.35
    return score


# ── Scenario classification ───────────────────────────────────────────────────

def classify_scenarios(numbers: tuple[int, ...]) -> list[str]:
    low    = low_value(numbers)
    high   = high_value(numbers)
    total  = sum(numbers)
    first  = numbers[0]
    consec = consecutive_count(numbers)
    names: list[str] = []
    if 100 <= total <= 159 and consec <= 1:
        names.append("balanced")
    if total >= 160:
        names.append("high_sum")
    if first >= 21:
        names.append("high_start")
    if consec >= 2:
        names.append("consecutive")
    if high >= 3:
        names.append("high_band")
    return names


# ── Portfolio ─────────────────────────────────────────────────────────────────

def build_portfolio(
    scored: list[tuple[float, tuple[int, ...]]],
    quotas: dict[str, int],
) -> list[tuple[int, ...]]:
    heaps: dict[str, list[tuple[float, tuple[int, ...]]]] = {s: [] for s in quotas}
    for score, numbers in scored:
        for scenario in classify_scenarios(numbers):
            if scenario not in heaps:
                continue
            heap = heaps[scenario]
            limit = quotas[scenario]
            if len(heap) < limit:
                heapq.heappush(heap, (score, numbers))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, numbers))

    portfolio: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    total_quota = sum(quotas.values())
    for scenario in quotas:
        for _, numbers in sorted(heaps[scenario], reverse=True):
            if numbers not in seen:
                portfolio.append(numbers)
                seen.add(numbers)
                if len(portfolio) >= total_quota:
                    return portfolio
    return portfolio


def prize_label(numbers: tuple[int, ...], actual: Draw) -> str | None:
    matched = len(set(numbers) & set(actual.numbers))
    if matched == 6:
        return "1등"
    if matched == 5 and actual.bonus in numbers:
        return "2등"
    if matched == 5:
        return "3등"
    return None


# ── Pattern accuracy ──────────────────────────────────────────────────────────

def predict_patterns(normalized_scores: dict[str, float]) -> dict[str, str]:
    """패밀리별로 가장 높은 점수를 받은 target을 반환한다."""
    best: dict[str, tuple[float, str]] = {}
    for target, score in normalized_scores.items():
        family = target.split("=", 1)[0]
        if family not in best or score > best[family][0]:
            best[family] = (score, target)
    return {fam: val for fam, (_, val) in best.items()}


def check_pattern_hits(predicted: dict[str, str], actual: Draw) -> tuple[int, int]:
    actual_features = set(candidate_target_features(actual.numbers))
    hits = total = 0
    for family, pred in predicted.items():
        match = next((f for f in actual_features if f.startswith(f"{family}=")), None)
        if match is not None:
            total += 1
            if pred == match:
                hits += 1
    return hits, total


# ── Single-round backtest ─────────────────────────────────────────────────────

def run_single_round(
    actual_draw: Draw,
    training_draws: list[Draw],
    config: AblationConfig,
    sample_size: int,
    recent_window: int,
    max_conditions: int,
    min_support: int,
    min_confidence: float,
    min_lift: float,
    rng: random.Random,
) -> RoundResult:
    previous_oe = odd_even(training_draws[-1].numbers)

    raw_scores        = build_raw_target_scores(
        training_draws, max_conditions, min_support, min_confidence, min_lift
    )
    normalized_scores = normalize_target_scores(raw_scores)
    number_scores     = build_number_scores(training_draws, recent_window)
    pair_scores       = build_pair_scores(training_draws, recent_window)
    cycle_scores      = build_cycle_scores(training_draws)

    # 표본 생성 (실제 당첨 조합은 항상 포함)
    sampled: set[tuple[int, ...]] = set()
    while len(sampled) < sample_size:
        nums = tuple(sorted(rng.sample(range(1, 46), 6)))
        sampled.add(nums)
    sampled.add(actual_draw.numbers)

    scored = [
        (
            score_combo(
                nums, config, normalized_scores, number_scores,
                pair_scores, cycle_scores, previous_oe,
            ),
            nums,
        )
        for nums in sampled
    ]

    actual_score = next(s for s, n in scored if n == actual_draw.numbers)
    # 표본 내 순위 (점수가 actual보다 높은 수 + 1)
    sample_rank = sum(1 for s, _ in scored if s > actual_score) + 1
    # 전체 공간으로 선형 외삽
    est_rank = max(1, round(sample_rank / len(sampled) * TOTAL_COMBINATIONS))

    # 포트폴리오 등수
    portfolio = build_portfolio(scored, DEFAULT_PORTFOLIO_QUOTAS)
    prizes: Counter = Counter()
    for nums in portfolio:
        lbl = prize_label(nums, actual_draw)
        if lbl:
            prizes[lbl] += 1

    # 패턴 적중
    predicted = predict_patterns(normalized_scores)
    pattern_hits, pattern_total = check_pattern_hits(predicted, actual_draw)

    return RoundResult(
        round_no=actual_draw.round_no,
        actual=actual_draw.numbers,
        bonus=actual_draw.bonus,
        est_rank=est_rank,
        sample_size=len(sampled),
        pattern_hits=pattern_hits,
        pattern_total=pattern_total,
        prizes=prizes,
    )


# ── Multi-round runner ────────────────────────────────────────────────────────

def run_multi_round(
    all_draws: list[Draw],
    start_round: int,
    end_round: int,
    config: AblationConfig,
    args: argparse.Namespace,
    rng: random.Random,
) -> MultiRoundSummary:
    targets = [d for d in all_draws if start_round <= d.round_no <= end_round]
    if args.limit_rounds > 0:
        targets = targets[-args.limit_rounds:]  # 최신 N개만 테스트

    summary = MultiRoundSummary(config_name=config.name)
    total = len(targets)

    for i, target in enumerate(targets, start=1):
        training = [d for d in all_draws if d.round_no < target.round_no]
        if len(training) < args.min_training:
            _progress(f"  [{config.name}] {target.round_no}회 스킵 — 학습데이터 {len(training)}회 < {args.min_training}")
            continue

        t0 = time.monotonic()
        result = run_single_round(
            actual_draw=target,
            training_draws=training,
            config=config,
            sample_size=args.sample_size,
            recent_window=args.recent_window,
            max_conditions=args.max_conditions,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift,
            rng=rng,
        )
        elapsed = time.monotonic() - t0
        summary.rounds.append(result)

        prize_str = ", ".join(f"{k}:{v}" for k, v in sorted(result.prizes.items())) or "없음"
        _progress(
            f"  [{config.name}] {i:>3}/{total}  {target.round_no}회  "
            f"~{result.est_rank:>9,}위 ({result.percentile:>5.1f}%)  "
            f"패턴 {result.pattern_hits}/{result.pattern_total}  "
            f"포트: {prize_str:<8}  ({elapsed:.1f}s)"
        )

    return summary


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── Output formatting ─────────────────────────────────────────────────────────

def format_summary_block(summary: MultiRoundSummary, top_k_values: tuple[int, ...]) -> list[str]:
    n = summary.n
    if n == 0:
        return [f"[{summary.config_name}] 결과 없음"]

    prizes = summary.prize_total()
    best   = summary.best_round()
    worst  = summary.worst_round()

    lines = [
        SEP,
        f"설정: {summary.config_name}   ({n}회 백테스트)",
        SEP,
        "",
        "[ 순위 통계 ]",
        f"  평균 백분위 (상위):    {summary.avg_percentile():.1f}%",
        f"  중앙값 백분위 (상위):  {summary.median_percentile():.1f}%",
    ]
    if best:
        lines.append(f"  최상 순위:  {best.round_no}회 → 약 {best.est_rank:>12,}위  ({best.percentile:.2f}%)")
    if worst:
        lines.append(f"  최하 순위:  {worst.round_no}회 → 약 {worst.est_rank:>12,}위  ({worst.percentile:.2f}%)")

    lines += [
        "",
        f"[ Top-N 포함 횟수 ({n}회 중) ]",
        "  ※ 표본 기반 추정 — Top 50/300은 분산이 높아 참고용으로만 활용",
    ]
    for k in top_k_values:
        hits = summary.topk_hits(k)
        lines.append(f"  Top {k:>8,}: {hits:>3}회  ({hits / n * 100:5.1f}%)")

    lines += ["", f"[ 포트폴리오 등수 집계 ({n}회 중) ]"]
    for lbl in ("1등", "2등", "3등"):
        cnt = prizes.get(lbl, 0)
        lines.append(f"  {lbl} 포함:  {cnt:>3}회  ({cnt / n * 100:5.1f}%)")

    lines += [
        "",
        "[ 패턴 적중률 ]",
        f"  평균: {summary.avg_pattern_accuracy():.1f}%  "
        "(패밀리별 최다득점 예측 vs 실제 비교)",
    ]
    return lines


def format_round_table(summary: MultiRoundSummary, show: int) -> list[str]:
    if show <= 0 or not summary.rounds:
        return []

    hdr = f"  {'회차':>5}  {'실제번호':<30}  {'추정순위':>13}  {'%':>6}  패턴  포트"
    sep = "  " + "─" * 72

    def row(r: RoundResult) -> str:
        prize_str = " [" + ",".join(r.prizes.keys()) + "]" if r.prizes else ""
        return (
            f"  {r.round_no:>5}회  {str(list(r.actual)):<30}  "
            f"{r.est_rank:>13,}위  {r.percentile:>5.1f}%"
            f"  {r.pattern_hits}/{r.pattern_total}{prize_str}"
        )

    worst_n = min(show, summary.n)
    best_n  = min(5, summary.n)
    worst_rounds = sorted(summary.rounds, key=lambda r: r.est_rank, reverse=True)[:worst_n]
    best_rounds  = sorted(summary.rounds, key=lambda r: r.est_rank)[:best_n]

    lines = ["", f"[ 회차별 상세 ]", hdr, sep, f"  ── 하위 {worst_n}개 (순위 최하) ──"]
    lines += [row(r) for r in worst_rounds]
    lines += [sep, f"  ── 상위 {best_n}개 (순위 최상) ──"]
    lines += [row(r) for r in best_rounds]
    return lines


def _topk_label(k: int) -> str:
    if k >= 1_000_000:
        return f"T{k // 1_000_000}M"
    if k >= 1_000:
        return f"T{k // 1_000}K"
    return f"T{k}"


def format_ablation_table(
    summaries: dict[str, MultiRoundSummary],
    top_k_values: tuple[int, ...],
) -> list[str]:
    if len(summaries) < 2:
        return []

    n = next(iter(summaries.values())).n
    # 기준선: baseline 있으면 그것, 없으면(stage2) np_all, 둘 다 없으면 첫 항목
    ref_name = next(
        (k for k in ("baseline", "np_all") if k in summaries),
        next(iter(summaries)),
    )
    lines = ["", SEP, f"Ablation 비교표  ({n}회 기준, 기준선={ref_name})", SEP, ""]

    tk_hdrs = "".join(f"  {_topk_label(k):>5}" for k in top_k_values)
    lines.append(f"  {'설정':<16}  {'평균%':>6}  {'중앙%':>6}{tk_hdrs}  {'패턴%':>6}  {'3등+':>5}")
    lines.append("  " + "─" * 62)

    baseline = summaries[ref_name]
    for name, summary in summaries.items():
        avg   = summary.avg_percentile()
        med   = summary.median_percentile()
        tkcol = "".join(f"  {summary.topk_hits(k):>5}" for k in top_k_values)
        pat   = summary.avg_pattern_accuracy()
        top3  = sum(summary.prize_total().get(r, 0) for r in ("1등", "2등", "3등"))

        flag = ""
        if name != ref_name:
            flag = "  ↑개선" if avg < baseline.avg_percentile() - 0.01 else (
                   "  ↓악화" if avg > baseline.avg_percentile() + 0.01 else "  ─동일")

        lines.append(
            f"  {name:<16}  {avg:>6.1f}  {med:>6.1f}{tkcol}  {pat:>6.1f}  {top3:>5}{flag}"
        )

    lines += [
        "",
        f"  ↑개선 = {ref_name} 대비 평균 백분위 감소 (더 상위권에 가까움)",
        f"  ↓악화 = {ref_name} 대비 평균 백분위 증가",
        "  ─동일 = 유의미한 차이 없음 (±0.1%p 이내)",
        "",
        "  해석 지침:",
        "  - 평균% 가 낮을수록 실제 당첨번호를 더 높은 순위로 끌어올림",
        "  - 특정 컴포넌트 제거 시 평균% 가 낮아지면 → 그 컴포넌트는 노이즈",
        "  - 30회 이상 백테스트 결과를 기준으로 판단 권장",
    ]
    return lines


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-round Lotto backtest engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 (baseline, 최근 129회, 표본 10만)
  python3 backtest_multi.py --start-round 1100 --end-round 1228

  # 빠른 확인 (최근 30회만)
  python3 backtest_multi.py --start-round 1100 --end-round 1228 --limit-rounds 30

  # Ablation 전체 비교 (30회 기준 권장, 시간 절약)
  python3 backtest_multi.py --start-round 1100 --end-round 1228 --ablation all --limit-rounds 30

  # 정밀 표본 (느리지만 분산↓)
  python3 backtest_multi.py --start-round 1100 --end-round 1228 --sample-size 300000

  # 특정 ablation 단독 실행
  python3 backtest_multi.py --start-round 1100 --end-round 1228 --ablation no_cycle
""",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="입력 CSV 경로")
    parser.add_argument("--start-round",  type=int,   default=1100,     help="백테스트 시작 회차")
    parser.add_argument("--end-round",    type=int,   default=1228,     help="백테스트 종료 회차")
    parser.add_argument("--limit-rounds", type=int,   default=0,
                        help="테스트할 최대 회차 수 (0=제한 없음, 최신 우선)")
    parser.add_argument("--sample-size",  type=int,   default=100_000,  help="회차당 표본 크기")
    parser.add_argument("--min-training", type=int,   default=100,      help="최소 학습 데이터 회차 수")
    parser.add_argument("--recent-window",type=int,   default=50)
    parser.add_argument("--max-conditions",type=int,  default=3)
    parser.add_argument("--min-support",  type=int,   default=15)
    parser.add_argument("--min-confidence",type=float,default=0.45)
    parser.add_argument("--min-lift",     type=float, default=1.25)
    parser.add_argument(
        "--ablation",
        default="baseline",
        help=(
            "실행할 ablation 설정. 'all' 로 모든 preset 비교. "
            f"선택지: {', '.join(ABLATION_PRESETS)}, all"
        ),
    )
    parser.add_argument("--seed",         type=int,   default=42,  help="난수 시드 (재현성)")
    parser.add_argument("--show-rounds",  type=int,   default=10,
                        help="상세 출력할 회차 수 (하위/상위 각 N개, 0=생략)")
    parser.add_argument(
        "--top-k", default=None,
        help="쉼표 구분 Top-K 임계값 (기본: 50,300,1000,10000,100000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values: tuple[int, ...] = (
        tuple(int(v) for v in args.top_k.split(","))
        if args.top_k
        else DEFAULT_TOP_K
    )

    all_draws = load_draws(Path(args.csv))
    _progress(
        f"데이터: {all_draws[0].round_no}회 ~ {all_draws[-1].round_no}회  ({len(all_draws)}회)"
    )
    limit_note = f"  최대 {args.limit_rounds}회" if args.limit_rounds > 0 else ""
    _progress(
        f"백테스트: {args.start_round}회 ~ {args.end_round}회{limit_note}  "
        f"표본: {args.sample_size:,}  시드: {args.seed}"
    )

    if args.ablation in ABLATION_GROUPS:
        configs = [ABLATION_PRESETS[name] for name in ABLATION_GROUPS[args.ablation]]
    elif args.ablation in ABLATION_PRESETS:
        configs = [ABLATION_PRESETS[args.ablation]]
    else:
        print(f"알 수 없는 ablation 설정: {args.ablation!r}", file=sys.stderr)
        print(f"그룹: {', '.join(ABLATION_GROUPS)}", file=sys.stderr)
        print(f"개별: {', '.join(ABLATION_PRESETS)}", file=sys.stderr)
        sys.exit(1)

    summaries: dict[str, MultiRoundSummary] = {}
    for config in configs:
        _progress(f"\n── {config.name} 시작 ──")
        rng = random.Random(args.seed)  # 설정마다 동일 시드 → 공정한 비교
        summaries[config.name] = run_multi_round(
            all_draws=all_draws,
            start_round=args.start_round,
            end_round=args.end_round,
            config=config,
            args=args,
            rng=rng,
        )

    _progress("\n")

    output: list[str] = []
    for summary in summaries.values():
        output += format_summary_block(summary, top_k_values)
        output += format_round_table(summary, args.show_rounds)
        output.append("")

    if len(summaries) > 1:
        output += format_ablation_table(summaries, top_k_values)

    print("\n".join(output))


if __name__ == "__main__":
    main()
