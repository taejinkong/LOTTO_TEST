#!/usr/bin/env python3
"""동행복권 공식 API로 로또 6/45 당첨번호를 받아 CSV를 갱신한다.

엔드포인트(공식):
  https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do
  → ltEpsd, ltRflYmd, tm1WnNo~tm6WnNo, bnsWnNo, rnk1WnNope,
    rnk1WnAmt(1인당 당첨금), wholEpsdSumNtslAmt(총판매금액)를 반환.

모드:
  append  (기본) 기존 CSV의 마지막 회차 다음부터 '아직 추첨 안 된 회차'를 만날
                 때까지만 받아 이어붙인다. 빠르고, 기존 행은 그대로 보존.
  rebuild        start-date~end-date 구간 전체를 다시 받아 CSV를 새로 쓴다.

예:
  python3 update_lotto_csv.py                 # 새 회차만 추가 + app/data.js 갱신
  python3 update_lotto_csv.py --no-rebuild-data
  python3 update_lotto_csv.py --mode rebuild --start-date 2014-07-12 --end-date 2026-06-20

주의: 동행복권은 클라우드/데이터센터 IP나 자동화 요청을 차단(홈·errorPage로
      리다이렉트)할 수 있다. 그 경우 일반 가정용 네트워크에서 직접 실행하거나,
      잠시 후 재시도하라. 차단 시 기존 CSV는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"
OUTPUT_COLUMNS = [
    "회차", "추첨일", "번호1", "번호2", "번호3", "번호4", "번호5", "번호6",
    "보너스", "1등 당첨자수(명)", "1등 당첨금액(원)", "총판매금액(원)",
]


class BlockedError(RuntimeError):
    """동행복권이 JSON 대신 HTML(차단/리다이렉트)을 돌려준 경우."""


@dataclass(frozen=True)
class OfficialDraw:
    round_no: int
    draw_date: date
    numbers: tuple[int, ...]
    bonus: int
    first_winner_count: int
    first_prize_amount: int
    total_sell_amount: int

    def to_row(self) -> list:
        return [
            f"{self.round_no}회", self.draw_date.isoformat(), *self.numbers,
            self.bonus, self.first_winner_count, self.first_prize_amount,
            self.total_sell_amount,
        ]


def expected_date_for_round(round_no: int) -> date:
    # 1회 = 2002-12-07(토), 이후 매주 토요일.
    return date(2002, 12, 7) + timedelta(days=(round_no - 1) * 7)


def round_for_date(draw_date: date) -> int:
    delta = (draw_date - date(2002, 12, 7)).days
    if delta < 0 or delta % 7 != 0:
        raise ValueError(f"{draw_date.isoformat()} 는 토요일 추첨일이 아닙니다")
    return delta // 7 + 1


def latest_drawn_round(today: date | None = None) -> int:
    """오늘 기준 이미 추첨이 끝났을 최신 회차(가장 가까운 과거 토요일)."""
    today = today or date.today()
    days_since_sat = (today.weekday() - 5) % 7  # 토=5
    last_sat = today - timedelta(days=days_since_sat)
    return round_for_date(last_sat)


def fetch_json(round_no: int, timeout: float) -> dict:
    query = urlencode({"srchDir": "center", "srchLtEpsd": round_no})
    request = Request(
        f"{API_URL}?{query}",
        headers={
            "Accept": "application/json,text/plain,*/*",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://www.dhlottery.co.kr/lt645/result?drawNo={round_no}",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body.strip()[:80].replace("\n", " ")
        raise BlockedError(
            f"{round_no}회 응답이 JSON이 아닙니다(차단/리다이렉트 추정): {preview}"
        ) from exc


def parse_draw(payload: dict, round_no: int) -> OfficialDraw:
    items = payload.get("data", {}).get("list", [])
    item = next((row for row in items if int(row.get("ltEpsd", 0)) == round_no), None)
    if item is None:
        raise RuntimeError(f"{round_no}회 결과 없음(미추첨): {payload!r}")
    numbers = tuple(int(item[f"tm{i}WnNo"]) for i in range(1, 7))
    bonus = int(item["bnsWnNo"])
    all_numbers = numbers + (bonus,)
    if len(set(all_numbers)) != 7 or any(n < 1 or n > 45 for n in all_numbers):
        raise RuntimeError(f"{round_no}회 번호 이상: {all_numbers}")
    draw_date = datetime.strptime(item["ltRflYmd"], "%Y%m%d").date()
    expected = expected_date_for_round(round_no)
    if draw_date != expected:
        raise RuntimeError(f"{round_no}회 날짜 불일치: 공식={draw_date}, 예상={expected}")
    return OfficialDraw(
        round_no=round_no,
        draw_date=draw_date,
        numbers=tuple(sorted(numbers)),
        bonus=bonus,
        first_winner_count=int(item.get("rnk1WnNope", 0)),
        first_prize_amount=int(item.get("rnk1WnAmt", 0)),
        total_sell_amount=int(item.get("wholEpsdSumNtslAmt", 0)),
    )


def load_existing(path: Path) -> tuple[list[list], int]:
    """기존 CSV 행(원본 그대로)과 최대 회차를 반환."""
    if not path.exists():
        return [], 0
    rows: list[list] = []
    max_round = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            rows.append(row)
            try:
                max_round = max(max_round, int(row[0].replace("회", "")))
            except (ValueError, IndexError):
                pass
    return rows, max_round


def write_rows(rows: list[list], output: Path) -> None:
    # 회차 내림차순으로 정렬해 저장(기존 파일 관례와 동일).
    def round_of(r: list) -> int:
        try:
            return int(r[0].replace("회", ""))
        except (ValueError, IndexError):
            return -1
    rows = sorted(rows, key=round_of, reverse=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)


def backup_existing(output: Path) -> Path | None:
    if not output.exists():
        return None
    backup = output.with_suffix(output.suffix + ".bak")
    i = 1
    while backup.exists():
        backup = output.with_suffix(output.suffix + f".bak{i}")
        i += 1
    shutil.copy2(output, backup)
    return backup


def run_append(output: Path, timeout: float, max_new: int) -> int:
    existing_rows, max_round = load_existing(output)
    if max_round == 0:
        print("기존 CSV가 없거나 비어 있습니다. --mode rebuild 로 먼저 만드세요.", file=sys.stderr)
        return 1

    target = latest_drawn_round()
    if max_round >= target:
        print(f"이미 최신입니다. CSV 최신 {max_round}회, 추첨 완료 최신 {target}회.")
        return 0

    print(f"CSV 최신 {max_round}회 → 추첨 완료 최신 {target}회. {max_round + 1}회부터 받습니다.")
    new_draws: list[OfficialDraw] = []
    round_no = max_round + 1
    while round_no <= target and len(new_draws) < max_new:
        try:
            draw = parse_draw(fetch_json(round_no, timeout), round_no)
        except BlockedError as exc:
            if new_draws:
                print(f"  {round_no}회에서 차단됨. 받은 {len(new_draws)}회까지만 반영합니다.")
                break
            print(f"동행복권이 요청을 차단했습니다: {exc}", file=sys.stderr)
            print("→ 가정용 네트워크에서 직접 실행하거나 잠시 후 재시도하세요. CSV는 그대로 둡니다.", file=sys.stderr)
            return 2
        except (HTTPError, URLError, TimeoutError) as exc:
            if new_draws:
                print(f"  {round_no}회 네트워크 오류. 받은 {len(new_draws)}회까지만 반영합니다: {exc}")
                break
            print(f"동행복권 요청 실패: {exc}", file=sys.stderr)
            print("CSV는 그대로 둡니다.", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f"  {round_no}회 중단: {exc}")
            break
        print(f"  + {draw.round_no}회 {list(draw.numbers)}+{draw.bonus} "
              f"(1등 {draw.first_winner_count}명, {draw.first_prize_amount/1e8:.1f}억)")
        new_draws.append(draw)
        round_no += 1

    if not new_draws:
        print("추가할 새 회차가 없습니다.")
        return 0

    backup = backup_existing(output)
    write_rows(existing_rows + [d.to_row() for d in new_draws], output)
    print(f"{len(new_draws)}개 회차 추가 → {output}")
    if backup:
        print(f"이전 CSV 백업: {backup}")
    return 0


def run_rebuild(output: Path, start: date, end: date, timeout: float) -> int:
    start_round, end_round = round_for_date(start), round_for_date(end)
    if start_round > end_round:
        raise ValueError("시작일이 종료일보다 빠를 수 없습니다")
    draws = []
    try:
        for r in range(start_round, end_round + 1):
            draws.append(parse_draw(fetch_json(r, timeout), r))
    except (HTTPError, URLError, TimeoutError, BlockedError, RuntimeError) as exc:
        print(f"가져오기 실패: {exc}", file=sys.stderr)
        print("기존 CSV는 수정하지 않았습니다.", file=sys.stderr)
        return 1
    backup = backup_existing(output)
    write_rows([d.to_row() for d in draws], output)
    print(f"{len(draws)}개 회차 기록 → {output}")
    if backup:
        print(f"이전 CSV 백업: {backup}")
    return 0


def run_add(output: Path, spec: str) -> int:
    """API가 막혔을 때, 동행복권 사이트에서 보고 한 줄을 직접 추가한다.

    형식: '회차,추첨일,n1,n2,n3,n4,n5,n6,보너스,당첨자수,1인당당첨금,총판매금액'
    예:   --add '1230,2026-06-27,1,2,3,4,5,6,7,10,3500000000,0'
    뒤쪽 당첨자수/당첨금/판매금액은 모르면 0으로 둬도 된다(통계엔 영향 적음).
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 9:
        print("형식 오류. 최소 '회차,추첨일,n1~n6,보너스' 12필드 권장.", file=sys.stderr)
        return 1
    parts += ["0"] * (12 - len(parts))  # 뒤쪽 3개 기본 0
    try:
        round_no = int(parts[0].replace("회", ""))
        draw_date = datetime.strptime(parts[1], "%Y-%m-%d").date()
        numbers = tuple(sorted(int(x) for x in parts[2:8]))
        bonus = int(parts[8])
    except ValueError as exc:
        print(f"형식 오류: {exc}", file=sys.stderr)
        return 1
    alln = numbers + (bonus,)
    if len(set(alln)) != 7 or any(n < 1 or n > 45 for n in alln):
        print(f"번호 이상: {alln}", file=sys.stderr)
        return 1
    draw = OfficialDraw(round_no, draw_date, numbers, bonus,
                        int(parts[9]), int(parts[10]), int(parts[11]))

    existing_rows, _ = load_existing(output)
    existing_rows = [r for r in existing_rows
                     if r and r[0].replace("회", "") != str(round_no)]  # 중복 회차 교체
    backup = backup_existing(output)
    write_rows(existing_rows + [draw.to_row()], output)
    print(f"{round_no}회 추가: {list(numbers)}+{bonus} → {output}")
    if backup:
        print(f"이전 CSV 백업: {backup}")
    return 0


def rebuild_app_data() -> None:
    builder = Path(__file__).parent / "build_app_data.py"
    if not builder.exists():
        print("build_app_data.py 가 없어 앱 데이터 갱신을 건너뜁니다.")
        return
    print("앱 데이터 재생성: python3 build_app_data.py")
    result = subprocess.run([sys.executable, str(builder)], cwd=str(builder.parent))
    if result.returncode != 0:
        print("build_app_data.py 실행 실패", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="동행복권 공식 API로 로또 CSV 갱신.")
    p.add_argument("--mode", choices=["append", "rebuild"], default="append")
    p.add_argument("--add", default=None,
                   help="API 차단 시 수동 추가: '회차,추첨일,n1~n6,보너스[,당첨자수,당첨금,판매금액]'")
    p.add_argument("--output", default="lotto_winners_2020_2026.csv")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--max-new", type=int, default=60, help="append 모드에서 한 번에 받을 최대 회차 수")
    p.add_argument("--start-date", default="2014-07-12", help="rebuild 모드 시작 추첨일")
    p.add_argument("--end-date", default=None, help="rebuild 모드 종료 추첨일(기본: 최신 추첨)")
    p.add_argument("--no-rebuild-data", action="store_true", help="app/data.js 자동 갱신 안 함")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if args.add:
        code = run_add(output, args.add)
    elif args.mode == "append":
        code = run_append(output, args.timeout, args.max_new)
    else:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = (datetime.strptime(args.end_date, "%Y-%m-%d").date()
               if args.end_date else expected_date_for_round(latest_drawn_round()))
        code = run_rebuild(output, start, end, args.timeout)
    if code == 0 and not args.no_rebuild_data:
        rebuild_app_data()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
