#!/usr/bin/env python3
"""Fetch Korean Lotto 6/45 draw data from the official API and rebuild the CSV."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={round_no}"
OUTPUT_COLUMNS = [
    "회차",
    "추첨일",
    "번호1",
    "번호2",
    "번호3",
    "번호4",
    "번호5",
    "번호6",
    "보너스",
    "1등 당첨자수(명)",
    "1등 당첨금액(원)",
    "총판매금액(원)",
]


@dataclass(frozen=True)
class OfficialDraw:
    round_no: int
    draw_date: date
    numbers: tuple[int, ...]
    bonus: int
    first_winner_count: int
    first_prize_amount: int
    total_sell_amount: int


def expected_date_for_round(round_no: int) -> date:
    # 1회는 2002-12-07 토요일이고 이후 매주 토요일 추첨입니다.
    return date(2002, 12, 7) + timedelta(days=(round_no - 1) * 7)


def round_for_date(draw_date: date) -> int:
    base = date(2002, 12, 7)
    delta = (draw_date - base).days
    if delta < 0 or delta % 7 != 0:
        raise ValueError(f"{draw_date.isoformat()} is not a Lotto Saturday draw date")
    return delta // 7 + 1


def fetch_json(round_no: int, timeout: float) -> dict:
    request = Request(
        API_URL.format(round_no=round_no),
        headers={
            "Accept": "application/json,text/plain,*/*",
            "User-Agent": "Mozilla/5.0 lotto-data-refresh",
            "Referer": "https://www.dhlottery.co.kr/gameResult.do?method=byWin",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body[:120].replace("\n", " ")
        raise RuntimeError(f"Official API did not return JSON for {round_no}: {preview}") from exc


def parse_draw(payload: dict, round_no: int) -> OfficialDraw:
    if payload.get("returnValue") != "success":
        raise RuntimeError(f"Official API returned no result for {round_no}: {payload!r}")

    numbers = tuple(int(payload[f"drwtNo{index}"]) for index in range(1, 7))
    bonus = int(payload["bnusNo"])
    all_numbers = numbers + (bonus,)
    if len(set(all_numbers)) != 7:
        raise RuntimeError(f"Duplicate number detected for {round_no}: {all_numbers}")
    if any(number < 1 or number > 45 for number in all_numbers):
        raise RuntimeError(f"Number out of range for {round_no}: {all_numbers}")

    draw_date = datetime.strptime(payload["drwNoDate"], "%Y-%m-%d").date()
    expected_date = expected_date_for_round(round_no)
    if draw_date != expected_date:
        raise RuntimeError(
            f"Date mismatch for {round_no}: official={draw_date}, expected={expected_date}"
        )

    return OfficialDraw(
        round_no=round_no,
        draw_date=draw_date,
        numbers=tuple(sorted(numbers)),
        bonus=bonus,
        first_winner_count=int(payload.get("firstPrzwnerCo", 0)),
        first_prize_amount=int(payload.get("firstWinamnt", 0)),
        total_sell_amount=int(payload.get("totSellamnt", 0)),
    )


def fetch_draws(start_round: int, end_round: int, timeout: float) -> list[OfficialDraw]:
    draws = []
    for round_no in range(start_round, end_round + 1):
        draws.append(parse_draw(fetch_json(round_no, timeout), round_no))
    return draws


def write_csv(draws: list[OfficialDraw], output: Path) -> None:
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(OUTPUT_COLUMNS)
        for draw in sorted(draws, key=lambda item: item.round_no, reverse=True):
            writer.writerow(
                [
                    f"{draw.round_no}회",
                    draw.draw_date.isoformat(),
                    *draw.numbers,
                    draw.bonus,
                    draw.first_winner_count,
                    draw.first_prize_amount,
                    draw.total_sell_amount,
                ]
            )


def backup_existing(output: Path) -> Path | None:
    if not output.exists():
        return None
    backup = output.with_suffix(output.suffix + ".bak")
    index = 1
    while backup.exists():
        backup = output.with_suffix(output.suffix + f".bak{index}")
        index += 1
    shutil.copy2(output, backup)
    return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Lotto CSV from official DH Lottery data.")
    parser.add_argument("--start-date", default="2020-01-04", help="First draw date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2026-06-13", help="Last draw date, YYYY-MM-DD.")
    parser.add_argument("--output", default="lotto_winners_2020_2026.csv", help="CSV output path.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    start_round = round_for_date(start_date)
    end_round = round_for_date(end_date)
    if start_round > end_round:
        raise ValueError("start date must be before end date")

    output = Path(args.output)
    try:
        draws = fetch_draws(start_round, end_round, args.timeout)
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        print(f"Failed to fetch official Lotto data: {exc}", file=sys.stderr)
        print("The existing CSV was not modified.", file=sys.stderr)
        return 1

    backup = backup_existing(output)
    write_csv(draws, output)
    print(f"Wrote {len(draws)} official draws to {output}")
    if backup:
        print(f"Previous CSV backed up to {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
