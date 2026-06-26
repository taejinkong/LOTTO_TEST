#!/usr/bin/env python3
"""Convert the downloaded Lotto Excel sheet to the analyzer CSV format."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
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
class ExcelDraw:
    round_no: int
    numbers: tuple[int, ...]
    bonus: int
    first_winner_count: int
    first_prize_amount: int


def draw_date(round_no: int) -> date:
    return date(2002, 12, 7) + timedelta(days=(round_no - 1) * 7)


def round_for_date(value: date) -> int:
    base = date(2002, 12, 7)
    delta = (value - base).days
    if delta < 0 or delta % 7 != 0:
        raise ValueError(f"{value.isoformat()} is not a Saturday Lotto draw date")
    return delta // 7 + 1


def parse_int(value: str) -> int:
    value = value.strip()
    if re.fullmatch(r"[0-9]+(?:\.0+)?", value):
        return int(float(value))
    digits = re.sub(r"[^0-9]", "", value)
    if not digits:
        return 0
    return int(digits)


def parse_xlsx_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as xlsx:
        shared_strings = []
        if "xl/sharedStrings.xml" in xlsx.namelist():
            root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", XLSX_NS):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//x:t", XLSX_NS))
                )

        sheet = ET.fromstring(xlsx.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//x:sheetData/x:row", XLSX_NS):
            values = []
            for cell in row.findall("x:c", XLSX_NS):
                value_node = cell.find("x:v", XLSX_NS)
                value = "" if value_node is None else value_node.text or ""
                if cell.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                values.append(value)
            rows.append(values)
    return rows


def parse_draws(path: Path) -> list[ExcelDraw]:
    rows = parse_xlsx_rows(path)
    draws = []
    for row in rows[1:]:
        if len(row) < 12:
            continue
        round_no = parse_int(row[1])
        numbers = tuple(parse_int(value) for value in row[2:8])
        bonus = parse_int(row[8])
        if not round_no or len(numbers) != 6:
            continue
        all_numbers = numbers + (bonus,)
        if len(set(all_numbers)) != 7 or any(number < 1 or number > 45 for number in all_numbers):
            raise ValueError(f"Invalid numbers at round {round_no}: {all_numbers}")
        draws.append(
            ExcelDraw(
                round_no=round_no,
                numbers=tuple(sorted(numbers)),
                bonus=bonus,
                first_winner_count=parse_int(row[10]),
                first_prize_amount=parse_int(row[11]),
            )
        )
    return sorted(draws, key=lambda item: item.round_no)


def write_csv(draws: list[ExcelDraw], output: Path) -> None:
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(OUTPUT_COLUMNS)
        for draw in sorted(draws, key=lambda item: item.round_no, reverse=True):
            writer.writerow(
                [
                    f"{draw.round_no}회",
                    draw_date(draw.round_no).isoformat(),
                    *draw.numbers,
                    draw.bonus,
                    draw.first_winner_count,
                    draw.first_prize_amount,
                    0,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Lotto Excel data to normalized CSV.")
    parser.add_argument("xlsx", help="Input XLSX file.")
    parser.add_argument("--output", default="lotto_winners_2020_2026.csv", help="Output CSV path.")
    parser.add_argument("--start-date", default="2014-07-12", help="Start draw date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2026-06-13", help="End draw date, YYYY-MM-DD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = round_for_date(datetime.strptime(args.start_date, "%Y-%m-%d").date())
    end = round_for_date(datetime.strptime(args.end_date, "%Y-%m-%d").date())
    draws = [draw for draw in parse_draws(Path(args.xlsx)) if start <= draw.round_no <= end]
    expected_count = end - start + 1
    if len(draws) != expected_count:
        raise RuntimeError(f"Expected {expected_count} draws, parsed {len(draws)}")
    write_csv(draws, Path(args.output))
    print(f"Wrote {len(draws)} draws ({start}~{end}) to {args.output}")


if __name__ == "__main__":
    main()
