#!/usr/bin/env python3
"""앱 아이콘 PNG 생성 (외부 라이브러리 없이 stdlib zlib만 사용).

로또 공 느낌: 짙은 네이비 배경 + 가운데 파란 공 + 좌상단 하이라이트.
생성: app/icon-192.png, app/icon-512.png
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def png_bytes(width: int, height: int, rgb) -> bytes:
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            raw += bytes(rgb(x, y))
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def make_icon(size: int) -> bytes:
    bg = (11, 16, 32)
    ball = (108, 140, 255)
    ball_dark = (60, 84, 190)
    highlight = (210, 222, 255)
    cx = cy = size / 2
    r = size * 0.36
    # 하이라이트 중심(좌상단)
    hx, hy = size * 0.38, size * 0.36
    hr = size * 0.10

    def rgb(x, y):
        dx, dy = x + 0.5 - cx, y + 0.5 - cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > r:
            return bg
        # 공 음영: 중심에서 멀수록 어둡게
        t = min(dist / r, 1.0)
        col = [
            int(ball[i] * (1 - t * 0.45) + ball_dark[i] * (t * 0.45))
            for i in range(3)
        ]
        # 하이라이트 블렌딩
        hd = ((x + 0.5 - hx) ** 2 + (y + 0.5 - hy) ** 2) ** 0.5
        if hd < hr:
            k = (1 - hd / hr) * 0.7
            col = [int(col[i] * (1 - k) + highlight[i] * k) for i in range(3)]
        return tuple(col)

    return png_bytes(size, size, rgb)


def main() -> None:
    out = Path("app")
    out.mkdir(exist_ok=True)
    for size in (192, 512):
        path = out / f"icon-{size}.png"
        path.write_bytes(make_icon(size))
        print(f"생성: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
