#!/usr/bin/env python3
"""로또 분석 PWA 로컬 서버.

PC 브라우저는 물론, 같은 와이파이의 폰에서도 접속해 '홈 화면에 추가'로
앱처럼 쓸 수 있게 0.0.0.0 으로 app/ 폴더를 서빙한다.

사용:
  python3 serve.py            # 기본 8000 포트
  python3 serve.py 9000       # 포트 지정

접속:
  - PC: http://localhost:8000
  - 폰: http://<PC의 LAN IP>:8000  (실행 시 출력되는 주소)
"""

from __future__ import annotations

import socket
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP_DIR = Path(__file__).parent / "app"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if not APP_DIR.exists():
        sys.exit("app/ 폴더가 없습니다. 먼저 build_app_data.py 와 gen_icons.py 를 실행하세요.")

    handler = partial(SimpleHTTPRequestHandler, directory=str(APP_DIR))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    ip = lan_ip()
    print("로또 분석 PWA 서버 실행 중")
    print(f"  PC  : http://localhost:{port}")
    print(f"  폰  : http://{ip}:{port}   (같은 와이파이)")
    print("  종료: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료했습니다.")


if __name__ == "__main__":
    main()
