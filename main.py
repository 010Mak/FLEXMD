import argparse
import os
import threading
import time
import webbrowser

from api.server import app

def _open_browser_when_ready(url: str, delay: float = 0.8) -> None:
    def _worker():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="MMDFLED micro-sim server")
    parser.add_argument("--host", default=os.environ.get("MMDFLED_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MMDFLED_PORT", "5000")))
    parser.add_argument("--open-browser", action="store_true",
                        help="Open http://HOST:PORT/demo after the server starts")
    args = parser.parse_args(argv)

    if args.open_browser:
        _open_browser_when_ready(f"http://{args.host}:{args.port}/demo")

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
