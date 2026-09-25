"""Serve the Connect Four dashboard."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

import connect4_engine
from connect4_laya import decide, load_agent

ROOT = Path(__file__).parent


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/preload":
            try:
                load_agent()
                body = json.dumps({"ready": True}).encode()
                self._send(200, body)
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}).encode())
            return
        if self.path != "/api/decide":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            board = payload.get("state", {}).get("board")
            if not board:
                raise ValueError("state.board is required")
            result = decide(board, mode=payload.get("mode", "mock"), player=connect4_engine.LAYA)
            self._send(200, json.dumps(result).encode())
        except Exception as exc:
            self._send(500, json.dumps({"error": str(exc)}).encode())

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    handler = lambda *args, **kwargs: Handler(*args, directory=str(ROOT), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), handler)
    print("Connect Four dashboard: http://127.0.0.1:8765/play_dashboard.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
