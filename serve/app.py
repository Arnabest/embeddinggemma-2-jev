#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gemma-Jev High-Throughput REST API & Web UI Service.

Usage:
  python serve/app.py --port 8765 --adapter models/adapter/adapter.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.engine import GemmaPrepEngine

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


class GemmaHTTPHandler(BaseHTTPRequestHandler):
    engine: GemmaPrepEngine

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            self._json({
                "status": "ok",
                "model": "EmbeddingGemma-2-Jev",
                "device": self.engine.dev,
                "max_len": self.engine.max_len
            })
        elif p == "/":
            self._json({
                "service": "EmbeddingGemma-2-Jev REST Service",
                "version": "1.0.0",
                "endpoints": ["/api/predict", "/api/intent", "/health"]
            })
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if self.path == "/api/predict":
            try:
                mode = body.get("mode", "auto")
                question = body.get("question", "").strip()
                options = body.get("options", [])
                state = body.get("state", "").strip()
                return self._json(self.engine.jev_decide(mode, question, options, state))
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        elif self.path == "/api/intent":
            try:
                instruction = body.get("instruction", "").strip()
                context = body.get("context", "").strip()
                return self._json(self.engine.classify_intent(instruction, context))
            except Exception as e:
                return self._json({"error": str(e)}, 500)
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description="Serve Gemma-Jev Model")
    parser.add_argument("--base-model", default="models/base", help="Path to base model directory")
    parser.add_argument("--adapter", default="models/adapter/adapter.pt", help="Path to adapter checkpoint")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument("--device", default="cuda", help="Execution device")
    args = parser.parse_args()

    engine = GemmaPrepEngine(
        base_model_dir=args.base_model,
        adapter_path=args.adapter,
        device=args.device
    )

    GemmaHTTPHandler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), GemmaHTTPHandler)
    print(f"[Gemma-Jev Server] Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Gemma-Jev Server] Stopped.")


if __name__ == "__main__":
    main()
