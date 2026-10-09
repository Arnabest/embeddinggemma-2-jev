#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EmbeddingGemma-2-Jev Standard MCP Server over stdio.
Compliant with Model Context Protocol (MCP) JSON-RPC 2.0 specifications.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import GemmaPrepEngine
from core.pipeline import ContextAssemblyPipeline


class GemmaJevMCPServer:
    def __init__(self, base_model: str = "models/base", adapter_path: str = "models/adapter/adapter.pt"):
        self.engine = GemmaPrepEngine(
            base_model_dir=PROJECT_ROOT / base_model,
            adapter_path=PROJECT_ROOT / adapter_path
        )
        self.pipeline = ContextAssemblyPipeline(self.engine)

    def get_tools_manifest(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "gemma_intent_classify",
                "description": "Pre-instruction intent classifier powered by EmbeddingGemma-Jev. Categorizes user goals into CODE_MUTATE, READONLY_QUERY, SYSTEM_COMMAND, MEDIA_PIPELINE, RAG_RECALL, or DANGEROUS_WRITE.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "description": "The target prompt or instruction."},
                        "context": {"type": "string", "description": "Optional background context (up to 8K)."}
                    },
                    "required": ["instruction"]
                }
            },
            {
                "name": "gemma_jev_decide",
                "description": "Deterministic Jev causal decision judge: 'noul' (binary safety gate), 'choice' (action routing), or 'score' (1-5 risk grading).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["noul", "choice", "score"]},
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "object"}},
                        "state": {"type": "string"}
                    },
                    "required": ["mode", "question"]
                }
            },
            {
                "name": "gemma_assemble_context",
                "description": "End-to-end Context Assembly Pipeline. Emits clean Attention HUD telemetry.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string"},
                        "state": {"type": "string"}
                    },
                    "required": ["instruction"]
                }
            },
            {
                "name": "gemma_hardware_status",
                "description": "Probe live NVIDIA GPU VRAM occupancy and safety status.",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "gemma_intent_classify":
            return self.engine.classify_intent(args.get("instruction", ""), args.get("context", ""))
        elif name == "gemma_jev_decide":
            return self.engine.jev_decide(
                mode=args.get("mode", "noul"),
                question=args.get("question", ""),
                options=args.get("options", []),
                state=args.get("state", "")
            )
        elif name == "gemma_assemble_context":
            return self.pipeline.assemble(
                instruction=args.get("instruction", ""),
                state=args.get("state", ""),
                include_hardware=True
            )
        elif name == "gemma_hardware_status":
            return self.engine.get_hardware_status()
        return {"error": f"Unknown tool: {name}"}

    def run(self):
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer

        while True:
            line = stdin.readline()
            if not line:
                break
            try:
                try:
                    line_str = line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    line_str = line.decode("utf-16").strip()
                if not line_str:
                    continue
                req = json.loads(line_str)
                msg_id = req.get("id")
                method = req.get("method")
                params = req.get("params", {})

                if method == "initialize":
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "embeddinggemma-2-jev", "version": "1.0.0"}
                        }
                    }
                    stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                    stdout.flush()

                elif method == "tools/list":
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"tools": self.get_tools_manifest()}
                    }
                    stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                    stdout.flush()

                elif method == "tools/call":
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    out = self.handle_tool_call(tool_name, arguments)
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=2)}]
                        }
                    }
                    stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                    stdout.flush()

                elif method == "ping":
                    res = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
                    stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                    stdout.flush()

                elif method == "notifications/initialized":
                    pass
                else:
                    if msg_id is not None:
                        res = {
                            "jsonrpc": "2.0", "id": msg_id,
                            "error": {"code": -32601, "message": f"Method not found: {method}"}
                        }
                        stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                        stdout.flush()
            except Exception:
                pass


if __name__ == "__main__":
    server = GemmaJevMCPServer()
    server.run()
