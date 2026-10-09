#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Antigravity / Harnessd Lifecycle Hook Adapter.

Integrates:
1. PreInvocation: Injects ephemeral Attention HUD before model thinking.
2. PreToolUse: Intercepts high-risk mutations and memory-exhaustion commands.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import GemmaPrepEngine
from core.pipeline import ContextAssemblyPipeline

engine = GemmaPrepEngine(
    base_model_dir=PROJECT_ROOT / "models" / "base",
    adapter_path=PROJECT_ROOT / "models" / "adapter" / "adapter.pt"
)
pipeline = ContextAssemblyPipeline(engine)


def evaluate_pre_invocation(payload: dict) -> dict:
    """PreInvocation hook."""
    transcript_path = payload.get("transcriptPath")
    query = ""
    if transcript_path and Path(transcript_path).exists():
        try:
            with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("type") == "USER_INPUT" or obj.get("source") == "USER_EXPLICIT":
                            query = obj.get("content", "")
            query = query.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()
        except Exception:
            pass
    if not query:
        return {"injectSteps": []}
    res = pipeline.assemble(instruction=query, state="", include_hardware=True)
    return {"injectSteps": res.get("assembled_steps", [])}


def evaluate_pre_tool_use(payload: dict) -> dict:
    """PreToolUse hook."""
    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    if tool_name in ("write_to_file", "replace_file_content", "run_command"):
        hw = engine.get_hardware_status()
        if hw.get("danger_level", 1) >= 5 and tool_name == "run_command":
            cmd = str(args.get("CommandLine", ""))
            if any(k in cmd.lower() for k in ("python", "torch", "comfy", "train")):
                return {
                    "decision": "ask",
                    "reason": f"Gemma-Jev Gate: High VRAM occupancy ({hw.get('vram_usage_pct')}%). Potential CUDA OOM hazard."
                }
    return {"decision": "allow"}


def main():
    if "--pre-invocation" in sys.argv:
        try:
            raw = sys.stdin.read()
            p = json.loads(raw) if raw.strip() else {}
        except Exception:
            p = {}
        print(json.dumps(evaluate_pre_invocation(p), ensure_ascii=False))
        return 0

    try:
        raw = sys.stdin.read()
        p = json.loads(raw) if raw.strip() else {}
        if "invocationNum" in p:
            print(json.dumps(evaluate_pre_invocation(p), ensure_ascii=False))
            return 0
        print(json.dumps(evaluate_pre_tool_use(p), ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"decision": "allow"}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
