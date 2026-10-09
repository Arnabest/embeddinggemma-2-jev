#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Antigravity Lifecycle Hook Adapter:
作为思考前 Hook 与 Agent Loop 门禁接入点：
1. PreInvocation: 介入 Antigravity 上下文编排，将意图分析、记忆召回与硬件态势以 ephemeralMessage 注入。
2. PreToolUse: 介入 Agent Loop 决策流，利用 Gemma-Jev noul/score 对高危变异工具进行安全拦截与物理放行。
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path

# 编码安全
for s in (sys.stdin, sys.stdout):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine import GemmaPrepEngine
from pipeline import ContextAssemblyPipeline

engine = GemmaPrepEngine()
pipeline = ContextAssemblyPipeline(engine)


def extract_last_user_query(payload: dict) -> str:
    """从 transcript.jsonl 中提取最近一条用户提问内容。"""
    transcript_path = payload.get("transcriptPath")
    if not transcript_path or not Path(transcript_path).exists():
        return ""
    try:
        last_user = ""
        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "USER_INPUT" or obj.get("source") == "USER_EXPLICIT":
                        last_user = obj.get("content", "")
                except Exception:
                    pass
        # 清洗掉 <USER_REQUEST> 等包裹标签
        last_user = last_user.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()
        return last_user
    except Exception:
        return ""


def evaluate_pre_invocation(payload: dict) -> dict:
    """思考前 Hook: 组装前置意图与记忆先验并注入到上下文。"""
    try:
        query = extract_last_query = extract_last_user_query(payload)
        if not query:
            return {"injectSteps": []}

        # 执行组装流水线
        res = pipeline.assemble(instruction=query, state="", include_recall=True, include_hardware=True)
        return {"injectSteps": res.get("assembled_steps", [])}
    except Exception as exc:
        return {"injectSteps": []}


def evaluate_pre_tool_use(payload: dict) -> dict:
    """Agent Loop 决策门禁: 对危险操作与显存饱和时段做 Jev 安全裁决。"""
    try:
        tool_call = payload.get("toolCall", {})
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        # 只读工具默认放行
        if tool_name in ("view_file", "search_web", "call_mcp_tool", "vault_search", "shelf_search"):
            return {"decision": "allow"}

        # 变异工具或命令执行
        if tool_name in ("write_to_file", "replace_file_content", "run_command"):
            # 1. 硬件显存饱和检查
            hw = engine.get_hardware_status()
            if hw.get("danger_level", 1) >= 5 and tool_name == "run_command":
                cmd = str(args.get("CommandLine", ""))
                if any(k in cmd.lower() for k in ("python", "torch", "comfy", "train")):
                    return {
                        "decision": "ask",
                        "reason": f"Gemma-Jev 硬件哨兵拦截: 当前 GPU 显存占用高达 {hw.get('vram_usage_pct')}%，并发下发大任务极可能引发 CUDA OOM 崩溃，请确认是否继续。"
                    }

            # 2. 危险写盘检查
            if tool_name == "write_to_file":
                target = str(args.get("TargetFile", ""))
                is_overwrite = bool(args.get("Overwrite"))
                if is_overwrite and any(k in target.lower() for k in ("config", "adapter.pt", "model.safetensors")):
                    # Jev noul 判定
                    dec = engine.jev_decide(
                        "noul",
                        question="检测到系统关键模型或配置文件覆盖写入，是否判定为潜在高危行为？",
                        options=[],
                        state=f"目标文件: {target}, 覆盖标志: {is_overwrite}"
                    )
                    if dec.get("pred") == 1:
                        return {
                            "decision": "ask",
                            "reason": f"Gemma-Jev 安全护栏: 正在覆写高危关键资产 [{Path(target).name}]，已触发安全确认。"
                        }

        return {"decision": "allow"}
    except Exception:
        return {"decision": "allow"}


def main():
    if "--post-task-audit" in sys.argv:
        idx = sys.argv.index("--post-task-audit")
        ws = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "."
        goal = sys.argv[idx + 2] if idx + 2 < len(sys.argv) else ""
        res = engine.post_task_codebase_audit(workspace_path=ws, session_goal=goal)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if "--pre-invocation" in sys.argv:
        try:
            raw = sys.stdin.read()
            payload = json.loads(raw) if raw.strip() else {}
        except Exception:
            payload = {}
        print(json.dumps(evaluate_pre_invocation(payload), ensure_ascii=False))
        return 0

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print(json.dumps({"decision": "allow"}))
            return 0
        payload = json.loads(raw)
        if "invocationNum" in payload:
            print(json.dumps(evaluate_pre_invocation(payload), ensure_ascii=False))
            return 0
        print(json.dumps(evaluate_pre_tool_use(payload), ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"decision": "allow"}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
