#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Antigravity Lifecycle Hook Adapter (Fail-Open Anti-Self-Lock Hardened):
1. PreInvocation: 意图分析 (含 DIRECTIONAL_PLANNING)、历史结果回填、TTL 配额监测 (<=15%/<=8%)、
   双轨交接续接提示、以及 Hook 自锁故障诊断恢复告警。
2. PreToolUse: 惰性容错导入；若内部模块受损触发自锁风险，则自动 fail-open 并写明故障现场，
   确保不会阻断修复工具；正常状态下执行方向性门禁锁与安全裁决。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

for s in (sys.stdin, sys.stdout):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_SELF_LOCK_FILE = HERE / "data" / "hook_self_lock_state.json"


def _self_lock_file() -> Path:
    override = os.environ.get("NEXUS_HOOK_SELF_LOCK_FILE")
    return Path(override) if override else DEFAULT_SELF_LOCK_FILE


def _record_bootstrap_self_lock(exc: Exception) -> None:
    try:
        p = _self_lock_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "hook": "gemma-prep-core",
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:300],
            "traceback": traceback.format_exc()[-500:],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clear_bootstrap_self_lock() -> None:
    try:
        p = _self_lock_file()
        if p.is_file():
            p.unlink()
    except Exception:
        pass


def _read_self_lock_warning() -> str:
    try:
        p = _self_lock_file()
        if not p.is_file():
            return ""
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict):
            return ""
        etype = data.get("error_type", "Error")
        emsg = data.get("error_msg", "")
        return (
            f"[HOOK_SELF_LOCK_CRITICAL] hook '{data.get('hook', 'gemma-prep-core')}' "
            f"bootstrap failed ({etype}: {emsg}). PreToolUse failed open to prevent tool deadlock; repair immediately."
        )
    except Exception:
        return ""


def _fallback_clean_user_text(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"<ADDITIONAL_METADATA>[\s\S]*?</ADDITIONAL_METADATA>", "", raw)
    text = re.sub(r"<USER_SETTINGS_CHANGE>[\s\S]*?</USER_SETTINGS_CHANGE>", "", text)
    text = text.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "")
    return text.strip()


def _load_runtime():
    """Lazily import runtime modules so import/syntax faults never exit(1) and deadlock tools."""
    try:
        from deciders.directional_gate import (
            clean_user_text,
            evaluate_directional_tool_gate,
            inspect_transcript_directional_state,
            is_readonly_tool_call,
            read_directional_lock,
            write_directional_lock,
        )
        from deciders.handoff_quota import (
            format_handoff_hud,
            load_pending_handoffs,
            probe_quota_status,
            record_and_backfill_turn_outcome,
            save_dual_track_handoff,
        )
        from deciders.session_context import classify_session_profile
        from decision_substrate import DecisionSubstrate
        from engine import GemmaPrepEngine
        from pipeline import ContextAssemblyPipeline

        _clear_bootstrap_self_lock()
        return {
            "clean_user_text": clean_user_text,
            "evaluate_directional_tool_gate": evaluate_directional_tool_gate,
            "inspect_transcript_directional_state": inspect_transcript_directional_state,
            "is_readonly_tool_call": is_readonly_tool_call,
            "read_directional_lock": read_directional_lock,
            "write_directional_lock": write_directional_lock,
            "format_handoff_hud": format_handoff_hud,
            "load_pending_handoffs": load_pending_handoffs,
            "probe_quota_status": probe_quota_status,
            "record_and_backfill_turn_outcome": record_and_backfill_turn_outcome,
            "save_dual_track_handoff": save_dual_track_handoff,
            "classify_session_profile": classify_session_profile,
            "DecisionSubstrate": DecisionSubstrate,
            "engine": GemmaPrepEngine(),
            "pipeline_cls": ContextAssemblyPipeline,
        }, None
    except Exception as exc:
        _record_bootstrap_self_lock(exc)
        return None, exc


def extract_last_user_query(payload: dict) -> str:
    """从 transcript.jsonl 中提取最近一条用户提问内容并清洗元数据标签。"""
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
        return _fallback_clean_user_text(last_user)
    except Exception:
        return ""


def evaluate_pre_invocation(payload: dict) -> dict:
    """思考前 Hook: 前置意图、历史结果闭环、TTL 配额监控、双轨交接续接与自锁修复告警。"""
    hud_parts: list[str] = []
    rt, boot_exc = _load_runtime()
    if boot_exc is not None or rt is None:
        warn = _read_self_lock_warning()
        if warn:
            hud_parts.append(warn)
        return {"injectSteps": [{"ephemeralMessage": "\n\n".join(hud_parts)}]} if hud_parts else {"injectSteps": []}

    try:
        conv_id = str(payload.get("conversationId") or "default")
        transcript_path = payload.get("transcriptPath")
        workspace_paths = payload.get("workspacePaths") or []
        s_profile = rt["classify_session_profile"](workspace_paths, transcript_path=transcript_path)
        primary_ws = s_profile.get("primary_workspace") or (str(workspace_paths[0]) if workspace_paths else None)

        t_state = rt["inspect_transcript_directional_state"](transcript_path, conversation_id=conv_id)
        query = t_state.get("latest_user_query") or extract_last_user_query(payload)
        user_step_index = int(t_state.get("user_step_index", -1))

        if query:
            substrate = rt["DecisionSubstrate"].get_instance()
            grounding = substrate.on_pre_turn(
                session_id=conv_id,
                prompt=query,
                state=t_state.get("dialogue_context", ""),
                session_profile=s_profile,
            )
            locked = bool(
                grounding.intent.intent == "DIRECTIONAL_PLANNING"
                and not t_state.get("ask_question_answered")
            )
            rt["write_directional_lock"](
                conversation_id=conv_id,
                locked=locked,
                query=query,
                user_step_index=user_step_index,
                signals=grounding.directional_gate.get("signals", []),
                reason=grounding.intent.description,
            )
            try:
                rt["record_and_backfill_turn_outcome"](
                    conversation_id=conv_id,
                    user_step_index=user_step_index,
                    query=query,
                    predicted_intent=grounding.intent.intent,
                    confidence=grounding.intent.confidence,
                    transcript_path=transcript_path,
                )
            except Exception:
                pass
            if grounding and grounding.hud_markdown:
                hud_parts.append(grounding.hud_markdown)

        # Purpose C: TTL-cached Connect-RPC quota inspection (>15% -> 0 token silent)
        try:
            qres = rt["probe_quota_status"](conversation_id=conv_id)
            if qres.get("level") == "CRITICAL":
                rt["save_dual_track_handoff"](
                    conversation_id=conv_id,
                    workspace_path=primary_ws,
                    summary=f"Auto-handoff triggered at low quota ({qres.get('min_pct')}%): {query[:160] or 'active session'}",
                    next_steps=["Resume pending task from checkpoint and verify modified modules."],
                    reason="quota_critical",
                    session_mode=s_profile.get("mode", ""),
                    min_interval_sec=300.0,
                )
            if qres.get("hud"):
                hud_parts.append(qres["hud"])
        except Exception:
            pass

        # Purpose C: Surface unresolved cross-session handoffs once per new session
        try:
            eff_handoff_ws = list(dict.fromkeys([*(workspace_paths or []), *(s_profile.get("targeted_repos") or [])]))
            pending_items = rt["load_pending_handoffs"](
                workspace_paths=eff_handoff_ws,
                current_conv_id=conv_id,
                mark_seen_for_session=True,
            )
            handoff_hud = rt["format_handoff_hud"](pending_items)
            if handoff_hud:
                hud_parts.append(handoff_hud)
        except Exception:
            pass

        if hud_parts:
            return {"injectSteps": [{"ephemeralMessage": "\n\n".join(hud_parts)}]}
        return {"injectSteps": []}
    except Exception:
        return {"injectSteps": []}


def evaluate_pre_tool_use(payload: dict) -> dict:
    """Agent Loop 决策门禁: 若门禁模块异常则自动 fail-open 防死锁，正常时校对方向性锁与安全边界。"""
    rt, boot_exc = _load_runtime()
    if boot_exc is not None or rt is None:
        return {"decision": "allow"}

    try:
        tool_call = payload.get("toolCall", {})
        tool_name = str(tool_call.get("name", ""))
        args = tool_call.get("args", {})
        conv_id = str(payload.get("conversationId") or "default")
        transcript_path = payload.get("transcriptPath")

        if rt["is_readonly_tool_call"](tool_name, args):
            return {"decision": "allow"}

        t_state = rt["inspect_transcript_directional_state"](transcript_path, conversation_id=conv_id)
        lock_rec = rt["read_directional_lock"](conv_id)
        locked = bool(t_state.get("is_locked"))

        if (
            not locked
            and lock_rec
            and lock_rec.get("locked")
            and not t_state.get("ask_question_answered")
            and "[clarification_resolved]" not in (t_state.get("dialogue_context") or "").lower()
        ):
            cur_step = t_state.get("user_step_index", -1)
            rec_step = lock_rec.get("user_step_index", -1)
            # Only carry over lock across identical step (or missing transcript step); step drift auto-clears stale lock.
            if cur_step == -1 or rec_step == -1 or cur_step == rec_step:
                locked = True

        if not locked and lock_rec and lock_rec.get("locked"):
            rt["write_directional_lock"](conv_id, locked=False)

        substrate = rt["DecisionSubstrate"].get_instance()
        if not locked:
            substrate.confirm_directional_plan(conv_id, note="resolved_by_transcript")

        if locked:
            gate = rt["evaluate_directional_tool_gate"](tool_name, args, locked=True)
            if gate:
                denial_count = int((lock_rec or {}).get("denial_count", 0)) + 1
                rt["write_directional_lock"](
                    conversation_id=conv_id,
                    locked=True,
                    query=str((lock_rec or {}).get("query") or ""),
                    user_step_index=int((lock_rec or {}).get("user_step_index", t_state.get("user_step_index", -1))),
                    signals=(lock_rec or {}).get("signals", []),
                    reason=str((lock_rec or {}).get("reason") or ""),
                )
                # Persist incremented denial_count cleanly if lock store exists
                try:
                    from deciders.directional_gate import DEFAULT_LOCK_STORE
                    if DEFAULT_LOCK_STORE.is_file():
                        ddata = json.loads(DEFAULT_LOCK_STORE.read_text(encoding="utf-8"))
                        if isinstance(ddata, dict) and isinstance(ddata.get(conv_id), dict):
                            ddata[conv_id]["denial_count"] = denial_count
                            DEFAULT_LOCK_STORE.write_text(json.dumps(ddata, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                if denial_count >= 3:
                    gate["reason"] = (
                        f"{gate['reason']}\n[SELF_LOCK_LOOP_DETECTED] Directional gate blocked mutating tool "
                        f"{denial_count}x in this step. Do not retry mutations; call ask_question or clarify with user."
                    )
                return gate

        hazard = substrate.on_hazard_check(
            tool_name,
            args,
            session_id=conv_id,
            context={"directional_locked": locked},
        )
        return hazard.to_proto_hook_dict()
    except Exception:
        return {"decision": "allow"}


def main():
    if "--post-task-audit" in sys.argv:
        rt, boot_exc = _load_runtime()
        if boot_exc is not None or rt is None:
            print(json.dumps({"error": str(boot_exc)}))
            return 1
        idx = sys.argv.index("--post-task-audit")
        ws = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "."
        goal = sys.argv[idx + 2] if idx + 2 < len(sys.argv) else ""
        res = rt["engine"].post_task_codebase_audit(workspace_path=ws, session_goal=goal)
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
