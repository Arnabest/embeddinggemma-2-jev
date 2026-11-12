#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Closed-loop tests for Directional Planning JEV Judgment & PreToolUse Hard Gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE.parent
COMPILER_DIR = CORE_DIR.parent / "nexus-reasoning-compiler"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from deciders.directional_gate import (
    DEFAULT_LOCK_STORE,
    evaluate_directional_intent,
    evaluate_directional_tool_gate,
    inspect_transcript_directional_state,
    write_directional_lock,
)
from decision_substrate import (
    DecisionSubstrate,
    HeuristicDecisionAdapter,
    IntentType,
)
from engine import GemmaPrepEngine
from pipeline import ContextAssemblyPipeline
import hook_adapter


def _assert_no_emojis(text: str) -> None:
    for ch in text:
        code = ord(ch)
        assert not (0x1F300 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF), f"Emoji detected: {ch!r}"


def test_intent_enum_includes_directional_planning():
    assert IntentType.DIRECTIONAL_PLANNING.value == "DIRECTIONAL_PLANNING"
    assert "DIRECTIONAL_PLANNING" in {m.value for m in IntentType}


def test_directional_detection_precision_and_recall():
    directional_prompts = [
        "我想重构一下整个项目的存储架构",
        "帮我做一个本地知识库检索系统",
        "我们往多智能体协同方向演进一下",
        "优化一下系统的整体性能和交互体验",
        "接下来我想给项目加上权限体系和插件生态",
        "先别动手，帮我规划一下整体方案",
    ]
    for p in directional_prompts:
        res = evaluate_directional_intent(p)
        assert res["is_directional"] is True, f"Expected directional for: {p}"

    concrete_prompts = [
        "编写一个快速排序算法并生成单测",
        "查看 /workspace/project/mcp/engine.py",
        "把 engine.py 第 10 行的默认值改成 0.9",
        "修复 hook_adapter.py 中的 AttributeError 报错",
        "立即执行 git reset 消除脏改动，回滚到上一个安全快照",
        "pytest tests/ -q",
        "可以",
        "按方案 B 开始执行",
    ]
    for p in concrete_prompts:
        res = evaluate_directional_intent(p)
        assert res["is_directional"] is False, f"Unexpected directional lock for: {p}"


def test_engine_and_pipeline_directional_integration():
    engine = GemmaPrepEngine()
    res_dir = engine.classify_intent("我想重构一下整个记忆模块的架构")
    assert res_dir["intent"] == "DIRECTIONAL_PLANNING"
    assert res_dir["requires_clarification"] is True

    res_code = engine.classify_intent("编写一个快速排序算法并生成单测")
    assert res_code["intent"] == "CODE_MUTATE"
    assert res_code["requires_clarification"] is False

    res_resolved = engine.classify_intent("就按刚讨论的模块拆法处理", context="[CLARIFICATION_RESOLVED]")
    assert res_resolved["intent"] != "DIRECTIONAL_PLANNING"
    assert res_resolved["requires_clarification"] is False

    pipe = ContextAssemblyPipeline(engine)
    assembled = pipe.assemble("帮我搭建一个自动化视频学习工作流系统")
    assert assembled["directional_gate"]["locked"] is True
    assert "[上下文 JEV 判定 - 方向性前置对齐]" in assembled["hud_markdown"]
    _assert_no_emojis(assembled["hud_markdown"])


def test_substrate_on_pre_turn_and_hazard_gate():
    substrate = DecisionSubstrate(adapter=HeuristicDecisionAdapter())
    sess = "test-directional-sess"

    grounding = substrate.on_pre_turn(session_id=sess, prompt="我想开发一个多源检索工具")
    assert grounding.intent.intent == "DIRECTIONAL_PLANNING"
    assert grounding.intent.requires_clarification is True
    assert grounding.directional_gate["locked"] is True
    assert "[上下文 JEV 判定 - 方向性前置对齐]" in grounding.hud_markdown
    _assert_no_emojis(grounding.hud_markdown)

    # Read-only & ask_question allowed
    assert substrate.on_hazard_check("read_file", {"Path": "a.py"}, session_id=sess).decision == "allow"
    assert substrate.on_hazard_check("ask_question", {"questions": []}, session_id=sess).decision == "allow"

    # Mutating & shell execution hard-denied
    for tool_name, args in (
        ("write_file", {"Path": "a.py", "Content": "x=1"}),
        ("edit_file", {"Path": "a.py", "Target": "a", "Replacement": "b"}),
        ("write_to_file", {"TargetFile": "a.py"}),
        ("replace_file_content", {"TargetFile": "a.py"}),
        ("run_command", {"CommandLine": "echo hi"}),
    ):
        dec = substrate.on_hazard_check(tool_name, args, session_id=sess)
        assert dec.decision == "deny", f"{tool_name} should be denied when directional lock active"
        assert "DIRECTIONAL_PLANNING_GATE" in dec.reason
        _assert_no_emojis(dec.reason)

    # Confirm unlocks
    substrate.confirm_directional_plan(sess, note="confirmed")
    assert substrate.on_hazard_check("write_file", {"Path": "a.py", "Content": "x=1"}, session_id=sess).decision == "allow"


def test_hook_lifecycle_transcript_lock_unlock_and_relock(tmp_path):
    conv_id = "test-hook-conv-directional"
    transcript = tmp_path / "transcript.jsonl"
    lock_store = DEFAULT_LOCK_STORE

    def _write_steps(steps):
        transcript.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in steps) + "\n", encoding="utf-8")

    try:
        # Turn 1: Directional prompt -> PreInvocation injects HUD & PreToolUse denies mutating tools
        steps = [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "<USER_REQUEST>\n我想把整个后端重构成插件化架构\n</USER_REQUEST>"}
        ]
        _write_steps(steps)

        pre_inv = hook_adapter.evaluate_pre_invocation({"conversationId": conv_id, "transcriptPath": str(transcript)})
        assert len(pre_inv["injectSteps"]) == 1
        hud = pre_inv["injectSteps"][0]["ephemeralMessage"]
        assert "DIRECTIONAL_PLANNING" in hud
        assert "[上下文 JEV 判定 - 方向性前置对齐]" in hud
        _assert_no_emojis(hud)

        # Read-only allowed, write_file / edit_file / run_command / mutating MCP denied
        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "read_file", "args": {"Path": "x.py"}},
        })["decision"] == "allow"

        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "ask_question", "args": {"questions": []}},
        })["decision"] == "allow"

        for blocked_tool, blocked_args in (
            ("write_file", {"Path": "x.py", "Content": "a=1"}),
            ("edit_file", {"Path": "x.py", "Target": "a", "Replacement": "b"}),
            ("run_command", {"CommandLine": "python test.py"}),
            ("call_mcp_tool", {"ServerName": "memory-vault", "ToolName": "vault_store", "Arguments": {}}),
        ):
            out = hook_adapter.evaluate_pre_tool_use({
                "conversationId": conv_id,
                "transcriptPath": str(transcript),
                "toolCall": {"name": blocked_tool, "args": blocked_args},
            })
            assert out["decision"] == "deny", f"{blocked_tool} must be denied"
            assert "DIRECTIONAL_PLANNING_GATE" in out["reason"]

        # Coordinate with nexus-reasoning-compiler PreInvocation hook when co-located
        compiler_hook = COMPILER_DIR / "hook_adapter.py"
        if compiler_hook.is_file():
            if str(COMPILER_DIR) not in sys.path:
                sys.path.insert(0, str(COMPILER_DIR))
            import importlib.util
            spec = importlib.util.spec_from_file_location("nrc_hook_adapter", str(compiler_hook))
            nrc_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nrc_mod)
            nrc_out = nrc_mod.evaluate_pre_invocation({"conversationId": conv_id, "transcriptPath": str(transcript)})
            assert len(nrc_out["injectSteps"]) == 1
            assert "DIRECTIONAL ALIGNMENT MODE" in nrc_out["injectSteps"][0]["ephemeralMessage"]
            assert "<final_execution>" not in nrc_out["injectSteps"][0]["ephemeralMessage"]
        else:
            from deciders.directional_gate import read_directional_lock
            lock_state = read_directional_lock(conv_id, lock_store_path=lock_store)
            assert lock_state.get("locked") is True
            assert "open_macro_goal" in lock_state.get("signals", [])
            assert lock_state.get("user_step_index") == 0

        # User answers ask_question -> unlocks immediately
        steps.append({"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "ask_question", "args": {}}], "content": "请确认范围"})
        steps.append({"step_index": 2, "source": "MODEL", "type": "GENERIC", "status": "DONE", "content": "Created At: t1\nCompleted At: t2\nA1: 只重构 core 模块"})
        _write_steps(steps)

        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "write_file", "args": {"Path": "core.py", "Content": "x=1"}},
        })["decision"] == "allow"

        # After implementing Turn 1, Turn 2 introduces a NEW directional prompt -> must RE-LOCK!
        steps.append({"step_index": 3, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "write_file", "args": {"Path": "core.py"}}], "content": "Done"})
        steps.append({"step_index": 4, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "接下来我想搭建一个自动化监控告警平台"})
        _write_steps(steps)

        t_state2 = inspect_transcript_directional_state(str(transcript), conversation_id=conv_id)
        assert t_state2["is_locked"] is True
        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "run_command", "args": {"CommandLine": "mkdir monitor"}},
        })["decision"] == "deny"

        # Plain-text clarification flow: Assistant asks in text (no mutations), User replies in next USER_INPUT -> unlocks!
        steps.append({"step_index": 5, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [], "content": "请确认监控指标和告警渠道"})
        steps.append({"step_index": 6, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "指标只看 GPU 显存和进程存活，渠道用本地日志，开始做吧"})
        _write_steps(steps)

        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "write_file", "args": {"Path": "monitor.py", "Content": "ok=True"}},
        })["decision"] == "allow"
    finally:
        write_directional_lock(conv_id, locked=False)
        write_directional_lock(conv_id, locked=False, lock_store_path=lock_store)


def test_interrupted_ask_question_followups_keep_lock_until_resolved(tmp_path: Path) -> None:
    conv_id = "test-conv-interrupted-followup-001"
    transcript = tmp_path / "transcript_interrupted.jsonl"
    lock_store = tmp_path / "directional_locks.json"

    def _write_steps(steps_list: list[dict]) -> None:
        with open(transcript, "w", encoding="utf-8") as f:
            for s in steps_list:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    try:
        # Turn 1: Directional prompt -> model truncates without calling ask_question
        steps = [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "我想新建一个插件化任务调度中心，帮我设计架构"},
            {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [], "content": "请先确认以下 4 项设计与范围边界："},
            {"step_index": 2, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "哪些边界？"},
        ]
        _write_steps(steps)

        t_state_q = inspect_transcript_directional_state(str(transcript), conversation_id=conv_id, lock_store_path=lock_store)
        assert t_state_q["prior_clarification_resolved"] is False
        assert t_state_q["is_locked"] is True

        pre_out = hook_adapter.evaluate_pre_invocation({"conversationId": conv_id, "transcriptPath": str(transcript)})
        hud_msg = pre_out["injectSteps"][0]["ephemeralMessage"]
        assert "DIRECTIONAL_PLANNING" in hud_msg
        assert "1500" in hud_msg

        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "write_file", "args": {"Path": "sched.py", "Content": "x=1"}},
        })["decision"] == "deny"
        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "ask_question", "args": {"questions": []}},
        })["decision"] == "allow"

        # Turn 3: Model failed again ("请通过下方选项确认……") and user reports "未显示选项"
        steps.append({"step_index": 3, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [], "content": "请通过下方选项确认……"})
        steps.append({"step_index": 4, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "未显示选项"})
        _write_steps(steps)

        t_state_missing = inspect_transcript_directional_state(str(transcript), conversation_id=conv_id, lock_store_path=lock_store)
        assert t_state_missing["prior_clarification_resolved"] is False
        assert t_state_missing["is_locked"] is True
        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "edit_file", "args": {"Path": "sched.py", "Target": "a", "Replacement": "b"}},
        })["decision"] == "deny"

        # Turn 4: User provides real plain-text design answer -> unlocks cleanly
        steps.append({"step_index": 5, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [], "content": "请确认调度持久化方式和并发模型：1) 内存队列 2) SQLite"})
        steps.append({"step_index": 6, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "队列只做内存版，配置用 JSON，开始实现吧"})
        _write_steps(steps)

        t_state_resolved = inspect_transcript_directional_state(str(transcript), conversation_id=conv_id, lock_store_path=lock_store)
        assert t_state_resolved["prior_clarification_resolved"] is True
        assert t_state_resolved["is_locked"] is False
        assert hook_adapter.evaluate_pre_tool_use({
            "conversationId": conv_id,
            "transcriptPath": str(transcript),
            "toolCall": {"name": "write_file", "args": {"Path": "sched.py", "Content": "ok=True"}},
        })["decision"] == "allow"
    finally:
        write_directional_lock(conv_id, locked=False)
        write_directional_lock(conv_id, locked=False, lock_store_path=lock_store)
