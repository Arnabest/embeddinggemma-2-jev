#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive verification suite for refactored Gemma-Jev Decision Micro-Kernel."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine import GemmaPrepEngine
from server import GemmaPrepCoreMCPServer


def test_command_guard():
    print("[TEST 1] Testing CommandGuardDecider...")
    engine = GemmaPrepEngine()

    safe_res = engine.audit_command_safety("git status", cwd=".")
    assert safe_res["danger_level"] <= 2, f"Expected safe level, got {safe_res['danger_level']}"
    assert safe_res["is_blocked"] is False

    danger_res = engine.audit_command_safety("Remove-Item -Recurse -Force /system/root", cwd=".")
    assert danger_res["danger_level"] >= 4, f"Expected danger level >= 4, got {danger_res['danger_level']}"
    assert danger_res["is_blocked"] is True
    assert danger_res["risk_category"] == "DESTRUCTIVE_DELETE"

    print(" -> Safe command evaluated successfully (Danger Lvl 1)")
    print(f" -> Dangerous command blocked successfully: {danger_res['risk_category']} (Lvl {danger_res['danger_level']})")
    print("[TEST 1] PASS\n")


def test_failure_triage():
    print("[TEST 2] Testing FailureTriageDecider...")
    engine = GemmaPrepEngine()

    tb_dep = "Traceback (most recent call last):\n  File 'app.py', line 1\nModuleNotFoundError: No module named 'torch'"
    res_dep = engine.triage_failure(tb_dep, exit_code=1)
    assert res_dep["failure_category"] == "DEPENDENCY_MISSING", f"Got {res_dep['failure_category']}"

    tb_assert = "Traceback:\nAssertionError: assert res == 200, got 500"
    res_assert = engine.triage_failure(tb_assert, exit_code=1)
    assert res_assert["failure_category"] == "ASSERTION_LOGIC", f"Got {res_assert['failure_category']}"

    print(f" -> Dependency missing classified: {res_dep['failure_category']} ({res_dep['remediation_capsule']})")
    print(f" -> Assertion failure classified: {res_assert['failure_category']}")
    print("[TEST 2] PASS\n")


def test_memory_gate():
    print("[TEST 3] Testing MemoryGateDecider...")
    engine = GemmaPrepEngine()

    chatter_res = engine.audit_memory_novelty("Chat log", "好的，收到，谢谢！", track="chat")
    assert chatter_res["should_ingest"] is True
    assert chatter_res["target_stream"] == "interaction"
    assert chatter_res["target_track"] == "chat"

    empty_res = engine.audit_memory_novelty("", "")
    assert empty_res["should_ingest"] is False

    tech_content = """
    ### [架构] 物理双库磁盘硬断开与1ms影子快照机制
    ```python
    def take_snapshot():
        git write-tree ...
    ```
    采用纯依赖下沉与 Shim 垫片设计，彻底阻断环状依赖，单测通过 (Exit Code 0)。
    """
    tech_res = engine.audit_memory_novelty("物理双库架构设计", tech_content, track="architecture")
    assert tech_res["should_ingest"] is True
    assert tech_res["target_stream"] == "code"
    assert tech_res["target_track"] == "architecture"
    assert tech_res["novelty_score"] >= 4.0

    print(f" -> Chatter routed to interaction/chat: stream={chatter_res['target_stream']}, track={chatter_res['target_track']}")
    print(f" -> High-value ADR routed to code/architecture: stream={tech_res['target_stream']}, track={tech_res['target_track']} (score: {tech_res['novelty_score']})")
    print("[TEST 3] PASS\n")


def test_model_router():
    print("[TEST 4] Testing ModelRouterDecider...")
    engine = GemmaPrepEngine()

    trivial_res = engine.route_model_tier("Python 字典推导式的基本语法怎么写？")
    assert trivial_res["recommended_tier"] == "CLOUD_FLASH"
    assert trivial_res["current_paradigm"] == "ONLINE_CLOUD_PRIMARY"
    assert trivial_res["local_harness_eligible"] is True
    assert trivial_res["future_harness_tier"] == "LOCAL_LIGHT"

    complex_res = engine.route_model_tier("如何重构千万级高并发消息网关并排查分布式死锁与内存泄漏？", context_len=5000)
    assert complex_res["recommended_tier"] == "CLOUD_PRO"
    assert complex_res["local_harness_eligible"] is False

    print(f" -> Trivial task routed to: {trivial_res['recommended_tier']} (Local Harness Eligible: {trivial_res['local_harness_eligible']})")
    print(f" -> Complex task routed to: {complex_res['recommended_tier']} (Complexity: {complex_res['estimated_complexity']})")
    print("[TEST 4] PASS\n")


def test_facade_and_mcp_dispatch():
    print("[TEST 5] Testing GemmaPrepCoreMCPServer tool dispatch...")
    server = GemmaPrepCoreMCPServer()
    manifest = server.get_tools_manifest()
    names = [t["name"] for t in manifest]
    for required in (
        "gemma_intent_classify", "gemma_memory_recall", "gemma_jev_decide",
        "gemma_assemble_context", "gemma_hardware_status", "gemma_post_task_audit",
        "gemma_command_audit", "gemma_failure_triage", "gemma_memory_gate", "gemma_model_route"
    ):
        assert required in names, f"Missing tool: {required}"

    # Test dispatching newly added tool
    res = server.handle_tool_call("gemma_command_audit", {"command_line": "dir"})
    assert "danger_level" in res
    assert res["is_blocked"] is False
    print(f" -> Dispatched gemma_command_audit successfully: Danger Lvl {res['danger_level']}")
    print("[TEST 5] PASS\n")


if __name__ == "__main__":
    test_command_guard()
    test_failure_triage()
    test_memory_gate()
    test_model_router()
    test_facade_and_mcp_dispatch()
    print("ALL 5 TESTS IN MODULAR KERNEL SUITE PASSED (Exit Code 0)!")
