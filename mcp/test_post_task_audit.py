#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Physical test suite for gemma_post_task_audit."""
import tempfile
import os
import shutil
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine import GemmaPrepEngine
from server import GemmaPrepCoreMCPServer


def test_real_workspace_audit():
    print("[TEST 1] Auditing gemma-prep-core workspace itself...")
    engine = GemmaPrepEngine()
    res = engine.post_task_codebase_audit(
        workspace_path=str(HERE),
        exit_code=0,
        session_goal="实现任务后代码仓库整理与架构同步裁决工具",
        max_line_limit=800
    )

    assert "audit_result" in res, "Missing audit_result"
    assert res["total_files"] >= 4, f"Expected >= 4 code files, got {res['total_files']}"
    assert len(res["violations"]) == 0, f"Expected 0 violations, got {res['violations']}"
    assert "structure_tree" in res, "Missing structure_tree"
    assert "symbol_matrix" in res, "Missing symbol_matrix"
    assert "memory_vault_payload" in res, "Missing memory_vault_payload"

    print(f" -> Result: {res['audit_result']} (conf: {res['confidence']:.2f})")
    print(f" -> Total Files: {res['total_files']}, Total Lines: {res['total_lines']}")
    print(f" -> Latency: {res['latency_ms']}ms (device: {res['device']})")
    print(" -> Structure Tree Snippet:\n" + "\n".join(res["structure_tree"].splitlines()[:5]))
    print("[TEST 1] PASS\n")


def test_violation_and_warn_refactor():
    print("[TEST 2] Testing workspace with file > 800 lines violation...")
    temp_dir = tempfile.mkdtemp(prefix="nexus_test_audit_")
    try:
        # Create a file with 850 lines
        big_file = Path(temp_dir) / "monolith.py"
        lines = ["# Monolithic giant module\n"] + [f"def func_{i}(): pass\n" for i in range(850)]
        big_file.write_text("".join(lines), encoding="utf-8")

        # Create a small compliant file
        small_file = Path(temp_dir) / "helper.py"
        small_file.write_text("def helper(): pass\n", encoding="utf-8")

        engine = GemmaPrepEngine()
        res = engine.post_task_codebase_audit(
            workspace_path=temp_dir,
            exit_code=0,
            session_goal="新增功能",
            max_line_limit=800
        )

        assert res["audit_result"] == "WARN_REFACTOR", f"Expected WARN_REFACTOR, got {res['audit_result']}"
        assert len(res["violations"]) == 1, f"Expected 1 violation, got {len(res['violations'])}"
        assert res["violations"][0]["lines"] > 800
        assert res["violations"][0]["excess"] > 0
        print(f" -> Correctly caught violation: {res['violations'][0]['path']} ({res['violations'][0]['lines']} lines, +{res['violations'][0]['excess']} over limit)")
        print(f" -> Recommendation: {res['recommendation']}")
        print("[TEST 2] PASS\n")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_mcp_server_dispatch():
    print("[TEST 3] Testing MCP server tool dispatch for gemma_post_task_audit...")
    server = GemmaPrepCoreMCPServer()
    manifest = server.get_tools_manifest()
    tool_names = [t["name"] for t in manifest]
    assert "gemma_post_task_audit" in tool_names, "gemma_post_task_audit not in MCP tools manifest"

    out = server.handle_tool_call("gemma_post_task_audit", {
        "workspace_path": str(HERE),
        "exit_code": 0,
        "session_goal": "解耦重构完成，验证架构同步",
        "max_line_limit": 800
    })

    assert "audit_result" in out, "Tool call output missing audit_result"
    assert out["audit_result"] in ("SYNC_VAULT", "PASS", "WARN_REFACTOR")
    print(f" -> MCP Tool Call Dispatched successfully: {out['audit_result']}")
    print("[TEST 3] PASS\n")


if __name__ == "__main__":
    test_real_workspace_audit()
    test_violation_and_warn_refactor()
    test_mcp_server_dispatch()
    print("ALL 3 TESTS PASSED (Exit Code 0)!")
