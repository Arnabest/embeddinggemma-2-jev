#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for new gemma-prep-core MCP query, handoff lifecycle, and HUD injection tools."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import GemmaPrepCoreMCPServer


def test_gemma_handoff_manage_save_list_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    db_path = tmp_path / "vault_code.db"
    monkeypatch.setenv("NEXUS_MEMORY_VAULT_DB", str(db_path))

    srv = GemmaPrepCoreMCPServer()
    names = [t["name"] for t in srv.get_tools_manifest()]
    assert "gemma_handoff_manage" in names
    assert "gemma_gate_status" in names
    assert "gemma_inject_hud" in names

    saved = srv.handle_tool_call(
        "gemma_handoff_manage",
        {
            "action": "save",
            "workspace_path": str(ws),
            "conversation_id": "conv_a",
            "summary": "Implemented blast radius lookup and preflight checks cleanly.",
            "next_steps": ["Run MCP stdio tests"],
            "reason": "session_wrapup",
        },
    )
    assert saved["success"] is True
    assert Path(saved["json_path"]).is_file()

    listed = srv.handle_tool_call(
        "gemma_handoff_manage",
        {"action": "list", "workspace_path": str(ws), "status_filter": "PENDING"},
    )
    assert listed["count"] == 1
    assert listed["items"][0]["handoff_id"] == saved["handoff_id"]

    resolved = srv.handle_tool_call(
        "gemma_handoff_manage",
        {
            "action": "resolve",
            "workspace_path": str(ws),
            "handoff_id": saved["handoff_id"],
            "resolution_note": "All MCP tests green",
        },
    )
    assert resolved["success"] is True
    assert resolved["status"] == "RESOLVED"

    listed_after = srv.handle_tool_call(
        "gemma_handoff_manage",
        {"action": "list", "workspace_path": str(ws), "status_filter": "PENDING"},
    )
    assert listed_after["count"] == 0

    conn = sqlite3.connect(str(db_path))
    meta_raw = conn.execute("SELECT metadata FROM memories WHERE node_id = ?", (saved["handoff_id"],)).fetchone()[0]
    conn.close()
    assert json.loads(meta_raw)["status"] == "RESOLVED"


def test_gemma_gate_status_and_inject_hud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws2"
    (ws / ".nexus").mkdir(parents=True)
    monkeypatch.setenv("NEXUS_MEMORY_VAULT_DB", str(tmp_path / "vault2.db"))
    monkeypatch.setenv("NEXUS_QUOTA_CACHE_FILE", str(tmp_path / "qcache.json"))
    (tmp_path / "qcache.json").write_text(json.dumps({"cached_at": 9999999999.0, "raw": {"success": False}}), encoding="utf-8")

    srv = GemmaPrepCoreMCPServer()
    srv.handle_tool_call(
        "gemma_handoff_manage",
        {
            "action": "save",
            "workspace_path": str(ws),
            "conversation_id": "conv_old",
            "summary": "Unfinished migration task pending verification.",
        },
    )

    gs = srv.handle_tool_call("gemma_gate_status", {"conversation_id": "conv_new", "workspace_path": str(ws)})
    assert gs["conversation_id"] == "conv_new"
    assert gs["pending_handoffs_count"] >= 1
    assert "directional_lock" in gs

    hud = srv.handle_tool_call(
        "gemma_inject_hud",
        {
            "prompt": "查看当前系统架构状态",
            "workspace_path": str(ws),
            "conversation_id": "conv_new",
            "include_hardware": False,
        },
    )
    assert hud["success"] is True
    assert hud["section_count"] >= 2
    assert "PENDING_HANDOFF" in hud["hud_markdown"]


def test_gemma_mcp_stdio_tools_list() -> None:
    server_py = ROOT / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_py)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert proc.stdin and proc.stdout
    req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode("utf-8") + b"\n"
    proc.stdin.write(req)
    proc.stdin.flush()
    line = proc.stdout.readline().decode("utf-8")
    proc.stdin.close()
    proc.terminate()
    proc.wait()
    resp = json.loads(line)
    names = [t["name"] for t in resp["result"]["tools"]]
    for expected in ("gemma_handoff_manage", "gemma_gate_status", "gemma_inject_hud"):
        assert expected in names
