#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for quota warning/critical HUD, dual-track handoff, intent calibration, and fail-open self-lock."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deciders.directional_gate import evaluate_directional_intent, is_readonly_query, read_directional_lock
from deciders.handoff_quota import (
    format_handoff_hud,
    load_pending_handoffs,
    probe_quota_status,
    record_and_backfill_turn_outcome,
    save_dual_track_handoff,
)
from decision_substrate.adapters.heuristic_adapter import HeuristicDecisionAdapter
from engine import GemmaPrepEngine
import hook_adapter


def _mock_quota_payload(pct: float, email: str = "tester@example.com") -> dict:
    return {
        "success": True,
        "matched_account": {
            "account_email": email,
            "gemini_5h_pct": pct,
            "gemini_5h_reset": "~2h10m",
            "gemini_weekly_pct": 85.0,
            "claude_gpt_5h_pct": 90.0,
            "claude_gpt_weekly_pct": 95.0,
        },
    }


def test_quota_thresholds_and_ttl_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_file = tmp_path / "quota_cache.json"
    monkeypatch.setenv("NEXUS_QUOTA_CACHE_FILE", str(cache_file))

    ok_res = probe_quota_status("c1", cache_path=cache_file, probe_fn=lambda _: _mock_quota_payload(45.0))
    assert ok_res["level"] == "OK"
    assert ok_res["hud"] == ""

    warn_res = probe_quota_status("c1", cache_path=cache_file, probe_fn=lambda _: _mock_quota_payload(12.5))
    assert warn_res["level"] == "WARN"
    assert "QUOTA_WARN" in warn_res["hud"]
    assert "12.5%" in warn_res["hud"]

    crit_res = probe_quota_status("c1", cache_path=cache_file, probe_fn=lambda _: _mock_quota_payload(6.0))
    assert crit_res["level"] == "CRITICAL"
    assert "QUOTA_CRITICAL <= 8%" in crit_res["hud"]

    # Subsequent default call without probe_fn reads TTL cache instead of spawning PS
    cached_res = probe_quota_status("c1", ttl_sec=90.0, cache_path=cache_file)
    assert cached_res["level"] == "CRITICAL"


def test_dual_track_handoff_write_and_new_session_resurface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    db_file = tmp_path / "vault_code.db"
    monkeypatch.setenv("NEXUS_MEMORY_VAULT_DB", str(db_file))

    saved = save_dual_track_handoff(
        conversation_id="session_old",
        workspace_path=ws,
        summary="Finished Safety Barrier; remaining work is adding Stop hook tests.",
        next_steps=["Run pytest on test_stop_hook.py"],
        reason="quota_critical",
        db_path=db_file,
    )
    assert Path(saved["json_path"]).is_file()
    assert saved["vault"]["success"] is True

    # Same session should not resurface its own handoff
    same = load_pending_handoffs([str(ws)], current_conv_id="session_old", db_path=db_file)
    assert same == []

    # New session sees it once and formats clean HUD
    new_items = load_pending_handoffs([str(ws)], current_conv_id="session_new", db_path=db_file)
    assert len(new_items) == 1
    hud = format_handoff_hud(new_items)
    assert "PENDING_HANDOFF" in hud
    assert "Finished Safety Barrier" in hud

    # Second check in same new session does not spam repeatedly
    assert load_pending_handoffs([str(ws)], current_conv_id="session_new", db_path=db_file) == []

    # Verify memory-vault sqlite actually holds the tagged row
    conn = sqlite3.connect(str(db_file))
    row = conn.execute("SELECT track, tags FROM memories WHERE node_id = ?", (saved["handoff_id"],)).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "trajectory"
    assert "handoff" in row[1]


def test_intent_calibration_questions_and_macro_redesign() -> None:
    q_gap = "根据历史工作记录，还有什么被遗漏的，当前仍有实用价值的机制未被保留"
    q_why = "为什么当时的制度没有被后面沿袭"
    q_eval_compound = "门禁判断是否与代码分支树以及上下游关联文件等形成完整的修改请求——查询代码位置上下游——评估后果——放行条件判断——工具请求批准的闭环"
    q_eval_arch = "评价当前本地agent运行整体机制"
    q_continue = "继续完善"
    q_imp_edit = "请修改 antigravity_gate.py 增加日志"
    q_design = "不能照搬，需要结合现有较为完善的基座下重新按用途进行设计"
    q_fix = "修复 deciders/base.py 第 10 行 KeyError"
    q_plan_num = "1.爆炸半径需要以从宿主环境——开发环境——生产环境三层受改动影响的范围\n2.默认注入 [BLAST_RADIUS_HUD] 提醒\n3.是"

    assert is_readonly_query(q_gap)
    assert is_readonly_query(q_why)
    assert is_readonly_query(q_eval_compound)
    assert is_readonly_query(q_eval_arch)
    assert not is_readonly_query(q_continue)
    assert not is_readonly_query(q_imp_edit)
    assert evaluate_directional_intent(q_design)["is_directional"] is True
    assert evaluate_directional_intent(q_fix)["is_directional"] is False

    heur = HeuristicDecisionAdapter()
    assert heur.infer_intent(q_gap).intent == "READONLY_QUERY"
    assert heur.infer_intent(q_why).intent == "READONLY_QUERY"
    assert heur.infer_intent(q_eval_compound).intent == "READONLY_QUERY"
    assert heur.infer_intent(q_eval_arch).intent == "READONLY_QUERY"
    assert heur.infer_intent(q_continue).intent == "CODE_MUTATE"
    assert heur.infer_intent(q_imp_edit).intent == "CODE_MUTATE"
    assert heur.infer_intent(q_design).intent == "DIRECTIONAL_PLANNING"
    assert heur.infer_intent(q_fix).intent == "CODE_MUTATE"

    eng = GemmaPrepEngine()
    assert eng.classify_intent(q_gap)["intent"] == "READONLY_QUERY"
    assert eng.classify_intent(q_eval_compound)["intent"] == "READONLY_QUERY"
    assert eng.classify_intent(q_eval_arch)["intent"] == "READONLY_QUERY"
    assert eng.classify_intent(q_continue)["intent"] == "CODE_MUTATE"
    assert eng.classify_intent(q_design)["intent"] == "DIRECTIONAL_PLANNING"
    assert eng.classify_intent(q_plan_num)["intent"] == "CODE_MUTATE"


def test_closed_loop_outcome_backfill(tmp_path: Path) -> None:
    tp = tmp_path / "transcript.jsonl"
    steps = [
        {"step_index": 1, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "Inspect gap"},
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "source": "MODEL",
            "tool_calls": [{"name": "read_file", "args": {"Path": "foo.py"}}],
        },
        {"step_index": 3, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "Next turn"},
    ]
    tp.write_text("\n".join(json.dumps(s) for s in steps) + "\n", encoding="utf-8")

    pred_file = tmp_path / "turn_preds.json"
    harvest_file = tmp_path / "harvested.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"

    # Turn 1 wrongly predicted CODE_MUTATE
    record_and_backfill_turn_outcome(
        conversation_id="c_out",
        user_step_index=1,
        query="Inspect gap",
        predicted_intent="CODE_MUTATE",
        confidence=0.99,
        transcript_path=str(tp),
        pred_store_path=pred_file,
        harvest_path=harvest_file,
        outcomes_log_path=outcomes_file,
    )
    # Turn 2 observes Turn 1 only ran read_file -> backfills READONLY_QUERY mismatch
    bf = record_and_backfill_turn_outcome(
        conversation_id="c_out",
        user_step_index=3,
        query="Next turn",
        predicted_intent="READONLY_QUERY",
        confidence=0.88,
        transcript_path=str(tp),
        pred_store_path=pred_file,
        harvest_path=harvest_file,
        outcomes_log_path=outcomes_file,
    )
    assert bf is not None
    assert bf["mismatch"] is True
    assert bf["observed_intent"] == "READONLY_QUERY"
    assert harvest_file.is_file()
    assert "READONLY_QUERY" in harvest_file.read_text(encoding="utf-8")


def test_hook_fail_open_self_lock_and_stale_lock_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sl_file = tmp_path / "hook_self_lock.json"
    monkeypatch.setenv("NEXUS_HOOK_SELF_LOCK_FILE", str(sl_file))

    def broken_loader():
        exc = ImportError("simulated broken module")
        hook_adapter._record_bootstrap_self_lock(exc)
        return None, exc

    monkeypatch.setattr(hook_adapter, "_load_runtime", broken_loader)
    pre_tool = hook_adapter.evaluate_pre_tool_use(
        {"conversationId": "c_bo", "toolCall": {"name": "edit_file", "args": {"Path": "broken.py"}}}
    )
    assert pre_tool == {"decision": "allow"}

    pre_inv = hook_adapter.evaluate_pre_invocation({"conversationId": "c_bo"})
    assert len(pre_inv["injectSteps"]) == 1
    assert "HOOK_SELF_LOCK_CRITICAL" in pre_inv["injectSteps"][0]["ephemeralMessage"]

    # Stale directional lock auto-expires
    lock_file = tmp_path / "locks.json"
    lock_file.write_text(
        json.dumps({"c_stale": {"locked": True, "updated_at": time.time() - 3600, "user_step_index": 5}}),
        encoding="utf-8",
    )
    assert read_directional_lock("c_stale", lock_store_path=lock_file, max_age_sec=1800.0) is None
