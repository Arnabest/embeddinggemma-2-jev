from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deciders.handoff_quota import load_pending_handoffs, save_dual_track_handoff
from deciders.session_context import classify_session_profile, stream_preference_for_session
from decision_substrate.adapters.heuristic_adapter import HeuristicDecisionAdapter
from decision_substrate.kernel import DecisionSubstrate
from engine import GemmaPrepEngine
from recall.memory_hub import MemoryHub


def _write_tx(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_session_context_four_modes_and_stream_bias(tmp_path: Path) -> None:
    repo = tmp_path / "svc_repo"
    (repo / ".git").mkdir(parents=True)
    code_file = repo / "service.py"
    code_file.write_text("def ok(): return True\n", encoding="utf-8")

    # 1. workspace_bound
    c1 = classify_session_profile([str(repo)])
    assert c1["mode"] == "workspace_bound"
    assert c1["stream_bias"] == "code"

    # 2. standalone_targeted_repo (even after >100 trailing EPHEMERAL_MESSAGE lines)
    recs = [{"type": "PLANNER_RESPONSE", "tool_calls": [{"name": "edit_file", "arguments": {"Path": str(code_file)}}]}]
    recs.extend({"type": "EPHEMERAL_MESSAGE", "content": "hud heartbeat"} for _ in range(110))
    tx_repo = _write_tx(tmp_path / "repo_tx.jsonl", recs)
    c2 = classify_session_profile([], transcript_path=tx_repo)
    assert c2["mode"] == "standalone_targeted_repo"
    assert c2["primary_workspace"] == str(repo.resolve()).replace("\\", "/")
    assert c2["stream_bias"] == "code"

    # 3. standalone_scratch
    c3 = classify_session_profile([], tool_name="write_file", target_path=str(tmp_path / ".gemini" / "antigravity" / "scratch" / "probe.py"))
    assert c3["mode"] == "standalone_scratch"
    assert c3["stream_bias"] == "interaction"

    # 4. standalone_chat_research
    c4 = classify_session_profile([])
    assert c4["mode"] == "standalone_chat_research"
    assert c4["stream_bias"] == "interaction"

    assert stream_preference_for_session("standalone_chat_research", "READONLY_QUERY") == "interaction"
    assert stream_preference_for_session("standalone_targeted_repo", "READONLY_QUERY") == "code"


def test_intent_avoids_code_mutate_default_in_chat_and_scratch() -> None:
    adapter = HeuristicDecisionAdapter()
    eng = GemmaPrepEngine()

    chat_decision = adapter.infer_intent(
        "Explain the tradeoffs between WAL mode and MVCC snapshots in SQLite",
        session_mode="standalone_chat_research",
    )
    assert chat_decision.intent == "READONLY_QUERY"

    scratch_decision = adapter.infer_intent(
        "Compare these two benchmark summaries",
        session_mode="standalone_scratch",
    )
    assert scratch_decision.intent == "READONLY_QUERY"

    eng_chat = eng.classify_intent(
        "Why do some hooks inject ephemeral messages cleanly",
        session_mode="standalone_chat_research",
    )
    assert eng_chat.get("intent") == "READONLY_QUERY"

    # Explicit code mutation signals still route to CODE_MUTATE even in standalone mode
    explicit_edit = adapter.infer_intent(
        "修复 server.py 中的分页逻辑",
        session_mode="standalone_chat_research",
    )
    assert explicit_edit.intent == "CODE_MUTATE"


def test_kernel_pre_turn_hud_and_interaction_stream_bias(tmp_path: Path, monkeypatch) -> None:
    def _seed_db(db_path: Path, node_id: str, title: str, content: str) -> None:
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT UNIQUE NOT NULL, track TEXT NOT NULL, "
            "title TEXT NOT NULL, content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', "
            "metadata TEXT NOT NULL DEFAULT '{}', version INTEGER NOT NULL DEFAULT 1, "
            "is_active INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL, updated_at REAL NOT NULL);"
        )
        now = time.time()
        conn.execute(
            "INSERT INTO memories (node_id, track, title, content, tags, metadata, version, is_active, created_at, updated_at) "
            "VALUES (?, 'trajectory', ?, ?, '[]', '{}', 1, 1, ?, ?);",
            (node_id, title, content, now, now),
        )
        conn.commit()
        conn.close()

    db_code = tmp_path / "vault_code.db"
    db_int = tmp_path / "vault_interaction.db"
    _seed_db(db_code, "c1", "architecture latency note", "architecture latency token")
    _seed_db(db_int, "i1", "user latency note", "architecture latency token")

    monkeypatch.setenv("VAULT_CODE_DB", str(db_code))
    monkeypatch.setenv("VAULT_INTERACT_DB", str(db_int))
    monkeypatch.setattr("recall.memory_hub._load_mv_engine_modules", lambda: None)

    hub = MemoryHub()
    biased_hits = hub.recall_vault_memory("latency token", top_k=1, stream_bias="interaction")
    assert len(biased_hits) == 1
    assert biased_hits[0]["source"] == "vault_interact"

    default_hits = hub.recall_vault_memory("latency token", top_k=1, stream_bias="code")
    assert len(default_hits) == 1
    assert default_hits[0]["source"] == "vault_code"

    sub = DecisionSubstrate(adapter=HeuristicDecisionAdapter())
    grounding = sub.on_pre_turn(
        session_id="conv-chat-1",
        prompt="Summarize design principles for latency isolation",
        include_recall=False,
        include_hardware=False,
        session_profile={
            "mode": "standalone_chat_research",
            "primary_workspace": None,
            "stream_bias": "interaction",
        },
    )
    assert "[会话画像]" in grounding.hud_markdown
    assert "standalone_chat_research" in grounding.hud_markdown
    assert "interaction" in grounding.hud_markdown


def test_handoff_preserves_session_mode_and_cross_session_aggregation(tmp_path: Path) -> None:
    repo = tmp_path / "repo_handoff"
    repo.mkdir()
    db_path = tmp_path / "vault_code.db"

    saved = save_dual_track_handoff(
        conversation_id="conv-prev",
        workspace_path=repo,
        summary="Finished stage 2 rollout and verified unit suite",
        next_steps=["Run stage 3 verification"],
        session_mode="standalone_targeted_repo",
        db_path=db_path,
    )
    rec = json.loads(Path(saved["json_path"]).read_text(encoding="utf-8"))
    assert rec["session_mode"] == "standalone_targeted_repo"

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT tags, metadata FROM memories WHERE node_id = ?;", (saved["handoff_id"],)).fetchone()
    conn.close()
    assert row is not None
    assert "session:standalone_targeted_repo" in json.loads(row[0])
    assert json.loads(row[1]).get("session_mode") == "standalone_targeted_repo"

    # Cross-session aggregation from a subsequent session must surface pending handoff
    pending = load_pending_handoffs(
        workspace_paths=[str(repo)],
        current_conv_id="conv-next",
        db_path=db_path,
        mark_seen_for_session=True,
    )
    assert len(pending) == 1
    assert pending[0]["handoff_id"] == saved["handoff_id"]
