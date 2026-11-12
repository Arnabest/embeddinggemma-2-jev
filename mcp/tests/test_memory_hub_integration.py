#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-seam integration tests for MemoryHub (gemma-prep-core <-> memory-vault <-> skills)."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

CORE_ROOT = Path(__file__).resolve().parent.parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from recall.memory_hub import MemoryHub


def _seed_memories_db(db_path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                track TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '',
                is_active INTEGER DEFAULT 1
            );
            """
        )
        for row in rows:
            conn.execute(
                "INSERT INTO memories (node_id, track, title, content, tags, is_active) VALUES (?, ?, ?, ?, ?, 1)",
                (
                    row["node_id"],
                    row["track"],
                    row["title"],
                    row["content"],
                    json.dumps(row["tags"], ensure_ascii=False),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def test_memory_hub_cross_seam_vault_dialogue_and_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code_db = tmp_path / "vault_code.db"
    inter_db = tmp_path / "vault_interaction.db"
    _seed_memories_db(
        code_db,
        [
            {
                "node_id": "arch:memory_hub_fix",
                "track": "architecture",
                "title": "MemoryHub 记忆召回中枢契约对齐",
                "content": "统一使用 memory-vault 的 memories 表与 FTS5 多路检索，禁用 cards 错表名。",
                "tags": ["memory_hub", "memories", "rrf"],
            }
        ],
    )
    _seed_memories_db(
        inter_db,
        [
            {
                "node_id": "pref:zero_emoji",
                "track": "preference",
                "title": "全局零 Emoji 与中文交流偏好",
                "content": "所有代码注释与报告严禁使用 Emoji，优先给出可验证步骤。",
                "tags": ["preference", "zero_emoji"],
            }
        ],
    )

    # Dialogue index jsonl + Skills dir
    dial_file = tmp_path / "brain_dialogue_index.jsonl"
    dial_file.write_text(
        json.dumps({
            "chunk_id": "conv1_s0_s5",
            "conv_id": "conv1",
            "title": "会话 conv1 [Step 0-5]",
            "time_range": ["2026-11-11T01:00:00Z", "2026-11-11T01:05:00Z"],
            "text": "[Step 0 用户] 修复 memory-vault 检索和 MemoryHub 断链问题",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    skills_dir = tmp_path / "skills" / "memory-audit"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: memory-audit\ndescription: \"记忆系统健康与召回审计工具\"\n---\n# Memory Audit\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MEMORY_VAULT_DIR", str(tmp_path / "no_mv"))
    monkeypatch.setenv("VAULT_CODE_DB", str(code_db))
    monkeypatch.setenv("VAULT_INTERACT_DB", str(inter_db))
    monkeypatch.setenv("DIALOGUE_INDEX_PATH", str(dial_file))
    monkeypatch.setenv("DIALOGUE_CACHE_DB", str(tmp_path / "dialogue_fts.db"))
    monkeypatch.setenv("SKILLS_ROOT", str(tmp_path / "skills"))

    hub = MemoryHub()
    diag = hub.recall_with_diagnostics("memory-vault 记忆召回 断链", top_k=6)
    assert diag["errors"] == []
    sources = {r["source"] for r in diag["results"]}
    assert "vault_code" in sources
    assert "dialogue_transcript" in sources
    assert "skill_shelf" in sources
    assert any(r.get("node_id") == "arch:memory_hub_fix" for r in diag["results"])


def test_memory_hub_surfaces_corrupt_db_error_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_db = tmp_path / "corrupt_code.db"
    bad_db.write_text("not a valid sqlite database", encoding="utf-8")
    monkeypatch.setenv("MEMORY_VAULT_DIR", str(tmp_path / "no_mv"))
    monkeypatch.setenv("VAULT_CODE_DB", str(bad_db))
    monkeypatch.setenv("VAULT_INTERACT_DB", str(tmp_path / "missing_inter.db"))
    monkeypatch.setenv("DIALOGUE_INDEX_PATH", str(tmp_path / "missing.jsonl"))
    monkeypatch.setenv("SKILLS_ROOT", str(tmp_path / "no_skills"))

    hub = MemoryHub()
    diag = hub.recall_with_diagnostics("记忆", top_k=3)
    assert diag["results"] == []
    assert len(diag["errors"]) >= 1
