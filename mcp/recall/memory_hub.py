# -*- coding: utf-8 -*-
r"""MemoryHub: 跨源数据混合召回中枢 (2,095段会话历史 + memory-vault SQLite + F:\AI SKILL 技能库)."""
from __future__ import annotations

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(os.getenv("JEV_TRAIN_ROOT", str(Path.home() / ".gemma" / "jev-train")))
DIALOGUE_INDEX_PATH = Path(os.getenv("DIALOGUE_INDEX_PATH", str(PROJECT_ROOT / "bench" / "brain_dialogue_index.jsonl")))
VAULT_CODE_DB = Path(os.getenv("VAULT_CODE_DB", str(Path.home() / ".memory-vault" / "vault_code.db")))
VAULT_INTERACT_DB = Path(os.getenv("VAULT_INTERACT_DB", str(Path.home() / ".memory-vault" / "vault_interaction.db")))
SKILLS_ROOT = Path(os.getenv("SKILLS_ROOT", str(Path.home() / ".antigravity" / "skills")))


class MemoryHub:
    def __init__(self):
        self._cached_skills: Optional[List[Dict[str, Any]]] = None

    def recall_dialogue_history(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """从 2,095 条物理会话时序索引中检索最相关的历史轨迹。"""
        if not DIALOGUE_INDEX_PATH.exists():
            return []
        tokens = set(re.findall(r"[\w\u4e00-\u9fa5]+", query.lower()))
        if not tokens:
            return []

        hits = []
        try:
            with open(DIALOGUE_INDEX_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        text_lower = obj["text"].lower()
                        overlap = sum(1 for tok in tokens if tok in text_lower)
                        if overlap > 0:
                            score = overlap / (len(text_lower) ** 0.1)
                            hits.append((score, obj))
                    except Exception:
                        pass
        except Exception:
            return []

        hits.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, obj in hits[:top_k]:
            results.append({
                "source": "dialogue_transcript",
                "chunk_id": obj.get("chunk_id"),
                "conv_id": obj.get("conv_id"),
                "title": obj.get("title"),
                "time_range": obj.get("time_range"),
                "snippet": obj.get("text", "")[:350],
                "score": round(score, 3)
            })
        return results

    def recall_skills(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        r"""检索本地 Skills 技能库。"""
        if self._cached_skills is None:
            self._cached_skills = []
            if SKILLS_ROOT.exists():
                for skill_md in SKILLS_ROOT.glob("**/SKILL.md"):
                    try:
                        content = skill_md.read_text(encoding="utf-8", errors="ignore")
                        name = skill_md.parent.name
                        desc = ""
                        if "---" in content:
                            parts = content.split("---", 2)
                            if len(parts) >= 3:
                                for line in parts[1].splitlines():
                                    if line.startswith("description:"):
                                        desc = line.replace("description:", "").strip()
                        self._cached_skills.append({
                            "name": name,
                            "path": str(skill_md),
                            "desc": desc,
                            "full_text": content[:500]
                        })
                    except Exception:
                        pass

        tokens = set(re.findall(r"[\w\u4e00-\u9fa5]+", query.lower()))
        hits = []
        for sk in self._cached_skills:
            target = f"{sk['name']} {sk['desc']} {sk['full_text']}".lower()
            score = sum(1 for tok in tokens if tok in target)
            if score > 0:
                hits.append((score, sk))
        hits.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, sk in hits[:top_k]:
            results.append({
                "source": "skill_shelf",
                "name": sk["name"],
                "path": sk["path"],
                "snippet": sk["desc"] or sk["full_text"][:200],
                "score": score
            })
        return results

    def recall_vault_memory(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """从 memory-vault SQLite 中检索历史代码与交互记录。"""
        results = []
        for db_path, stream_name in [(VAULT_CODE_DB, "vault_code"), (VAULT_INTERACT_DB, "vault_interact")]:
            if not db_path.exists():
                continue
            try:
                conn = sqlite3.connect(str(db_path), timeout=1.0)
                cur = conn.cursor()
                q_like = f"%{query[:30]}%"
                cur.execute(
                    "SELECT id, title, content FROM cards WHERE title LIKE ? OR content LIKE ? LIMIT ?",
                    (q_like, q_like, top_k)
                )
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "source": stream_name,
                        "card_id": row[0],
                        "title": row[1],
                        "snippet": str(row[2])[:300],
                        "score": 1.0
                    })
                conn.close()
            except Exception:
                pass
        return results[:top_k]

    def hybrid_recall(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """统一混合召回入口。"""
        all_hits = []
        all_hits.extend(self.recall_dialogue_history(query, top_k=2))
        all_hits.extend(self.recall_skills(query, top_k=2))
        all_hits.extend(self.recall_vault_memory(query, top_k=1))
        return all_hits[:top_k]
