# -*- coding: utf-8 -*-
r"""MemoryHub: 跨源数据混合召回中枢 (会话历史 FTS5 缓存 + memory-vault 多模式检索 + Skills 库)."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CORE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MV_DIR = Path(os.getenv("MEMORY_VAULT_DIR", str(Path.home() / ".antigravity" / "memory-vault")))
DEFAULT_SKILLS_ROOT = Path(os.getenv("SKILLS_ROOT", str(Path.home() / ".antigravity" / "skills")))
DEFAULT_JEV_ROOT = Path(os.getenv("JEV_TRAIN_ROOT", str(Path.home() / ".gemma" / "jev-train")))


def _resolve_jev_root() -> Path:
    env = os.getenv("JEV_TRAIN_ROOT")
    if env:
        return Path(env)
    if DEFAULT_JEV_ROOT.is_dir():
        return DEFAULT_JEV_ROOT
    return Path.home() / ".gemma" / "jev-train"


def _resolve_dialogue_index(jev_root: Path) -> Path:
    env = os.getenv("DIALOGUE_INDEX_PATH")
    if env:
        return Path(env)
    return jev_root / "bench" / "brain_dialogue_index.jsonl"


def _resolve_vault_db(stream: str) -> Path:
    if stream == "code":
        env = os.getenv("VAULT_CODE_DB") or os.getenv("NEXUS_MEMORY_VAULT_DB")
        if env:
            return Path(env)
        live = DEFAULT_MV_DIR / "data" / "vault_code.db"
        if live.is_file():
            return live
        return Path.home() / ".memory-vault" / "vault_code.db"
    env = os.getenv("VAULT_INTERACT_DB") or os.getenv("VAULT_INTERACTION_DB")
    if env:
        return Path(env)
    live = DEFAULT_MV_DIR / "data" / "vault_interaction.db"
    if live.is_file():
        return live
    return Path.home() / ".memory-vault" / "vault_interaction.db"


def _resolve_skills_root() -> Path:
    env = os.getenv("SKILLS_ROOT")
    if env:
        return Path(env)
    if DEFAULT_SKILLS_ROOT.is_dir():
        return DEFAULT_SKILLS_ROOT
    return Path.home() / ".antigravity" / "skills"


def _tokenize_bilingual(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    seen = set()
    for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-\.]*", text.lower()):
        for part in [w] + re.split(r"[_\-\.]+", w):
            if len(part) >= 2 and part not in seen:
                seen.add(part)
                tokens.append(part)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) >= 2:
            if len(run) <= 8 and run not in seen:
                seen.add(run)
                tokens.append(run)
            for i in range(len(run) - 1):
                bg = run[i : i + 2]
                if bg not in seen:
                    seen.add(bg)
                    tokens.append(bg)
        elif run not in seen:
            seen.add(run)
            tokens.append(run)
    return tokens


def _load_mv_engine_modules() -> Optional[Dict[str, Any]]:
    """Import memory-vault engine submodules under an isolated package alias to avoid engine.py shadowing."""
    mv_engine_dir = Path(os.getenv("MEMORY_VAULT_ENGINE_DIR", str(DEFAULT_MV_DIR / "engine")))
    if not mv_engine_dir.is_dir():
        return None
    pkg_name = "_mv_recall_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(mv_engine_dir)]
        sys.modules[pkg_name] = pkg

    needed = ("db", "tokenizer", "vectors", "graph", "searcher")
    loaded: Dict[str, Any] = {}
    for mod_name in needed:
        full_name = f"{pkg_name}.{mod_name}"
        if full_name in sys.modules:
            loaded[mod_name] = sys.modules[full_name]
            continue
        mod_path = mv_engine_dir / f"{mod_name}.py"
        if not mod_path.is_file():
            return None
        spec = importlib.util.spec_from_file_location(full_name, str(mod_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        loaded[mod_name] = mod
    return loaded


class MemoryHub:
    def __init__(self):
        self._cached_skills: Optional[List[Dict[str, Any]]] = None
        self._cached_skills_root: Optional[Path] = None
        self._db_cache: Dict[str, Any] = {}
        self._dialogue_cache_db = Path(
            os.getenv("DIALOGUE_CACHE_DB", str(CORE_ROOT / "data" / "dialogue_fts.db"))
        )
        self.last_errors: List[str] = []

    def _ensure_dialogue_fts(self, index_path: Path) -> Optional[Path]:
        """Build or refresh incremental SQLite FTS5 index with fast read-only hit and lock protection."""
        if not index_path.is_file():
            return None
        st = index_path.stat()
        sig = f"{index_path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
        cache_db = self._dialogue_cache_db

        # Fast zero-lock read-only check before opening any write transaction
        if cache_db.is_file():
            try:
                ro_conn = sqlite3.connect(f"file:{cache_db.as_posix()}?mode=ro", uri=True, timeout=1.5)
                try:
                    row = ro_conn.execute("SELECT v FROM meta WHERE k = 'source_sig';").fetchone()
                    if row and row[0] == sig:
                        return cache_db
                finally:
                    ro_conn.close()
            except sqlite3.Error:
                pass

        cache_db.parent.mkdir(parents=True, exist_ok=True)
        lock_file = cache_db.with_suffix(".lock")
        acquired_lock = False
        for attempt in range(4):
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.close(fd)
                acquired_lock = True
                break
            except FileExistsError:
                try:
                    if time.time() - lock_file.stat().st_mtime > 30.0:
                        lock_file.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                time.sleep(0.08 * (attempt + 1))
            except OSError:
                break

        try:
            conn = sqlite3.connect(str(cache_db), timeout=5.0)
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS meta (
                        k TEXT PRIMARY KEY,
                        v TEXT NOT NULL
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dialogue_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chunk_id TEXT,
                        conv_id TEXT,
                        title TEXT,
                        time_range TEXT,
                        text TEXT
                    );
                """)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS dialogue_chunks_fts USING fts5(
                        title,
                        text,
                        content='dialogue_chunks',
                        content_rowid='id',
                        tokenize='trigram'
                    );
                """)
                row = conn.execute("SELECT v FROM meta WHERE k = 'source_sig';").fetchone()
                if row and row[0] == sig:
                    return cache_db

                conn.execute("DELETE FROM dialogue_chunks;")
                conn.execute("INSERT INTO dialogue_chunks_fts(dialogue_chunks_fts) VALUES('rebuild');")
                batch: List[Tuple[str, str, str, str, str]] = []
                with open(index_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        tr = obj.get("time_range")
                        tr_json = json.dumps(tr, ensure_ascii=False) if not isinstance(tr, str) else tr
                        batch.append((
                            str(obj.get("chunk_id") or ""),
                            str(obj.get("conv_id") or ""),
                            str(obj.get("title") or ""),
                            tr_json,
                            str(obj.get("text") or ""),
                        ))
                conn.executemany(
                    "INSERT INTO dialogue_chunks (chunk_id, conv_id, title, time_range, text) VALUES (?, ?, ?, ?, ?);",
                    batch,
                )
                conn.execute("INSERT INTO dialogue_chunks_fts(dialogue_chunks_fts) VALUES('rebuild');")
                conn.execute(
                    "INSERT INTO meta (k, v) VALUES ('source_sig', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v;",
                    (sig,),
                )
                conn.commit()
                return cache_db
            finally:
                conn.close()
        finally:
            if acquired_lock:
                try:
                    lock_file.unlink(missing_ok=True)
                except OSError:
                    pass

    def recall_dialogue_history(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant historical dialogue chunks via cached FTS5 + tokenized fallback."""
        jev_root = _resolve_jev_root()
        index_path = _resolve_dialogue_index(jev_root)
        if not index_path.is_file():
            return []

        tokens = _tokenize_bilingual(query)
        if not tokens:
            return []

        try:
            cache_db = self._ensure_dialogue_fts(index_path)
            if cache_db is None:
                return []
            conn = sqlite3.connect(str(cache_db), timeout=2.0)
            conn.row_factory = sqlite3.Row
            try:
                fts_terms = [f'"{t.replace(chr(34), "")}"' for t in tokens if len(t) >= 3][:8]
                rows = []
                if fts_terms:
                    try:
                        rows = conn.execute(
                            """
                            SELECT d.chunk_id, d.conv_id, d.title, d.time_range, d.text,
                                   (-bm25(dialogue_chunks_fts)) AS score
                            FROM dialogue_chunks_fts f
                            JOIN dialogue_chunks d ON f.rowid = d.id
                            WHERE dialogue_chunks_fts MATCH ?
                            ORDER BY bm25(dialogue_chunks_fts)
                            LIMIT ?;
                            """,
                            (" OR ".join(fts_terms), top_k * 3),
                        ).fetchall()
                    except sqlite3.OperationalError as exc:
                        self.last_errors.append(f"dialogue_fts:{exc}")

                if not rows:
                    like_terms = [t for t in tokens if len(t) >= 2][:6]
                    if like_terms:
                        clauses = " OR ".join(["LOWER(text) LIKE ? OR LOWER(title) LIKE ?" for _ in like_terms])
                        params: List[Any] = []
                        for t in like_terms:
                            p = f"%{t}%"
                            params.extend([p, p])
                        params.append(top_k * 4)
                        rows = conn.execute(
                            f"SELECT chunk_id, conv_id, title, time_range, text, 1.0 AS score FROM dialogue_chunks WHERE {clauses} LIMIT ?;",
                            params,
                        ).fetchall()

                scored: List[Tuple[float, Dict[str, Any]]] = []
                for r in rows:
                    txt_low = (r["title"] + " " + r["text"]).lower()
                    overlap = sum(1 for tok in tokens if tok in txt_low)
                    if overlap == 0:
                        continue
                    score = overlap + float(r["score"] or 0.0) * 0.25
                    try:
                        tr_val = json.loads(r["time_range"]) if r["time_range"] and r["time_range"].startswith("[") else r["time_range"]
                    except Exception:
                        tr_val = r["time_range"]
                    scored.append((
                        score,
                        {
                            "source": "dialogue_transcript",
                            "chunk_id": r["chunk_id"],
                            "conv_id": r["conv_id"],
                            "title": r["title"],
                            "time_range": tr_val,
                            "snippet": str(r["text"])[:350],
                            "score": round(score, 4),
                        },
                    ))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [item for _, item in scored[:top_k]]
            finally:
                conn.close()
        except Exception as exc:
            self.last_errors.append(f"recall_dialogue_history:{type(exc).__name__}: {exc}")
            return []

    def recall_skills(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve relevant local SKILL.md entries using bilingual tokenized scoring."""
        skills_root = _resolve_skills_root()
        if self._cached_skills is None or self._cached_skills_root != skills_root:
            self._cached_skills = []
            self._cached_skills_root = skills_root
            if skills_root.exists():
                for skill_md in skills_root.glob("**/SKILL.md"):
                    try:
                        content = skill_md.read_text(encoding="utf-8", errors="ignore")
                        name = skill_md.parent.name
                        desc = ""
                        if "---" in content:
                            parts = content.split("---", 2)
                            if len(parts) >= 3:
                                for line in parts[1].splitlines():
                                    if line.startswith("description:"):
                                        desc = line.replace("description:", "").strip().strip('"')
                        self._cached_skills.append({
                            "name": name,
                            "path": str(skill_md),
                            "desc": desc,
                            "full_text": content[:600],
                        })
                    except Exception as exc:
                        self.last_errors.append(f"skill_read({skill_md.name}): {exc}")

        tokens = _tokenize_bilingual(query)
        if not tokens:
            return []

        hits: List[Tuple[float, Dict[str, Any]]] = []
        for sk in self._cached_skills:
            name_low = sk["name"].lower()
            desc_low = sk["desc"].lower()
            body_low = sk["full_text"].lower()
            score = 0.0
            for tok in tokens:
                if tok in name_low:
                    score += 2.5
                elif tok in desc_low:
                    score += 1.6
                elif tok in body_low:
                    score += 1.0
            if score > 0.0:
                hits.append((score, sk))
        hits.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "source": "skill_shelf",
                "name": sk["name"],
                "title": sk["name"],
                "path": sk["path"],
                "snippet": sk["desc"] or sk["full_text"][:200],
                "score": round(score, 3),
            }
            for score, sk in hits[:top_k]
        ]

    def recall_vault_memory(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid",
        stream_bias: str = "",
    ) -> List[Dict[str, Any]]:
        """Recall from memory-vault SQLite stores using MemorySearcher (correct `memories` schema)."""
        if not (query or "").strip():
            return []
        mv_mods = _load_mv_engine_modules()
        results: List[Dict[str, Any]] = []

        streams = ("interaction", "code") if stream_bias == "interaction" else ("code", "interaction")
        for stream in streams:
            db_path = _resolve_vault_db(stream)
            if not db_path.is_file():
                continue
            source_label = "vault_code" if stream == "code" else "vault_interact"
            try:
                if mv_mods is not None:
                    db_key = str(db_path.resolve())
                    cached_bundle = self._db_cache.get(db_key)
                    if cached_bundle is None:
                        db_mgr = mv_mods["db"].DatabaseManager(str(db_path))
                        graph = mv_mods["graph"].TemporalGraphEngine(db_mgr, max_depth=2)
                        vec_idx = mv_mods["vectors"].VectorStoreIndex()
                        cached_bundle = (db_mgr, graph, vec_idx)
                        self._db_cache[db_key] = cached_bundle
                    else:
                        db_mgr, graph, vec_idx = cached_bundle
                    searcher = mv_mods["searcher"].MemorySearcher(
                        db_mgr, graph, vector_index=vec_idx, default_mode=mode
                    )
                    hits = searcher.search(
                        query=query,
                        track="all",
                        limit=top_k,
                        detail=False,
                        include_graph=False,
                        stream_name=stream,
                        mode=mode,
                        record_access=False,
                    )
                    for h in hits:
                        results.append({
                            "source": source_label,
                            "node_id": h.get("node_id"),
                            "track": h.get("track"),
                            "title": h.get("title"),
                            "snippet": h.get("snippet", "")[:300],
                            "score": float(h.get("score") or 0.0),
                        })
                else:
                    # Pure SQLite fallback aligned to `memories` table
                    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=2.0)
                    conn.row_factory = sqlite3.Row
                    try:
                        tokens = _tokenize_bilingual(query)[:6]
                        clauses = " OR ".join(["LOWER(title) LIKE ? OR LOWER(content) LIKE ? OR LOWER(tags) LIKE ?" for _ in tokens])
                        params: List[Any] = []
                        for t in tokens:
                            p = f"%{t}%"
                            params.extend([p, p, p])
                        params.append(top_k)
                        rows = conn.execute(
                            f"SELECT node_id, track, title, content FROM memories WHERE is_active = 1 AND ({clauses}) LIMIT ?;",
                            params,
                        ).fetchall()
                        for r in rows:
                            results.append({
                                "source": source_label,
                                "node_id": r["node_id"],
                                "track": r["track"],
                                "title": r["title"],
                                "snippet": str(r["content"])[:300],
                                "score": 1.0,
                            })
                    finally:
                        conn.close()
            except Exception as exc:
                self.last_errors.append(f"recall_vault_memory({stream}): {type(exc).__name__}: {exc}")

        results.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
        return results[:top_k]

    def hybrid_recall(self, query: str, top_k: int = 4, mode: str = "hybrid", stream_bias: str = "") -> List[Dict[str, Any]]:
        """Unified RRF-fused cross-source recall (memory-vault + dialogue history + skills)."""
        self.last_errors = []
        vault_hits = self.recall_vault_memory(query, top_k=max(top_k, 3), mode=mode, stream_bias=stream_bias)
        dial_hits = self.recall_dialogue_history(query, top_k=max(top_k, 3))
        skill_hits = self.recall_skills(query, top_k=max(top_k, 2))

        rrf_k = 60.0
        pool: Dict[str, Dict[str, Any]] = {}

        for source_weight, lane in (
            (1.35, vault_hits),
            (1.0, dial_hits),
            (0.9, skill_hits),
        ):
            for rank_idx, item in enumerate(lane, start=1):
                key = str(
                    item.get("node_id")
                    or item.get("chunk_id")
                    or item.get("path")
                    or item.get("title")
                    or f"{item.get('source')}:{rank_idx}"
                )
                contrib = source_weight / (rrf_k + rank_idx)
                if key not in pool:
                    copy_item = dict(item)
                    copy_item["rrf_score"] = contrib
                    pool[key] = copy_item
                else:
                    pool[key]["rrf_score"] += contrib

        fused = sorted(pool.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        for it in fused:
            it["rrf_score"] = round(float(it["rrf_score"]), 6)
        return fused[:top_k]

    def recall_with_diagnostics(
        self,
        query: str,
        top_k: int = 4,
        mode: str = "hybrid",
        stream_bias: str = "",
    ) -> Dict[str, Any]:
        """Return both hybrid recall results and explicit per-source diagnostics."""
        results = self.hybrid_recall(query, top_k=top_k, mode=mode, stream_bias=stream_bias)
        return {
            "results": results,
            "errors": list(self.last_errors),
            "paths": {
                "vault_code_db": str(_resolve_vault_db("code")),
                "vault_interact_db": str(_resolve_vault_db("interaction")),
                "skills_root": str(_resolve_skills_root()),
                "dialogue_index": str(_resolve_dialogue_index(_resolve_jev_root())),
            },
        }
