#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Connect-RPC Quota Probe, Dual-Track Session Handoff, and Closed-Loop Outcome Ledger.

Purpose C & D substrate integration:
1. TTL-cached Connect-RPC quota inspection (delegating to antigravity-quota-monitor).
   - > 15% remaining: 0-token silent.
   - 8% < pct <= 15%: concise [QUOTA_WARN] HUD.
   - <= 8%: urgent [QUOTA_CRITICAL] HUD + automatic dual-track handoff snapshot.
2. Dual-track handoff persistence (.nexus/handoff.json + memory-vault vault_code.db)
   and new-session HUD resurfacing until acknowledged/superseded.
3. Closed-loop JEV turn outcome verification & mismatch harvesting into harvested_cases.jsonl.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUOTA_ENGINE_PATH = Path(os.getenv("QUOTA_ENGINE_PATH", str(Path.home() / ".antigravity" / "skills" / "quota-monitor" / "scripts" / "quota_engine.py")))
MEMORY_VAULT_ENGINE_DIR = Path(os.getenv("MEMORY_VAULT_ENGINE_DIR", str(Path.home() / ".antigravity" / "memory-vault" / "engine")))
DEFAULT_VAULT_DB = Path(os.getenv("MEMORY_VAULT_DB", str(Path.home() / ".antigravity" / "memory-vault" / "data" / "vault_code.db")))
DEFAULT_QUOTA_CACHE = DATA_DIR / "quota_cache.json"
DEFAULT_TURN_PRED_FILE = DATA_DIR / "turn_predictions.json"
DEFAULT_HARVESTED_CASES = DATA_DIR / "harvested_cases.jsonl"
DEFAULT_OUTCOMES_LOG = Path(os.getenv("JEV_OUTCOMES_LOG", str(Path.home() / ".gemma" / "decision_outcomes.jsonl")))

WARN_THRESHOLD_PCT = 15.0
CRITICAL_THRESHOLD_PCT = 8.0
DEFAULT_QUOTA_TTL_SEC = 90.0


def _quota_cache_path(override: Optional[Path] = None) -> Path:
    env = os.environ.get("NEXUS_QUOTA_CACHE_FILE")
    if env:
        return Path(env)
    return override or DEFAULT_QUOTA_CACHE


def _vault_db_path(override: Optional[Path] = None) -> Path:
    env = os.environ.get("NEXUS_MEMORY_VAULT_DB")
    if env:
        return Path(env)
    return override or DEFAULT_VAULT_DB


def _turn_pred_path(override: Optional[Path] = None) -> Path:
    env = os.environ.get("NEXUS_TURN_PRED_FILE")
    if env:
        return Path(env)
    return override or DEFAULT_TURN_PRED_FILE


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _live_quota_subprocess(conv_id: str, timeout_sec: float = 2.2) -> Dict[str, Any]:
    """Query Connect-RPC quota in a bounded child process so PreInvocation never stalls."""
    if os.environ.get("NEXUS_QUOTA_DISABLE_LIVE") == "1" or not QUOTA_ENGINE_PATH.is_file():
        return {"success": False, "error": "live_probe_disabled"}
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(QUOTA_ENGINE_PATH.parent)!r}); "
        "import quota_engine; "
        f"print(json.dumps(quota_engine.get_all_active_quotas(target_session_id={conv_id!r}), ensure_ascii=False))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout.strip())
            if isinstance(data, dict):
                return data
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    return {"success": False, "error": "empty_rpc_response"}


def _extract_lowest_bucket(account: Dict[str, Any]) -> tuple[float, str, str]:
    buckets = [
        ("Gemini-5h", account.get("gemini_5h_pct"), account.get("gemini_5h_reset", "")),
        ("Gemini-Weekly", account.get("gemini_weekly_pct"), account.get("gemini_weekly_reset", "")),
        ("Claude/GPT-5h", account.get("claude_gpt_5h_pct"), account.get("claude_gpt_5h_reset", "")),
        ("Claude/GPT-Weekly", account.get("claude_gpt_weekly_pct"), account.get("claude_gpt_weekly_reset", "")),
    ]
    valid: list[tuple[float, str, str]] = []
    for label, pct, reset_hint in buckets:
        if isinstance(pct, (int, float)) and 0.0 <= float(pct) <= 100.0:
            valid.append((float(pct), label, str(reset_hint or "")))
    if not valid:
        return 100.0, "unknown", ""
    valid.sort(key=lambda x: x[0])
    return valid[0]


def probe_quota_status(
    conversation_id: str = "",
    *,
    ttl_sec: float = DEFAULT_QUOTA_TTL_SEC,
    cache_path: Optional[Path] = None,
    probe_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return TTL-cached quota state plus conditional HUD guidance (>15% -> empty)."""
    cpath = _quota_cache_path(cache_path)
    now = time.time()
    cached = _read_json(cpath)
    raw: Dict[str, Any] = {}

    if cached and (now - float(cached.get("cached_at", 0.0))) < ttl_sec and probe_fn is None:
        raw = cached.get("raw") if isinstance(cached.get("raw"), dict) else {}
    else:
        raw = probe_fn(conversation_id) if probe_fn is not None else _live_quota_subprocess(conversation_id)
        _write_json(cpath, {"cached_at": now, "raw": raw})

    acct = raw.get("matched_account") if isinstance(raw, dict) else None
    if not raw.get("success") or not isinstance(acct, dict):
        return {"ok": False, "level": "OK", "min_pct": None, "bucket": "", "reset": "", "hud": ""}

    min_pct, bucket, reset = _extract_lowest_bucket(acct)
    reset_str = f", reset {reset}" if reset else ""
    email = str(acct.get("account_email") or "")
    acct_tag = f" [{email}]" if email else ""

    if min_pct > WARN_THRESHOLD_PCT:
        return {"ok": True, "level": "OK", "min_pct": min_pct, "bucket": bucket, "reset": reset, "hud": ""}
    if min_pct > CRITICAL_THRESHOLD_PCT:
        hud = f"[QUOTA_WARN]{acct_tag} Remaining model quota is low ({bucket}: {min_pct:.1f}%{reset_str}). Avoid unnecessary detours."
        return {"ok": True, "level": "WARN", "min_pct": min_pct, "bucket": bucket, "reset": reset, "hud": hud}

    hud = (
        f"[QUOTA_CRITICAL <= 8%]{acct_tag} Remaining quota critical ({bucket}: {min_pct:.1f}%{reset_str}). "
        f"Wrap up cleanly now; dual-track handoff checkpoint synced."
    )
    return {"ok": True, "level": "CRITICAL", "min_pct": min_pct, "bucket": bucket, "reset": reset, "hud": hud}


def _import_memory_vault_consolidator(db_path: Path):
    """Load memory-vault consolidator under an isolated namespace with SSOT and vector wiring."""
    if not MEMORY_VAULT_ENGINE_DIR.is_dir():
        return None
    pkg_name = "_nexus_mv_engine"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(MEMORY_VAULT_ENGINE_DIR)]
        sys.modules[pkg_name] = pkg
    db_mod = importlib.import_module(f"{pkg_name}.db")
    graph_mod = importlib.import_module(f"{pkg_name}.graph")
    san_mod = importlib.import_module(f"{pkg_name}.sanitizer")
    cons_mod = importlib.import_module(f"{pkg_name}.consolidator")
    db = db_mod.DatabaseManager(str(db_path))
    graph = graph_mod.TemporalGraphEngine(db)
    projector = None
    ssot_sync = None
    vec_index = None
    try:
        proj_mod = importlib.import_module(f"{pkg_name}.projector")
        ssot_mod = importlib.import_module(f"{pkg_name}.ssot_sync")
        vec_mod = importlib.import_module(f"{pkg_name}.vectors")
        vfs_dir = db_path.resolve().parent / "vault_fs"
        is_live = db_path.resolve() == DEFAULT_VAULT_DB.resolve()
        projector = proj_mod.VaultProjector(
            base_dir=vfs_dir,
            git_tracked=is_live,
            commit_mode="batch" if is_live else "manual",
        )
        ssot_sync = ssot_mod.VaultSSOTSyncManager(projector)
        vec_index = vec_mod.VectorStoreIndex(local_dim=256)
    except Exception:
        pass
    return cons_mod.MemoryConsolidator(
        db=db,
        graph=graph,
        sanitizer=san_mod.MemorySanitizer(),
        stream_name="code",
        projector=projector,
        ssot_sync=ssot_sync,
        vector_index=vec_index,
    )


def _workspace_key(workspace_path: Optional[str | Path], conversation_id: str) -> str:
    if workspace_path and str(workspace_path).strip():
        norm = str(Path(str(workspace_path)).resolve()).lower().replace("\\", "/")
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:10]
        stem = Path(norm).name or "root"
        return f"{stem}_{digest}"
    safe_conv = re.sub(r"[^a-zA-Z0-9_\-]", "_", conversation_id or "default")[:24]
    return f"conv_{safe_conv}"


def _handoff_json_path(workspace_path: Optional[str | Path], key: str) -> Path:
    if workspace_path and Path(str(workspace_path)).is_dir():
        return Path(str(workspace_path)) / ".nexus" / "handoff.json"
    return DATA_DIR / "handoffs" / f"{key}.json"


def _upsert_sqlite_fallback(db_path: Path, node_id: str, title: str, content: str, tags: List[str], meta: Dict[str, Any]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT UNIQUE NOT NULL, track TEXT NOT NULL, "
            "title TEXT NOT NULL, content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', "
            "metadata TEXT NOT NULL DEFAULT '{}', version INTEGER NOT NULL DEFAULT 1, "
            "is_active INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL, updated_at REAL NOT NULL);"
        )
        now = time.time()
        tags_json = json.dumps(tags, ensure_ascii=False)
        meta_json = json.dumps(meta, ensure_ascii=False)
        cur.execute("SELECT id, version FROM memories WHERE node_id = ?;", (node_id,))
        row = cur.fetchone()
        if row:
            cur.execute(
                "UPDATE memories SET track=?, title=?, content=?, tags=?, metadata=?, version=?, is_active=1, updated_at=? WHERE id=?;",
                ("trajectory", title, content, tags_json, meta_json, int(row[1]) + 1, now, row[0]),
            )
        else:
            cur.execute(
                "INSERT INTO memories (node_id, track, title, content, tags, metadata, version, is_active, created_at, updated_at) "
                "VALUES (?, 'trajectory', ?, ?, ?, ?, 1, 1, ?, ?);",
                (node_id, title, content, tags_json, meta_json, now, now),
            )
        conn.commit()
    finally:
        conn.close()


def save_dual_track_handoff(
    *,
    conversation_id: str,
    workspace_path: Optional[str | Path] = None,
    summary: str,
    next_steps: Optional[List[str]] = None,
    reason: str = "quota_critical",
    session_mode: str = "",
    db_path: Optional[Path] = None,
    min_interval_sec: float = 0.0,
) -> Dict[str, Any]:
    """Persist session handoff simultaneously to workspace .nexus/handoff.json and memory-vault."""
    key = _workspace_key(workspace_path, conversation_id)
    node_id = f"handoff:{key}"
    now = time.time()
    json_path = _handoff_json_path(workspace_path, key)
    if min_interval_sec > 0.0:
        prev_rec = _read_json(json_path)
        if (
            prev_rec
            and prev_rec.get("conversation_id") == conversation_id
            and (now - float(prev_rec.get("created_at") or 0.0)) < min_interval_sec
        ):
            return {"json_path": str(json_path), "vault": {"success": True, "action": "DEBOUNCED"}, "handoff_id": node_id}

    steps = [str(s).strip() for s in (next_steps or []) if str(s).strip()]
    ws_str = str(Path(str(workspace_path)).resolve()) if workspace_path and Path(str(workspace_path)).is_dir() else ""
    eff_mode = session_mode or ("workspace_bound" if ws_str else "standalone_chat_research")
    clean_summary = str(summary or "Session handoff checkpoint").strip()
    if len(clean_summary) < 16:
        clean_summary = f"{clean_summary} (conversation={conversation_id}, reason={reason})"

    record = {
        "handoff_id": node_id,
        "conversation_id": conversation_id,
        "workspace": ws_str,
        "session_mode": eff_mode,
        "summary": clean_summary,
        "next_steps": steps,
        "reason": reason,
        "status": "PENDING",
        "created_at": now,
        "surfaced_sessions": [],
    }
    _write_json(json_path, record)

    title = f"Handoff {Path(ws_str).name if ws_str else conversation_id[:16]}"
    md_body = f"Reason: {reason}\nSessionMode: {eff_mode}\nConversation: {conversation_id}\nWorkspace: {ws_str or 'global'}\n\nSummary:\n{clean_summary}"
    if steps:
        md_body += "\n\nNext Steps:\n" + "\n".join(f"- {s}" for s in steps)
    tags = ["handoff", "continuity", f"session:{eff_mode}"]
    meta = {
        "status": "PENDING",
        "conversation_id": conversation_id,
        "workspace": ws_str,
        "session_mode": eff_mode,
        "reason": reason,
        "created_at": now,
        "next_steps": steps,
    }

    target_db = _vault_db_path(db_path)
    vault_res: Dict[str, Any] = {"success": False}
    try:
        cons = _import_memory_vault_consolidator(target_db)
        if cons is not None:
            vault_res = cons.consolidate_and_store(
                track="trajectory",
                node_id=node_id,
                title=title,
                content=md_body,
                tags=tags,
                metadata=meta,
                detect_conflicts=False,
            )
    except Exception:
        vault_res = {"success": False}

    if not vault_res.get("success"):
        _upsert_sqlite_fallback(target_db, node_id, title, md_body, tags, meta)
        vault_res = {"success": True, "action": "SQLITE_FALLBACK", "node_id": node_id}

    return {"json_path": str(json_path), "vault": vault_res, "handoff_id": node_id}


def load_pending_handoffs(
    workspace_paths: Optional[List[str]] = None,
    current_conv_id: str = "",
    *,
    db_path: Optional[Path] = None,
    ttl_seconds: int = 7 * 86400,
    mark_seen_for_session: bool = True,
) -> List[Dict[str, Any]]:
    """Collect unresolved cross-session handoff items from .nexus/handoff.json and memory-vault."""
    now = time.time()
    found: Dict[str, Dict[str, Any]] = {}
    already_seen_ids: set[str] = set()

    # 1. Local workspace JSON track
    for wp in (workspace_paths or []):
        hpath = Path(str(wp)) / ".nexus" / "handoff.json"
        rec = _read_json(hpath)
        if not rec or rec.get("status") != "PENDING":
            continue
        if current_conv_id and rec.get("conversation_id") == current_conv_id:
            continue
        if (now - float(rec.get("created_at") or 0.0)) > ttl_seconds:
            continue
        hid = str(rec.get("handoff_id") or hpath)
        seen = rec.get("surfaced_sessions") if isinstance(rec.get("surfaced_sessions"), list) else []
        if current_conv_id and mark_seen_for_session and current_conv_id in seen:
            already_seen_ids.add(hid)
            continue
        found[hid] = {
            "handoff_id": hid,
            "summary": str(rec.get("summary") or ""),
            "next_steps": rec.get("next_steps") or [],
            "reason": str(rec.get("reason") or ""),
            "source_conv": str(rec.get("conversation_id") or ""),
            "source": "workspace_json",
        }
        if current_conv_id and mark_seen_for_session:
            seen.append(current_conv_id)
            rec["surfaced_sessions"] = seen[-10:]
            _write_json(hpath, rec)

    # 2. Memory-vault SQLite track (synced with workspace JSON surfaced_sessions)
    target_db = _vault_db_path(db_path)
    if target_db.is_file():
        try:
            conn = sqlite3.connect(str(target_db), timeout=2.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT node_id, title, content, metadata, updated_at FROM memories "
                "WHERE is_active = 1 AND tags LIKE '%\"handoff\"%' ORDER BY updated_at DESC LIMIT 8;"
            ).fetchall()
            for r in rows:
                if (now - float(r["updated_at"] or 0.0)) > ttl_seconds:
                    continue
                meta = json.loads(r["metadata"] or "{}") if isinstance(r["metadata"], str) else {}
                if meta.get("status") != "PENDING":
                    continue
                src_conv = str(meta.get("conversation_id") or "")
                if current_conv_id and src_conv == current_conv_id:
                    continue
                nid = str(r["node_id"])
                seen_vault = meta.get("surfaced_sessions") if isinstance(meta.get("surfaced_sessions"), list) else []

                if nid in found:
                    if current_conv_id and mark_seen_for_session and current_conv_id not in seen_vault:
                        seen_vault.append(current_conv_id)
                        meta["surfaced_sessions"] = seen_vault[-10:]
                        conn.execute("UPDATE memories SET metadata = ? WHERE node_id = ?;", (json.dumps(meta, ensure_ascii=False), nid))
                        conn.commit()
                    continue

                if nid in already_seen_ids:
                    continue

                if current_conv_id and mark_seen_for_session and current_conv_id in seen_vault:
                    continue

                found[nid] = {
                    "handoff_id": nid,
                    "summary": str(r["title"] or nid),
                    "next_steps": meta.get("next_steps") or [],
                    "reason": str(meta.get("reason") or ""),
                    "source_conv": src_conv,
                    "source": "memory_vault",
                }
                if current_conv_id and mark_seen_for_session:
                    seen_vault.append(current_conv_id)
                    meta["surfaced_sessions"] = seen_vault[-10:]
                    conn.execute("UPDATE memories SET metadata = ? WHERE node_id = ?;", (json.dumps(meta, ensure_ascii=False), nid))
                    conn.commit()
            conn.close()
        except Exception:
            pass

    return list(found.values())


def format_handoff_hud(items: List[Dict[str, Any]]) -> str:
    """Format pending handoff items into a compact single-block PreInvocation HUD notice."""
    if not items:
        return ""
    lines = ["[PENDING_HANDOFF] Unresolved cross-session checkpoint detected:"]
    for item in items[:2]:
        steps = item.get("next_steps") or []
        step_str = f" | next: {'; '.join(str(s)[:60] for s in steps[:2])}" if steps else ""
        lines.append(f"- ({item.get('source')}) {str(item.get('summary') or '')[:140]}{step_str}")
    return "\n".join(lines)


def _inspect_turn_physical_behavior(transcript_path: Optional[str], target_user_step: int) -> Dict[str, Any]:
    """Inspect whether a prior user step actually performed mutations or only read-only actions."""
    if not transcript_path or not Path(transcript_path).is_file():
        return {"found": False}
    mutating_tools = {"write_file", "edit_file", "write_to_file", "replace_file_content", "multi_replace_file_content", "nexus_patch_chunk"}
    steps: List[Dict[str, Any]] = []
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    try:
                        steps.append(json.loads(line))
                    except Exception:
                        continue
    except Exception:
        return {"found": False}

    start_idx = None
    for i, st in enumerate(steps):
        if (st.get("type") == "USER_INPUT" or st.get("source") == "USER_EXPLICIT") and st.get("step_index") == target_user_step:
            start_idx = i
            break
    if start_idx is None:
        return {"found": False}

    end_idx = len(steps)
    for j in range(start_idx + 1, len(steps)):
        if steps[j].get("type") == "USER_INPUT" or steps[j].get("source") == "USER_EXPLICIT":
            end_idx = j
            break

    mutated = False
    ran_cmd = False
    used_ask_question = False
    tools_used: List[str] = []
    for st in steps[start_idx + 1:end_idx]:
        for tc in st.get("tool_calls") or []:
            tname = str(tc.get("name") or "").lower()
            tools_used.append(tname)
            if tname in mutating_tools:
                mutated = True
            elif tname == "run_command":
                ran_cmd = True
            elif tname == "ask_question":
                used_ask_question = True
    return {"found": True, "mutated": mutated, "ran_cmd": ran_cmd, "used_ask_question": used_ask_question, "tools_used": tools_used}


def record_and_backfill_turn_outcome(
    *,
    conversation_id: str,
    user_step_index: int,
    query: str,
    predicted_intent: str,
    confidence: float,
    transcript_path: Optional[str] = None,
    pred_store_path: Optional[Path] = None,
    harvest_path: Optional[Path] = None,
    outcomes_log_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Backfill previous turn's JEV outcome against transcript reality and store the new prediction."""
    pfile = _turn_pred_path(pred_store_path)
    store = _read_json(pfile)
    prev = store.get(conversation_id)
    backfilled: Optional[Dict[str, Any]] = None

    if isinstance(prev, dict) and prev.get("user_step_index") != user_step_index and not prev.get("backfilled"):
        obs = _inspect_turn_physical_behavior(transcript_path, int(prev.get("user_step_index", -1)))
        if obs.get("found"):
            pred_label = str(prev.get("intent") or "")
            actual_label = pred_label
            if pred_label == "CODE_MUTATE" and not obs["mutated"] and not obs["ran_cmd"]:
                actual_label = "DIRECTIONAL_PLANNING" if obs["used_ask_question"] else "READONLY_QUERY"
            mismatch = actual_label != pred_label
            backfilled = {
                "ts": time.time(),
                "conv_id": conversation_id,
                "user_step_index": prev.get("user_step_index"),
                "predicted_intent": pred_label,
                "observed_intent": actual_label,
                "mismatch": mismatch,
                "tools_used": obs["tools_used"][:20],
            }
            out_log = outcomes_log_path or DEFAULT_OUTCOMES_LOG
            try:
                out_log.parent.mkdir(parents=True, exist_ok=True)
                with open(out_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(backfilled, ensure_ascii=False) + "\n")
            except Exception:
                pass
            if mismatch:
                hfile = harvest_path or DEFAULT_HARVESTED_CASES
                try:
                    hfile.parent.mkdir(parents=True, exist_ok=True)
                    sample = {
                        "ts": time.time(),
                        "source": "closed_loop_outcome_backfill",
                        "prompt": str(prev.get("query") or "")[:400],
                        "predicted_intent": pred_label,
                        "label": actual_label,
                        "confidence": float(prev.get("confidence") or 0.0),
                    }
                    with open(hfile, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                except Exception:
                    pass

    store[conversation_id] = {
        "user_step_index": user_step_index,
        "query": str(query)[:400],
        "intent": predicted_intent,
        "confidence": round(float(confidence), 4),
        "ts": time.time(),
        "backfilled": False,
    }
    _write_json(pfile, store)
    return backfilled


def list_all_handoffs(
    workspace_paths: Optional[List[str]] = None,
    *,
    status_filter: str = "PENDING",
    db_path: Optional[Path] = None,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """Read-only listing of dual-track handoffs without marking any session seen."""
    results: Dict[str, Dict[str, Any]] = {}
    want_upper = (status_filter or "PENDING").upper()

    for wp in (workspace_paths or []):
        hpath = Path(str(wp)) / ".nexus" / "handoff.json"
        rec = _read_json(hpath)
        if not rec:
            continue
        st = str(rec.get("status") or "PENDING").upper()
        if want_upper != "ALL" and st != want_upper:
            continue
        hid = str(rec.get("handoff_id") or hpath)
        results[hid] = {
            "handoff_id": hid,
            "status": st,
            "summary": rec.get("summary", ""),
            "next_steps": rec.get("next_steps") or [],
            "reason": rec.get("reason", ""),
            "conversation_id": rec.get("conversation_id", ""),
            "workspace": rec.get("workspace", str(wp)),
            "created_at": rec.get("created_at"),
            "source": "workspace_json",
        }

    target_db = _vault_db_path(db_path)
    if target_db.is_file():
        try:
            conn = sqlite3.connect(str(target_db), timeout=2.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT node_id, title, metadata, updated_at FROM memories "
                "WHERE is_active = 1 AND tags LIKE '%\"handoff\"%' ORDER BY updated_at DESC LIMIT ?;",
                (max(1, int(limit)),),
            ).fetchall()
            for r in rows:
                meta = json.loads(r["metadata"] or "{}") if isinstance(r["metadata"], str) else {}
                st = str(meta.get("status") or "PENDING").upper()
                if want_upper != "ALL" and st != want_upper:
                    continue
                nid = str(r["node_id"])
                if nid not in results:
                    results[nid] = {
                        "handoff_id": nid,
                        "status": st,
                        "summary": str(r["title"] or nid),
                        "next_steps": meta.get("next_steps") or [],
                        "reason": str(meta.get("reason") or ""),
                        "conversation_id": str(meta.get("conversation_id") or ""),
                        "workspace": str(meta.get("workspace") or ""),
                        "updated_at": r["updated_at"],
                        "source": "memory_vault",
                    }
            conn.close()
        except Exception:
            pass
    return list(results.values())


def resolve_dual_track_handoff(
    *,
    handoff_id: str = "",
    workspace_path: Optional[str | Path] = None,
    resolution_note: str = "",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Explicitly mark a pending handoff as RESOLVED across both workspace JSON and memory-vault."""
    now = time.time()
    updated_json = False
    updated_vault = False
    resolved_id = handoff_id

    if workspace_path and Path(str(workspace_path)).is_dir():
        hpath = Path(str(workspace_path)) / ".nexus" / "handoff.json"
        rec = _read_json(hpath)
        if rec and (not resolved_id or rec.get("handoff_id") == resolved_id):
            resolved_id = resolved_id or str(rec.get("handoff_id") or "")
            rec["status"] = "RESOLVED"
            rec["resolved_at"] = now
            if resolution_note:
                rec["resolution_note"] = resolution_note[:300]
            _write_json(hpath, rec)
            updated_json = True

    target_db = _vault_db_path(db_path)
    if target_db.is_file() and resolved_id:
        try:
            conn = sqlite3.connect(str(target_db), timeout=3.0)
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT metadata FROM memories WHERE node_id = ?;", (resolved_id,)).fetchone()
            if row:
                meta = json.loads(row["metadata"] or "{}") if isinstance(row["metadata"], str) else {}
                meta["status"] = "RESOLVED"
                meta["resolved_at"] = now
                if resolution_note:
                    meta["resolution_note"] = resolution_note[:300]
                conn.execute(
                    "UPDATE memories SET metadata = ?, updated_at = ? WHERE node_id = ?;",
                    (json.dumps(meta, ensure_ascii=False), now, resolved_id),
                )
                conn.commit()
                updated_vault = True
            conn.close()
        except Exception:
            pass

    return {
        "success": updated_json or updated_vault,
        "handoff_id": resolved_id,
        "updated_json": updated_json,
        "updated_vault": updated_vault,
        "status": "RESOLVED" if (updated_json or updated_vault) else "NOT_FOUND",
    }


def inspect_gate_status(
    *,
    conversation_id: str = "default",
    workspace_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Read-only status inspection of directional lock, quota, Stop hook nudges, and JEV feedback."""
    from deciders.directional_gate import read_directional_lock

    lock_rec = read_directional_lock(conversation_id)
    quota = probe_quota_status(conversation_id=conversation_id)

    stop_file = Path(os.environ.get("NEXUS_STOP_NUDGE_FILE") or (Path.home() / ".antigravity" / "reasoning-compiler" / "stop_nudges.json"))
    stop_data = _read_json(stop_file)
    conv_nudges = [k for k in stop_data.keys() if str(k).startswith(f"{conversation_id}::")]

    preds = _read_json(_turn_pred_path())
    last_pred = preds.get(conversation_id)

    harvest_count = 0
    if DEFAULT_HARVESTED_CASES.is_file():
        try:
            harvest_count = sum(1 for ln in DEFAULT_HARVESTED_CASES.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())
        except Exception:
            pass

    pending = list_all_handoffs(
        workspace_paths=[str(workspace_path)] if workspace_path else None,
        status_filter="PENDING",
    )

    return {
        "conversation_id": conversation_id,
        "directional_lock": lock_rec or {"locked": False},
        "quota": quota,
        "stop_hook_nudged_turns": conv_nudges,
        "last_turn_prediction": last_pred,
        "harvested_mismatch_count": harvest_count,
        "pending_handoffs_count": len(pending),
        "pending_handoffs": pending[:5],
    }


def assemble_full_hud(
    *,
    prompt: str = "",
    workspace_path: Optional[str | Path] = None,
    conversation_id: str = "default",
    include_hardware: bool = True,
    pipeline: Any = None,
) -> Dict[str, Any]:
    """On-demand non-mutating full Attention + Blast Radius + Quota + Handoff HUD synthesis."""
    sections: List[str] = []

    if prompt and pipeline is not None:
        try:
            res = pipeline.assemble(instruction=prompt, include_recall=True, include_hardware=include_hardware)
            if isinstance(res, dict) and res.get("hud_markdown"):
                sections.append(str(res["hud_markdown"]))
        except Exception:
            pass

    if workspace_path:
        try:
            nexus_src = Path(os.getenv("NEXUS_STATE_SRC", str(Path.home() / ".antigravity" / "nexus-state" / "src")))
            if nexus_src.is_dir() and str(nexus_src) not in sys.path:
                sys.path.insert(0, str(nexus_src))
            from nexus_state.core.blast_radius import render_hud
            from nexus_state.core.safety_barrier import get_dead_branch_warning, get_hook_health_warning, get_self_lock_warnings

            br_hud = render_hud(workspace_path, conversation_id=conversation_id)
            if br_hud:
                sections.append(br_hud)
            data_dir = Path(os.getenv("NEXUS_STATE_DATA", str(Path.home() / ".antigravity" / "nexus-state" / "data")))
            stats_path = data_dir / "telemetry_stats.json"
            for warn in (
                get_dead_branch_warning(workspace_path, data_dir),
                get_hook_health_warning(stats_path),
                get_self_lock_warnings(stats_path),
            ):
                if warn:
                    sections.append(warn)
        except Exception:
            pass

    qres = probe_quota_status(conversation_id=conversation_id)
    if qres.get("hud"):
        sections.append(qres["hud"])

    pending = load_pending_handoffs(
        workspace_paths=[str(workspace_path)] if workspace_path else None,
        current_conv_id=conversation_id,
        mark_seen_for_session=False,
    )
    h_hud = format_handoff_hud(pending)
    if h_hud:
        sections.append(h_hud)

    return {
        "success": True,
        "section_count": len(sections),
        "hud_markdown": "\n\n".join(sections),
        "quota": qres,
        "pending_handoffs_count": len(pending),
    }
