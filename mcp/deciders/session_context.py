# -*- coding: utf-8 -*-
"""Session profile classification and dual-stream bias for gemma-prep-core."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple


SESSION_MODES = (
    "workspace_bound",
    "standalone_targeted_repo",
    "standalone_scratch",
    "standalone_chat_research",
)

MUTATING_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "write_to_file",
    "replace_file_content",
    "multi_replace_file_content",
    "apply_patch",
    "delete_file",
    "patch_file_chunk",
})

COMMAND_TOOLS = frozenset({"run_command", "bash", "execute_command", "run_shell"})

_REPO_MARKERS = (
    ".git",
    ".nexus",
    "pyproject.toml",
    "package.json",
    "cargo.toml",
    "Cargo.toml",
    "go.mod",
    "server.py",
)

_SET_LOC_RE = re.compile(
    r"""(?:^|[;&|]\s*)(?:Set-Location|sl|cd|pushd|chdir)\s+(?:-Path\s+)?(?:"([^"]+)"|'([^']+)'|([^\s;|&><]+))""",
    re.IGNORECASE,
)
_GIT_C_RE = re.compile(
    r"""\bgit\s+-C\s+(?:"([^"]+)"|'([^']+)'|([^\s;|&><]+))""",
    re.IGNORECASE,
)
_WIN_PATH_RE = re.compile(r"""[A-Za-z]:[/\\][^\s'"`;|&><()]+""")

_CODE_EDIT_SIGNAL_RE = re.compile(
    r"(?:\.(?:py|ts|tsx|js|jsx|go|rs|json|yaml|yml|toml|sql)\b"
    r"|修改|修复|重构|编写|实现|调试|排查|加个|补齐|升级|拆解"
    r"|\b(?:fix|refactor|implement|edit|write|patch|debug|pytest|unittest)\b)",
    re.IGNORECASE,
)


def _norm_posix(path: str | Path) -> str:
    raw = str(path).strip().replace("\\", "/")
    if len(raw) > 3 and raw.endswith("/"):
        raw = raw.rstrip("/")
    return raw


def _norm_low(path: str | Path) -> str:
    return _norm_posix(path).lower()


def _same_or_under(path_low: str, root_low: str) -> bool:
    if not path_low or not root_low:
        return False
    return path_low == root_low or path_low.startswith(root_low.rstrip("/") + "/")


def _has_repo_marker(path_obj: Path) -> bool:
    return any((path_obj / marker).exists() for marker in _REPO_MARKERS)


def is_explicit_scratch_path(path: str | Path | None) -> bool:
    """Return True strictly for Antigravity scratch and byproduct directories."""
    if not path:
        return False
    low = _norm_low(path)
    if not low:
        return False
    if low.endswith("/.nexus") or "/.nexus/" in low:
        return True
    if ("/.gemini/antigravity/brain/" in low or "/.gemini/antigravity-ide/brain/" in low) and (
        "/scratch/" in low or low.endswith("/scratch")
    ):
        return True
    if "/.gemini/antigravity/scratch/" in low or low.endswith("/.gemini/antigravity/scratch"):
        return True
    if "/.gemini/scratch/" in low or low.endswith("/.gemini/scratch"):
        return True
    if "/scratch/" in low or low.endswith("/scratch"):
        return True
    return False


def is_scratch_or_ephemeral_path(path: str | Path | None) -> bool:
    """Return True for scratch directories, `.nexus` files, or un-repo'd OS temp files."""
    if not path:
        return False
    raw = str(path).strip()
    if not raw:
        return False
    if is_explicit_scratch_path(raw):
        return True
    low = _norm_low(raw)
    try:
        temp_low = _norm_low(Path(tempfile.gettempdir()).resolve())
    except Exception:
        temp_low = "/tmp"
    is_temp = (
        _same_or_under(low, temp_low)
        or "/appdata/local/temp/" in low
        or low.startswith("/tmp/")
        or low == "/tmp"
    )
    if is_temp:
        try:
            p = Path(raw)
            cur = p if p.is_dir() else p.parent
            for _ in range(5):
                if _norm_low(cur) == temp_low or cur.parent == cur:
                    break
                if _has_repo_marker(cur):
                    return False
                cur = cur.parent
        except Exception:
            pass
        return True
    return False


def is_excluded_fallback_root(
    path: str | Path | None,
    explicit_workspaces: Tuple[str, ...] | List[str] = (),
) -> bool:
    """Return True if path must never act as an implicit default workspace."""
    if not path:
        return True
    try:
        resolved = Path(str(path)).resolve()
    except Exception:
        resolved = Path(str(path))
    low = _norm_low(resolved)
    if not low or resolved == resolved.parent:
        return True
    explicit_lows = tuple(_norm_low(Path(w).resolve()) if w else "" for w in explicit_workspaces)
    if any(_same_or_under(low, ew) and _same_or_under(ew, low) for ew in explicit_lows if ew):
        return False
    if low == _norm_low(Path.home().resolve()):
        return True
    if _same_or_under(low, _norm_low((Path.home() / ".gemini").resolve())):
        return True
    if "/appdata/local/programs/antigravity" in low or "/appdata/local/programs/python" in low:
        return True
    return is_scratch_or_ephemeral_path(resolved)


def extract_command_target_paths(
    command_line: str | None,
    args: Dict[str, Any] | None = None,
) -> List[str]:
    """Extract ordered path candidates from shell commands and tool arguments."""
    out: List[str] = []
    seen: set[str] = set()

    def _add(val: Optional[str]) -> None:
        if not isinstance(val, str):
            return
        cleaned = val.strip().strip('"').strip("'").rstrip(".,;:")
        if not cleaned or cleaned.startswith(("http://", "https://", "-", "--")):
            return
        key = _norm_low(cleaned)
        if key and key not in seen:
            seen.add(key)
            out.append(cleaned)

    cmd = str(command_line or "")
    for pat in (_SET_LOC_RE, _GIT_C_RE):
        for m in pat.finditer(cmd):
            _add(m.group(1) or m.group(2) or m.group(3))

    if isinstance(args, dict):
        for key in (
            "Cwd", "cwd", "WorkingDirectory", "AbsolutePath", "DirectoryPath",
            "TargetFile", "target_file", "Path", "path",
        ):
            v = args.get(key)
            if isinstance(v, str):
                _add(v)

    for m in _WIN_PATH_RE.finditer(cmd):
        _add(m.group(0))
    return out


def resolve_repo_root(
    path_or_str: str | Path | None,
    explicit_workspaces: Tuple[str, ...] | List[str] = (),
) -> Optional[Path]:
    """Resolve enclosing real non-excluded repository root for a path."""
    if not path_or_str:
        return None
    raw = str(path_or_str).strip()
    if not raw or is_scratch_or_ephemeral_path(raw):
        return None
    low = _norm_low(raw)
    if "/.gemini/antigravity/brain/" in low or "/.gemini/antigravity-ide/brain/" in low:
        return None
    try:
        p = Path(raw).resolve()
    except Exception:
        return None
    cur = p if p.is_dir() else p.parent
    home = Path.home().resolve()
    for _ in range(8):
        if is_excluded_fallback_root(cur, explicit_workspaces):
            return None
        if _has_repo_marker(cur):
            return cur
        if cur.parent == cur or cur == home:
            break
        cur = cur.parent
    return None


def _inspect_transcript_tail(
    transcript_path: str | None,
    explicit_workspaces: Tuple[str, ...],
) -> Tuple[List[str], List[str]]:
    if not transcript_path:
        return [], []
    p = Path(transcript_path)
    if not p.is_file():
        return [], []
    try:
        size = p.stat().st_size
        with p.open("rb") as fh:
            if size > 524288:
                fh.seek(-524288, os.SEEK_END)
                fh.readline()
            lines = fh.read().decode("utf-8", errors="replace").splitlines()[-400:]
    except Exception:
        return [], []

    targeted: List[str] = []
    scratches: List[str] = []
    t_seen: set[str] = set()
    s_seen: set[str] = set()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except Exception:
            continue
        for tc in (rec.get("tool_calls") or []) if isinstance(rec, dict) else []:
            if not isinstance(tc, dict):
                continue
            tname = str(tc.get("name") or tc.get("tool_name") or "").strip().lower()
            args = tc.get("arguments") or tc.get("args") or tc.get("tool_input") or {}
            if not isinstance(args, dict):
                continue
            cands: List[str] = []
            if tname in COMMAND_TOOLS:
                cmd = str(args.get("CommandLine") or args.get("command") or args.get("cmd") or "")
                cands = extract_command_target_paths(cmd, args)
            elif tname in MUTATING_TOOLS:
                cands = [str(args.get(k)) for k in ("AbsolutePath", "TargetFile", "target_file", "Path", "path") if isinstance(args.get(k), str)]
            for cand in cands:
                if is_scratch_or_ephemeral_path(cand):
                    k = _norm_low(cand)
                    if k not in s_seen:
                        s_seen.add(k)
                        scratches.append(_norm_posix(cand))
                elif tname in MUTATING_TOOLS or tname in COMMAND_TOOLS:
                    repo = resolve_repo_root(cand, explicit_workspaces)
                    if repo is not None:
                        np = _norm_posix(repo)
                        k = np.lower()
                        if k not in t_seen:
                            t_seen.add(k)
                            targeted.append(np)
    return targeted, scratches


def has_explicit_code_mutation_signal(prompt: str) -> bool:
    """Return True when prompt explicitly names source files or code edit/test verbs."""
    return bool(_CODE_EDIT_SIGNAL_RE.search(prompt or ""))


def stream_preference_for_session(mode: str, intent: str = "") -> str:
    """Return preferred memory-vault stream (`code` vs `interaction`) for a session profile."""
    if mode in ("workspace_bound", "standalone_targeted_repo"):
        return "code"
    if intent in ("CODE_MUTATE", "STATE_ROLLBACK"):
        return "code"
    return "interaction"


def classify_session_profile(
    workspace_paths: List[str] | Tuple[str, ...] | None = None,
    *,
    transcript_path: str | None = None,
    tool_name: str | None = None,
    target_path: str | None = None,
    command_line: str | None = None,
    args: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Classify the current session into one of the 4 Antigravity lifecycle profiles."""
    explicit_clean: List[str] = []
    for raw in workspace_paths or []:
        if isinstance(raw, str) and raw.strip():
            try:
                explicit_clean.append(_norm_posix(Path(raw.strip()).resolve()))
            except Exception:
                explicit_clean.append(_norm_posix(raw.strip()))
    explicit_tuple = tuple(dict.fromkeys(explicit_clean))

    if explicit_tuple:
        return {
            "mode": "workspace_bound",
            "primary_workspace": explicit_tuple[0],
            "explicit_workspaces": list(explicit_tuple),
            "targeted_repos": list(explicit_tuple),
            "scratch_paths": [],
            "stream_bias": "code",
            "reason": "explicit_workspace_paths",
        }

    targeted: List[str] = []
    scratches: List[str] = []
    t_seen: set[str] = set()
    s_seen: set[str] = set()

    tname_low = str(tool_name or "").strip().lower()
    active = tname_low in MUTATING_TOOLS or tname_low in COMMAND_TOOLS
    cands: List[str] = []
    if target_path and str(target_path).strip():
        cands.append(str(target_path).strip())
    if command_line or isinstance(args, dict):
        cands.extend(extract_command_target_paths(command_line, args))
    for cand in cands:
        if is_scratch_or_ephemeral_path(cand):
            np = _norm_posix(cand)
            if np.lower() not in s_seen:
                s_seen.add(np.lower())
                scratches.append(np)
        elif active:
            repo = resolve_repo_root(cand, explicit_tuple)
            if repo is not None:
                np = _norm_posix(repo)
                if np.lower() not in t_seen:
                    t_seen.add(np.lower())
                    targeted.append(np)

    tx_targeted, tx_scratches = _inspect_transcript_tail(transcript_path, explicit_tuple)
    for r in tx_targeted:
        if r.lower() not in t_seen:
            t_seen.add(r.lower())
            targeted.append(r)
    for sp in tx_scratches:
        if sp.lower() not in s_seen:
            s_seen.add(sp.lower())
            scratches.append(sp)

    if targeted:
        return {
            "mode": "standalone_targeted_repo",
            "primary_workspace": targeted[0],
            "explicit_workspaces": [],
            "targeted_repos": targeted,
            "scratch_paths": scratches,
            "stream_bias": "code",
            "reason": "targeted_repo_activity" if tx_targeted else "command_or_mutation_target",
        }
    if scratches:
        return {
            "mode": "standalone_scratch",
            "primary_workspace": None,
            "explicit_workspaces": [],
            "targeted_repos": [],
            "scratch_paths": scratches,
            "stream_bias": "interaction",
            "reason": "scratch_path_activity",
        }
    return {
        "mode": "standalone_chat_research",
        "primary_workspace": None,
        "explicit_workspaces": [],
        "targeted_repos": [],
        "scratch_paths": [],
        "stream_bias": "interaction",
        "reason": "no_workspace_no_repo_mutation",
    }
