# -*- coding: utf-8 -*-

"""JevDeciderBase: Jev 三原语底层通信、守护进程拉起与硬件探针."""

from __future__ import annotations

import os

import sys

import json

import time

import re

import urllib.request

import urllib.error

from pathlib import Path

from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(os.getenv("JEV_TRAIN_ROOT", str(Path.home() / ".gemma" / "jev-train")))

if not PROJECT_ROOT.is_dir():
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "train" / "serve_gemma.py").is_file() or (repo_root / "models").is_dir():
        PROJECT_ROOT = repo_root

DEFAULT_CKPT = Path(os.getenv("GEMMA_CHECKPOINT", str(PROJECT_ROOT / "train" / "runs" / "jev_harness_v4" / "best.pt")))

DEFAULT_DAEMON_URL = os.getenv("GEMMA_DAEMON_URL", "http://127.0.0.1:8765")

class JevDeciderBase:

    def __init__(self, daemon_url: str = DEFAULT_DAEMON_URL):

        self.daemon_url = daemon_url.rstrip("/")

        self._spawn_attempted = False

    def get_hardware_status(self) -> Dict[str, Any]:

        """探查本地 GPU 与显存占用情况（用于决策门禁）。"""

        status = {

            "gpu_available": False,

            "vram_total_mb": 0,

            "vram_used_mb": 0,

            "vram_free_mb": 0,

            "vram_usage_pct": 0.0,

            "danger_level": 1,

            "active_python_processes": 0,

        }

        try:
            import subprocess

            cmd = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,nounits,noheader"

            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)

            if res.returncode == 0 and res.stdout.strip():
                parts = [int(p.strip()) for p in res.stdout.strip().split(",")]

                if len(parts) >= 3:
                    tot, used, free = parts[0], parts[1], parts[2]

                    status["gpu_available"] = True

                    status["vram_total_mb"] = tot

                    status["vram_used_mb"] = used

                    status["vram_free_mb"] = free

                    status["vram_usage_pct"] = round((used / max(1, tot)) * 100, 1)

                    if status["vram_usage_pct"] > 90:
                        status["danger_level"] = 5

                    elif status["vram_usage_pct"] > 80:
                        status["danger_level"] = 4

                    elif status["vram_usage_pct"] > 65:
                        status["danger_level"] = 3

                    else:
                        status["danger_level"] = 1

        except Exception:
            pass

        try:
            import psutil
            count = sum(1 for p in psutil.process_iter(['name']) if p.info.get('name') in ('python.exe', 'pythonw.exe'))
            status["active_python_processes"] = count
        except Exception:
            try:
                import subprocess
                flags = 0x08000000 if sys.platform == "win32" else 0
                si = subprocess.STARTUPINFO() if sys.platform == "win32" else None
                if si:
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    si.wShowWindow = subprocess.SW_HIDE
                p_chk = subprocess.run(
                    'tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH',
                    shell=True, capture_output=True, text=True, timeout=2,
                    creationflags=flags, startupinfo=si
                )
                if "python.exe" in p_chk.stdout:
                    status["active_python_processes"] = p_chk.stdout.count("python.exe")
            except Exception:
                pass

        return status

    def _try_spawn_daemon(self) -> None:
        """若守护服务离线，在后台静默拉起 serve_gemma.py（具备单例排他锁与 zero-console 保证）。"""
        import socket
        # 1. 探查 8765 端口是否已经就绪或已有监听连接
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", 8765)) == 0:
                    return
        except Exception:
            pass

        # 2. 跨进程启动锁防并发雪崩（防止多个MCP/工具调用同时拉起重复进程）
        lock_file = PROJECT_ROOT / "train" / "serve_gemma.boot.lock"
        now = time.time()
        try:
            if lock_file.exists():
                mtime = lock_file.stat().st_mtime
                if now - mtime < 45.0:  # 45秒内已有其他进程正在启动
                    return
            lock_file.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass

        # 3. 确定解释器：优先使用无黑框 GUI 子系统的 pythonw.exe
        python_exe = os.getenv("GEMMA_PYTHON_EXE", "")
        if not python_exe or not os.path.isfile(python_exe):
            venv_pyw = PROJECT_ROOT / ".venv" / "Scripts" / "pythonw.exe"
            venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
            if venv_pyw.is_file():
                python_exe = str(venv_pyw)
            elif venv_py.is_file():
                python_exe = str(venv_py)
            else:
                pyw_sys = Path(sys.executable).parent / "pythonw.exe"
                python_exe = str(pyw_sys) if pyw_sys.is_file() else sys.executable

        script_path = str(PROJECT_ROOT / "train" / "serve_gemma.py")
        ckpt_path = str(DEFAULT_CKPT)

        if not (os.path.exists(python_exe) and os.path.exists(script_path) and os.path.exists(ckpt_path)):
            return

        try:
            import subprocess
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008
            flags = CREATE_NO_WINDOW | DETACHED_PROCESS

            si = None
            if sys.platform == "win32":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = subprocess.SW_HIDE

            log_dir = PROJECT_ROOT / "train" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_out = open(log_dir / "serve_gemma.stdout.log", "a", encoding="utf-8")
            log_err = open(log_dir / "serve_gemma.stderr.log", "a", encoding="utf-8")

            subprocess.Popen(
                [python_exe, script_path, "--ckpt", ckpt_path, "--port", "8765", "--device", "cuda", "--no-open"],
                creationflags=flags,
                startupinfo=si,
                stdin=subprocess.DEVNULL,
                stdout=log_out,
                stderr=log_err,
                close_fds=True,
                cwd=str(PROJECT_ROOT)
            )
        except Exception:
            pass

    def _call_daemon(self, endpoint: str, payload: dict, timeout: float = 3.0) -> Optional[dict]:

        """向本地 serve_gemma 守护进程发送请求，离线时安全自愈并降级。"""

        url = f"{self.daemon_url}{endpoint}"

        try:
            req_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            req = urllib.request.Request(

                url, data=req_data,

                headers={"Content-Type": "application/json; charset=utf-8"}

            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))

        except Exception:
            if not self._spawn_attempted:
                self._spawn_attempted = True

                self._try_spawn_daemon()

        return None

    def _log_audit(self, mode: str, question: str, options: list[dict], state: str, res: dict) -> None:

        try:
            audit_file = Path(os.getenv("JEV_DECISIONS_LOG", str(Path.home() / ".gemma" / "decisions.jsonl")))

            if not audit_file.parent.is_dir():
                audit_file.parent.mkdir(parents=True, exist_ok=True)

            pred = res.get("pred", 0)

            conf = res.get("conf", 1.0)

            label = "unknown"

            if mode == "choice" and 0 <= pred < len(options):
                label = options[pred].get("t", f"option_{pred}")

            elif mode == "noul":
                label = "high_risk" if pred == 1 else "low_risk"

            elif mode == "score":
                label = str(pred + 1)

            entry = {

                "ts": time.time(),

                "node": "antigravity_mcp",

                "tool": "gemma_jev_decide",

                "state": state[:4000],

                "questions": [

                    {

                        "name": "jev_query",

                        "primitive": mode,

                        "prompt": question,

                        "labels": options,

                    }

                ],

                "outcome": {

                    "decisions": {

                        "jev_query": {

                            "primitive": mode,

                            "label": label,

                            "confidence": float(conf),

                            "backend": res.get("device", "cuda"),

                            "latency_ms": res.get("ms", 0.0),

                        }

                    }

                },

                "escalated": False,

                "disagreement": False,

                "allowed": bool(res.get("allowed", (pred == 0) if mode == "noul" else True)),

                "outcome_status": "PENDING",

            }

            with audit_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception:
            pass

    def jev_decide(self, mode: str, question: str, options: list[dict], state: str = "") -> dict:

        """执行 Jev 三原语因果决策（优先通过守护进程，守护离线时执行确定性语义降级）。"""

        res = self._call_daemon("/api/predict", {

            "mode": mode, "question": question, "options": options, "state": state

        })

        if res and "pred" in res:
            self._log_audit(mode, question, options, state, res)
            return res

        t0 = time.time()

        text_corpus = f"{state} {question}".lower()

        if mode == "noul":
            is_risky = any(k in text_corpus for k in (

                "oom", "error", "fail", "危险", "覆写", "覆盖", "高危", "关键",

                "rmdir", "delete", "format", "超限", "panic", "crash"

            ))

            p = 0.85 if is_risky else 0.15

            pred = int(p >= 0.5)

            return {

                "task": "noul", "pred": pred, "conf": p if pred == 1 else 1 - p,

                "ms": round((time.time() - t0) * 1000, 2), "device": "rule_fallback",

                "items": [{"label": "安全 / 正常", "p": 1 - p}, {"label": "风险 / 异常", "p": p}]

            }

        elif mode == "choice":
            best_idx = 0

            best_score = -1

            scores = []

            for idx, opt in enumerate(options):
                score = 0

                label = (opt.get("t", "") + " " + opt.get("d", "")).lower()

                for token in re.findall(r"[\w]+", label):
                    if len(token) > 1 and token in text_corpus:
                        score += 1

                scores.append(score)

                if score > best_score:
                    best_score = score

                    best_idx = idx

            probs = [round((s + 1) / max(1, sum(scores) + len(options)), 3) for s in scores]

            return {

                "task": "choice", "pred": best_idx, "conf": probs[best_idx],

                "ms": round((time.time() - t0) * 1000, 2), "device": "rule_fallback",

                "items": [{"label": opt["t"], "p": probs[i]} for i, opt in enumerate(options)]

            }

        else: # score

            lvl = 1

            if any(k in text_corpus for k in ("crash", "oom", "panic", "致命", "destructive")): lvl = 5

            elif any(k in text_corpus for k in ("error", "fail", "告警", "中断", "high")): lvl = 4

            elif any(k in text_corpus for k in ("warn", "超时", "排队", "重试", "medium")): lvl = 3

            elif any(k in text_corpus for k in ("notice", "info", "low")): lvl = 2

            idx = min(len(options) - 1, max(0, lvl - 1)) if options else lvl

            return {

                "task": "score", "pred": idx, "expected_level": float(lvl), "conf": 0.8,

                "ms": round((time.time() - t0) * 1000, 2), "device": "rule_fallback",

                "items": [{"label": opt["t"], "p": 0.8 if i == idx else 0.05} for i, opt in enumerate(options)] if options else []

            }
