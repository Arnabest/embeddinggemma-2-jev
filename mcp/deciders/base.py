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
            import subprocess
            p_chk = subprocess.run(
                'tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH',
                shell=True, capture_output=True, text=True, timeout=2
            )
            if "python.exe" in p_chk.stdout:
                status["active_python_processes"] = p_chk.stdout.count("python.exe")
        except Exception:
            pass

        return status

    def _try_spawn_daemon(self) -> None:
        """若守护服务离线，在后台静默拉起 serve_gemma.py。"""
        python_exe = os.getenv("GEMMA_PYTHON_EXE", sys.executable)
        script_path = str(PROJECT_ROOT / "train" / "serve_gemma.py")
        if not (os.path.exists(python_exe) and os.path.exists(script_path)):
            return
        try:
            import subprocess
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008
            flags = CREATE_NO_WINDOW | DETACHED_PROCESS
            subprocess.Popen(
                [python_exe, script_path, "--port", "8765", "--no-open"],
                creationflags=flags,
                close_fds=True
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

    def jev_decide(self, mode: str, question: str, options: list[dict], state: str = "") -> dict:
        """执行 Jev 三原语因果决策（优先通过守护进程，守护离线时执行确定性语义降级）。"""
        res = self._call_daemon("/api/predict", {
            "mode": mode, "question": question, "options": options, "state": state
        })
        if res and "pred" in res:
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
