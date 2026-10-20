#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""GemmaPrepEngine: 决策微内核顶级门面 (Facade / Shim)。
整合：
1. 决策微内核集 (JevDeciderBase, CommandGuard, FailureTriage, MemoryGate, CodebaseAudit, ModelRouter)
2. 跨源记忆与历史召回中枢 (MemoryHub)
3. 硬件态势感知与全场景意图流分类 (11大原子意图 + 伴随槽位抽取)
4. JEV 70/30 经验回放增量微调与自进化闭环 (JevEvolutionRunner)
完全保持向后兼容性与 < 200 行的高内聚门面规范。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from deciders import (
    JevDeciderBase,
    CommandGuardDecider,
    FailureTriageDecider,
    MemoryGateDecider,
    CodebaseAuditDecider,
    ModelRouterDecider,
)
from recall import MemoryHub

DEFAULT_DAEMON_URL = "http://127.0.0.1:8765"


class GemmaPrepEngine:
    def __init__(self, daemon_url: str = DEFAULT_DAEMON_URL):
        self.base = JevDeciderBase(daemon_url=daemon_url)
        self.command_guard = CommandGuardDecider(self.base)
        self.failure_triage = FailureTriageDecider(self.base)
        self.memory_gate = MemoryGateDecider(self.base)
        self.codebase_audit = CodebaseAuditDecider(self.base)
        self.model_router = ModelRouterDecider(self.base)
        self.memory_hub = MemoryHub()

    # ------------------------------------------------------------------------- 硬件探针
    def get_hardware_status(self) -> Dict[str, Any]:
        """探查本地 GPU 与显存占用情况（用于决策门禁）。"""
        return self.base.get_hardware_status()

    # ------------------------------------------------------------------------- Jev 因果三原语
    def jev_decide(self, mode: str, question: str, options: list[dict], state: str = "") -> dict:
        """执行 Jev 三原语因果决策。"""
        return self.base.jev_decide(mode, question, options, state)

    # ------------------------------------------------------------------------- 前置意图分类
    def classify_intent(self, instruction: str, context: str = "") -> dict:
        """评估用户指令的核心意图，做动作分流建议 (涵盖11大系统与工程原子意图)."""
        slots: Dict[str, Any] = {}
        urls = re.findall(r'https?://[^\s<>"]+', instruction)
        if urls:
            slots["urls"] = urls
        files = re.findall(r'[\w\-./\\]+\.(?:py|md|json|ts|tsx|js|html|bat|sh|yaml|yml)', instruction)
        if files:
            slots["target_files"] = list(set(files))
        git_match = re.search(r'git\s+(?:checkout|reset|revert|switch|branch)\s+([a-zA-Z0-9_\-./]+)', instruction)
        if git_match:
            slots["git_target"] = git_match.group(1)

        options = [
            {"t": "CODE_MUTATE", "d": "编写、修改、重构、修Bug或生成源代码"},
            {"t": "READONLY_QUERY", "d": "只读查询文件、阅读文档、解释概念、询问状态"},
            {"t": "SYSTEM_COMMAND", "d": "运行终端命令、环境诊断、包安装或进程管理"},
            {"t": "MEDIA_COMFYUI", "d": "ComfyUI出图、生视频、模型推理调度或图像处理"},
            {"t": "RAG_RECALL", "d": "检索历史对话记录、查询知识库经验、召回技能"},
            {"t": "DANGEROUS_WRITE", "d": "覆写或删除核心配置、高危磁盘写入操作"},
            {"t": "BROWSER_ACTION", "d": "Chrome CDP 网络嗅探、网页抓取与前端DOM自动化交互"},
            {"t": "MULTIMODAL_PROCESS", "d": "语音合成TTS、音视频转写FunASR或OCR文字识别"},
            {"t": "SWARM_ORCHESTRATE", "d": "多路径子代理派发、并发探索或蜂群编排"},
            {"t": "STATE_ROLLBACK", "d": "Git Shadow 快照恢复、撤销更改或消除脏改动"},
            {"t": "DEEP_RESEARCH", "d": "学术前沿论文研读、数学公式推导与复杂消融实验提炼"}
        ]
        question = "请判定当前用户指令的核心意图类别："
        state = f"用户指令: {instruction}\n上下文摘要: {context[:400]}"
        dec = self.base.jev_decide("choice", question, options, state)
        chosen = options[dec["pred"]]["t"]
        conf = dec.get("conf", 0.0)

        q_lower = instruction.lower()
        if any(w in q_lower for w in ("git reset", "git revert", "git checkout .", "回滚", "撤销修改", "还原快照", "rollback", "undo", "消除脏改动")):
            chosen = "STATE_ROLLBACK"
            conf = max(conf, 0.92)
        elif any(w in q_lower for w in ("browser", "cdp", "网页", "嗅探", "sniffer", "bilibili", "b站", "弹幕", "三连", "油猴", "tampermonkey")):
            chosen = "BROWSER_ACTION"
            conf = max(conf, 0.91)
        elif any(w in q_lower for w in ("语音合成", "tts", "gpt-sovits", "sovits", "音频转写", "字幕", "funasr", "asr", "ocr", "提取文字", "mineru")):
            chosen = "MULTIMODAL_PROCESS"
            conf = max(conf, 0.90)
        elif any(w in q_lower for w in ("子代理", "subagent", "并发探索", "蜂群", "swarm", "多路径推理", "对立探索", "并行执行", "fork_agent")):
            chosen = "SWARM_ORCHESTRATE"
            conf = max(conf, 0.89)
        elif any(w in q_lower for w in ("论文", "arxiv", "公式推导", "消融实验", "paper", "研究报告", "研报总结", "算法提炼")):
            chosen = "DEEP_RESEARCH"
            conf = max(conf, 0.88)
        elif any(w in q_lower for w in ("怎么样", "如何", "查看", "查询", "多少", "状态", "为什么", "运行情况", "check", "status", "how")):
            if not any(w in q_lower for w in ("删除", "rmdir", "delete", "覆写", "覆盖", "重构", "修改", "写一个")):
                chosen = "READONLY_QUERY"
                conf = max(conf, 0.78)

        desc = next((opt["d"] for opt in options if opt["t"] == chosen), options[0]["d"])

        return {
            "intent": chosen,
            "confidence": conf,
            "description": desc,
            "slots": slots,
            "all_probs": dec.get("items", []),
            "latency_ms": dec.get("ms", 0.0),
            "device": dec.get("device", "unknown")
        }

    # ------------------------------------------------------------------------- 决策微内核特化能力
    def audit_command_safety(self, command_line: str, cwd: str = "", is_daemon: bool = False) -> Dict[str, Any]:
        """终端高危命令静态语义审计与破坏力分级。"""
        return self.command_guard.audit_command_safety(command_line, cwd, is_daemon)

    def triage_failure(self, traceback_text: str, exit_code: int = 1) -> Dict[str, Any]:
        """报错堆栈与单测失败秒级归因分流。"""
        return self.failure_triage.triage_failure(traceback_text, exit_code)

    def audit_memory_novelty(self, title: str, content: str, track: str = "all", tags: List[str] = None) -> Dict[str, Any]:
        """记忆入库高价值熵值与新颖度初筛门禁。"""
        return self.memory_gate.audit_memory_novelty(title, content, track, tags)

    def route_model_tier(self, prompt: str, context_len: int = 0) -> Dict[str, Any]:
        """任务认知复杂度与模型档位推荐分流。"""
        return self.model_router.route_model_tier(prompt, context_len)

    def post_task_codebase_audit(
        self,
        workspace_path: str,
        changed_files: Optional[List[str]] = None,
        exit_code: int = 0,
        session_goal: str = "",
        max_line_limit: int = 800
    ) -> Dict[str, Any]:
        """任务后代码仓库整理与 800 行硬上限审计。"""
        return self.codebase_audit.post_task_codebase_audit(
            workspace_path, changed_files, exit_code, session_goal, max_line_limit
        )

    # ------------------------------------------------------------------------- JEV 自进化与增量训练
    def trigger_evolution(self, epochs: int = 5, lr: float = 1e-4, force: bool = False) -> Dict[str, Any]:
        """触发 JEV 决策微内核的 70/30 经验回放极速增量蒸馏与就地热重载。"""
        from harvest.evolution_runner import JevEvolutionRunner
        runner = JevEvolutionRunner()
        return runner.run_incremental_distillation(epochs=epochs, lr=lr, force=force)

    def get_evolution_status(self) -> Dict[str, Any]:
        """获取当前适配器权重与最近一次自进化训练状态。"""
        from harvest.evolution_runner import JevEvolutionRunner
        runner = JevEvolutionRunner()
        return runner.get_status()

    # ------------------------------------------------------------------------- 跨源记忆召回
    def recall_dialogue_history(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.memory_hub.recall_dialogue_history(query, top_k)

    def recall_skills(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        return self.memory_hub.recall_skills(query, top_k)

    def recall_vault_memory(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        return self.memory_hub.recall_vault_memory(query, top_k)

    def hybrid_recall(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        return self.memory_hub.hybrid_recall(query, top_k)
