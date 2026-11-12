#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Context Assembly Pipeline:
将前置意图分类、多源记忆召回、硬件态势与本地依赖适用路由编排组装为结构化上下文卡片（Attention HUD），
注入到 Antigravity 思考前 Hook 与在线模型（Gemini 等）的会话上下文中。
"""
from __future__ import annotations

import json
from typing import Dict, Any, Optional
from engine import GemmaPrepEngine
from deciders.directional_gate import format_directional_hud_block


class ContextAssemblyPipeline:
    def __init__(self, engine: Optional[GemmaPrepEngine] = None):
        self.engine = engine or GemmaPrepEngine()

    def assemble(
        self,
        instruction: str,
        state: str = "",
        include_recall: bool = True,
        include_hardware: bool = True
    ) -> Dict[str, Any]:
        """执行端到端上下文编排组装流水线。"""
        # 1. 前置意图判断
        intent_info = self.engine.classify_intent(instruction, context=state)
        is_directional = intent_info.get("intent") == "DIRECTIONAL_PLANNING"
        directional_gate = {
            "locked": is_directional,
            "phase": "CLARIFY_FIRST" if is_directional else "EXECUTION_READY",
            "intent": intent_info.get("intent", "UNKNOWN"),
            "requires_clarification": is_directional,
            "signals": intent_info.get("directional_signals", []),
        }

        # 2. 硬件态势感知
        hw_info = self.engine.get_hardware_status() if include_hardware else {}

        # 3. 依赖适用路由引导感知
        dep_guidance = []
        hud_dep_text = ""
        if instruction:
            dep_res = self.engine.route_dependencies(instruction, context=state)
            dep_guidance = dep_res.get("guidances", [])
            hud_dep_text = dep_res.get("hud_text", "")

        # 4. 记忆与先验数据召回
        recalls = []
        if include_recall and instruction:
            recalls = self.engine.hybrid_recall(instruction, top_k=3)

        # 5. 生成高密度 Attention HUD Markdown
        hud_lines = ["[Gemma-Core: 前置态势与先验装配]"]

        # 意图行
        intent_tag = intent_info.get("intent", "UNKNOWN")
        intent_conf = intent_info.get("confidence", 0.0)
        hud_lines.append(f"- [核心意图] `{intent_tag}` (置信度: {intent_conf:.1%}) | {intent_info.get('description', '')}")
        if is_directional:
            hud_lines.append(format_directional_hud_block(True, signals=directional_gate.get("signals")))

        # 依赖路由指引行
        if hud_dep_text:
            hud_lines.append(hud_dep_text)

        # 硬件行
        if hw_info.get("gpu_available"):
            used = hw_info.get("vram_used_mb", 0)
            tot = hw_info.get("vram_total_mb", 0)
            pct = hw_info.get("vram_usage_pct", 0.0)
            lvl = hw_info.get("danger_level", 1)
            warn = "[警告: 显存高水位]" if lvl >= 4 else "[正常: 显存充裕]"
            hud_lines.append(f"- [硬件状态] {warn} (RTX 4060 Ti 显存: {used}/{tot}MB, {pct}% 占用)")

        # 召回先验行
        if recalls:
            hud_lines.append("- [记忆召回] 历史会话与技能先验:")
            for item in recalls:
                src = item.get("source", "memory")
                title = item.get("title") or item.get("name") or item.get("chunk_id") or "片段"
                snip = (item.get("snippet", "")).replace("\n", " ").strip()
                if len(snip) > 120:
                    snip = snip[:120] + "..."
                hud_lines.append(f"  * [{src}: {title}] {snip}")

        hud_markdown = "\n".join(hud_lines)

        return {
            "intent": intent_info,
            "directional_gate": directional_gate,
            "hardware": hw_info,
            "dependency_guidance": dep_guidance,
            "recalls": recalls,
            "hud_markdown": hud_markdown,
            "assembled_steps": [
                {
                    "ephemeralMessage": hud_markdown
                }
            ]
        }
