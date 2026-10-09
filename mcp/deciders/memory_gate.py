# -*- coding: utf-8 -*-
"""MemoryGateDecider: 记忆入库双流分流与熵值初筛门禁."""
from __future__ import annotations

import re
from typing import Dict, Any, List
from .base import JevDeciderBase


class MemoryGateDecider:
    def __init__(self, base: JevDeciderBase):
        self.base = base

    def audit_memory_novelty(
        self,
        title: str,
        content: str,
        track: str = "all",
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """
        评估待存入 memory-vault 的内容的信息熵、技术新颖度与知识密度，
        并智能分流至 code（技术库）或 interaction（日常会话/偏好库）。
        返回:
        - novelty_score: 1.0 ~ 5.0
        - should_ingest: 是否准予入库 (True/False)
        - target_stream: "code" | "interaction" (物理双库分流目标)
        - target_track: 目标分类轨 (chat, memo, architecture, errors 等)
        - reason: 判定归因
        - suggested_tags: 推荐标签
        """
        text_clean = f"{title}\n{content}".strip()
        tags_list = tags or []

        if len(text_clean) < 2:
            return {
                "title": title,
                "novelty_score": 0.0,
                "should_ingest": False,
                "target_stream": "interaction",
                "target_track": "chat",
                "reason": "内容为空，无需入库。",
                "suggested_tags": []
            }

        text_lower = text_clean.lower()

        # 1. 检测是否属于日常会话、客套或备忘交互
        is_chatter = any(p in text_lower for p in (
            "收到", "好的", "没问题", "ok, got it", "hello", "hi there", "谢谢", "感谢",
            "辛苦", "拜拜", "在吗", "测试一下", "对话记录", "聊天"
        )) and not any(k in text_lower for k in ("def ", "class ", "```", "traceback", "commit"))

        is_explicit_interaction = track in ("chat", "preference", "memo", "news") or any(
            t in ("chat", "interaction", "dialogue", "preference") for t in tags_list
        )

        # 2. 技术硬特征检测
        has_architecture = any(k in text_lower for k in ("架构", "topology", "adr", "结构树", "解耦", "shim", "interface"))
        has_error_fix = any(k in text_lower for k in ("踩坑", "排障", "root cause", "解决", "exit code 0", "workaround", "error", "exception"))
        has_code_block = "```" in text_clean or "def " in text_clean or "class " in text_clean

        is_tech = has_architecture or has_error_fix or has_code_block or track in ("architecture", "errors", "trajectory", "media", "literature")

        # 3. 双流导流与分类轨决策
        if is_chatter or (is_explicit_interaction and not is_tech):
            # 录入日常会话记录区 (interaction 物理流)
            target_stream = "interaction"
            target_track = "chat" if track in ("all", "chat") else track
            novelty_score = 2.0
            should_ingest = True
            reason = "日常会话与客套交互记录，准予录入 interaction 物理流的 chat/memo 专区。"
            suggested_tags = list(set(tags_list + ["interaction", "chat"]))
            latency_ms = 0.0
            device = "heuristic"
        else:
            # 录入严肃工程与技术资产区 (code 物理流)
            target_stream = "code"
            if track not in ("all", "chat", "interaction"):
                target_track = track
            elif has_error_fix:
                target_track = "errors"
            else:
                target_track = "architecture"

            base_score = 3.0
            if has_architecture: base_score += 1.0
            if has_error_fix: base_score += 0.8
            if has_code_block: base_score += 0.5
            final_score = min(5.0, base_score)

            # Jev 因果校验
            options = [
                {"t": "MODERATE", "d": "一般性项目日志或常规代码片段 (Score 3)"},
                {"t": "HIGH_VALUE", "d": "高密度技术决策、架构图谱、严谨排障经验或重构规范 (Score 4-5)"}
            ]
            question = "请对该技术记忆片段的长效复用价值与知识密度进行评分："
            state = f"分类轨: {target_track}\n标题: {title}\n基准分: {final_score}"

            dec = self.base.jev_decide("choice", question, options, state)
            jev_choice = options[dec.get("pred", 0)]["t"]

            if jev_choice == "HIGH_VALUE":
                final_score = max(4.0, final_score)

            novelty_score = round(final_score, 1)
            should_ingest = True
            reason = f"高价值技术资产与因果沉淀，准予录入 code 物理流的 {target_track} 专区。"

            suggested_tags = list(set(tags_list))
            if has_architecture and "architecture" not in suggested_tags:
                suggested_tags.append("architecture")
            if has_error_fix and "troubleshooting" not in suggested_tags:
                suggested_tags.append("troubleshooting")

            latency_ms = dec.get("ms", 0.0)
            device = dec.get("device", "unknown")

        return {
            "title": title,
            "novelty_score": novelty_score,
            "should_ingest": should_ingest,
            "target_stream": target_stream,
            "target_track": target_track,
            "reason": reason,
            "suggested_tags": suggested_tags,
            "latency_ms": latency_ms,
            "device": device
        }
