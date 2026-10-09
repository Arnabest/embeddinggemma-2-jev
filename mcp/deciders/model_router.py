# -*- coding: utf-8 -*-
"""ModelRouterDecider: 模型复杂度分级与任务分流路由建议 (当前在线为主，预留本地 Harness 演进)."""
from __future__ import annotations

import re
from typing import Dict, Any
from .base import JevDeciderBase


class ModelRouterDecider:
    def __init__(self, base: JevDeciderBase):
        self.base = base

    def route_model_tier(
        self,
        prompt: str,
        context_len: int = 0
    ) -> Dict[str, Any]:
        """
        评估用户指令与任务上下文的认知复杂度。
        - 当前在 Antigravity 环境下：以在线模型为主（推荐 CLOUD_FLASH 或 CLOUD_PRO）。
        - 长期演进方向：随本地 Harness 逐步完善，标注 local_harness_eligible 与 future_harness_tier。
        返回:
        - recommended_tier: CLOUD_FLASH | CLOUD_PRO (当前生效档位)
        - current_paradigm: "ONLINE_CLOUD_PRIMARY"
        - local_harness_eligible: 是否适合在本地 Harness 完善后下沉至本地模型
        - future_harness_tier: LOCAL_LIGHT | CLOUD_PRO
        - estimated_complexity: 1.0 ~ 5.0
        - rationale: 推荐与演进理由
        """
        prompt_clean = prompt.strip()
        p_lower = prompt_clean.lower()

        # 1. 复杂度启发式基准
        is_complex = any(k in p_lower for k in (
            "架构", "重构", "解耦", "refactor", "decouple", "多模块", "并发",
            "deadlock", "死锁", "内存泄漏", "设计模式", "adr", "microservice"
        )) or context_len > 4000

        is_trivial = any(k in p_lower for k in (
            "怎么写", "如何实现", "语法", "syntax", "正则", "regex", "单词", "翻译",
            "格式化", "时间戳", "print", "查看", "解释一下"
        )) and not is_complex and context_len < 1000

        # 2. Jev 因果分类
        options = [
            {"t": "FLASH_ROUTINE", "d": "轻量单点代码、语法查询、常规增删查改或快速单测 (推荐 Cloud Flash，后续可下沉本地)"},
            {"t": "PRO_COMPLEX", "d": "复杂跨文件解耦、系统架构设计、复杂死锁排查与深度综合推演 (推荐 Cloud Pro)"}
        ]
        question = "请对该任务在当前 Antigravity 在线环境下推荐最优算力档位，并评估本地下沉潜力："
        state = f"提示词: {prompt_clean[:300]}\n上下文长度: {context_len}\n初步感知: {'COMPLEX' if is_complex else ('TRIVIAL' if is_trivial else 'NORMAL')}"

        dec = self.base.jev_decide("choice", question, options, state)
        jev_choice = options[dec.get("pred", 0)]["t"]

        # 规则校准
        if is_complex:
            recommended_tier = "CLOUD_PRO"
            local_eligible = False
            future_tier = "CLOUD_PRO"
            complexity = 4.8
            rationale = "任务涉及跨模块解耦或深层系统架构，在 Antigravity 当前环境下建议选用 Cloud Pro 深度模型保障推演完整性。"
        else:
            recommended_tier = "CLOUD_FLASH"
            local_eligible = is_trivial or context_len < 2000
            future_tier = "LOCAL_LIGHT" if local_eligible else "CLOUD_FLASH"
            complexity = 1.5 if is_trivial else 3.0
            rationale = (
                "轻量或常规任务，当前在 Antigravity 优先采用 Cloud Flash 保持极速响应；未来本地 Harness 就绪后，可直接无缝下沉至本地模型执行。"
                if local_eligible else
                "常规单模块代码任务，当前推荐使用 Cloud Flash 兼顾低延迟与高经济性。"
            )

        return {
            "recommended_tier": recommended_tier,
            "current_paradigm": "ONLINE_CLOUD_PRIMARY",
            "confidence": dec.get("conf", 0.0),
            "estimated_complexity": complexity,
            "local_harness_eligible": local_eligible,
            "future_harness_tier": future_tier,
            "rationale": rationale,
            "latency_ms": dec.get("ms", 0.0),
            "device": dec.get("device", "unknown")
        }
