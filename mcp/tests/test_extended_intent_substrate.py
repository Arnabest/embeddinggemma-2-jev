#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression Test Suite for Extended Intent Decision Substrate & Slot Filling.

Verifies:
1. Extended IntentType enum integrity (all 11 intents present).
2. IntentDecision data contract with lightweight slot extraction.
3. HeuristicDecisionAdapter deterministic intent routing & skill recommendation.
4. GemmaPrepEngine facade intent classification with dual-track calibration.
5. Microsecond execution latency guarantee (< 10ms).
"""

import sys
import time
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from decision_substrate.protocol import IntentType, IntentDecision
from decision_substrate.adapters.heuristic_adapter import HeuristicDecisionAdapter
from engine import GemmaPrepEngine


class TestExtendedIntentSubstrate:

    def test_intent_type_enum_completeness(self):
        """1. 验证意图枚举完整包含 11 类原子意图。"""
        expected_intents = {
            "CODE_MUTATE",
            "READONLY_QUERY",
            "SYSTEM_COMMAND",
            "HIGH_HAZARD",
            "GENERAL_CONVERSATION",
            "BROWSER_ACTION",
            "MULTIMODAL_PROCESS",
            "SWARM_ORCHESTRATE",
            "STATE_ROLLBACK",
            "DEEP_RESEARCH",
            "COMPOUND_CHAIN",
        }
        actual_intents = {member.value for member in IntentType}
        assert expected_intents.issubset(actual_intents), f"Missing intents: {expected_intents - actual_intents}"

    def test_heuristic_adapter_extended_intent_routing(self):
        """2. 验证符号规则适配器对 6 大新空白意图的精确分流与技能推荐。"""
        adapter = HeuristicDecisionAdapter()

        test_cases = [
            # 浏览器与 CDP 自动化
            (
                "请使用 browser-sniffer 监听当前 B 站标签页的 WebSocket 弹幕流",
                "BROWSER_ACTION",
                "browser-sniffer",
            ),
            # 泛多模态 (TTS/ASR/OCR)
            (
                "用 gpt-sovits 为这段解说词生成语音合成音频，并调用 funasr 校验字幕",
                "MULTIMODAL_PROCESS",
                "gpt-sovits-synthesize",
            ),
            # 子代理并发编排
            (
                "针对该数学竞赛题拉起两个对立子代理进行并发探索与多路径推理",
                "SWARM_ORCHESTRATE",
                "nexus-scaffold",
            ),
            # 状态守恒与安全回滚
            (
                "立即执行 git reset 消除脏改动，回滚到上一个安全快照",
                "STATE_ROLLBACK",
                "nexus-scaffold",
            ),
            # 深度学术科研
            (
                "精读这篇 arxiv 论文，推导其第 4 节公式并提炼消融实验证据",
                "DEEP_RESEARCH",
                "paper-distiller",
            ),
            # 高危命令拦截
            (
                "rmdir /s /q c:\\test_folder",
                "HIGH_HAZARD",
                "pre-action-audit",
            ),
        ]

        for prompt, expected_intent, expected_skill in test_cases:
            t0 = time.perf_counter()
            decision: IntentDecision = adapter.infer_intent(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            assert decision.intent == expected_intent, f"Prompt '{prompt}' expected {expected_intent}, got {decision.intent}"
            assert expected_skill in decision.recommended_skills, f"Expected {expected_skill} in {decision.recommended_skills}"
            assert latency_ms < 10.0, f"Latency too high: {latency_ms:.2f}ms"

    def test_slot_extraction(self):
        """3. 验证伴随槽位（URL、文件、Git分支）的轻量抽取能力。"""
        adapter = HeuristicDecisionAdapter()
        prompt = "分析 https://huggingface.co/datasets/XingChen-AGI/test 里的 model.py 并执行 git checkout feature-branch"
        
        decision = adapter.infer_intent(prompt)
        slots = decision.slots

        assert "urls" in slots
        assert any("huggingface.co" in u for u in slots["urls"])
        assert "target_files" in slots
        assert any("model.py" in f for f in slots["target_files"])
        assert slots.get("git_target") == "feature-branch"

    def test_engine_facade_classify_intent(self):
        """4. 验证 GemmaPrepEngine 顶级门面意图分类与槽位集成。"""
        engine = GemmaPrepEngine()
        
        # 测试浏览器动作
        res_browser = engine.classify_intent("打开当前网页进行 CDP 嗅探并提取 DOM 结构")
        assert res_browser["intent"] == "BROWSER_ACTION"
        assert "slots" in res_browser

        # 测试回滚操作
        res_rollback = engine.classify_intent("帮我回滚刚才的修改，撤销修改还原快照")
        assert res_rollback["intent"] == "STATE_ROLLBACK"

        # 测试论文精读
        res_paper = engine.classify_intent("研读这篇 paper 的消融实验部分")
        assert res_paper["intent"] == "DEEP_RESEARCH"

        # 测试默认代码生成
        res_code = engine.classify_intent("编写一个快速排序算法并生成单测")
        assert res_code["intent"] == "CODE_MUTATE"
