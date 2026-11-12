#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Xing Intent Distiller & Schema-to-JEV Projector (星辰意图数据集精选提纯与投影器).

从 Xing4.0-29B-Intent-oss 百万级生成式语料中：
1. 精准挑选：仅截取“系统配置调节(SYSTEM_COMMAND)”与“运行状态查询(READONLY_QUERY)”两类高价值样本；
2. 毒性剪枝：严格剔除智能座舱车控(雨刮/座椅)、智能家居(电饭煲/扫地机)与垂直业务工单(挂号/报销)等生活语义噪音；
3. 降维投影：将生成式 JSON Schema 投影转换为 JEV 标准 choice 三原语结构，并保留伴随槽位 (slots)；
4. 经验守恒：控制精选样本容量在 200~300 条以内，适配本地 70/30 经验回放微调屏障。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JEV_STANDARD_OPTIONS = [
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

# 严格白名单关键词 (仅针对系统运维与状态探测)
SYSTEM_CMD_KEYWORDS = {
    "system", "setting", "volume", "display", "config", "bluetooth", "wifi",
    "network", "service", "brightness", "restart", "shutdown", "sound", "screen",
    "系统设置", "网络配置", "声音调节", "屏幕亮度", "服务重启", "蓝牙连接", "终端参数"
}

READONLY_QUERY_KEYWORDS = {
    "query", "status", "info", "battery", "storage", "memory", "cpu", "version",
    "check", "detail", "usage", "配额查询", "状态查看", "剩余空间", "系统版本", "硬件状态"
}

# 绝对黑名单关键词 (严格过滤日常生活与车辆物理硬件干扰)
EXCLUDE_KEYWORDS = {
    "seat", "window", "wiper", "air_conditioner", "hvac", "sunroof", "trunk",
    "door", "seat_belt", "ticket", "hotel", "flight", "train", "hospital",
    "medical", "doctor", "reimburse", "invoice", "cooker", "cleaner",
    "车窗", "座椅", "雨刮", "天窗", "后备箱", "空调温度", "定机票", "预订酒店",
    "看病挂号", "发票报销", "扫地机器人", "电饭煲"
}


class XingIntentDistiller:
    def __init__(self, max_samples: int = 300):
        self.max_samples = max_samples

    def is_eligible_sample(self, text: str, category: str) -> Tuple[bool, Optional[str]]:
        """判定样本是否命中系统运维/状态查询且远离生活语义噪音."""
        combined = f"{text.lower()} {category.lower()}"

        # 1. 命中黑名单直接剪枝
        if any(w in combined for w in EXCLUDE_KEYWORDS):
            return False, None

        # 2. 检查是否命中只读查询
        if any(w in combined for w in READONLY_QUERY_KEYWORDS):
            return True, "READONLY_QUERY"

        # 3. 检查是否命中系统配置与调节
        if any(w in combined for w in SYSTEM_CMD_KEYWORDS):
            return True, "SYSTEM_COMMAND"

        return False, None

    def transform_record(self, raw_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将单条 Xing 生成式样本转换为 JEV Choice 标准样本并保留 slots."""
        user_text = raw_record.get("instruction", "") or raw_record.get("query", "")
        raw_output = raw_record.get("output", "")

        if not user_text or not raw_output:
            return None

        # 解析生成式 JSON
        if isinstance(raw_output, str):
            try:
                parsed_output = json.loads(raw_output)
            except Exception:
                return None
        elif isinstance(raw_output, dict):
            parsed_output = raw_output
        elif isinstance(raw_output, list) and len(raw_output) > 0:
            parsed_output = raw_output[0]
        else:
            return None

        category = str(parsed_output.get("category", "") or parsed_output.get("intent", ""))
        args = parsed_output.get("args", {}) if isinstance(parsed_output, dict) else {}

        eligible, mapped_intent = self.is_eligible_sample(user_text, category)
        if not eligible or not mapped_intent:
            return None

        target_idx = 1 if mapped_intent == "READONLY_QUERY" else 2

        return {
            "mode": "choice",
            "category": "INTENT_PROJECTION",
            "question": "请判定当前用户指令的核心意图类别：",
            "instruction": user_text,
            "state": f"用户指令: {user_text}\n意图抽取先验: category={category}",
            "options": JEV_STANDARD_OPTIONS,
            "target_idx": target_idx,
            "ground_truth_label": target_idx,
            "slots": args,
            "source_origin": "Xing4.0-29B-Intent-oss",
            "distillation_meta": {
                "original_category": category,
                "projected_intent": mapped_intent,
                "confidence_anchor": 0.95
            }
        }

    def distill_from_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量筛选并降维投影，返回严格受控数量的黄金增量样本."""
        distilled = []
        for r in records:
            transformed = self.transform_record(r)
            if transformed:
                distilled.append(transformed)
                if len(distilled) >= self.max_samples:
                    break
        return distilled

    def save_increment_dataset(self, samples: List[Dict[str, Any]], output_path: Path) -> int:
        """物理写入 jsonl 数据集."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return len(samples)


def generate_seed_distilled_corpus(output_path: Path) -> int:
    """生成一批高精度的星辰意图系统级对齐种子样本 (用于本地离线立即验证与热重载)."""
    seed_xing_raw_data = [
        # --- 只读状态与配额查询 (READONLY_QUERY -> target_idx: 1) ---
        {"instruction": "查看系统当前剩余磁盘配额与各分区使用情况", "output": json.dumps({"category": "query_storage_usage", "args": {"target": "disk", "metric": "quota"}})},
        {"instruction": "查询当前机器剩余可用物理内存与交换分区状态", "output": json.dumps({"category": "check_memory_status", "args": {"target": "ram_swap", "unit": "mb"}})},
        {"instruction": "获取系统内核版本号与当前操作系统详细发行版信息", "output": json.dumps({"category": "query_system_version", "args": {"detail": True}})},
        {"instruction": "显示当前后台活跃的网络连接与监听端口状态", "output": json.dumps({"category": "query_network_status", "args": {"protocol": "all", "state": "listening"}})},
        {"instruction": "查询系统当前的电池剩余电量和充电健康度", "output": json.dumps({"category": "query_battery_info", "args": {"include_health": True}})},
        {"instruction": "检索当前进程列表中占用 CPU 最多的前5个任务", "output": json.dumps({"category": "check_cpu_usage", "args": {"top_k": 5, "sort_by": "cpu"}})},
        {"instruction": "查看当前本地蓝牙服务是否开启并列出已配对设备", "output": json.dumps({"category": "query_bluetooth_status", "args": {"paired_only": True}})},
        {"instruction": "检查本地 Python 运行时的依赖包安装列表与版本", "output": json.dumps({"category": "query_package_status", "args": {"runtime": "python"}})},

        # --- 系统配置调节与服务管理 (SYSTEM_COMMAND -> target_idx: 2) ---
        {"instruction": "将系统屏幕亮度调高到百分之八十", "output": json.dumps({"category": "setting_display_brightness", "args": {"level": 80, "relative": False}})},
        {"instruction": "把系统主音频输出音量设置为百分之五十并解除静音", "output": json.dumps({"category": "setting_audio_volume", "args": {"volume": 50, "unmute": True}})},
        {"instruction": "重启后台 network-manager 网络服务以刷新配置", "output": json.dumps({"category": "restart_system_service", "args": {"service_name": "network-manager"}})},
        {"instruction": "配置系统静态 IP 地址为 192.168.1.100 并应用更改", "output": json.dumps({"category": "config_network_ip", "args": {"ip": "192.168.1.100", "apply": True}})},
        {"instruction": "开启系统的节能电源管理模式并设置休眠超时为15分钟", "output": json.dumps({"category": "setting_power_mode", "args": {"mode": "energy_saving", "timeout_min": 15}})},
        {"instruction": "开启本地蓝牙适配器并开始扫描周边广播设备", "output": json.dumps({"category": "setting_bluetooth_power", "args": {"state": "on", "scan": True}})},
        {"instruction": "清空系统 DNS 本地解析缓存并重置套接字连接", "output": json.dumps({"category": "flush_network_cache", "args": {"target": "dns"}})},
        {"instruction": "将终端默认字符编码设置为 UTF-8 并保存环境配置", "output": json.dumps({"category": "config_terminal_encoding", "args": {"encoding": "utf-8"}})},

        # --- 对抗性排除样本 (测试黑名单剪枝能力) ---
        {"instruction": "帮我把副驾驶座椅往后调一点", "output": json.dumps({"category": "seat_control", "args": {"seat": "passenger"}})},
        {"instruction": "今天下雨了把汽车雨刮器开到自动模式", "output": json.dumps({"category": "wiper_control", "args": {"mode": "auto"}})},
        {"instruction": "帮我查一下明天北京到上海的高铁机票", "output": json.dumps({"category": "query_flight_ticket", "args": {"date": "tomorrow"}})},
        {"instruction": "把电饭煲设置为晚上六点开始煮粥", "output": json.dumps({"category": "cooker_schedule", "args": {"mode": "porridge"}})},
    ]

    distiller = XingIntentDistiller(max_samples=300)
    samples = distiller.distill_from_records(seed_xing_raw_data)
    count = distiller.save_increment_dataset(samples, output_path)
    return count


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "xing_system_intent_increment.jsonl"
    cnt = generate_seed_distilled_corpus(out)
    print(f"[XingIntentDistiller] Successfully extracted and saved {cnt} pure system intent samples to {out}")
