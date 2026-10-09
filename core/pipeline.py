#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Context Assembly Pipeline:
Composes pre-instruction intent classification, hardware telemetry,
and retrieved knowledge priors into a structured Attention HUD.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from .engine import GemmaPrepEngine


class ContextAssemblyPipeline:
    def __init__(self, engine: Optional[GemmaPrepEngine] = None):
        self.engine = engine

    def assemble(
        self,
        instruction: str,
        state: str = "",
        include_hardware: bool = True
    ) -> Dict[str, Any]:
        """Executes end-to-end context assembly pipeline."""
        intent_info = self.engine.classify_intent(instruction, context=state) if self.engine else {
            "intent": "UNKNOWN", "confidence": 0.0, "description": ""
        }
        hw_info = self.engine.get_hardware_status() if (self.engine and include_hardware) else {}

        hud_lines = ["[Gemma-Core: Telemetry & Attention HUD]"]
        intent_tag = intent_info.get("intent", "UNKNOWN")
        intent_conf = intent_info.get("confidence", 0.0)
        hud_lines.append(f"- [Intent] `{intent_tag}` (Confidence: {intent_conf:.1%}) | {intent_info.get('description', '')}")

        if hw_info.get("gpu_available"):
            used = hw_info.get("vram_used_mb", 0)
            tot = hw_info.get("vram_total_mb", 0)
            pct = hw_info.get("vram_usage_pct", 0.0)
            lvl = hw_info.get("danger_level", 1)
            warn = "[ALERT: High VRAM Occupancy]" if lvl >= 4 else "[OK: Normal Headroom]"
            hud_lines.append(f"- [Hardware] {warn} (GPU Memory: {used}/{tot}MB, {pct}% used)")

        hud_markdown = "\n".join(hud_lines)

        return {
            "intent": intent_info,
            "hardware": hw_info,
            "hud_markdown": hud_markdown,
            "assembled_steps": [
                {
                    "ephemeralMessage": hud_markdown
                }
            ]
        }
