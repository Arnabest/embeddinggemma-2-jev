#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gemma-Jev Model Architecture:
Non-destructive multimodal base with modular 7KB decision adapter.

Supports:
1. Pure text long context (up to 8K) causal decision making.
2. Multimodal image + text state gating.
3. Jev three primitives:
   - Noul: Binary anomaly & safety gating
   - Choice: Multi-option routing
   - Score: Risk & resource 1-5 scale grading
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class GemmaJevModel(nn.Module):
    """Gemma-Jev Core Decision Model.
    Wraps the EmbeddingGemma-2 language backbone (or full multimodal model)
    with a lightweight (~7KB) universal scoring adapter.
    """
    def __init__(
        self,
        base_model_dir: str | Path,
        freeze_base: bool = True,
        hidden_dim: int = 768,
        marker_dim: int = 512,
        adapter_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.base_model_dir = str(base_model_dir)

        # Load backbone
        self.backbone = AutoModel.from_pretrained(
            self.base_model_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Lightweight decision adapter (~7KB)
        self.marker = nn.Embedding(3, marker_dim)  # 0: noul, 1: choice, 2: score
        nn.init.normal_(self.marker.weight, std=0.02)
        self.scorer = nn.Linear(hidden_dim, 1)

        if adapter_path and Path(adapter_path).exists():
            self.load_adapter(adapter_path)

    def load_adapter(self, adapter_path: str | Path):
        state = torch.load(str(adapter_path), map_location="cpu")
        if "marker" in state and "scorer" in state:
            self.marker.load_state_dict(state["marker"])
            self.scorer.load_state_dict(state["scorer"])
        else:
            self.load_state_dict(state, strict=False)

    def save_adapter(self, output_path: str | Path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "marker": self.marker.state_dict(),
            "scorer": self.scorer.state_dict(),
            "architecture": "embeddinggemma-2-jev",
            "version": "1.0.0"
        }, str(output_path))

    def forward(self, batch: Dict[str, torch.Tensor]):
        input_ids = batch["ids"]
        attention_mask = batch["mask"]

        # Embed tokens through backbone
        embeds = self.backbone.get_input_embeddings()(input_ids)

        # Inject task marker tokens
        marker_idx = batch["task_idx"]  # [B]
        marker_emb = self.marker(marker_idx)  # [B, 512]
        embeds[:, 1, :marker_emb.shape[-1]] = marker_emb

        out = self.backbone(inputs_embeds=embeds, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, L, 768]

        # Universal scoring
        logits = self.scorer(last_hidden).squeeze(-1)  # [B, L]

        # Extract predictions for Jev primitives
        noul_pos = batch.get("noul_pos")
        choice_pos = batch.get("choice_pos")
        score_pos = batch.get("score_pos")

        noul_logits = logits[torch.arange(len(logits)), noul_pos] if noul_pos is not None else None

        choice_logits = None
        if choice_pos is not None:
            B, K = choice_pos.shape
            choice_logits = torch.gather(logits, 1, choice_pos)

        score_logits = None
        if score_pos is not None:
            B, S = score_pos.shape
            score_logits = torch.gather(logits, 1, score_pos)

        return noul_logits, choice_logits, score_logits
