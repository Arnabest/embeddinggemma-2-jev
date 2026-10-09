#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Incremental Distillation & Active Learning Pipeline.
Allows rapid fine-tuning of the 7KB adapter on newly harvested cases in <30 seconds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from core.model import GemmaJevModel
from train.train import JevDataset, collate_fn


def incremental_finetune(
    base_model_dir: str | Path,
    adapter_path: str | Path,
    new_data_path: str | Path,
    output_path: str | Path,
    epochs: int = 1,
    lr: float = 1e-4
):
    print(f"[Incremental] Fine-tuning 7KB adapter on new cases: {new_data_path}")
    tok = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    dataset = JevDataset(new_data_path, tok)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda b: collate_fn(b, tok.pad_token_id or 0))

    model = GemmaJevModel(base_model_dir, freeze_base=True, adapter_path=adapter_path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(torch.bfloat16).cuda() if dev == "cuda" else model.to(torch.float32)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        for batch in loader:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            noul, choice, score = model(batch)
            loss = F.cross_entropy(choice.float().squeeze(1), batch["golds"]) if choice is not None else 0.0
            optimizer.zero_grad()
            if torch.is_tensor(loss):
                loss.backward()
                optimizer.step()

    model.save_adapter(output_path)
    print(f"[Incremental] Done in {time.time()-t0:.2f}s! Updated adapter: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental Adapter Fine-Tuning")
    parser.add_argument("--base-model", default="models/base")
    parser.add_argument("--adapter", default="models/adapter/adapter.pt")
    parser.add_argument("--data", default="data/sample_dataset.jsonl")
    parser.add_argument("--output", default="models/adapter/adapter.pt")
    args = parser.parse_args()

    incremental_finetune(args.base_model, args.adapter, args.data, args.output)
