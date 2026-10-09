#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adapter Training Script for EmbeddingGemma-2-Jev.

Freezes 100% of the multimodal base parameters.
Optimizes only the lightweight marker embedding and universal scorer.
Enforces native bfloat16 mixed precision.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from core.model import GemmaJevModel


class JevDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer: AutoTokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        state = row.get("state", "")
        # Format task
        if "choice" in row:
            task_idx = 1
            q = row["choice"]["question"]
            opts = row["choice"]["options"]
            prompt = f"{state}\nQuestion: {q}\n" + "\n".join(f"- {o['t']}" for o in opts)
        elif "score" in row:
            task_idx = 2
            q = row["score"]["question"]
            levels = row["score"]["levels"]
            prompt = f"{state}\nQuestion: {q}\n" + "\n".join(f"- {lv}" for lv in levels)
        else:
            task_idx = 0
            stmt = row["noul"]["statement"]
            prompt = f"{state}\nStatement: {stmt}"

        enc = self.tokenizer(prompt, max_length=self.max_len, truncation=True, return_tensors="pt")
        return {
            "ids": enc["input_ids"][0],
            "mask": enc["attention_mask"][0],
            "task_idx": torch.tensor(task_idx, dtype=torch.long),
            "gold": row.get("choice", {}).get("y", row.get("score", {}).get("y", row.get("noul", {}).get("y", 0)))
        }


def collate_fn(batch, pad_token_id: int):
    max_l = max(b["ids"].shape[0] for b in batch)
    ids = torch.full((len(batch), max_l), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_l), dtype=torch.long)
    task_idx = torch.stack([b["task_idx"] for b in batch])
    golds = torch.tensor([b["gold"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        l = b["ids"].shape[0]
        ids[i, :l] = b["ids"]
        mask[i, :l] = b["mask"]

    return {
        "ids": ids,
        "mask": mask,
        "task_idx": task_idx,
        "golds": golds,
        "noul_pos": ids.argmax(dim=-1),  # Last token placeholder
        "choice_pos": ids.argmax(dim=-1).unsqueeze(-1),
        "score_pos": ids.argmax(dim=-1).unsqueeze(-1)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Gemma-Jev Adapter")
    parser.add_argument("--base-model", default="models/base", help="Path to base model")
    parser.add_argument("--data", default="data/sample_dataset.jsonl", help="Training dataset JSONL")
    parser.add_argument("--output", default="models/adapter/adapter.pt", help="Output adapter path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    print(f"[Train] Initializing adapter training with base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    dataset = JevDataset(args.data, tok)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tok.pad_token_id or 0))

    model = GemmaJevModel(args.base_model, freeze_base=True)
    model = model.to(torch.bfloat16).cuda() if torch.cuda.is_available() else model.to(torch.float32)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    for ep in range(args.epochs):
        tot_loss = 0.0
        for step, batch in enumerate(loader):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            noul, choice, score = model(batch)
            loss = F.cross_entropy(choice.float().squeeze(1), batch["golds"]) if choice is not None else 0.0

            optimizer.zero_grad()
            if torch.is_tensor(loss):
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
        print(f"Epoch {ep+1}/{args.epochs} - Loss: {tot_loss/max(1, len(loader)):.4f}")

    model.save_adapter(args.output)
    print(f"[Train] Adapter successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
