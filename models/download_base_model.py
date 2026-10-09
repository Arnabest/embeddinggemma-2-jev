#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download Google DeepMind's official EmbeddingGemma-2 base weights from Hugging Face.

Usage:
  python models/download_base_model.py --target-dir models/base/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_base_model(target_dir: str | Path, model_id: str = "google/embeddinggemma-2") -> Path:
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"[Download] Fetching official base model weights from Hugging Face: {model_id}")
    print(f"[Download] Target directory: {target_path}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("[Error] huggingface_hub is required. Run: pip install huggingface-hub")

    downloaded = snapshot_download(
        repo_id=model_id,
        local_dir=str(target_path),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    print(f"[Download] Successfully downloaded base model to: {downloaded}")
    return Path(downloaded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EmbeddingGemma-2 base weights")
    parser.add_argument("--target-dir", default="models/base", help="Target directory for weights")
    parser.add_argument("--repo-id", default="google/embeddinggemma-2", help="Hugging Face repository ID")
    args = parser.parse_args()

    download_base_model(args.target_dir, args.repo_id)
