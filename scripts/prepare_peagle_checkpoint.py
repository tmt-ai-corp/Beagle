#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a P-EAGLE checkpoint and patch it for SpecForge's SGLang runtime."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="amazon/GPT-OSS-20B-P-EAGLE",
        help="Hugging Face model id to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/peagle/GPT-OSS-20B-P-EAGLE"),
        help="Local directory that will contain the prepared checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the local config even if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.safetensors", "*.bin"],
    )

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {output_dir}")

    config = json.loads(config_path.read_text())
    original_architectures = config.get("architectures", [])
    if args.force or config.get("architectures") != ["LlamaForCausalLMPeagle"]:
        config["architectures"] = ["LlamaForCausalLMPeagle"]
    config["specforge_parallel_drafting"] = True
    config["specforge_original_architectures"] = original_architectures
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"Prepared P-EAGLE checkpoint at {output_dir}")
    print(f"Draft model path: {output_dir}")


if __name__ == "__main__":
    main()
