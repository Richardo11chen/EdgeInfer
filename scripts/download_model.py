#!/usr/bin/env python3

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}


ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "tokenizer*",
    "generation_config.json",
]


IGNORE_PATTERNS = [
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.onnx",
    "*.msgpack",
    "*.h5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen3 model files for EdgeInfer experiments."
    )

    parser.add_argument(
        "model",
        choices=MODEL_REPOS.keys(),
        help="Model to download.",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory used to store downloaded models. Default: ./models",
    )

    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face model revision, branch, tag, or commit hash.",
    )

    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Usually not needed for public Qwen models.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_id = MODEL_REPOS[args.model]
    local_dir = args.models_dir / args.model

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id}")
    print(f"Target directory: {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        revision=args.revision,
        token=args.token,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    safetensors_files = sorted(local_dir.glob("*.safetensors"))
    config_file = local_dir / "config.json"
    tokenizer_file = local_dir / "tokenizer.json"

    if not config_file.exists():
        raise SystemExit(f"Missing config.json in {local_dir}")

    if not tokenizer_file.exists():
        raise SystemExit(f"Missing tokenizer.json in {local_dir}")

    if not safetensors_files:
        raise SystemExit(f"No safetensors weights found in {local_dir}")

    print()
    print("Download completed.")
    print(f"Model path: {local_dir.resolve()}")
    print()
    print("Use it with:")
    print(f"  export EDGEINFER_MODEL_PATH={local_dir.resolve()}")


if __name__ == "__main__":
    main()