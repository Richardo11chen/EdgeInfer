import os
import subprocess
import sys
from pathlib import Path

import torch


def main() -> int:
    print("python:", sys.version.replace("\n", " "))
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("device count:", torch.cuda.device_count())

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        print("nvidia-smi:", result.stdout.strip() or result.stderr.strip())
    except FileNotFoundError:
        print("nvidia-smi: not found")

    model_path = os.environ.get("EDGEINFER_MODEL_PATH")
    if not model_path:
        print("EDGEINFER_MODEL_PATH: not set")
        return 0

    path = Path(model_path)
    print("model path:", path)

    if not path.exists():
        print(f"model path does not exist: {path}")
        return 1

    safetensors = list(path.glob("*.safetensors"))
    print("safetensors files:", len(safetensors))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
