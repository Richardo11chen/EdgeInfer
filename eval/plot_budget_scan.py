from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot budget scan outputs")
    parser.add_argument("--summary-csv", default="outputs/required/8b_budget_scan/budget_scan_summary.csv")
    parser.add_argument("--output-dir", default="outputs/required/figures")
    args = parser.parse_args()

    rows = _load_rows(Path(args.summary_csv))
    budgets = [int(row["gpu_layer_budget"]) for row in rows]
    decode = [float(row["avg_decode_tokens_per_sec"]) for row in rows]
    memory = [float(row["peak_memory_mib"]) for row in rows]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot(budgets, decode, marker="o", color="#006d77")
    plt.xlabel("GPU Layer Budget")
    plt.ylabel("Decode Tokens/s")
    plt.title("Qwen3-8B Prefetch Decode Throughput vs Budget")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    decode_path = output_dir / "8b_budget_decode_tokens_per_sec.png"
    plt.savefig(decode_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(budgets, memory, marker="o", color="#bc6c25")
    plt.xlabel("GPU Layer Budget")
    plt.ylabel("Peak Memory (MiB)")
    plt.title("Qwen3-8B Prefetch Peak Memory vs Budget")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    memory_path = output_dir / "8b_budget_memory.png"
    plt.savefig(memory_path, dpi=160)
    plt.close()

    print(f"Wrote {decode_path}")
    print(f"Wrote {memory_path}")


if __name__ == "__main__":
    main()
