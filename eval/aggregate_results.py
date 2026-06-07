from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_RESULTS = (
    ("1_7b_baseline/edgeinfer_resident", "edgeinfer", "edgeinfer_result.json"),
    ("1_7b_baseline/vllm", "vllm", "vllm_result.json"),
    ("8b_naive_offload", "edgeinfer", "edgeinfer_result.json"),
    ("8b_prefetch_offload", "edgeinfer", "edgeinfer_result.json"),
    ("8b_vllm_offload", "vllm", "vllm_result.json"),
)


def _load_result(base_dir: Path, directory: str, expected_framework: str, filename: str) -> dict[str, object]:
    path = base_dir / directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required result: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["framework"] != expected_framework:
        raise ValueError(f"Unexpected framework in {path}: {data['framework']}")
    return data


def _load_budget_scan_rows(base_dir: Path) -> list[dict[str, object]]:
    path = base_dir / "8b_budget_scan" / "budget_scan_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing budget scan summary: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "experiment": "8b_budget_scan",
            "framework": row["framework"],
            "mode": row["mode"],
            "gpu_layer_budget": row["gpu_layer_budget"],
            "avg_ttft_ms": row["avg_ttft_ms"],
            "avg_decode_tokens_per_sec": row["avg_decode_tokens_per_sec"],
            "peak_memory_mib": row["peak_memory_mib"],
            "avg_prefill_latency_ms": row["avg_prefill_latency_ms"],
            "measurement_notes": row.get("measurement_notes", ""),
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate required experiment results")
    parser.add_argument("--base-dir", default="outputs/required")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    summary_rows: list[dict[str, object]] = []

    for directory, expected_framework, filename in EXPECTED_RESULTS:
        data = _load_result(base_dir, directory, expected_framework, filename)
        summary_rows.append(
            {
                "experiment": directory,
                "framework": data["framework"],
                "mode": data.get("mode", ""),
                "gpu_layer_budget": data.get("gpu_layer_budget", ""),
                "avg_prefill_latency_ms": data["avg_prefill_latency_ms"],
                "avg_ttft_ms": data["avg_ttft_ms"],
                "avg_decode_tokens_per_sec": data["avg_decode_tokens_per_sec"],
                "peak_memory_mib": data["peak_memory_mib"],
                "measurement_notes": data.get("measurement_notes", ""),
            }
        )

    summary_rows.extend(_load_budget_scan_rows(base_dir))

    summary_csv = base_dir / "summary.csv"
    summary_md = base_dir / "summary.md"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "framework",
                "mode",
                "gpu_layer_budget",
                "avg_prefill_latency_ms",
                "avg_ttft_ms",
                "avg_decode_tokens_per_sec",
                "peak_memory_mib",
                "measurement_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Required Experiment Summary",
        "",
        "| experiment | framework | mode | gpu_layer_budget | avg_prefill_latency_ms | avg_ttft_ms | avg_decode_tokens_per_sec | peak_memory_mib | measurement_notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {experiment} | {framework} | {mode} | {gpu_layer_budget} | {avg_prefill_latency_ms} | "
            "{avg_ttft_ms} | {avg_decode_tokens_per_sec} | {peak_memory_mib} | {measurement_notes} |".format(
                **row
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
