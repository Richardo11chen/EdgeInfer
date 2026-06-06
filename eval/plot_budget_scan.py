from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_budget_data(base_dir: str) -> dict[str, dict]:
    label_order = ["resident", "naive", "1", "2", "4", "8", "12", "16", "28"]
    mapping: dict[str, dict] = {}
    for name in label_order:
        if name in ("resident", "naive"):
            fname = f"{base_dir}/{name}.json"
        else:
            fname = f"{base_dir}/prefetch_{name}.json"
        p = Path(fname)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        mapping[name] = data
    return mapping


def _plot_model(axs, data: dict[str, dict], title: str, color: str):
    budgets = []
    decode_speeds = []
    ttfts = []
    memories = []
    copy_times = []
    labels = []

    label_map = {"resident": "Res", "naive": "Nv"}
    for name, d in data.items():
        budgets.append(name)
        labels.append(label_map.get(name, f"P{name}"))
        decode_speeds.append(d["decode_tokens_per_sec"])
        ttfts.append(d.get("ttft_ms", d.get("prefill_latency_ms", 0)))
        memories.append(d["peak_memory_mib"])
        copy_times.append(d["total_copy_ms"])

    x = range(len(budgets))

    axs[0].plot(x, decode_speeds, "-o", color=color)
    axs[0].set_ylabel("Decode (tok/s)")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, fontsize=8)

    ax2 = axs[0].twinx()
    ax2.bar(x, memories, alpha=0.3, color=color)
    ax2.set_ylabel("Peak Memory (MiB)")

    axs[1].plot(x, ttfts, "-s", color=color)
    axs[1].set_ylabel("TTFT (ms)")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, fontsize=8)

    axs[2].plot(x, copy_times, "-^", color=color)
    axs[2].set_ylabel("Total Copy (ms)")
    axs[2].set_xlabel("Offload Strategy / GPU Layer Budget")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, fontsize=8)

    for ax in axs:
        ax.set_title(title, fontsize=11)


def main() -> None:
    for model, model_label, color in [
        ("1.7B", "Qwen3-1.7B (28 layers)", "#2196F3"),
        ("8B", "Qwen3-8B (36 layers)", "#FF5722"),
    ]:
        data = _load_budget_data(f"outputs/budget_scan/{model}")
        if not data:
            print(f"No data for {model}")
            continue

        fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        _plot_model(axs, data, model_label, color)
        plt.tight_layout()
        fig.savefig(f"outputs/budget_scan_{model}.png", dpi=150)
        print(f"Saved outputs/budget_scan_{model}.png")

    # Combined decode speed comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, color, marker in [
        ("1.7B", "#2196F3", "o"),
        ("8B", "#FF5722", "s"),
    ]:
        data = _load_budget_data(f"outputs/budget_scan/{model}")
        if not data:
            continue
        x_vals = []
        y_vals = []
        for name, d in data.items():
            b = int(name) if name not in ("resident", "naive") else (0 if name == "resident" else -1)
            x_vals.append(b)
            y_vals.append(d["decode_tokens_per_sec"])
        ax.plot(x_vals, y_vals, f"-{marker}", color=color, label=model)
    ax.set_xlabel("GPU Layer Budget (0 = resident, -1 = naive)")
    ax.set_ylabel("Decode Speed (tok/s)")
    ax.set_title("Decode Speed vs GPU Layer Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/budget_scan_combined.png", dpi=150)
    print("Saved outputs/budget_scan_combined.png")


if __name__ == "__main__":
    main()
