"""
avg_metrics.py
--------------
Computes average metrics across multiple runs for a given dataset/model/context.

Usage:
    python avg_metrics.py <results_folder> <dataset> <model> <context>

Example:
    python avg_metrics.py defi_anomalies_results gpt5.2 poor
    python avg_metrics.py defi_anomalies_results gemini3_pro rich

This will:
- Look for run1/, run2/, run3/ (etc.) folders inside <results_folder>/<model>/
- Read viz1_<context>_context_output_metrics.json from each run folder
- Compute averages for mean_delta_e, visual_entropy, distinguishability_ratio
- Save avg_metrics_<context>_context.json in <results_folder>/<model>/
"""

import json
import os
import sys
from pathlib import Path


def avg_metrics(results_folder: str, model: str, context: str):
    base_path = Path(results_folder) / model
    
    if not base_path.exists():
        print(f"ERROR: Path not found: {base_path}")
        sys.exit(1)

    # Find all run folders matching run*_<context>_context
    run_folders = sorted([
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith("run") and f"{context}_context" in d.name
    ])

    if not run_folders:
        print(f"ERROR: No run folders found in {base_path}")
        sys.exit(1)

    print(f"\nFound {len(run_folders)} run(s): {[r.name for r in run_folders]}")

    # Collect metrics from each run
    metrics_file = f"viz1_{context}_context_output_metrics.json"
    
    all_mean_delta_e = []
    all_entropy = []
    all_distinguishability = []
    all_exec = []
    all_semantic = []
    run_data = []

    for run_folder in run_folders:
        metrics_path = run_folder / metrics_file
        
        if not metrics_path.exists():
            print(f"WARNING: Missing {metrics_path} — skipping this run")
            continue

        with open(metrics_path, "r") as f:
            data = json.load(f)

        mean_de = data["color_delta_e"]["mean_delta_e"]
        entropy = data["visual_entropy"]
        dist_ratio = data["color_delta_e"]["distinguishability_ratio"]
        exec_success = data["code_accuracy"]["execution_success"]

        all_mean_delta_e.append(mean_de)
        all_entropy.append(entropy)
        all_distinguishability.append(dist_ratio)
        all_exec.append(exec_success)

        run_data.append({
            "run": run_folder.name,
            "mean_delta_e": mean_de,
            "visual_entropy": entropy,
            "distinguishability_ratio": dist_ratio,
            "execution_success": exec_success
        })

        print(f"  {run_folder.name}: ΔE={mean_de:.3f}, Entropy={entropy:.3f}, Exec={exec_success}")

    if not all_mean_delta_e:
        print("ERROR: No valid metrics files found.")
        sys.exit(1)

    # Compute averages
    n = len(all_mean_delta_e)
    avg = {
        "dataset_model_context": f"{results_folder} | {model} | {context} context",
        "num_runs": n,
        "per_run_results": run_data,
        "averages": {
            "mean_delta_e": round(sum(all_mean_delta_e) / n, 4),
            "visual_entropy": round(sum(all_entropy) / n, 4),
            "distinguishability_ratio": round(sum(all_distinguishability) / n, 4),
            "execution_pass_rate": round(sum(1 for e in all_exec if e) / n, 4)
        },
        "raw_values": {
            "mean_delta_e_per_run": all_mean_delta_e,
            "visual_entropy_per_run": all_entropy,
            "distinguishability_ratio_per_run": all_distinguishability
        }
    }

    # Save output
    out_file = base_path / f"avg_metrics_{context}_context.json"
    with open(out_file, "w") as f:
        json.dump(avg, f, indent=2)

    print(f"\n--- AVERAGES ({n} runs) ---")
    print(f"  Mean ΔE:               {avg['averages']['mean_delta_e']}")
    print(f"  Visual Entropy:        {avg['averages']['visual_entropy']}")
    print(f"  Distinguishability:    {avg['averages']['distinguishability_ratio']}")
    print(f"  Execution Pass Rate:   {avg['averages']['execution_pass_rate']}")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python avg_metrics.py <results_folder> <model> <context>")
        print("Example: python avg_metrics.py defi_anomalies_results gpt5.2 poor")
        sys.exit(1)

    results_folder = sys.argv[1]
    model = sys.argv[2]
    context = sys.argv[3]

    avg_metrics(results_folder, model, context)