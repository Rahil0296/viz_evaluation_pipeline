"""
batch_regrade.py  [v2 - corrected for actual folder structure]
----------------
Re-grades all existing runs using the fixed metrics.py
(K-Means based color extraction instead of pixel frequency sampling).

Place this file in your project root (same level as grade_manual.py).
Run with:
    python batch_regrade.py

What it covers (54 total grades):
    sleep_health_results:      gpt5.2 + gemini3_pro x poor + rich x 3 runs = 12
    defi_anomalies_results:    gpt5.2 + gemini3_pro x poor + rich x 3 runs = 12
    titanic_results/dashboard: gpt5.2 + gemini3_pro x poor + rich x 3 runs = 12
    titanic_results/survival:  gpt5.2 + gemini3_pro x poor + rich x 3 runs = 12
    customer_segments/gpt5.2:  poor + rich x 3 runs                        =  6

What it SKIPS:
    customer_segments/gemini3_pro  -> run folders exist but are empty
    ev_charging                    -> no run folders (only single viz1_ files)
"""

import json
import sys
import os
from pathlib import Path

# PATH FIX — metrics.py lives in src/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from metrics import calculate_metrics_for_visualization
# ── Run specs: (results_folder, model, viz_sub_or_None, file_prefix) ─────────
#
# file_prefix is the part before _viz_<context>_context_output.png
# defi uses "viz1" (image = viz1_poor_context_output.png)
# others use a short dataset name (sleep, customer, dashboard, survival)

RUN_SPECS = [
    # results_folder               model           sub          prefix
    ("sleep_health_results",       "gpt5.2",       None,        "sleep"),
    ("sleep_health_results",       "gemini3_pro",  None,        "sleep"),
    ("defi_anomalies_results",     "gpt5.2",       None,        "viz1"),
    ("defi_anomalies_results",     "gemini3_pro",  None,        "viz1"),
    ("titanic_results",            "gpt5.2",       "dashboard", "dashboard"),
    ("titanic_results",            "gpt5.2",       "survival",  "survival"),
    ("titanic_results",            "gemini3_pro",  "dashboard", "dashboard"),
    ("titanic_results",            "gemini3_pro",  "survival",  "survival"),
    ("customer_segments_results",  "gpt5.2",       None,        "customer"),
]

CONTEXTS = ["poor", "rich"]
RUNS     = ["run1", "run2", "run3"]


def regrade_run(run_folder: Path, prefix: str, context: str) -> bool:
    image_name = f"{prefix}_viz_{context}_context_output.png"
    code_name  = f"{prefix}_viz_{context}_context.py"

    image_path = run_folder / image_name
    code_path  = run_folder / code_name

    # Fallback to any .png/.py if exact name not found
    if not image_path.exists():
        pngs = list(run_folder.glob("*.png"))
        image_path = pngs[0] if pngs else None

    if image_path is None or not image_path.exists():
        print(f"    [SKIP] No image found in {run_folder}")
        return False

    code_content = None
    if code_path.exists():
        try:
            code_content = code_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"    [WARN] Could not read code: {e}")
    else:
        pys = list(run_folder.glob("*.py"))
        if pys:
            try:
                code_content = pys[0].read_text(encoding="utf-8")
            except Exception:
                pass

    try:
        results = calculate_metrics_for_visualization(
            image_path=str(image_path),
            code=code_content,
            original_data_path=None
        )
    except Exception as e:
        print(f"    [ERROR] {e}")
        return False

    # Overwrite the existing metrics JSON
    out_file = run_folder / f"viz1_{context}_context_output_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    de      = results.get("color_delta_e", {}).get("mean_delta_e")
    entropy = results.get("visual_entropy")
    exec_ok = results.get("code_accuracy", {}).get("execution_success", "N/A")
    de_str  = f"{de:.3f}"      if de      is not None else "N/A"
    ent_str = f"{entropy:.3f}" if entropy is not None else "N/A"
    print(f"    [OK]  ΔE={de_str}, Entropy={ent_str}, Exec={exec_ok}")
    return True


def main():
    root    = Path(".")
    total   = 0
    success = 0
    skipped = 0

    for (results_folder, model, sub, prefix) in RUN_SPECS:
        base = root / results_folder / model
        if sub:
            base = base / sub

        for context in CONTEXTS:
            for run in RUNS:
                run_folder = base / f"{run}_{context}_context"
                total += 1

                label = "/".join(filter(None, [results_folder, model, sub, f"{run}_{context}_context"]))

                if not run_folder.exists():
                    print(f"  [MISSING] {label}")
                    skipped += 1
                    continue

                if not any(run_folder.iterdir()):
                    print(f"  [EMPTY]   {label}")
                    skipped += 1
                    continue

                print(f"  Grading: {label}")
                ok = regrade_run(run_folder, prefix, context)
                if ok:
                    success += 1
                else:
                    skipped += 1

    print(f"\n{'='*60}")
    print(f"Done.  {success}/{total} runs re-graded.  {skipped} skipped/failed.")
    print(f"{'='*60}")

    if success > 0:
        print("\nNext — re-run avg_metrics.py for each completed combo:\n")
        simple = [
            ("sleep_health_results",    "gpt5.2",      "poor"),
            ("sleep_health_results",    "gpt5.2",      "rich"),
            ("sleep_health_results",    "gemini3_pro", "poor"),
            ("sleep_health_results",    "gemini3_pro", "rich"),
            ("defi_anomalies_results",  "gpt5.2",      "poor"),
            ("defi_anomalies_results",  "gpt5.2",      "rich"),
            ("defi_anomalies_results",  "gemini3_pro", "poor"),
            ("defi_anomalies_results",  "gemini3_pro", "rich"),
            ("customer_segments_results", "gpt5.2",    "poor"),
            ("customer_segments_results", "gpt5.2",    "rich"),
        ]
        for rf, m, ctx in simple:
            print(f"  python avg_metrics.py {rf} {m} {ctx}")

        print()
        print("  # Titanic (avg_metrics uses results_folder=titanic_results/<model>, model=sub):")
        for model in ["gpt5.2", "gemini3_pro"]:
            for sub in ["dashboard", "survival"]:
                for ctx in ["poor", "rich"]:
                    print(f"  python avg_metrics.py titanic_results/{model} {sub} {ctx}")


if __name__ == "__main__":
    main()