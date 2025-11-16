"""src/evaluate.py – independent evaluation & visualisation
-----------------------------------------------------------------
Fetches runs from WandB and creates:
• per-run metrics.json, learning-curve & confusion-matrix figures
• cross-run bar chart, box plot & significance tests
• comparison/aggregated_metrics.json  (spec-compliant)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind

PRIMARY_METRIC = "top1_accuracy"

# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------

def _fig_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{filename}.pdf"


# ----------------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=str)
    p.add_argument("run_ids", type=str, help='JSON list – e.g. "[\"run-1\", \"run-2\"]"')
    return p.parse_args()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    args = _parse()
    results_root = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)

    # global WandB config -----------------------------------------------------
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    entity, project = cfg.wandb.entity, cfg.wandb.project
    api = wandb.Api()

    collected_metrics: Dict[str, Dict[str, float]] = {}
    proposed_vals, baseline_vals = [], []  # for significance test

    for rid in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{rid}")
        except wandb.CommError as e:
            print(f"[WARN] cannot fetch {rid}: {e}")
            continue

        # per-run directory ---------------------------------------------------
        run_dir = results_root / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # history & summary ---------------------------------------------------
        hist_df: pd.DataFrame = run.history(samples=10000)  # full history
        summary: Dict = dict(run.summary)
        config: Dict = dict(run.config)

        # save raw metrics ----------------------------------------------------
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "config": config}, f, indent=2)
        print(run_dir / "metrics.json")

        # learning curve ------------------------------------------------------
        if "train_loss" in hist_df.columns:
            plt.figure()
            sns.lineplot(data=hist_df, y="train_loss", x=hist_df.index, label="train_loss")
            for col in ["val_loss", "epoch_train_loss"]:
                if col in hist_df.columns:
                    sns.lineplot(data=hist_df, y=col, x=hist_df.index, label=col)
            plt.title(f"{rid} loss curve")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.tight_layout()
            fpath = _fig_path(run_dir, f"{rid}_learning_curve")
            plt.savefig(fpath)
            plt.close()
            print(fpath)

        # confusion matrix figure (if present) -------------------------------
        if "confusion_matrix" in summary:
            cm = summary["confusion_matrix"]
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Pred")
            plt.ylabel("True")
            plt.title(f"{rid} confusion matrix")
            plt.tight_layout()
            fpath = _fig_path(run_dir, f"{rid}_confusion_matrix")
            plt.savefig(fpath)
            plt.close()
            print(fpath)

        # collect metrics -----------------------------------------------------
        for key, val in summary.items():
            if isinstance(val, (int, float)):
                collected_metrics.setdefault(key, {})[rid] = float(val)
        # separate lists for significance
        if "proposed" in rid:
            proposed_vals.append(collected_metrics.get(PRIMARY_METRIC, {}).get(rid, 0.0))
        if any(k in rid for k in ("baseline", "comparative")):
            baseline_vals.append(collected_metrics.get(PRIMARY_METRIC, {}).get(rid, 0.0))

    # ---------------------------------------------------------------------
    # Aggregated comparison figures
    # ---------------------------------------------------------------------
    comp_dir = results_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # bar chart --------------------------------------------------------------
    if PRIMARY_METRIC in collected_metrics:
        labels, values = zip(*collected_metrics[PRIMARY_METRIC].items())
        plt.figure(figsize=(max(6, 0.8 * len(labels)), 4))
        sns.barplot(x=list(labels), y=list(values))
        plt.xticks(rotation=45, ha="right")
        for idx, v in enumerate(values):
            plt.text(idx, v, f"{v:.3f}", ha="center", va="bottom")
        plt.title("Primary metric across runs")
        plt.ylabel(PRIMARY_METRIC)
        plt.tight_layout()
        fpath = _fig_path(comp_dir, "comparison_primary_metric_bar_chart")
        plt.savefig(fpath)
        plt.close()
        print(fpath)

        # box plot -----------------------------------------------------------
        plt.figure()
        sns.boxplot(y=list(values))
        plt.title("Distribution – primary metric")
        plt.tight_layout()
        fpath2 = _fig_path(comp_dir, "comparison_primary_metric_box_plot")
        plt.savefig(fpath2)
        plt.close()
        print(fpath2)

    # significance test ------------------------------------------------------
    sig_p = None
    if len(proposed_vals) > 1 and len(baseline_vals) > 1:
        _, sig_p = ttest_ind(proposed_vals, baseline_vals, equal_var=False)

    # best runs selection -----------------------------------------------------
    def _best(ids_subset: List[str]) -> Tuple[str, float]:
        best_id, best_val = "", -float("inf")
        for _id in ids_subset:
            val = collected_metrics.get(PRIMARY_METRIC, {}).get(_id, None)
            if val is not None and val > best_val:
                best_id, best_val = _id, val
        return best_id, best_val

    proposed_ids = [i for i in run_ids if "proposed" in i]
    baseline_ids = [i for i in run_ids if any(t in i for t in ("baseline", "comparative"))]
    best_prop_id, best_prop_val = _best(proposed_ids)
    best_base_id, best_base_val = _best(baseline_ids)

    # gap (% change) ----------------------------------------------------------
    if best_base_val != 0:
        gap = (best_prop_val - best_base_val) / best_base_val * 100.0
    else:
        gap = 0.0

    aggregated_json = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": collected_metrics,
        "best_proposed": {"run_id": best_prop_id, "value": best_prop_val},
        "best_baseline": {"run_id": best_base_id, "value": best_base_val},
        "gap": gap,
        "significance_pvalue": sig_p,
    }
    with open(comp_dir / "aggregated_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregated_json, f, indent=2)
    print(comp_dir / "aggregated_metrics.json")


if __name__ == "__main__":
    main()