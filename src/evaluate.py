import json
import os
from statistics import mean
from typing import Dict, List

import hydra
from omegaconf import DictConfig
import wandb

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _load_single(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_all_results(result_dir: str) -> List[Dict]:
    results = []
    for sub in os.listdir(result_dir):
        fp = os.path.join(result_dir, sub, "results.json")
        if os.path.isfile(fp):
            results.append(_load_single(fp))
    return results


def baseline_lookup(runs: List[Dict]) -> Dict[str, Dict]:
    base = {}
    for r in runs:
        if r["method_name"].startswith(("baseline", "comparative")) and r["dataset_name"] not in base:
            base[r["dataset_name"]] = r
    return base

# -----------------------------------------------------------------------------
# Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Flatten run config into root level
    from omegaconf import OmegaConf
    from hydra.utils import get_original_cwd
    
    # Ensure we're in the repository root
    os.chdir(get_original_cwd())
    
    OmegaConf.set_struct(cfg, False)
    if "run" in cfg:
        cfg = OmegaConf.merge(cfg, cfg.run)
    
    results_dir = os.path.abspath(cfg.results_dir)
    run_id = cfg.get("run_id", "default_run")
    trial_mode = bool(cfg.trial_mode)

    # ----------------------- Select runs -----------------------------------
    if trial_mode:
        result_path = os.path.join(results_dir, f"run_{run_id}", "results.json")
        if not os.path.isfile(result_path):
            raise FileNotFoundError(f"results.json not found for run {run_id}")
        runs = [_load_single(result_path)]
    else:
        runs = load_all_results(results_dir)

    baselines = baseline_lookup(runs) if not trial_mode else {}

    summary = {}
    for r in runs:
        ds = r["dataset_name"]
        base = baselines.get(ds)
        if base and base["run_id"] != r["run_id"]:
            rel_acc = (
                r["final_metrics"]["accuracy"] - base["final_metrics"]["accuracy"]
            ) / max(1e-12, base["final_metrics"]["accuracy"])
            rel_f1 = (
                r["final_metrics"]["f1_score"] - base["final_metrics"]["f1_score"]
            ) / max(1e-12, base["final_metrics"]["f1_score"])
        else:
            rel_acc, rel_f1 = 0.0, 0.0
        summary[r["run_id"]] = {
            "accuracy": r["final_metrics"]["accuracy"],
            "f1_score": r["final_metrics"]["f1_score"],
            "inference_time": r["final_metrics"]["inference_time"],
            "rel_improvement_accuracy": rel_acc,
            "rel_improvement_f1": rel_f1,
        }

    mean_accuracy = mean(v["accuracy"] for v in summary.values()) if summary else 0.0

    # ----------------------- WandB logging ---------------------------------
    if cfg.wandb.mode not in {"disabled", "none", None}:
        wb = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name="evaluation_summary" if not trial_mode else f"eval_{run_id}",
        )
        for rid, metrics in summary.items():
            wb.log({f"{rid}/{k}": v for k, v in metrics.items()})
        wb.log({"mean_accuracy": mean_accuracy})
        wb.finish()

    # ----------------------- Structured output -----------------------------
    print(json.dumps({"evaluation_summary": summary, "mean_accuracy": mean_accuracy}))


if __name__ == "__main__":
    main()