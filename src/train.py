import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb

# -----------------------------------------------------------------------------
# Local imports (absolute paths so PYTHONPATH is not required when executed via
# subprocess from an arbitrary CWD)
# -----------------------------------------------------------------------------
from src.model import build_model  # noqa: E402
from src.preprocess import prepare_dataloaders, prepare_tokenizer  # noqa: E402

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Deterministic behaviour as far as reasonably possible on CPU."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Primary metrics required by the paper."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(labels, preds, average="weighted"),
    }


def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)

# -----------------------------------------------------------------------------
# Core training logic
# -----------------------------------------------------------------------------

def run_training(cfg: DictConfig, trial_mode: bool, use_wandb: bool) -> Tuple[Dict, Dict]:
    """Run the train/val/test loop once and return final + per-epoch metrics."""

    set_seed()

    # --------------------------- Data -------------------------------------
    tokenizer = prepare_tokenizer(cfg)
    train_loader, val_loader, test_loader = prepare_dataloaders(cfg, tokenizer, trial_mode)

    # --------------------------- Model ------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    # --------------------------- Optimiser --------------------------------
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )

    total_iters = max(1, (1 if trial_mode else cfg.training.epochs) * len(train_loader))
    scheduler = optim.lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.0, total_iters=total_iters)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
    }

    # --------------------------- Epoch loop -------------------------------
    for epoch in range(1 if trial_mode else cfg.training.epochs):
        # ----------------------- Training -------------------------------
        model.train()
        tr_losses, tr_preds, tr_lbls = [], [], []
        for step, batch in enumerate(train_loader):
            if trial_mode and step > 1:
                break
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimiser.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            scheduler.step()

            tr_losses.append(loss.item())
            tr_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            tr_lbls.extend(labels.cpu().numpy())

            if use_wandb:
                wandb.log({"train/step_loss": loss.item()}, commit=False)

        train_metrics = compute_metrics(np.array(tr_preds), np.array(tr_lbls))
        train_loss = float(np.mean(tr_losses))

        # ----------------------- Validation -----------------------------
        model.eval()
        val_losses, val_preds, val_lbls = [], [], []
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if trial_mode and step > 1:
                    break
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                logits = model(**inputs)
                loss = criterion(logits, labels)

                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                val_lbls.extend(labels.cpu().numpy())

        val_metrics = compute_metrics(np.array(val_preds), np.array(val_lbls))
        val_loss = float(np.mean(val_losses))

        # --------------- Logging & history -----------------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_score"])

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_loss,
                "val/accuracy": val_metrics["accuracy"],
                "val/f1": val_metrics["f1_score"],
            })

    # --------------------------- Testing ----------------------------------
    model.eval()
    tst_preds, tst_lbls = [], []
    start_time = time.time()
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            if trial_mode and step > 1:
                break
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs)
            tst_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            tst_lbls.extend(labels.cpu().numpy())
    inference_time = (time.time() - start_time) / max(1, len(tst_lbls))

    test_metrics = compute_metrics(np.array(tst_preds), np.array(tst_lbls))
    final_metrics = {**test_metrics, "inference_time": inference_time}

    # Free memory (important for 500 MB cap on CI)
    del model
    torch.cuda.empty_cache()

    return final_metrics, history

# -----------------------------------------------------------------------------
# Optuna helpers
# -----------------------------------------------------------------------------

def apply_optuna_params(cfg: DictConfig, params: Dict):
    mapping = {
        "learning_rate": ("training", "learning_rate"),
        "weight_decay": ("training", "weight_decay"),
        "batch_size": ("dataset", "batch_size"),
        "adapter_size": ("model", "adapter", "adapter_size"),
    }
    for pname, val in params.items():
        node = cfg
        for k in mapping[pname][:-1]:
            node = node[k]
        node[mapping[pname][-1]] = val


def build_objective(base_cfg: DictConfig, trial_mode: bool):
    def objective(trial: optuna.Trial):
        cfg = copy.deepcopy(base_cfg)
        sampled = {}
        for param, spec in cfg.optuna.search_space.items():
            ptype = spec["type"].lower()
            if ptype == "loguniform":
                sampled[param] = trial.suggest_float(param, spec["low"], spec["high"], log=True)
            elif ptype == "uniform":
                sampled[param] = trial.suggest_float(param, spec["low"], spec["high"], log=False)
            elif ptype == "int":
                sampled[param] = trial.suggest_int(param, spec["low"], spec["high"], step=1)
            elif ptype == "categorical":
                sampled[param] = trial.suggest_categorical(param, spec["choices"])
            else:
                raise ValueError(f"Unsupported Optuna parameter type {ptype}")

        apply_optuna_params(cfg, sampled)
        cfg.wandb.mode = "disabled"  # disable WandB during HPO to save bandwidth
        final_metrics, hist = run_training(cfg, trial_mode=True, use_wandb=False)
        return hist["val_acc"][-1]

    return objective

# -----------------------------------------------------------------------------
# Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Ensure path is repository root regardless of Hydra's CWD change
    os.chdir(get_original_cwd())

    results_dir = os.path.abspath(cfg.results_dir)
    run_dir = os.path.join(results_dir, f"run_{cfg.run_id}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    trial_mode = bool(cfg.trial_mode)

    # ---------------------- Hyper-parameter optimisation -------------------
    if int(cfg.optuna.n_trials) > 0 and not trial_mode:
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(build_objective(cfg, trial_mode=True), n_trials=int(cfg.optuna.n_trials), timeout=int(cfg.optuna.timeout) if cfg.optuna.timeout else None, show_progress_bar=False)
        apply_optuna_params(cfg, study.best_params)
        save_json(os.path.join(run_dir, "optuna_best_params.json"), study.best_params)

    # ---------------------- WandB initialisation ---------------------------
    use_wandb = cfg.wandb.mode not in {"disabled", "none", None}
    wb_run = None
    if use_wandb:
        wb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            name=cfg.run_id,
        )
        # Immediately print URL so that CI workflow can capture it.
        print(json.dumps({"run_id": cfg.run_id, "wandb_url": wb_run.url}))

    # ---------------------- Training --------------------------------------
    final_metrics, history = run_training(cfg, trial_mode, use_wandb)

    # ---------------------- WandB final logging ---------------------------
    if use_wandb:
        wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
        wandb.finish()
        save_json(
            os.path.join(run_dir, "wandb_metadata.json"),
            {
                "wandb_entity": cfg.wandb.entity,
                "wandb_project": cfg.wandb.project,
                "wandb_run_id": wb_run.id if wb_run else None,
                "url": wb_run.url if wb_run else None,
            },
        )

    # ---------------------- Persist results -------------------------------
    result_payload = {
        "run_id": cfg.run_id,
        "method_name": cfg.method,
        "model_name": cfg.model.name,
        "dataset_name": cfg.dataset.name,
        "final_metrics": final_metrics,
        "training_history": history,
        "hyperparameters": {
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.dataset.batch_size,
            "epochs": 1 if trial_mode else cfg.training.epochs,
            "optimizer": cfg.training.optimizer,
            "weight_decay": cfg.training.weight_decay,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    save_json(os.path.join(run_dir, "results.json"), result_payload)

    # ---------------------- Console summary -------------------------------
    summary_out = {"run_id": cfg.run_id, **final_metrics, "wandb_url": wb_run.url if wb_run else None}
    print(json.dumps(summary_out))


if __name__ == "__main__":
    # Guarantee repository root on PYTHONPATH when executed directly.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()