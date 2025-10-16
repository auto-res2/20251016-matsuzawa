import json
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    repo_root = hydra.utils.get_original_cwd()
    os.chdir(repo_root)

    # --------------------------- Train -----------------------------------
    train_cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run_id}",
        f"results_dir={cfg.results_dir}",
        f"wandb.mode={cfg.wandb.mode}",
    ]
    if cfg.trial_mode:
        train_cmd.append("trial_mode=true")
    subprocess.run(train_cmd, check=True)

    # --------------------------- Evaluate --------------------------------
    eval_cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.evaluate",
        f"run={cfg.run_id}",
        f"results_dir={cfg.results_dir}",
        f"wandb.mode={cfg.wandb.mode}",
    ]
    if cfg.trial_mode:
        eval_cmd.append("trial_mode=true")
    subprocess.run(eval_cmd, check=True)

    print(json.dumps({"workflow": "completed", "run_id": cfg.run_id}))

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()