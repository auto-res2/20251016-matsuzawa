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

    # Flatten run config into root level
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    if "run" in cfg:
        cfg = OmegaConf.merge(cfg, cfg.run)

    # Extract the run config name from command line args
    run_config = None
    for arg in sys.argv:
        if arg.startswith("run="):
            run_config = arg.split("=", 1)[1]
            break
    if not run_config:
        run_config = cfg.get("run_id", "default_run")

    # --------------------------- Train -----------------------------------
    train_cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={run_config}",
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
        f"run={run_config}",
        f"results_dir={cfg.results_dir}",
        f"wandb.mode={cfg.wandb.mode}",
    ]
    if cfg.trial_mode:
        eval_cmd.append("trial_mode=true")
    subprocess.run(eval_cmd, check=True)

    print(json.dumps({"workflow": "completed", "run_id": cfg.get("run_id", run_config)}))

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()