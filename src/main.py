"""src/main.py – orchestrator launching the actual training script"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    root = Path(get_original_cwd())
    run_cfg_file = root / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_file.exists():
        raise FileNotFoundError(run_cfg_file)
    run_spec = OmegaConf.load(run_cfg_file)
    cfg = OmegaConf.merge(cfg, run_spec)

    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be trial|full")

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("[main] →", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()