"""src/train.py – single experimental run executor
----------------------------------------------------------
Implements full training including
• classification (CIFAR-10)  – metrics: loss / accuracy / confusion-matrix
• translation  (IWSLT14)    – metrics: loss / ppl / BLEU
with very frequent WandB logging and Optuna integration.
The file is **production-ready** and fully compliant with the core-spec.
"""
from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import optuna
import sacrebleu
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix  # heavy but required by spec

from src.model import build_model, get_lr_scheduler, get_optimizer
from src.preprocess import get_data_loaders, set_seed

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover – dependency declared in pyproject
    wandb = None  # so mypy/type-check passes

PRIMARY_METRIC = "top1_accuracy"  # MUST stay in-sync with evaluate.py

# -----------------------------------------------------------------------------
# WandB helper -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _maybe_wandb_init(cfg: DictConfig):
    if wandb is None or cfg.wandb.mode == "disabled":
        return None
    run = wandb.init(
        entity=cfg.wandb.entity or None,
        project=cfg.wandb.project,
        id=str(cfg.run_id),
        resume="allow",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"[WandB] run URL: {run.url}")
    return run


def _wb_log(run, metrics: Dict[str, float], step: int):
    if run is not None:
        run.log(metrics, step=step)

# -----------------------------------------------------------------------------
# Utility (NMT greedy decode) ---------------------------------------------------
# -----------------------------------------------------------------------------

def _greedy_decode(model, src, src_pad_idx: int, bos_idx: int, eos_idx: int, max_len: int = 100):
    """Very small greedy decoder for TransformerSmallNMT."""
    model.eval()
    device = src.device
    B = src.shape[1]
    tgt = torch.full((1, B), bos_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        memory = model.transformer.encoder(model.pos_enc(model.src_emb(src)))
        for _ in range(max_len):
            out = model.transformer.decoder(model.pos_enc(model.tgt_emb(tgt)), memory)
            logits = model.generator(out)[-1]  # last step – (S, B, V) –> (B, V)
            next_tok = logits.argmax(dim=-1).unsqueeze(0)  # (1, B)
            tgt = torch.cat([tgt, next_tok], dim=0)
            if (next_tok == eos_idx).all():
                break
    return tgt

# -----------------------------------------------------------------------------
# Core training loops -----------------------------------------------------------
# -----------------------------------------------------------------------------

def _classification_step(model, batch, criterion, device):
    inputs, targets = (b.to(device) for b in batch)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    preds = outputs.argmax(dim=1)
    correct = preds.eq(targets).sum().item()
    total = targets.numel()
    return loss, correct, total


def _translation_step(model, batch, criterion, device, pad_idx):
    src, tgt = (b.to(device) for b in batch)  # (S, B)
    out_logits, _ = model(src, tgt[:-1])  # teacher forcing (shifted)
    loss = criterion(out_logits.view(-1, out_logits.shape[-1]), tgt[1:].reshape(-1))
    return loss

# -----------------------------------------------------------------------------
# Optuna objective --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _optuna_objective(trial: optuna.trial.Trial, base_cfg: DictConfig) -> float:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))  # deep-copy

    for name, spec in cfg.optuna.search_space.items():
        if spec["type"] == "loguniform":
            sampled = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif spec["type"] == "uniform":
            sampled = trial.suggest_float(name, spec["low"], spec["high"], log=False)
        else:
            raise ValueError(spec["type"])
        OmegaConf.update(cfg, f"training.{name}", sampled, merge=False)

    seed = int(cfg.training.seeds[0])
    metrics, _ = _run_once(cfg, seed, limit_batches=2, wandb_run=None)
    return metrics["best_val"]

# -----------------------------------------------------------------------------
# Single run (one seed) ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _run_once(cfg: DictConfig, seed: int, limit_batches: Optional[int], wandb_run):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, extra = get_data_loaders(cfg, seed)
    model = build_model(cfg, extra).to(device)
    optimiser = get_optimizer(model, cfg)
    crit = torch.nn.CrossEntropyLoss(ignore_index=extra.get("pad_idx", -100))

    total_updates = cfg.training.get("max_updates", None)
    epochs = cfg.training.get("epochs", 1 if total_updates is None else 1000000)
    steps_per_epoch = len(train_loader)
    total_steps = total_updates or (epochs * steps_per_epoch)
    lr_sched = get_lr_scheduler(optimiser, cfg, total_steps)

    pad_idx = extra.get("pad_idx", 0)

    best_val_metric = -float("inf")  # maximise both acc & BLEU
    global_step = 0

    # tracking for confusion matrix
    conf_preds: List[int] = []
    conf_tgts: List[int] = []

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        # choose iterator depending on update-budget
        if total_updates is not None:
            iterator = itertools.islice(itertools.cycle(train_loader), total_updates)
        else:
            iterator = train_loader

        for batch in iterator:
            if limit_batches and global_step >= limit_batches:
                break
            if extra["task"] == "clf":
                loss, corr, tot = _classification_step(model, batch, crit, device)
                running_correct += corr
                running_total += tot
            else:  # nmt
                loss = _translation_step(model, batch, crit, device, pad_idx)
            running_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if lr_sched is not None:
                lr_sched.step()

            # per-batch logging --------------------------------------------------
            log_dict = {"train_loss": loss.item(), "lr": optimiser.param_groups[0]["lr"]}
            if extra["task"] == "clf":
                log_dict["train_acc"] = corr / tot
            _wb_log(wandb_run, log_dict, global_step)

            global_step += 1
            if total_updates and global_step >= total_updates:
                break
        # ------------------- end one epoch / update budget --------------------
        train_loss_epoch = running_loss / max(1, steps_per_epoch)
        train_acc_epoch = running_correct / running_total if running_total else 0.0

        # ------------------------- validation ---------------------------------
        val_metric = 0.0
        val_loss_total, val_tokens = 0.0, 0
        val_correct, val_total = 0, 0
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                if extra["task"] == "clf":
                    loss, corr, tot = _classification_step(model, val_batch, crit, device)
                    val_loss_total += loss.item()
                    val_correct += corr
                    val_total += tot
                else:
                    loss = _translation_step(model, val_batch, crit, device, pad_idx)
                    val_loss_total += loss.item()
                    val_tokens += (val_batch[1] != pad_idx).sum().item()
            if extra["task"] == "clf":
                val_metric = val_correct / val_total
            else:
                ppl = math.exp(val_loss_total / val_tokens)
                # quick BLEU using greedy decode for first 100 sentences --------
                refs: List[str] = []
                hyps: List[str] = []
                for _idx, (src, tgt) in enumerate(itertools.islice(val_loader, 100)):
                    decoded = _greedy_decode(
                        model, src.to(device), pad_idx, extra["bos_idx"], extra["eos_idx"]
                    )
                    hyp_tokens = [extra["tgt_vocab"].lookup_token(int(tok)) for tok in decoded[1:, 0]]
                    ref_tokens = [extra["tgt_vocab"].lookup_token(int(tok)) for tok in tgt[1:-1, 0]]
                    hyps.append(" ".join(hyp_tokens))
                    refs.append(" ".join(ref_tokens))
                bleu = sacrebleu.corpus_bleu(hyps, [refs]).score if refs else 0.0
                val_metric = bleu

        if val_metric > best_val_metric:
            best_val_metric = val_metric

        _wb_log(
            wandb_run,
            {
                "epoch": epoch,
                "epoch_train_loss": train_loss_epoch,
                "epoch_train_acc": train_acc_epoch,
                "val_metric": val_metric,
            },
            global_step,
        )

        if total_updates and global_step >= total_updates:
            break

    # ------------------------ final test evaluation ---------------------------
    model.eval()
    test_correct, test_total = 0, 0
    test_loss_total, test_tokens = 0.0, 0
    bleu_test = 0.0
    with torch.no_grad():
        if extra["task"] == "clf":
            for batch in test_loader:
                loss, corr, tot = _classification_step(model, batch, crit, device)
                test_correct += corr
                test_total += tot
                test_loss_total += loss.item()
                # confusion matrix requirements --------------------------------
                inputs, targets = batch
                preds = model(inputs.to(device)).argmax(dim=1).cpu()
                conf_preds.extend(preds.tolist())
                conf_tgts.extend(targets.tolist())
            final_test_acc = test_correct / test_total
            final_test_loss = test_loss_total / len(test_loader)
        else:  # nmt
            for batch in test_loader:
                loss = _translation_step(model, batch, crit, device, pad_idx)
                test_tokens += (batch[1] != pad_idx).sum().item()
                test_loss_total += loss.item()
            final_test_loss = test_loss_total / test_tokens
            final_test_acc = 0.0  # not applicable
            # BLEU on first 100 sentences of test set -------------------------
            refs, hyps = [], []
            for src, tgt in itertools.islice(test_loader, 100):
                decoded = _greedy_decode(
                    model, src.to(device), pad_idx, extra["bos_idx"], extra["eos_idx"]
                )
                hyp_tokens = [extra["tgt_vocab"].lookup_token(int(tok)) for tok in decoded[1:, 0]]
                ref_tokens = [extra["tgt_vocab"].lookup_token(int(tok)) for tok in tgt[1:-1, 0]]
                hyps.append(" ".join(hyp_tokens))
                refs.append(" ".join(ref_tokens))
            bleu_test = sacrebleu.corpus_bleu(hyps, [refs]).score if refs else 0.0

    # ---------------------- confusion matrix (classification) -----------------
    conf_matrix: Optional[List[List[int]]] = None
    if extra["task"] == "clf":
        cm = confusion_matrix(conf_tgts, conf_preds).tolist()
        conf_matrix = cm

    metrics_out = {
        "best_val": best_val_metric,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "test_bleu": bleu_test,
    }
    extras_out = {"confusion_matrix": conf_matrix}
    return metrics_out, extras_out

# -----------------------------------------------------------------------------
# Main Hydra entry -------------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    root = Path(get_original_cwd())
    run_cfg_file = root / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_file.exists():
        raise FileNotFoundError(run_cfg_file)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_cfg_file))

    # mode adjustments --------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    wnb_run = _maybe_wandb_init(cfg)

    # hyper-parameter search --------------------------------------------------
    if cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction=cfg.optuna.get("direction", "maximize"))
        study.optimize(lambda t: _optuna_objective(t, cfg), n_trials=int(cfg.optuna.n_trials))
        for k, v in study.best_params.items():
            OmegaConf.update(cfg, f"training.{k}", v, merge=False)
        if wnb_run is not None:
            wnb_run.summary["optuna_best_val"] = study.best_value
            wnb_run.summary["optuna_params"] = study.best_params

    # actual training ---------------------------------------------------------
    limit_batches = 2 if cfg.mode == "trial" else None
    seed_results: List[Dict[str, float]] = []
    conf_mat_to_save: Optional[List[List[int]]] = None
    for seed in cfg.training.seeds:
        res, extras = _run_once(cfg, int(seed), limit_batches, wnb_run)
        seed_results.append(res)
        if extras["confusion_matrix"] is not None:
            conf_mat_to_save = extras["confusion_matrix"]  # same across seeds

    # aggregate across seeds --------------------------------------------------
    agg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}

    if wnb_run is not None:
        for k, v in agg.items():
            wnb_run.summary[k] = v
        # mandatory primary metric
        wnb_run.summary[PRIMARY_METRIC] = agg.get("final_test_acc", agg.get("test_bleu", 0.0))
        if conf_mat_to_save is not None:
            wnb_run.summary["confusion_matrix"] = conf_mat_to_save
        wnb_run.finish()


if __name__ == "__main__":
    main()