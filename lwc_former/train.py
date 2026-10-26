"""Train LWC-former (mini-batch) on the shared pipeline.

    python -m lwc_former.train --config lwc_former/config.yaml [--data x.xlsx] [--out runs/x]
                               [--set training.epochs=300 ...] [--repeat-splits N]

Loss is MSE (or Huber) on standardised targets; val RMSE (mean over targets) drives early
stopping and the best state is restored. The LR schedule is stepped every batch.
With augmentation.mode=online each mini-batch gets a fresh offset/slope jitter of the raw
spectra; validation and test use the clean features.
For each seed the output folder seed_<s>/ gets model.pt, metrics.json,
predictions_{train,val,test}.csv, metrics_test_by_Day.json, parity_plot.png and
loss_curve.png; summary.json holds mean/std across all runs plus per-split means.
data.split_seed may be a list (or use --repeat-splits N): every split seed x training
seed is run, in split_<ss>/seed_<s>/ folders when there is more than one split seed.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from lwc_former.common import SplitData, get_device, load_config, prepare_dataset, set_seed
from lwc_former.common.metrics import metrics_by_group, save_loss_curve, save_results, summarize_runs
from lwc_former.common.utils import apply_overrides, ensure_dir, repeat_splits, split_seeds

from .model import LWCFormer

MODEL_KEYS = ("patch_size", "embed_dim", "n_layers", "n_heads", "ff_dim", "attn_dropout",
              "head_hidden", "head_dropout", "pooling", "use_cls", "tail")


def _val_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean over targets of RMSE, in standardised units."""
    return float(np.sqrt(((pred - target) ** 2).mean(axis=0)).mean())


def _predict(model: torch.nn.Module, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).cpu().numpy())
    return np.vstack(out)


def _make_scheduler(opt, tcfg: Dict[str, Any], steps_per_epoch: int, epochs: int):
    name = str(tcfg.get("scheduler", "cyclic")).lower()
    base_lr, max_lr = float(tcfg.get("base_lr", 1e-5)), float(tcfg.get("max_lr", 1e-3))
    total = max(epochs * steps_per_epoch, 1)
    if name == "cyclic":
        up = max(int(round(float(tcfg.get("step_size_up", 25)) * steps_per_epoch)), 1)
        return torch.optim.lr_scheduler.CyclicLR(opt, base_lr=base_lr, max_lr=max_lr, step_size_up=up,
                                                 mode="triangular", cycle_momentum=False)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=base_lr)
    if name == "none":
        return None
    raise ValueError(f"unknown scheduler {name!r}; use cyclic, onecycle, cosine or none")


def run_one(cfg: Dict[str, Any], seed: int, out_dir, data: Optional[SplitData] = None,
            plots: bool = True, verbose: bool = True,
            split_seed: Optional[int] = None) -> Dict[str, Any]:
    """Train one seed and write its outputs. Returns the metrics dict.

    Pass `data` to reuse a prepared dataset (e.g. during tuning); it must have been
    built with the same seed. `split_seed` overrides the first data.split_seed.
    """
    out_dir = ensure_dir(out_dir)
    set_seed(seed)
    device = get_device()
    tcfg, mcfg = cfg["training"], cfg["model"]
    if data is None:
        data = prepare_dataset(cfg, seed, split_seed)
    used_split_seed = int(split_seed if split_seed is not None else split_seeds(cfg)[0])

    n_bands, n_targets = data.X_train.shape[1], data.Y_train.shape[1]
    model_kwargs = {key: mcfg[key] for key in MODEL_KEYS if key in mcfg}
    model = LWCFormer(n_bands, n_targets, **model_kwargs).to(device)

    lr0 = float(tcfg.get("base_lr", 1e-5)) if str(tcfg.get("scheduler", "cyclic")) != "none" \
        else float(tcfg.get("max_lr", 1e-3))
    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[str(tcfg.get("optimizer", "adam")).lower()]
    opt = opt_cls(model.parameters(), lr=lr0, weight_decay=float(tcfg.get("weight_decay", 1e-3)))

    bs = int(tcfg.get("batch_size", 32))
    n_tr = len(data.X_train)
    drop_last = n_tr % bs == 1  # BatchNorm cannot train on a batch of one
    steps_per_epoch = n_tr // bs if drop_last else int(np.ceil(n_tr / bs))
    epochs, patience = int(tcfg["epochs"]), int(tcfg["early_stopping_patience"])
    sched = _make_scheduler(opt, tcfg, steps_per_epoch, epochs)
    loss_name = str(tcfg.get("loss", "mse")).lower()
    loss_fn = {"mse": torch.nn.MSELoss(), "huber": torch.nn.HuberLoss()}[loss_name]
    clip = float(tcfg.get("grad_clip", 0) or 0)

    Xtr = torch.as_tensor(data.X_train, dtype=torch.float32, device=device)
    Ytr = torch.as_tensor(data.Y_train, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    online = data.augmenter is not None and data.X_train_raw is not None

    best_val, best_epoch, best_state, bad = float("inf"), 0, None, 0
    hist = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=gen).to(device)
        if drop_last:
            perm = perm[:steps_per_epoch * bs]
        run_loss, n_seen = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            if online:  # fresh jitter for this mini-batch, features recomputed from raw spectra
                xb = torch.as_tensor(data.augmenter.jitter(data.X_train_raw[idx.cpu().numpy()]),
                                     dtype=torch.float32, device=device)
            else:
                xb = Xtr[idx]
            loss = loss_fn(model(xb), Ytr[idx])
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched is not None:
                sched.step()
            run_loss += loss.item() * len(idx)
            n_seen += len(idx)
        train_loss = run_loss / n_seen

        pred_val = _predict(model, data.X_val, device)
        val_loss = float(((pred_val - data.Y_val) ** 2).mean())
        val_rmse = _val_rmse(pred_val, data.Y_val)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)

        if val_rmse < best_val:
            best_val, best_epoch, bad = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"[seed {seed}] epoch {epoch:4d} train_loss={train_loss:.4f} "
                  f"val_rmse={val_rmse:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        if bad >= patience:
            if verbose:
                print(f"[seed {seed}] early stop at epoch {epoch} (best {best_epoch}, val_rmse={best_val:.4f})")
            break
    if verbose:
        print(f"[seed {seed}] done in {time.time() - t0:.0f}s, best epoch {best_epoch}, val_rmse={best_val:.4f}")
    model.load_state_dict(best_state)

    inv = data.y_scaler.inverse_transform
    splits = {
        "train": (inv(data.Y_train), inv(_predict(model, data.X_train, device))),
        "val": (inv(data.Y_val), inv(_predict(model, data.X_val, device))),
        "test": (inv(data.Y_test), inv(_predict(model, data.X_test, device))),
    }
    metrics = save_results(splits, data.target_cols, out_dir, meta={"test": data.meta_test},
                           verbose=verbose, plot=plots)
    metrics["val_rmse_scaled"] = best_val
    metrics["best_epoch"] = best_epoch
    metrics["split_seed"] = used_split_seed

    group_col = (cfg.get("evaluation") or {}).get("group_by", "Day")
    if group_col in data.meta_test.columns:
        by_group = metrics_by_group(splits["test"][0], splits["test"][1],
                                    data.meta_test[group_col].to_numpy(), data.target_cols)
        with open(out_dir / f"metrics_test_by_{group_col}.json", "w") as f:
            json.dump(by_group, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if plots:
        save_loss_curve(hist, out_dir)

    torch.save({
        "state_dict": {key: v.cpu() for key, v in model.state_dict().items()},
        "model_name": "lwc_former",
        "model_kwargs": {"n_bands": n_bands, "n_targets": n_targets, **model_kwargs},
        "preprocessor": data.preprocessor, "y_scaler": data.y_scaler,
        "target_cols": data.target_cols, "spectral_cols": data.spectral_cols,
        "config": cfg, "seed": seed,
    }, out_dir / "model.pt")
    return metrics


def run_seeds(cfg: Dict[str, Any], out_dir, plots: bool = True) -> Dict[str, Any]:
    """Run every split seed x training seed; write summary.json (mean/std over all runs).

    With one split seed the folders are seed_<s>/; with several they are split_<ss>/seed_<s>/.
    summary.json also lists "split_seeds", "seeds" and a "per_split" block of per-split means.
    """
    out_dir = ensure_dir(out_dir)
    sseeds, seeds = split_seeds(cfg), [int(s) for s in cfg.get("seeds", [42])]
    runs_by_split: Dict[int, List[dict]] = {}
    for ss in sseeds:
        runs_by_split[ss] = []
        for seed in seeds:
            sub = out_dir / f"seed_{seed}" if len(sseeds) == 1 else out_dir / f"split_{ss}" / f"seed_{seed}"
            m = run_one(cfg, seed, sub, plots=plots, split_seed=ss)
            runs_by_split[ss].append({s: m[s] for s in ("train", "val", "test")})
    summary = summarize_runs(runs_by_split, seeds)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default="lwc_former/config.yaml")
    ap.add_argument("--data", default=None, help="override data.path")
    ap.add_argument("--out", default=None, help="override output_dir")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override a config value, e.g. --set training.epochs=300 (repeatable)")


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config, args.overrides)
    extra = []
    if args.data:
        extra.append(f"data.path={json.dumps(args.data)}")
    if args.out:
        extra.append(f"output_dir={json.dumps(args.out)}")
    cfg = apply_overrides(cfg, extra)
    if getattr(args, "repeat_splits", None):
        cfg = repeat_splits(cfg, args.repeat_splits)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    ap.add_argument("--repeat-splits", type=int, default=None, metavar="N",
                    help="evaluate on N test splits (first N of 42, 7, 123, 2024, 99, 11, 3, 77, 500, 2025)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    out_dir = Path(cfg["output_dir"])
    summary = run_seeds(cfg, out_dir)
    print(f"[done] {out_dir / 'summary.json'}")
    for tgt, m in summary["test"].items():
        print(f"test {tgt:8s} R2={m['R2']['mean']:.4f}+-{m['R2']['std']:.4f} "
              f"RMSE={m['RMSE']['mean']:.4f}+-{m['RMSE']['std']:.4f}")


if __name__ == "__main__":
    main()
