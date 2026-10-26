"""Predict with a trained LWC-former.

    python -m lwc_former.predict --model runs/lwc_former/seed_42/model.pt --data new.xlsx --out preds.csv

New spectra are preprocessed with the saved preprocessor and passed through the model;
outputs are returned in original units. If the target columns are present, R2/RMSE are printed.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from lwc_former.common.io import load_table
from lwc_former.common.metrics import score

from .model import LWCFormer
from .train import _predict


def predict_table(ckpt: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Return df's non-spectral columns plus <target>_predicted columns."""
    spectral, targets = ckpt["spectral_cols"], ckpt["target_cols"]
    missing = [c for c in spectral if c not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} spectral columns used in training are missing from the data "
                         f"(e.g. {missing[:5]}). Expected {spectral[0]}..{spectral[-1]} ({len(spectral)} bands).")
    X = df[spectral].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("New data contains NaN in spectral columns.")
    X = ckpt["preprocessor"].transform(X)

    model = LWCFormer(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    pred = ckpt["y_scaler"].inverse_transform(_predict(model, X, torch.device("cpu")))

    result = df.drop(columns=spectral).copy()
    for j, tgt in enumerate(targets):
        result[f"{tgt}_predicted"] = pred[:, j]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="model.pt written by lwc_former.train")
    ap.add_argument("--data", required=True, help="xlsx/csv with the same spectral columns")
    ap.add_argument("--out", required=True, help="output predictions csv")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    df = load_table(args.data)
    result = predict_table(ckpt, df)
    result.to_csv(args.out, index=False)
    print(f"[predict] wrote {args.out}: {len(result)} rows")

    targets = ckpt["target_cols"]
    if all(t in df.columns for t in targets):
        ok = df[targets].notna().all(axis=1).to_numpy()
        y_true = df.loc[ok, targets].to_numpy(dtype=float)
        y_pred = result.loc[ok, [f"{t}_predicted" for t in targets]].to_numpy()
        for tgt, m in score(y_true, y_pred, targets).items():
            print(f"{tgt:8s} R2={m['R2']:.4f} RMSE={m['RMSE']:.4f}")


if __name__ == "__main__":
    main()
