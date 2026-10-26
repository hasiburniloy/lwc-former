# LWC-former

Spectral transformer that predicts leaf water content (relative water content, RWC) of blueberry plants from canopy hyperspectral reflectance (514–1000 nm).

Code for: Rahman, M. H., Busby, S., Ru, S., Hanif, S., Sanz-Saez, A., Zheng, J., Rehman, T. U. (2025). *Transformer-based hyperspectral image analysis for phenotyping drought tolerance in blueberries*. **Computers and Electronics in Agriculture** 228:109684, doi:[10.1016/j.compag.2024.109684](https://doi.org/10.1016/j.compag.2024.109684).

No data is included; see [Data format](#data-format).

## How it works

The preprocessed spectrum is cut into non-overlapping patches of 13 bands. Each patch is projected to an 8-dimensional token by its own linear map (patch-based dimensionality reduction, PDR: `H_i = b_i + X_i W_i` with a separate `W_i`, `b_i` per patch). Sinusoidal positional encoding is added, three post-norm transformer encoder layers (2 heads, feed-forward 32) mix the tokens, the tokens are flattened, and an MLP head (64, 64, 32; BatchNorm, ReLU, dropout 0.4) regresses RWC. Training is mini-batch (32) with Adam, weight decay 1e-3, a triangular cyclic learning rate and early stopping on validation RMSE.

Data pipeline (`common/`):

1. Stratified 85:15 test split, made first and never augmented or balanced.
2. Optional SMOTE-ENN balancing of the train-val set (`balance.enabled`, off by default; low vs high RWC at `balance.threshold`).
3. Offset/slope augmentation, 5 copies per spectrum (offset 6 % of the band std, slope ±5 %).
4. Group-aware 70:30 train/val split (copies of one plant stay on one side).
5. log10(1/R) → multiplicative scatter correction → mean-centring, fitted on the training split only.
6. Target standardised with the training mean and std; metrics are reported in original units.

```
spectrum (180) -> patches (13 x 13) -> PDR (13 x 8) -> +pos -> Encoder x 3 -> flatten -> MLP 64-64-32 -> RWC
                  each patch has its own W_i, b_i
```

## Install

```bash
pip install -r requirements.txt
```

PyTorch may need a platform-specific install; see [pytorch.org](https://pytorch.org).

## Data format

One row per plant, `.xlsx` or `.csv`: metadata columns (e.g. `Day`, `PlantID`, `Genotype`, `Treatment`), the target column `RWC`, then reflectance columns whose headers are wavelengths in nm (values in (0, 1]). Spectral columns are detected automatically as the numeric-named columns; `data.spectral_columns` and `data.wavelength_range` in `config.yaml` can restrict them. If your table has no `Treatment`/`Day` columns, set `data.stratify_columns: []`. Details in `data/README.md`.

## Train

Run from the repo root.

```bash
python -m lwc_former.train --config lwc_former/config.yaml --data path/to/data.xlsx --out runs/lwc_former
```

Any config value can be overridden: `--set model.patch_size=10 --set model.embed_dim=6 --set training.epochs=500`.

Test sets are small, so a single split is noisy. `--repeat-splits 5` runs the first 5 of the split seeds `[42, 7, 123, 2024, 99, 11, 3, 77, 500, 2025]` (or set `data.split_seed` to a list) and reports mean ± std over splits:

```bash
python -m lwc_former.train --config lwc_former/config.yaml --data path/to/data.xlsx --out runs/lwc_former --repeat-splits 5
```

Outputs per run (`runs/lwc_former/seed_<s>/`, or `split_<ss>/seed_<s>/` with several split seeds): `model.pt` (weights and fitted preprocessing), `metrics.json` (train / val / test R2 and RMSE, original units), `predictions_{train,val,test}.csv`, `metrics_test_by_Day.json`, `parity_plot.png`, `loss_curve.png`. `summary.json` has mean and std across all runs and a `per_split` block.

## Predict

```bash
python -m lwc_former.predict --model runs/lwc_former/seed_42/model.pt --data new_spectra.xlsx --out predictions.csv
```

The new file needs the same spectral columns as training. New spectra are preprocessed with the training statistics stored in `model.pt`. Output has the non-spectral columns plus `RWC_predicted`; R2/RMSE are printed if `RWC` is present.

## Configuration

All keys are documented in `lwc_former/config.yaml`. Main groups: `model` (patch_size, embed_dim, n_layers, n_heads, ff_dim, attn_dropout, head_hidden, head_dropout, pooling flatten|mean|cls, use_cls, tail drop|pad), `training` (batch_size, optimizer adam|adamw, weight_decay, loss mse|huber, scheduler cyclic|onecycle|cosine|none, base_lr, max_lr, step_size_up in epochs, grad_clip, epochs, early_stopping_patience), `data`, `balance`, `augmentation` (mode static|online, n_copies, offset_percent, slope_variation), `split`, `seeds`.

`augmentation.mode: static` (default, the paper's protocol) draws fixed jittered copies once. `online` skips the copies and jitters every mini-batch afresh; validation and test always use clean spectra.

## Differences from the paper

- SMOTE-ENN is applied after the test split and only to the train-val set, so the test set holds real spectra only. It is off by default because it did not improve held-out accuracy on our data; `--set balance.enabled=true` (with `balance.threshold` set for your RWC range) reproduces the paper's balancing.
- The train/val split is group-aware (`split.group_aware: false` restores a random split).
- Augmentation copies the target unchanged (no target jitter).
- The cyclic learning rate is stepped every batch, with `step_size_up` given in epochs.
- Early stopping on validation RMSE (patience 100), best weights restored.

## Citation

```bibtex
@article{rahman2025lwcformer,
  title   = {Transformer-based hyperspectral image analysis for phenotyping drought tolerance in blueberries},
  author  = {Rahman, Md. Hasibur and Busby, Savannah and Ru, Sushan and Hanif, Sajid and Sanz-Saez, Alvaro and Zheng, Jingyi and Rehman, Tanzeel U.},
  journal = {Computers and Electronics in Agriculture},
  volume  = {228},
  pages   = {109684},
  year    = {2025},
  doi     = {10.1016/j.compag.2024.109684}
}
```

## License

MIT, see `LICENSE`.
