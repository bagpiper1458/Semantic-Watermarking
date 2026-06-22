# WAVES-Core and StirMark-Compatible Benchmark

This benchmark evaluates five code-level methods:

- `Tree-Ring`
- `METR`
- `RingID`
- `HSTR` (reported as `HSFW-HSTR`)
- `HSQR` (reported as `HSFW-HSQR`)

HSTR and HSQR are the two HSFW variants.

## Runner

Run from the repository root:

```bash
python src/benchmark_robustness.py \
  --methods Tree-Ring METR RingID HSTR HSQR \
  --suite all \
  --waves_strengths 0.4 0.8 \
  --num_samples 20 \
  --attack_batch_size 12 \
  --result_dir benchmark_results/waves_stirmark_n20 \
  --seed 20260622 \
  --lpips \
  --resume
```

The runner uses the repository's 50-step DDIM inversion and native L1 decoder.
Identification searches the complete 2,048-key codebook. The vectorized search
was numerically checked against the original key-by-key functions for all methods.

## WAVES-Core Track

The implementation follows the official WAVES distortion ranges for:

- rotation
- resized crop
- erasing
- brightness
- contrast
- Gaussian blur
- Gaussian noise
- JPEG compression

The default pilot uses normalized strengths `0.4` and `0.8`. Use five strengths
for a larger run:

```bash
--waves_strengths 0.2 0.4 0.6 0.8 1.0
```

## StirMark-Compatible Track

The Linux runner applies a compact geometric subset using values from StirMark
Benchmark 4.0 `SMBsettings.ini`:

- rotation: 2 and 10 degrees
- center crop: 10% and 25%
- rescale: 75%
- affine shear: 0.05
- periodic row/column removal: frequency 10
- smooth random geometric distortion: amplitude 2 pixels

The official StirMark 4 repository is a legacy Windows-oriented benchmark. Its
binary was not executed on the Linux GPU server. Therefore, all outputs and plots
use the label `stirmark_compatible`; they must not be described as a
binary-identical StirMark benchmark result.

## Metrics

For each method and condition, the runner saves:

- ROC-AUC
- TPR at 0.1% FPR
- TPR at 1% FPR
- identification top-1 and top-5 accuracy
- 11-bit key accuracy
- PSNR, SSIM, and optional LPIPS to the pre-attack watermarked image

The benchmark applies the same stochastic attack seed to paired clean and
watermarked images. Clean attacked inversions are cached and reused across methods.

## Outputs

The result folder contains:

```text
summary.csv
REPORT.md
run_config.json
conditions.json
raw_<method>.npz
tpr_1fpr_heatmap.png
id_top1_heatmap.png
quality_vs_verification.png
```

The `raw_<method>.npz` files preserve per-image distances, predictions, and quality
metrics for later confidence intervals or alternative threshold analyses.
