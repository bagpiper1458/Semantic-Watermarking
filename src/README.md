# Source Entry Points

This directory contains the runnable experiment code. Run commands from `src/` unless you use the wrapper scripts in `../scripts/`.

## Main Pipeline

| File | Purpose |
| --- | --- |
| `generate.py` | Generate clean and watermarked images for multi-key semantic watermarking. |
| `metric.py` | Compute CLIP score for all datasets and FID for COCO. |
| `diff_attack/diff_wm_attack.py` | Run diffusion-regeneration attack images used by `detect.py`. |
| `detect.py` | Evaluate verification and identification across clean, distortion, diffusion, crop, and rotation cases. |

Example:

```bash
python generate.py --wm_type HSQR --dataset_id coco --output_dir outputs
python metric.py --wm_type HSQR --dataset_id coco --output_dir outputs
python diff_attack/diff_wm_attack.py --wm_type HSQR --dataset_id coco --output_dir outputs
python detect.py --wm_type HSQR --dataset_id coco --output_dir outputs --no_save_inverted
```

## Fixed-Key Experiments

| File | Purpose |
| --- | --- |
| `generate_fixed_key.py` | Generate fixed-key watermarked images while symlinking clean images and/or pattern lists from an existing multi-key run. |
| `generate_fixed_key_fast.py` | Generate fixed-key watermarked images and create/reuse the method codebook in the fixed-key output folder. |
| `detect_avg_attack.py` | Evaluate clean-only or externally attacked fixed-key outputs with optional folder overrides. |

Example:

```bash
python generate_fixed_key_fast.py \
  --wm_type Tree-Ring \
  --dataset_id coco \
  --fixed_key 0 \
  --output_dir outputs_fixedkey_k0 \
  --shared_clean_dir ../outputs/coco/Tree-Ring/img_pil

python detect_avg_attack.py \
  --wm_type Tree-Ring \
  --dataset_id coco \
  --output_dir outputs_fixedkey_k0 \
  --only_clean \
  --max_trials 1000 \
  --no_save_inverted
```

## Method Names

Accepted `--wm_type` values:

```text
Tree-Ring
RingID
HSTR
HSQR
METR
```

## Output Layout

Each run writes under:

```text
<output_dir>/<dataset_id>/<wm_type>/
```

Generated image folders, latent inversions, codebooks, and metric arrays are experiment artifacts. They are ignored by Git and should be regenerated on the GPU machine when needed.
