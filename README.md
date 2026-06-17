# Semantic Watermarking Experiment Code

This repository contains the experiment code used for semantic watermarking experiments in a survey-paper project. It is not an official method release. The code is organized around four experiment groups:

1. Image quality: CLIP score and FID.
2. Verification robustness: without image editing and with image editing.
3. Identification robustness: without image editing and with image editing.
4. Averaging steganalysis under fixed-key deployment: `N=4000`.

The original base project is acknowledged at the end of this README.

## Supported Methods

The current scripts support:

```text
Tree-Ring
RingID
HSTR
HSQR
METR
```

`METR` and the fixed-key / averaging-evaluation workflow are local experiment additions in this fork.

## Installation

```bash
git clone https://github.com/bagpiper1458/Semantic-Watermarking.git
cd Semantic-Watermarking

conda create -n sfw python=3.10 -y
conda activate sfw
bash install.sh
```

If dependency versions are difficult to reproduce, compare against `requirements-lock.txt`.

## Dataset Setup

Place the prompt datasets under:

```text
src/text_dataset/
```

The supported dataset names are:

```text
coco
Gustavo
DB1k
```

FID is computed only for `coco`, because the code expects the COCO reference image folder returned by `get_text_dataset()`.

## Common Variables

The examples below use bash variables. Change them for the method or dataset you want to run.

```bash
cd src

export WM_TYPE=HSQR
export DATASET_ID=coco
export OUTPUT_DIR=outputs
```

All generated files are written to:

```text
<output_dir>/<dataset_id>/<wm_type>/
```

Large experiment outputs are ignored by Git. Do not commit `outputs/`, `outputs_fixedkey*/`, generated images, latent inversions, pattern lists, or `.npz` metric arrays.

## 1. Image Quality: CLIP Score and FID

First generate clean and watermarked images:

```bash
python generate.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}"
```

Then compute CLIP score and, for COCO, FID:

```bash
python metric.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}"
```

The output files are:

```text
outputs/<dataset_id>/<wm_type>/clip.npz
outputs/<dataset_id>/<wm_type>/fid.npz        # COCO only
```

`metric.py` prints the mean CLIP scores for clean and watermarked images. For COCO it also prints FID for clean and watermarked images.

## 2. Verification Robustness

Verification asks whether a given image is watermarked by a claimed key. The main reported metrics are AUC and TPR@1%FPR.

### 2.1 Without Image Editing

Use the clean-only evaluator. This runs DDIM inversion on clean and watermarked images without applying image-editing attacks:

```bash
python detect_avg_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --only_clean \
  --skip_identification \
  --run_tag verify_clean \
  --max_trials 1000 \
  --no_save_inverted
```

Read:

```text
outputs/<dataset_id>/<wm_type>/metrics_<wm_type>_verify_clean.txt
```

### 2.2 With Image Editing

`detect.py` evaluates verification under the built-in editing and regeneration attacks. It includes:

```text
Clean, Brightness, Contrast, JPEG, Blur, Noise, BM3D,
VAE-B, VAE-C, Rotation, CS75, Diff, CC, RC
```

The `Diff` case requires precomputed diffusion-regeneration attack images:

```bash
python diff_attack/diff_wm_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}"
```

Then run detection:

```bash
python detect.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --no_save_inverted
```

Read the "Verification Metrics" table in:

```text
outputs/<dataset_id>/<wm_type>/metrics_<wm_type>.txt
outputs/<dataset_id>/<wm_type>/verify-l1.npz
```

## 3. Identification Robustness

Identification asks whether the detector can recover the correct key ID among the 2048 candidate keys. The main reported metrics are PerfectMatch and BitAcc.

### 3.1 Without Image Editing

Use the clean-only evaluator without `--skip_identification`:

```bash
python detect_avg_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --only_clean \
  --run_tag identify_clean \
  --max_trials 1000 \
  --no_save_inverted
```

Read the "Identification Metrics" table in:

```text
outputs/<dataset_id>/<wm_type>/metrics_<wm_type>_identify_clean.txt
outputs/<dataset_id>/<wm_type>/identify-acc_identify_clean.npz
```

### 3.2 With Image Editing

Use the same edited-image run as verification robustness:

```bash
python diff_attack/diff_wm_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}"

python detect.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --no_save_inverted
```

Read the "Identification Metrics" table in:

```text
outputs/<dataset_id>/<wm_type>/metrics_<wm_type>.txt
outputs/<dataset_id>/<wm_type>/identify-acc.npz
```

## 4. Averaging Steganalysis: Fixed-Key Deployment, N=4000

This experiment evaluates large-scale fixed-key deployment. The intended setting is that all watermarked images are generated with the same key, then an averaging steganalysis stage estimates a pattern from `N=4000` images and applies removal or forgery.

### 4.1 Generate Fixed-Key Images

The fixed-key generator reuses the clean images from a standard multi-key run.

First make sure the clean image folder exists:

```bash
export WM_TYPE=Tree-Ring
export DATASET_ID=coco

python generate.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir outputs
```

Then generate fixed-key watermarked images:

```bash
python generate_fixed_key_fast.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --fixed_key 0 \
  --output_dir outputs_fixedkey_k0 \
  --shared_clean_dir "../outputs/${DATASET_ID}/${WM_TYPE}/img_pil"
```

This creates:

```text
outputs_fixedkey_k0/<dataset_id>/<wm_type>/img_pil
outputs_fixedkey_k0/<dataset_id>/<wm_type>/img_pil_wm
outputs_fixedkey_k0/<dataset_id>/<wm_type>/identify_gt_indices_*.npy
outputs_fixedkey_k0/<dataset_id>/<wm_type>/pattern_list-2048.pt
```

### 4.2 Prepare Averaging Attack Images

The detector expects the `N=4000` averaging-removal and averaging-forgery outputs to be available under:

```text
outputs_fixedkey_k0/coco/<wm_type>/avg_attack/removal/
outputs_fixedkey_k0/coco/<wm_type>/avg_attack/forgery/
```

The folder layout used by the experiments is:

```text
avg_attack/removal/Subtract_Pattern___[Greybox]_
avg_attack/removal/Subtract_Pattern___[Blackbox]
avg_attack/forgery/No_Operation_______[Greybox]_
avg_attack/forgery/Add_Pattern________[Greybox]_
avg_attack/forgery/No_Operation_______[Blackbox]
avg_attack/forgery/Add_Pattern________[Blackbox]
```

Those folders should contain images named by index, for example `0.png`, `1.png`, ..., matching the original dataset indices.

### 4.3 Evaluate Averaging Removal

Grey-box removal:

```bash
python detect_avg_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id coco \
  --output_dir outputs_fixedkey_k0 \
  --override_wm_dir "../outputs_fixedkey_k0/coco/${WM_TYPE}/avg_attack/removal/Subtract_Pattern___[Greybox]_" \
  --only_clean \
  --skip_identification \
  --run_tag fixedKey_removal_GB_N4000 \
  --max_trials 4000 \
  --no_save_inverted
```

Black-box removal:

```bash
python detect_avg_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id coco \
  --output_dir outputs_fixedkey_k0 \
  --override_wm_dir "../outputs_fixedkey_k0/coco/${WM_TYPE}/avg_attack/removal/Subtract_Pattern___[Blackbox]" \
  --only_clean \
  --skip_identification \
  --run_tag fixedKey_removal_BB_N4000 \
  --max_trials 4000 \
  --no_save_inverted
```

Read:

```text
outputs_fixedkey_k0/coco/<wm_type>/metrics_<wm_type>_fixedKey_removal_GB_N4000.txt
outputs_fixedkey_k0/coco/<wm_type>/metrics_<wm_type>_fixedKey_removal_BB_N4000.txt
```

### 4.4 Evaluate Averaging Forgery

Grey-box forgery with identification:

```bash
python detect_avg_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id coco \
  --output_dir outputs_fixedkey_k0 \
  --override_clean_dir "../outputs_fixedkey_k0/coco/${WM_TYPE}/avg_attack/forgery/No_Operation_______[Greybox]_" \
  --override_wm_dir "../outputs_fixedkey_k0/coco/${WM_TYPE}/avg_attack/forgery/Add_Pattern________[Greybox]_" \
  --only_clean \
  --run_tag fixedKey_forgery_GB_N4000_withID \
  --max_trials 4000 \
  --no_save_inverted
```

Black-box forgery with identification:

```bash
python detect_avg_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id coco \
  --output_dir outputs_fixedkey_k0 \
  --override_clean_dir "../outputs_fixedkey_k0/coco/${WM_TYPE}/avg_attack/forgery/No_Operation_______[Blackbox]" \
  --override_wm_dir "../outputs_fixedkey_k0/coco/${WM_TYPE}/avg_attack/forgery/Add_Pattern________[Blackbox]" \
  --only_clean \
  --run_tag fixedKey_forgery_BB_N4000_withID \
  --max_trials 4000 \
  --no_save_inverted
```

Read:

```text
outputs_fixedkey_k0/coco/<wm_type>/metrics_<wm_type>_fixedKey_forgery_GB_N4000_withID.txt
outputs_fixedkey_k0/coco/<wm_type>/metrics_<wm_type>_fixedKey_forgery_BB_N4000_withID.txt
outputs_fixedkey_k0/coco/<wm_type>/verify-l1_fixedKey_forgery_*_N4000_withID.npz
outputs_fixedkey_k0/coco/<wm_type>/identify-acc_fixedKey_forgery_*_N4000_withID.npz
```

## Convenience Scripts

Run the standard generation-quality-detection pipeline:

```bash
bash scripts/run_paper_pipeline.sh HSQR coco outputs
```

Run fixed-key generation plus clean-only detection:

```bash
MAX_TRIALS=1000 bash scripts/run_fixed_key_pipeline.sh Tree-Ring coco 0 outputs_fixedkey_k0
```

## Output Summary

Important files:

```text
clip.npz                    # CLIP score arrays
fid.npz                     # COCO FID values
verify-l1*.npz              # verification scores
identify-acc*.npz           # identification results
metrics_*.txt               # readable tables
pattern_list-2048.pt        # 2048 candidate key patterns
identify_gt_indices_*.npy   # ground-truth key index per image
```

## Acknowledgement

This repository modifies and builds on the official SFWMark codebase:

```text
Semantic Watermarking Reinvented:
Enhancing Robustness and Generation Quality with Fourier Integrity
Sung Ju Lee and Nam Ik Cho, ICCV 2025
```

Official SFWMark resources:

- Code: https://github.com/thomas11809/SFWMark
- Project page: https://thomas11809.github.io/SFWMark/
- Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Semantic_Watermarking_Reinvented_Enhancing_Robustness_and_Generation_Quality_with_Fourier_ICCV_2025_paper.html
- arXiv: https://arxiv.org/abs/2509.07647

The original SFWMark project also builds on and acknowledges Tree-Ring, RingID, ZoDiac, invisible-watermark, Stable Signature, and Gaussian Shading.

## Citation

If you use the original SFWMark method or code, cite:

```bibtex
@inproceedings{lee2025semantic,
  title={Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity},
  author={Lee, Sung Ju and Cho, Nam Ik},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18759--18769},
  year={2025}
}
```

## License

This fork preserves the upstream `CC BY-NC 4.0` license. Use this code only for research and non-commercial purposes, subject to the upstream license terms.
