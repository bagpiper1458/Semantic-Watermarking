# src/generate_fixed_key.py
import os
import itertools
from tqdm import tqdm
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from utils import *

def _ensure_chdir_for_text_dataset():
    """
    get_text_dataset() uses relative path 'text_dataset/...'.
    Some repos keep text_dataset under src/, some under root.
    This makes it robust regardless of where you run from.
    """
    here = Path(__file__).resolve().parent           # .../SFWMark/src
    root = here.parent                               # .../SFWMark
    if (root / "text_dataset").exists():
        os.chdir(root)
    elif (here / "text_dataset").exists():
        os.chdir(here)
    else:
        # don't chdir; let it error with clear message if missing
        pass

def _safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(str(src), str(dst))

@torch.no_grad()
def _make_ringid_combo_at_index(fixed_key: int):
    # Mirrors generate.py but avoids enumerating all 2048 patterns.
    single_channel_num_slots = RADIUS - RADIUS_CUTOFF  # e.g., 11 slots
    per_slot = [list(combo) for combo in itertools.product(np.linspace(-64, 64, 2).tolist(),
                                                          repeat=len(RING_WATERMARK_CHANNEL))]
    key_value_list = [per_slot for _ in range(single_channel_num_slots)]
    # pick the fixed_key-th combination without materializing all combinations
    combo = next(itertools.islice(itertools.product(*key_value_list), fixed_key, None))
    return list(combo)

def main(args):
    set_random_seed(42)

    # Make text_dataset resolution robust
    _ensure_chdir_for_text_dataset()

    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / f"{args.output_dir}/{args.dataset_id}/{args.wm_type}"
    save_dir.mkdir(parents=True, exist_ok=True)

    img_clean_dir = save_dir / "img_pil"
    img_wm_dir = save_dir / "img_pil_wm"
    img_wm_dir.mkdir(parents=True, exist_ok=True)

    # Reuse multi-key clean images (so benchmark greybox has ind_clean)
    # Default source: outputs/<dataset>/<wm_type>/img_pil
    src_clean = project_root / f"{args.src_output_dir}/{args.dataset_id}/{args.wm_type}/img_pil"
    if args.link_clean:
        if not src_clean.exists():
            raise FileNotFoundError(f"Clean source folder not found: {src_clean}")
        _safe_symlink(src_clean, img_clean_dir)
    else:
        img_clean_dir.mkdir(parents=True, exist_ok=True)

    # Reuse multi-key pattern_list-2048.pt by symlink (saves 257MB * 5 methods)
    src_pattern_list = project_root / f"{args.src_output_dir}/{args.dataset_id}/{args.wm_type}/pattern_list-{wm_capacity}.pt"
    dst_pattern_list = save_dir / f"pattern_list-{wm_capacity}.pt"
    if args.link_pattern_list:
        if not src_pattern_list.exists():
            raise FileNotFoundError(f"pattern_list source not found: {src_pattern_list}")
        _safe_symlink(src_pattern_list, dst_pattern_list)
    else:
        if not dst_pattern_list.exists():
            if not src_pattern_list.exists():
                raise FileNotFoundError(f"pattern_list source not found: {src_pattern_list}")
            # copy if you prefer
            import shutil
            shutil.copy2(src_pattern_list, dst_pattern_list)

    # Dataset
    meta_annot, prompt_key, gt_folder = get_text_dataset(args.dataset_id)
    num_dataset = len(meta_annot)

    # Fixed key assignment (needed by detect scripts)
    fixed_key = int(args.fixed_key)
    assert 0 <= fixed_key < wm_capacity
    identify_gt_indices = np.full(num_dataset, fixed_key, dtype=np.int64)
    np.save(save_dir / f"identify_gt_indices_{num_dataset}.npy", identify_gt_indices)

    # Stable Diffusion
    model_id = "SagiPolaczek/stable-diffusion-2-1-base"
    resolution = 512
    torch_dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Choose masks and build ONE fixed pattern (no 2048 pattern generation here)
    w_seed_fixed = w_seed + fixed_key  # consistent with multi-key seed mapping

    if args.wm_type == "Tree-Ring":
        masks = tree_masks
        fixed_pattern = make_Fourier_treering_pattern(pipe, shape, w_seed_fixed)  # (1,4,64,64) complex
    elif args.wm_type == "METR":
        masks = metr_masks
        fixed_pattern = make_Fourier_metr_pattern(shape, key_index=fixed_key, imag_mode="zero")  # (1,4,64,64) complex
    elif args.wm_type == "RingID":
        masks = ringid_masks
        combo = _make_ringid_combo_at_index(fixed_key)
        fixed_pattern = make_Fourier_ringid_pattern(
            pipe, shape, combo, w_seed=w_seed_fixed,
            radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
            ring_watermark_channel=RING_WATERMARK_CHANNEL,
            heter_watermark_channel=HETER_WATERMARK_CHANNEL,
            heter_watermark_region_mask=heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL) > 0 else None
        )
        # match official postprocessing in your generate.py
        fixed_pattern = fft(ifft(fixed_pattern).real)
        fixed_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(
            torch.fft.fftshift(ifft(fixed_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2))
        )
    elif args.wm_type == "HSTR":
        masks = tree_masks.clone()
        masks[:, HETER_WATERMARK_CHANNEL] = single_channel_heter_watermark_mask
        fixed_pattern = make_Fourier_treering_pattern(pipe, shape, w_seed_fixed, hs=True, center=True, heter=True)
    elif args.wm_type == "HSQR":
        assert box_size == 2
        fixed_pattern = make_hsqr_pattern(idx=w_seed_fixed)  # (c_wm,42,42) or similar
    else:
        raise ValueError(f"Unsupported wm_type: {args.wm_type}")

    # Generation: only generate watermarked images (clean images are reused)
    batch_size = int(args.batch_size)
    RANGE_EVAL = range(0, num_dataset)

    print(f"[FixedKey] Generation Starts: wm_type={args.wm_type}, fixed_key={fixed_key}, num={num_dataset}")

    for batch_start in tqdm(range(0, len(RANGE_EVAL), batch_size)):
        batch_indices = list(RANGE_EVAL)[batch_start:batch_start + batch_size]
        n = len(batch_indices)
        gen_prompts = [meta_annot[idx][prompt_key] for idx in batch_indices]
        file_names = [f"{idx}.png" for idx in batch_indices]

        set_random_seed(42 + batch_start)

        with torch.no_grad():
            # prepare fixed pattern batch
            if fixed_pattern.ndim == 4:  # (1,4,64,64)
                pattern_gt_batch = fixed_pattern.repeat(n, 1, 1, 1)  # (n,4,64,64)
            elif fixed_pattern.ndim == 3:  # (c_wm,42,42)
                pattern_gt_batch = fixed_pattern.unsqueeze(0).repeat(n, 1, 1, 1)  # (n,c_wm,42,42)
            else:
                raise ValueError(f"Unexpected fixed_pattern shape: {fixed_pattern.shape}")

            # sample random latents
            no_watermark_latents = get_random_latents(pipe, batch_size=n)

            # inject watermark
            if args.wm_type in ["Tree-Ring", "METR", "RingID"]:
                wm_latents, _ = inject_wm(no_watermark_latents, pattern_gt_batch, masks, cut_real=True, device=device)
            elif args.wm_type == "HSTR":
                wm_latents, _ = inject_wm(no_watermark_latents, pattern_gt_batch, masks, center=True, cut_real=False, device=device)
            elif args.wm_type == "HSQR":
                wm_latents = inject_hsqr(no_watermark_latents, pattern_gt_batch, center=True, device=device)
            else:
                raise ValueError

            # generate ONLY watermarked images
            wm_images = pipe(gen_prompts, latents=wm_latents, guidance_scale=7.5,
                             num_inference_steps=50, num_images_per_prompt=1).images
            torch.cuda.empty_cache()

        for i, idx in enumerate(batch_indices):
            wm_images[i].save(img_wm_dir / file_names[i])

    print(f"[Done] Saved watermarked images to: {img_wm_dir}")
    print(f"[Done] Clean images reused from: {src_clean}  (link_clean={args.link_clean})")
    print(f"[Done] pattern_list linked from: {src_pattern_list}  (link_pattern_list={args.link_pattern_list})")
    print(f"[Done] identify_gt_indices saved: {save_dir / f'identify_gt_indices_{num_dataset}.npy'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_type", required=True, choices=["Tree-Ring", "RingID", "HSTR", "HSQR", "METR"])
    parser.add_argument("--dataset_id", required=True, choices=["coco", "Gustavo", "DB1k"])
    parser.add_argument("--output_dir", default="outputs_fixedkey", help="New output root for fixed-key run")
    parser.add_argument("--src_output_dir", default="outputs", help="Existing multi-key outputs to reuse (clean + pattern_list)")
    parser.add_argument("--fixed_key", type=int, required=True, help="Fixed key in [0,2047]")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--link_clean", action="store_true", help="Symlink clean img_pil from src_output_dir")
    parser.add_argument("--link_pattern_list", action="store_true", help="Symlink pattern_list-2048.pt from src_output_dir")

    args = parser.parse_args()
    main(args)
