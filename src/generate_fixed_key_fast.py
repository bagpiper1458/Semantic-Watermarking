# src/generate_fixed_key_fast.py
import os
import itertools
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from utils import *


def _ensure_text_dataset_cwd():
    # get_text_dataset() uses relative path "text_dataset/..."
    here = Path(__file__).resolve().parent      # .../SFWMark/src
    root = here.parent                          # .../SFWMark
    if (root / "text_dataset").exists():
        os.chdir(root)
    elif (here / "text_dataset").exists():
        os.chdir(here)


def _symlink_dir(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(str(src), str(dst))


@torch.no_grad()
def _build_patterns(pipe, wm_type: str):
    # build full codebook once (needed for identification)
    w_seed_list = list(range(w_seed, w_seed + wm_capacity))

    if wm_type == "Tree-Ring":
        patterns = [make_Fourier_treering_pattern(pipe, shape, s) for s in w_seed_list]
    elif wm_type == "METR":
        patterns = [make_Fourier_metr_pattern(shape, key_index=i, imag_mode="zero") for i in range(wm_capacity)]
    elif wm_type == "RingID":
        single_channel_num_slots = RADIUS - RADIUS_CUTOFF
        key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-64, 64, 2).tolist(),
                                                                      repeat=len(RING_WATERMARK_CHANNEL))]
                          for _ in range(single_channel_num_slots)]
        key_value_combinations = list(itertools.product(*key_value_list))
        patterns = [
            make_Fourier_ringid_pattern(
                pipe, shape, list(combo), w_seed=w_seed_list[i],
                radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
                ring_watermark_channel=RING_WATERMARK_CHANNEL,
                heter_watermark_channel=HETER_WATERMARK_CHANNEL,
                heter_watermark_region_mask=heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL) > 0 else None
            )
            for i, combo in enumerate(key_value_combinations)
        ]
        # official fixes
        patterns = [fft(ifft(p).real) for p in patterns]
        for p in patterns:
            p[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(p[:, RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2)))
    elif wm_type == "HSTR":
        patterns = [make_Fourier_treering_pattern(pipe, shape, s, hs=True, center=True, heter=True) for s in w_seed_list]
    elif wm_type == "HSQR":
        assert box_size == 2
        patterns = [make_hsqr_pattern(idx=s) for s in w_seed_list]
    else:
        raise ValueError(wm_type)

    return torch.stack(patterns, 0).detach().cpu()


@torch.no_grad()
def main(args):
    _ensure_text_dataset_cwd()
    set_random_seed(42)

    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / f"{args.output_dir}/{args.dataset_id}/{args.wm_type}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # link shared clean set
    if args.shared_clean_dir:
        _symlink_dir(Path(args.shared_clean_dir).expanduser().resolve(), save_dir / "img_pil")
    else:
        (save_dir / "img_pil").mkdir(parents=True, exist_ok=True)

    (save_dir / "img_pil_wm").mkdir(parents=True, exist_ok=True)

    # dataset
    meta_annot, prompt_key, _ = get_text_dataset(args.dataset_id)
    num_dataset = len(meta_annot)

    fixed_key = int(args.fixed_key)
    assert 0 <= fixed_key < wm_capacity

    # fixed identify indices (all same key)
    identify_gt_indices = np.full(num_dataset, fixed_key, dtype=np.int64)
    np.save(save_dir / f"identify_gt_indices_{num_dataset}.npy", identify_gt_indices)

    # SD pipeline
    model_id = "SagiPolaczek/stable-diffusion-2-1-base"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # ensure pattern_list exists (needed by detect for identification)
    pat_path = save_dir / f"pattern_list-{wm_capacity}.pt"
    if not pat_path.exists():
        print(f"[Build] pattern_list-{wm_capacity}.pt for {args.wm_type} ...")
        patterns = _build_patterns(pipe, args.wm_type)
        torch.save(patterns, pat_path)
        print(f"[Saved] {pat_path}")
    else:
        print(f"[Reuse] {pat_path}")

    # load pattern list and select fixed pattern
    patterns = torch.load(pat_path, map_location="cpu")
    fixed_pattern = patterns[fixed_key]  # (1,4,64,64) or (c,42,42) depending on method

    # masks
    if args.wm_type == "Tree-Ring":
        masks = tree_masks
    elif args.wm_type == "METR":
        masks = metr_masks
    elif args.wm_type == "RingID":
        masks = ringid_masks
    elif args.wm_type == "HSTR":
        masks = tree_masks.clone()
        masks[:, HETER_WATERMARK_CHANNEL] = single_channel_heter_watermark_mask
    elif args.wm_type == "HSQR":
        masks = None
    else:
        raise ValueError

    # generate ONLY WM images
    batch_size = int(args.batch_size)
    print(f"[FixedKey] wm_type={args.wm_type} fixed_key={fixed_key} N={num_dataset} batch={batch_size}")

    for batch_start in tqdm(range(0, num_dataset, batch_size)):
        batch_indices = list(range(batch_start, min(num_dataset, batch_start + batch_size)))
        n = len(batch_indices)
        gen_prompts = [meta_annot[idx][prompt_key] for idx in batch_indices]

        # fixed pattern batch
        if fixed_pattern.ndim == 4:  # (1,4,64,64)
            pattern_gt_batch = fixed_pattern.repeat(n, 1, 1, 1)
        elif fixed_pattern.ndim == 3:  # (c,42,42)
            pattern_gt_batch = fixed_pattern.unsqueeze(0).repeat(n, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected fixed_pattern shape: {fixed_pattern.shape}")

        no_wm_latents = get_random_latents(pipe, batch_size=n)

        if args.wm_type in ["Tree-Ring", "METR", "RingID"]:
            wm_latents, _ = inject_wm(no_wm_latents, pattern_gt_batch, masks, cut_real=True, device=device)
        elif args.wm_type == "HSTR":
            wm_latents, _ = inject_wm(no_wm_latents, pattern_gt_batch, masks, center=True, cut_real=False, device=device)
        elif args.wm_type == "HSQR":
            wm_latents = inject_hsqr(no_wm_latents, pattern_gt_batch, center=True, device=device)
        else:
            raise ValueError

        wm_images = pipe(gen_prompts, latents=wm_latents, guidance_scale=7.5,
                         num_inference_steps=50, num_images_per_prompt=1).images
        torch.cuda.empty_cache()

        for i, idx in enumerate(batch_indices):
            wm_images[i].save(save_dir / "img_pil_wm" / f"{idx}.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--wm_type", required=True, choices=["Tree-Ring", "METR", "RingID", "HSTR", "HSQR"])
    p.add_argument("--dataset_id", required=True, choices=["coco", "Gustavo", "DB1k"])
    p.add_argument("--output_dir", default="outputs_fixedkey_k0")
    p.add_argument("--fixed_key", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--shared_clean_dir", default=None, help="Path to shared img_pil (clean) folder")
    args = p.parse_args()
    main(args)
