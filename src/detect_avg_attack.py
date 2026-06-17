import os
import itertools
import re
from tqdm import tqdm
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from utils import *



def _list_indices_in_dir(d: Path):
    """Return sorted integer indices for files like 123.png/jpg/jpeg under directory d."""
    if d is None or (not d.exists()):
        return []
    idxs = []
    for fn in os.listdir(d):
        m = re.match(r"^(\d+)\.(png|jpg|jpeg)$", fn, flags=re.IGNORECASE)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))

def _find_image_file(d: Path, idx: int) -> str:
    """Find idx.(png|jpg|jpeg) in directory d; raise FileNotFoundError if missing."""
    for ext in ("png", "jpg", "jpeg"):
        p = d / f"{idx}.{ext}"
        if p.exists():
            return str(p)
        p2 = d / f"{idx}.{ext.upper()}"
        if p2.exists():
            return str(p2)
    # fallback: try any extension that starts with idx.
    cand = list(d.glob(f"{idx}.*"))
    if cand:
        return str(cand[0])
    raise FileNotFoundError(f"Could not find image for idx={idx} under {d}")


@torch.no_grad()
def _candidate_distances_fast_single_channel(
    patterns: torch.Tensor,
    target_fft: torch.Tensor,
    mask_2d: torch.Tensor,
    ch: int,
    mode: str = "complex",
    p: int = 1,
) -> np.ndarray:
    """Fast path for Tree-Ring / METR identification.

    patterns: (K,1,4,64,64) complex
    target_fft: (1,4,64,64) complex
    mask_2d: (64,64) bool
    ch: channel index (int)

    Returns: (K,) float distances (CPU numpy)
    """
    assert patterns.ndim == 5 and target_fft.ndim == 4
    assert patterns.shape[1] == 1
    assert mode in {"complex", "real", "imag"}

    pat = patterns[:, 0, ch]  # (K,64,64)
    tgt = target_fft[0, ch]   # (64,64)

    if mode == "complex":
        diff = torch.abs(pat - tgt)  # (K,64,64)
    elif mode == "real":
        diff = torch.abs(pat.real - tgt.real)
    else:  # imag
        diff = torch.abs(pat.imag - tgt.imag)

    # apply mask
    m = mask_2d
    masked = diff[:, m]  # (K, M)

    if p == 1:
        d = masked.mean(dim=1)
    else:
        d = torch.norm(masked, p=p, dim=1) / masked.shape[1]

    return d.detach().cpu().numpy()


# main 함수
def main(args):
    set_random_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / f"{args.output_dir}/{args.dataset_id}/{args.wm_type}"
    # Optional: override input image folders (useful for averaging removal/forgery outputs)
    clean_dir = Path(args.override_clean_dir) if args.override_clean_dir else (save_dir / "img_pil")
    wm_dir = Path(args.override_wm_dir) if args.override_wm_dir else (save_dir / "img_pil_wm")
    run_tag = ("_" + args.run_tag) if args.run_tag else ""


    inverted_path = os.path.join(save_dir, "inverted_latents")
    if not args.no_save_inverted:
        os.makedirs(inverted_path, exist_ok=True)

    # [Datasets]
    meta_annot, prompt_key, gt_folder = get_text_dataset(args.dataset_id)

    # [Attack Settings]
    case_names = [
        "Clean", "Brightness", "Contrast", "JPEG", "Blur", "Noise", "BM3D",
        "VAE-B", "VAE-C", "Rotation", "CS75", "Diff", "CC", "RC"
    ]
    attack_dict = {
        "Brightness": 6,
        "Contrast": 0.5,
        "JPEG": 25,
        "Blur": 5,
        "Noise": 0.05,
        "BM3D": 0.1,
        "VAE-B": 3,
        "VAE-C": 3,
        "Rotation": 75,
        "CS75": 0.75,
        "CC": 0.5,
        "RC": 0.7,
    }

    if args.only_clean:
        case_names = ["Clean"]

    # [Evaluation Settings]
    num_dataset = len(meta_annot)
    identify_gt_indices = np.load(os.path.join(save_dir, f"identify_gt_indices_{num_dataset}.npy"))

    # robust to small datasets
    detect_trials = min(int(args.max_trials), int(len(identify_gt_indices)), int(num_dataset))

    # Determine which indices to evaluate.
    # If override dirs are provided, auto-detect available filenames to avoid missing-file errors.
    avail = []
    if args.override_wm_dir:
        avail = _list_indices_in_dir(wm_dir)
    elif args.override_clean_dir:
        avail = _list_indices_in_dir(clean_dir)
    else:
        avail = list(range(num_dataset))

    # keep indices that are within [0, num_dataset)
    avail = [i for i in avail if 0 <= i < num_dataset]
    if len(avail) == 0:
        raise RuntimeError(f"No evaluable images found. clean_dir={clean_dir} wm_dir={wm_dir}")

    detect_trials = min(detect_trials, len(avail))
    RANGE_EVAL = avail[:detect_trials]


    # [Stable-Diffusion-v2-1-base Settings]
    model_id = "SagiPolaczek/stable-diffusion-2-1-base"
    torch_dtype = torch.float32

    # [Load Stable-Diffusion pipeline]
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # [Result list] (kept in memory for AUC / TPR@1%FPR / mean acc)
    no_watermark_results_list = []
    Fourier_watermark_results_list = []
    id_acc_results_list = []
    bit_acc_results_list = []

    save_verify_name = f"verify-l1{run_tag}.npz"
    save_identify_name = f"identify-acc{run_tag}.npz"

    with torch.no_grad():
        # [Load Ground-Truth patterns]
        Fourier_watermark_pattern_list = torch.load(os.path.join(save_dir, f"pattern_list-{wm_capacity}.pt")).cpu()

        # [Evaluation methods]
        is_center = args.wm_type in ["HSTR", "HSQR"]
        channel_min = args.wm_type in ["RingID", "HSTR"]

        if args.wm_type in ["Tree-Ring", "METR", "RingID", "HSTR"]:
            eval_method = {
                "Distance": "L1",
                "Metrics": "|a-b|",
                "func": get_distance,
                "kwargs": {
                    "p": 1,
                    "center": is_center,
                    # Keep complex mode even if METR imag is mostly 0.
                    "mode": "complex",
                    "channel_min": channel_min,
                },
            }
        elif args.wm_type == "HSQR":
            eval_method = {
                "Distance": "L1",
                "Metrics": "|a-b|",
                "func": get_distance_hsqr,
                "kwargs": {"p": 1, "center": is_center},
            }
        else:
            raise ValueError(f"Unsupported wm_type: {args.wm_type}")

        # [Detection Area] Channel and Mask
        channel = RINGID_WATERMARK_CHANNEL if args.wm_type in ["RingID", "HSTR"] else TREE_WATERMARK_CHANNEL

        if args.wm_type == "Tree-Ring":
            mask = watermark_region_mask_tree.cpu()  # (1,64,64)
        elif args.wm_type == "METR":
            mask = watermark_region_mask_metr.cpu()  # (1,64,64)
        elif args.wm_type == "RingID":
            mask = watermark_region_mask_ringid.cpu()  # (C_R+C_H,64,64)
        elif args.wm_type == "HSTR":
            mask = watermark_region_mask_hstr.cpu()  # (C_R+C_H,64,64)
        elif args.wm_type == "HSQR":
            mask = None
        else:
            raise ValueError(f"Unsupported wm_type: {args.wm_type}")

        print("Attack-Detection Starts")

        for idx in tqdm(RANGE_EVAL):
            key_index = int(identify_gt_indices[idx])
            pattern_gt = Fourier_watermark_pattern_list[key_index].cpu()

            # Set random seed
            this_seed = 42 + idx
            set_random_seed(this_seed)

            # File inputs
            # robust to png/jpg/jpeg and to alternative folders (override_clean_dir/override_wm_dir)
            img_pil = Image.open(_find_image_file(clean_dir, idx))
            img_pil_wm = Image.open(_find_image_file(wm_dir, idx))
            if not args.only_clean:
                # diffusion attack images are precomputed by src/diff_attack
                img_pil_diff_attacked = Image.open(os.path.join(save_dir, f"img_pil-diffatt_fp16/{idx}.png"))
                img_pil_wm_diff_attacked = Image.open(os.path.join(save_dir, f"img_pil_wm-diffatt_fp16/{idx}.png"))
            else:
                img_pil_diff_attacked = None
                img_pil_wm_diff_attacked = None


            if args.only_clean:
                distorted_image_list = [[img_pil, img_pil_wm]]
            else:
                distorted_image_list = [
                    [img_pil, img_pil_wm],
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, brightness_factor=attack_dict["Brightness"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, contrast_factor=attack_dict["Contrast"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, jpeg_ratio=attack_dict["JPEG"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, gaussian_blur_r=attack_dict["Blur"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, gaussian_std=attack_dict["Noise"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, bm3d_sigma=attack_dict["BM3D"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, vaeb_quality=attack_dict["VAE-B"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, vaec_quality=attack_dict["VAE-C"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, r_degree=attack_dict["Rotation"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, crop_scale_area_ratio=attack_dict["CS75"]),
                    [img_pil_diff_attacked, img_pil_wm_diff_attacked],
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, center_crop_area_ratio=attack_dict["CC"]),
                    image_distortion(img_pil, img_pil_wm, seed=this_seed, random_crop_area_ratio=attack_dict["RC"]),
                ]
    
            img_pil_distorted_list = [pair[0] for pair in distorted_image_list]
            img_pil_wm_distorted_list = [pair[1] for pair in distorted_image_list]

            # [DDIM inversion]
            no_wm_distorted_zT = ddim_invert(pipe, img_pil_distorted_list, invert_guidance=0).cpu()  # (N_attack,4,64,64)
            Fourier_wm_distorted_zT = ddim_invert(pipe, img_pil_wm_distorted_list, invert_guidance=0).cpu()  # (N_attack,4,64,64)

            if not args.no_save_inverted:
                np.save(os.path.join(inverted_path, f"{idx}-no_latents.npy"), no_wm_distorted_zT.numpy())
                np.save(os.path.join(inverted_path, f"{idx}-wm_latents.npy"), Fourier_wm_distorted_zT.numpy())

            # Latent Fourier
            if args.wm_type in ["Tree-Ring", "METR", "RingID"]:
                no_wm_distorted_zT_fft = fft(no_wm_distorted_zT)
                Fourier_wm_distorted_zT_fft = fft(Fourier_wm_distorted_zT)
            elif args.wm_type in ["HSTR", "HSQR"]:
                no_wm_distorted_zT_fft = torch.zeros_like(no_wm_distorted_zT, dtype=torch.complex64)
                Fourier_wm_distorted_zT_fft = torch.zeros_like(Fourier_wm_distorted_zT, dtype=torch.complex64)
                no_wm_distorted_zT_fft[center_slice] = fft(no_wm_distorted_zT[center_slice])
                Fourier_wm_distorted_zT_fft[center_slice] = fft(Fourier_wm_distorted_zT[center_slice])
            else:
                raise ValueError

            # [Verification]
            no_wm_result = []
            Fourier_wm_result = []
            for distortion_index in range(len(distorted_image_list)):
                no_wm_zT_fft = no_wm_distorted_zT_fft[distortion_index][None, ...]
                Fourier_wm_zT_fft = Fourier_wm_distorted_zT_fft[distortion_index][None, ...]

                if args.wm_type in ["Tree-Ring", "METR", "RingID", "HSTR"]:
                    no_wm_verify_l1 = -eval_method["func"](pattern_gt, no_wm_zT_fft, mask=mask, channel=channel, **eval_method["kwargs"])
                    Fourier_wm_verify_l1 = -eval_method["func"](pattern_gt, Fourier_wm_zT_fft, mask=mask, channel=channel, **eval_method["kwargs"])
                elif args.wm_type == "HSQR":
                    no_wm_verify_l1 = -eval_method["func"](pattern_gt, no_wm_zT_fft, channel=channel, **eval_method["kwargs"])
                    Fourier_wm_verify_l1 = -eval_method["func"](pattern_gt, Fourier_wm_zT_fft, channel=channel, **eval_method["kwargs"])
                else:
                    raise ValueError

                no_wm_result.append(no_wm_verify_l1)
                Fourier_wm_result.append(Fourier_wm_verify_l1)

            no_watermark_results_list.append(no_wm_result)
            Fourier_watermark_results_list.append(Fourier_wm_result)

            if not args.skip_identification:
                # [Identification] nearest-pattern over all 2048 candidates
                id_acc_result = []
                bit_acc_result = []
                num_bits = int(np.log2(wm_capacity))
    
                # Fast path only for single-channel, non-center methods (Tree-Ring / METR)
                use_fast = (
                    (args.wm_type in ["Tree-Ring", "METR"])
                    and (not eval_method["kwargs"].get("center", False))
                    and (not eval_method["kwargs"].get("channel_min", False))
                    and (len(channel) == 1)
                    and (mask is not None)
                    and (mask.ndim == 3 and mask.shape[0] == 1)
                )
                if use_fast:
                    mask2d = mask[0].bool()
                    ch = int(channel[0])
    
                for distortion_index in range(len(distorted_image_list)):
                    Fourier_wm_zT_fft = Fourier_wm_distorted_zT_fft[distortion_index][None, ...]
    
                    if use_fast:
                        candidate_distances = _candidate_distances_fast_single_channel(
                            Fourier_watermark_pattern_list,
                            Fourier_wm_zT_fft,
                            mask2d,
                            ch=ch,
                            mode=eval_method["kwargs"].get("mode", "complex"),
                            p=eval_method["kwargs"].get("p", 1),
                        )
                        pred_index = int(np.argmin(candidate_distances))
                    else:
                        candidate_distances_list = []
                        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
                            if args.wm_type in ["Tree-Ring", "METR", "RingID", "HSTR"]:
                                candidate_distance = eval_method["func"](
                                    Fourier_watermark_pattern,
                                    Fourier_wm_zT_fft,
                                    mask=mask,
                                    channel=channel,
                                    **eval_method["kwargs"],
                                )
                            elif args.wm_type == "HSQR":
                                candidate_distance = eval_method["func"](
                                    Fourier_watermark_pattern,
                                    Fourier_wm_zT_fft,
                                    channel=channel,
                                    **eval_method["kwargs"],
                                )
                            else:
                                raise ValueError
                            candidate_distances_list.append(candidate_distance)
                        pred_index = int(np.argmin(np.array(candidate_distances_list)))
    
                    id_acc = (pred_index == key_index)
    
                    xor = pred_index ^ key_index
                    hd = 0
                    while xor:
                        hd += (xor & 1)
                        xor >>= 1
                    bit_acc = 1.0 - (hd / max(1, num_bits))
    
                    id_acc_result.append(id_acc)
                    bit_acc_result.append(bit_acc)
    
                id_acc_results_list.append(id_acc_result)
                bit_acc_results_list.append(bit_acc_result)
    
    # Optional: save raw arrays
    if not args.no_save_npz:
        no_watermark_results_list_array = np.array(no_watermark_results_list)
        Fourier_watermark_results_list_array = np.array(Fourier_watermark_results_list)

        np.savez(
            os.path.join(save_dir, save_verify_name),
            no_wm=no_watermark_results_list_array,
            wm=Fourier_watermark_results_list_array,
        )

        if not args.skip_identification:
            id_acc_results_list_array = np.array(id_acc_results_list)
            bit_acc_results_list_array = np.array(bit_acc_results_list)
            np.savez(
                os.path.join(save_dir, save_identify_name),
                wm=id_acc_results_list_array,
                bit=bit_acc_results_list_array,
        )

    # [Print + Save results]
    from prettytable import PrettyTable

    no_wms = np.array(no_watermark_results_list)
    wms = np.array(Fourier_watermark_results_list)

    auc_list, tpr1_list = [], []
    for j in range(no_wms.shape[1]):
        no_wm = no_wms[:, j].tolist()
        wm = wms[:, j].tolist()
        distances = no_wm + wm
        labels = [0] * len(no_wm) + [1] * len(wm)
        fpr, tpr, _ = metrics.roc_curve(labels, distances, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        tpr1 = tpr[np.where(fpr < 0.01)[0][-1]] if np.any(fpr < 0.01) else 0.0
        auc_list.append(float(auc))
        tpr1_list.append(float(tpr1))

    table_verify = PrettyTable()
    table_verify.field_names = ["Metric"] + case_names + ["Mean"]
    table_verify.add_row(["AUC"] + [f"{v:.3f}" for v in auc_list] + [f"{np.mean(auc_list):.3f}"])
    table_verify.add_row(["TPR@1%FPR"] + [f"{v:.3f}" for v in tpr1_list] + [f"{np.mean(tpr1_list):.3f}"])

    table_identify = None
    if not args.skip_identification:
        i_wms = np.array(id_acc_results_list)
        b_wms = np.array(bit_acc_results_list)
        id_accs = np.mean(i_wms.astype(np.float32), axis=0).tolist()
        bit_accs = np.mean(b_wms.astype(np.float32), axis=0).tolist()

        table_identify = PrettyTable()
        table_identify.field_names = ["Metric"] + case_names + ["Mean"]
        table_identify.add_row(["PerfectMatch"] + [f"{v:.3f}" for v in id_accs] + [f"{np.mean(id_accs):.3f}"])
        table_identify.add_row(["BitAcc"] + [f"{v:.3f}" for v in bit_accs] + [f"{np.mean(bit_accs):.3f}"])

    print()
    print("#" * 60)
    print("Verification Metrics (semantic)")
    print(table_verify)
    print()

    if table_identify is not None:
        print("#" * 60)
        print("Identification Metrics")
        print(table_identify)
        print()

    out_txt = os.path.join(save_dir, f"metrics_{args.wm_type}{run_tag}.txt")
    with open(out_txt, "w") as f:
        f.write("Verification Metrics (semantic)\n")
        f.write(table_verify.get_string())
        f.write("\n\n")
        if table_identify is not None:
            f.write("Identification Metrics\n")
            f.write(table_identify.get_string())
            f.write("\n")

    print(f"[Saved] {out_txt}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--wm_type",
        required=True,
        choices=["Tree-Ring", "RingID", "HSTR", "HSQR", "METR"],
        help="Choose watermarking methods",
    )
    parser.add_argument(
        "--dataset_id",
        choices=["coco", "Gustavo", "DB1k"],
        required=True,
        help="Choose dataset_id",
    )
    parser.add_argument("--output_dir", default="outputs", help="output directory: ./[output_dir]/")

    # Override input folders (for averaging removal/forgery outputs)
    parser.add_argument("--override_clean_dir", default=None, help="Optional: folder to replace img_pil")
    parser.add_argument("--override_wm_dir", default=None, help="Optional: folder to replace img_pil_wm")
    parser.add_argument("--only_clean", action="store_true", help="Evaluate only the Clean case (recommended for external attacks)")
    parser.add_argument("--run_tag", default="", help="Tag appended to output filenames, e.g., avg_removal_greybox_n1000")
    parser.add_argument("--max_trials", type=int, default=1000, help="Max number of images to evaluate")
    parser.add_argument("--skip_identification", action="store_true", help="Skip 2048-way identification (useful for forgery-only evaluation)")

    # If you only want the final averaged tables, these can greatly reduce disk usage.
    parser.add_argument(
        "--no_save_inverted",
        action="store_true",
        help="Do NOT save per-image inverted latents (*.npy).",
    )
    parser.add_argument(
        "--no_save_npz",
        action="store_true",
        help="Do NOT save verify-l1.npz / identify-acc.npz.",
    )

    args = parser.parse_args()
    main(args)
