"""WAVES-Core and StirMark-compatible robustness benchmark for SFWMark forks.

This runner is intentionally self-contained. It uses the official WAVES distortion
ranges and the published StirMark 4 profile values, while keeping the repository's
native DDIM inversion and 2048-key decoders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn import metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from diffusers import DDIMScheduler, DiffusionPipeline

import utils


METHOD_LABELS = {
    "Tree-Ring": "Tree-Ring",
    "METR": "METR",
    "RingID": "RingID",
    "HSTR": "HSFW-HSTR",
    "HSQR": "HSFW-HSQR",
}

WAVES_RANGES = {
    "rotation": (0.0, 45.0),
    "resizedcrop": (1.0, 0.5),
    "erasing": (0.0, 0.25),
    "brightness": (1.0, 2.0),
    "contrast": (1.0, 2.0),
    "blurring": (0.0, 20.0),
    "noise": (0.0, 0.1),
    "compression": (90.0, 10.0),
}


@dataclass(frozen=True)
class Condition:
    name: str
    suite: str
    attack: str
    strength: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Tree-Ring", "METR", "RingID", "HSTR", "HSQR"],
        choices=list(METHOD_LABELS),
    )
    parser.add_argument("--dataset_id", default="coco", choices=["coco"])
    parser.add_argument("--input_dir", default="outputs_fixedkey_k0")
    parser.add_argument("--result_dir", default="benchmark_results/waves_stirmark")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--attack_batch_size", type=int, default=12)
    parser.add_argument("--key_chunk_size", type=int, default=256)
    parser.add_argument("--waves_strengths", nargs="+", type=float, default=[0.4, 0.8])
    parser.add_argument(
        "--suite",
        choices=["waves", "stirmark", "all"],
        default="all",
    )
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_id", default="SagiPolaczek/stable-diffusion-2-1-base")
    return parser.parse_args()


def build_conditions(args: argparse.Namespace) -> list[Condition]:
    conditions = [Condition("clean", "baseline", "identity", 0.0)]
    if args.suite in {"waves", "all"}:
        for attack in WAVES_RANGES:
            for strength in args.waves_strengths:
                if not 0.0 <= strength <= 1.0:
                    raise ValueError("WAVES strengths must be in [0, 1]")
                conditions.append(
                    Condition(
                        f"waves_{attack}_s{strength:.2f}",
                        "waves_core",
                        attack,
                        float(strength),
                    )
                )
    if args.suite in {"stirmark", "all"}:
        # Compact subset from StirMark Benchmark 4.0 SMBsettings.ini.
        conditions.extend(
            [
                Condition("stirmark_rotation_2", "stirmark_compatible", "stir_rotation", 2.0),
                Condition("stirmark_rotation_10", "stirmark_compatible", "stir_rotation", 10.0),
                Condition("stirmark_crop_10", "stirmark_compatible", "stir_crop", 10.0),
                Condition("stirmark_crop_25", "stirmark_compatible", "stir_crop", 25.0),
                Condition("stirmark_rescale_75", "stirmark_compatible", "stir_rescale", 75.0),
                Condition("stirmark_affine_005", "stirmark_compatible", "stir_affine", 0.05),
                Condition("stirmark_remove_lines_10", "stirmark_compatible", "stir_remove_lines", 10.0),
                Condition("stirmark_random_distortion_2", "stirmark_compatible", "stir_random", 2.0),
            ]
        )
    return conditions


def absolute_waves_strength(relative: float, attack: str) -> float:
    start, end = WAVES_RANGES[attack]
    value = start + relative * (end - start)
    return min(max(value, min(start, end)), max(start, end))


def apply_waves(image: Image.Image, attack: str, relative: float, seed: int) -> Image.Image:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    strength = absolute_waves_strength(relative, attack)

    if attack == "rotation":
        return TF.rotate(image, strength)
    if attack == "resizedcrop":
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(strength, strength), ratio=(1.0, 1.0)
        )
        return TF.resized_crop(image, i, j, h, w, image.size)
    if attack == "erasing":
        tensor = TF.pil_to_tensor(image).float() / 255.0
        i, j, h, w, value = T.RandomErasing.get_params(
            tensor, scale=(strength, strength), ratio=(1.0, 1.0), value=[0]
        )
        erased = TF.erase(tensor, i, j, h, w, value)
        return TF.to_pil_image(erased)
    if attack == "brightness":
        return ImageEnhance.Brightness(image).enhance(strength)
    if attack == "contrast":
        return ImageEnhance.Contrast(image).enhance(strength)
    if attack == "blurring":
        return image.filter(ImageFilter.GaussianBlur(int(strength)))
    if attack == "noise":
        tensor = TF.pil_to_tensor(image).float() / 255.0
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(tensor.shape, generator=generator) * strength
        return TF.to_pil_image((tensor + noise).clamp(0, 1))
    if attack == "compression":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(strength))
        buffer.seek(0)
        output = Image.open(buffer).convert("RGB")
        output.load()
        return output
    raise ValueError(f"Unknown WAVES attack: {attack}")


def _center_crop_percent(image: Image.Image, percent: float) -> Image.Image:
    retained_area = max(0.01, 1.0 - percent / 100.0)
    retained_side = math.sqrt(retained_area)
    width, height = image.size
    crop_w = max(1, round(width * retained_side))
    crop_h = max(1, round(height * retained_side))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def _remove_periodic_lines(image: Image.Image, frequency: int) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    rows = np.arange(array.shape[0]) % frequency != 0
    cols = np.arange(array.shape[1]) % frequency != 0
    return Image.fromarray(array[rows][:, cols])


def _random_geometric_distortion(
    image: Image.Image, amplitude: float, seed: int
) -> Image.Image:
    """Smooth random remap approximating StirMark desynchronization."""
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    rng = np.random.default_rng(seed)
    grid_h = grid_w = 5
    dx_small = rng.uniform(-amplitude, amplitude, (grid_h, grid_w)).astype(np.float32)
    dy_small = rng.uniform(-amplitude, amplitude, (grid_h, grid_w)).astype(np.float32)
    dx = cv2.resize(dx_small, (width, height), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_small, (width, height), interpolation=cv2.INTER_CUBIC)
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    remapped = cv2.remap(
        array,
        x + dx,
        y + dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(remapped)


def apply_stirmark_compatible(
    image: Image.Image, attack: str, strength: float, seed: int
) -> Image.Image:
    if attack == "stir_rotation":
        return TF.rotate(image, strength)
    if attack == "stir_crop":
        return _center_crop_percent(image, strength)
    if attack == "stir_rescale":
        scale = strength / 100.0
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, Image.Resampling.BICUBIC)
    if attack == "stir_affine":
        shear_degrees = math.degrees(math.atan(strength))
        return TF.affine(image, angle=0.0, translate=[0, 0], scale=1.0, shear=[shear_degrees, shear_degrees])
    if attack == "stir_remove_lines":
        return _remove_periodic_lines(image, int(strength))
    if attack == "stir_random":
        return _random_geometric_distortion(image, strength, seed)
    raise ValueError(f"Unknown StirMark-compatible attack: {attack}")


def apply_condition(image: Image.Image, condition: Condition, seed: int) -> Image.Image:
    image = image.convert("RGB")
    if condition.attack == "identity":
        return image.copy()
    if condition.suite == "waves_core":
        return apply_waves(image, condition.attack, condition.strength, seed)
    if condition.suite == "stirmark_compatible":
        return apply_stirmark_compatible(image, condition.attack, condition.strength, seed)
    raise ValueError(condition)


def chunks(values: list, size: int) -> Iterable[list]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def list_image_indices(folder: Path) -> set[int]:
    indices = set()
    for path in folder.iterdir():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            try:
                indices.add(int(path.stem))
            except ValueError:
                pass
    return indices


def image_path(folder: Path, index: int) -> Path:
    for suffix in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        path = folder / f"{index}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for index {index} in {folder}")


def prepare_target_fft(latents: torch.Tensor, method: str) -> torch.Tensor:
    latents = latents.to("cpu")
    if method in {"Tree-Ring", "METR", "RingID"}:
        return utils.fft(latents)
    if method in {"HSTR", "HSQR"}:
        output = torch.zeros_like(latents, dtype=torch.complex64)
        output[utils.center_slice] = utils.fft(latents[utils.center_slice])
        return output
    raise ValueError(method)


@torch.no_grad()
def distance_matrix(
    patterns_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    method: str,
    key_chunk_size: int,
) -> torch.Tensor:
    """Return B x K distances using the repository's native L1 definitions."""
    device = torch.device("cuda")
    targets = targets_cpu.to(device)
    batch = targets.shape[0]
    capacity = patterns_cpu.shape[0]
    output = torch.empty((batch, capacity), dtype=torch.float32)

    if method == "Tree-Ring":
        channels = utils.TREE_WATERMARK_CHANNEL
        masks = utils.watermark_region_mask_tree.bool()
        center = False
        channel_min = False
    elif method == "METR":
        channels = utils.METR_WATERMARK_CHANNEL
        masks = utils.watermark_region_mask_metr.bool()
        center = False
        channel_min = False
    elif method == "RingID":
        channels = utils.RINGID_WATERMARK_CHANNEL
        masks = utils.watermark_region_mask_ringid.bool()
        center = False
        channel_min = True
    elif method == "HSTR":
        channels = utils.RINGID_WATERMARK_CHANNEL
        masks = utils.watermark_region_mask_hstr.bool()
        center = True
        channel_min = True
    elif method == "HSQR":
        channels = utils.HSQR_WATERMARK_CHANNEL
        masks = None
        center = True
        channel_min = False
    else:
        raise ValueError(method)

    for start in range(0, capacity, key_chunk_size):
        end = min(capacity, start + key_chunk_size)
        patterns = patterns_cpu[start:end].to(device)

        if method == "HSQR":
            qr = torch.where(patterns[:, 0], 45.0, -45.0)
            qr_complex = torch.complex(qr[:, :, :21], qr[:, :, 21:])
            target = targets[:, channels[0], 11:53, 33:54]
            distances = torch.abs(target[:, None] - qr_complex[None]).mean(dim=(-1, -2))
        else:
            if center:
                pattern_view = patterns[:, 0, channels, 10:54, 10:54]
                target_view = targets[:, channels, 10:54, 10:54]
                mask_view = masks[:, 10:54, 10:54]
            else:
                pattern_view = patterns[:, 0, channels]
                target_view = targets[:, channels]
                mask_view = masks

            per_channel = []
            for channel_index in range(len(channels)):
                mask = mask_view[channel_index].to(device)
                pattern_feature = pattern_view[:, channel_index][:, mask]
                target_feature = target_view[:, channel_index][:, mask]
                per_channel.append(
                    torch.abs(target_feature[:, None] - pattern_feature[None]).mean(-1)
                )
            distances = torch.stack(per_channel, dim=-1)
            distances = distances.min(dim=-1).values if channel_min else distances.mean(dim=-1)

        output[:, start:end] = distances.cpu()
        del patterns, distances

    return output


def score_metrics(no_wm_dist: np.ndarray, wm_dist: np.ndarray) -> dict[str, float]:
    labels = np.concatenate([np.zeros_like(no_wm_dist), np.ones_like(wm_dist)])
    scores = -np.concatenate([no_wm_dist, wm_dist])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)

    def tpr_at(max_fpr: float) -> float:
        valid = np.where(fpr <= max_fpr)[0]
        return float(tpr[valid[-1]]) if len(valid) else 0.0

    return {
        "auc": float(metrics.auc(fpr, tpr)),
        "tpr_0_1fpr": tpr_at(0.001),
        "tpr_1fpr": tpr_at(0.01),
    }


def quality_metrics(reference: Image.Image, attacked: Image.Image) -> tuple[float, float]:
    attacked = attacked.convert("RGB").resize(reference.size, Image.Resampling.BICUBIC)
    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32)
    attacked_array = np.asarray(attacked, dtype=np.float32)
    psnr = peak_signal_noise_ratio(reference_array, attacked_array, data_range=255.0)
    ssim = structural_similarity(
        reference_array, attacked_array, channel_axis=2, data_range=255.0
    )
    return float(psnr), float(ssim)


def bit_accuracy(predicted: np.ndarray, target: np.ndarray, bits: int = 11) -> np.ndarray:
    output = []
    for prediction, truth in zip(predicted.astype(int), target.astype(int)):
        output.append(1.0 - ((prediction ^ truth).bit_count() / bits))
    return np.asarray(output, dtype=np.float32)


def load_pipeline(model_id: str) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def cache_key(indices: list[int], conditions: list[Condition]) -> str:
    payload = json.dumps(
        {"indices": indices, "conditions": [asdict(c) for c in conditions]},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def invert_condition_images(
    pipe: DiffusionPipeline,
    source: Image.Image,
    conditions: list[Condition],
    sample_seed: int,
    attack_batch_size: int,
) -> torch.Tensor:
    outputs = []
    for chunk_start in range(0, len(conditions), attack_batch_size):
        condition_chunk = conditions[chunk_start : chunk_start + attack_batch_size]
        attacked = [
            apply_condition(
                source,
                condition,
                sample_seed + (chunk_start + offset) * 1009,
            )
            for offset, condition in enumerate(condition_chunk)
        ]
        outputs.append(utils.ddim_invert(pipe, attacked, invert_guidance=0).cpu())
    return torch.cat(outputs, dim=0)


def build_clean_cache(
    pipe: DiffusionPipeline,
    clean_dir: Path,
    indices: list[int],
    conditions: list[Condition],
    args: argparse.Namespace,
    result_dir: Path,
) -> torch.Tensor:
    path = result_dir / f"clean_latents_{cache_key(indices, conditions)}.pt"
    if args.resume and path.exists():
        print(f"[cache] Loading {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    all_latents = []
    for position, index in enumerate(indices):
        started = time.time()
        with Image.open(image_path(clean_dir, index)) as image:
            latents = invert_condition_images(
                pipe,
                image.convert("RGB"),
                conditions,
                args.seed + index * 100003,
                args.attack_batch_size,
            )
        all_latents.append(latents)
        print(
            f"[clean {position + 1}/{len(indices)}] index={index} "
            f"{time.time() - started:.1f}s"
        )
    cache = torch.stack(all_latents, dim=0)
    torch.save(cache, path)
    return cache


def evaluate_method(
    pipe: DiffusionPipeline,
    method: str,
    indices: list[int],
    conditions: list[Condition],
    clean_latents: torch.Tensor,
    args: argparse.Namespace,
    project_root: Path,
    result_dir: Path,
) -> list[dict]:
    method_dir = project_root / args.input_dir / args.dataset_id / method
    raw_path = result_dir / f"raw_{method}.npz"
    summary_path = result_dir / f"summary_{method}.json"
    if args.resume and raw_path.exists() and summary_path.exists():
        print(f"[resume] Skipping completed method {method}")
        return json.loads(summary_path.read_text())

    patterns = torch.load(
        method_dir / "pattern_list-2048.pt", map_location="cpu", weights_only=True
    )
    gt_all = np.load(method_dir / "identify_gt_indices_5000.npy")
    gt = gt_all[np.asarray(indices)].astype(np.int64)
    wm_dir = method_dir / "img_pil_wm"
    num_samples = len(indices)
    num_conditions = len(conditions)

    no_wm_dist = np.zeros((num_samples, num_conditions), dtype=np.float32)
    wm_dist = np.zeros_like(no_wm_dist)
    predicted = np.zeros((num_samples, num_conditions), dtype=np.int64)
    top5_correct = np.zeros((num_samples, num_conditions), dtype=bool)
    psnr = np.zeros_like(no_wm_dist)
    ssim = np.zeros_like(no_wm_dist)
    lpips_values = np.full_like(no_wm_dist, np.nan)

    for position, index in enumerate(indices):
        started = time.time()
        clean_fft = prepare_target_fft(clean_latents[position], method)
        clean_matrix = distance_matrix(
            patterns, clean_fft, method, args.key_chunk_size
        ).numpy()
        no_wm_dist[position] = clean_matrix[np.arange(num_conditions), gt[position]]

        with Image.open(image_path(wm_dir, index)) as source_image:
            source = source_image.convert("RGB")
            attacked_images = [
                apply_condition(source, condition, args.seed + index * 100003 + offset * 1009)
                for offset, condition in enumerate(conditions)
            ]
            for condition_index, attacked in enumerate(attacked_images):
                psnr[position, condition_index], ssim[position, condition_index] = quality_metrics(
                    source, attacked
                )
                if args.lpips:
                    lpips_values[position, condition_index] = utils.get_lpips(
                        source, attacked.resize(source.size, Image.Resampling.BICUBIC)
                    )

            wm_latents = []
            for attacked_chunk in chunks(attacked_images, args.attack_batch_size):
                wm_latents.append(
                    utils.ddim_invert(pipe, attacked_chunk, invert_guidance=0).cpu()
                )
            wm_fft = prepare_target_fft(torch.cat(wm_latents, dim=0), method)

        wm_matrix = distance_matrix(patterns, wm_fft, method, args.key_chunk_size).numpy()
        wm_dist[position] = wm_matrix[np.arange(num_conditions), gt[position]]
        predicted[position] = np.argmin(wm_matrix, axis=1)
        top5 = np.argpartition(wm_matrix, kth=4, axis=1)[:, :5]
        top5_correct[position] = np.any(top5 == gt[position], axis=1)
        print(
            f"[{method} {position + 1}/{num_samples}] index={index} "
            f"{time.time() - started:.1f}s"
        )

    bit_acc = bit_accuracy(predicted.reshape(-1), np.repeat(gt, num_conditions)).reshape(
        num_samples, num_conditions
    )
    summaries = []
    for condition_index, condition in enumerate(conditions):
        row = {
            "method": METHOD_LABELS[method],
            "code_method": method,
            **asdict(condition),
            "num_samples": num_samples,
            **score_metrics(no_wm_dist[:, condition_index], wm_dist[:, condition_index]),
            "id_top1": float(np.mean(predicted[:, condition_index] == gt)),
            "id_top5": float(np.mean(top5_correct[:, condition_index])),
            "bit_acc": float(np.mean(bit_acc[:, condition_index])),
            "psnr": float(np.mean(psnr[:, condition_index])),
            "ssim": float(np.mean(ssim[:, condition_index])),
            "lpips": float(np.nanmean(lpips_values[:, condition_index]))
            if args.lpips
            else None,
        }
        summaries.append(row)

    np.savez_compressed(
        raw_path,
        indices=np.asarray(indices),
        conditions=np.asarray([c.name for c in conditions]),
        gt=gt,
        no_wm_dist=no_wm_dist,
        wm_dist=wm_dist,
        predicted=predicted,
        top5_correct=top5_correct,
        bit_acc=bit_acc,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips_values,
    )
    summary_path.write_text(json.dumps(summaries, indent=2))
    del patterns
    torch.cuda.empty_cache()
    return summaries


def write_combined_outputs(rows: list[dict], result_dir: Path, args: argparse.Namespace) -> None:
    csv_path = result_dir / "summary.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    config = vars(args).copy()
    config["stirmark_note"] = (
        "StirMark-compatible Python transforms using published SMBsettings.ini values; "
        "the legacy Windows StirMark binary was not executed."
    )
    (result_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    methods = list(dict.fromkeys(row["method"] for row in rows))
    conditions = list(dict.fromkeys(row["name"] for row in rows))
    lookup = {(row["method"], row["name"]): row for row in rows}

    for metric_name, title in [
        ("tpr_1fpr", "TPR at 1% FPR"),
        ("id_top1", "Identification Top-1"),
    ]:
        matrix = np.asarray(
            [[lookup[(method, condition)][metric_name] for method in methods] for condition in conditions]
        )
        fig_height = max(6.0, 0.28 * len(conditions))
        fig, axis = plt.subplots(figsize=(8, fig_height))
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
        axis.set_yticks(range(len(conditions)), conditions)
        axis.set_title(title)
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(result_dir / f"{metric_name}_heatmap.png", dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^", "D", "P"]
    for marker, method in zip(markers, methods):
        method_rows = [row for row in rows if row["method"] == method and row["suite"] != "baseline"]
        axis.scatter(
            [row["ssim"] for row in method_rows],
            [row["tpr_1fpr"] for row in method_rows],
            label=method,
            alpha=0.75,
            marker=marker,
        )
    axis.set_xlabel("SSIM to pre-attack watermarked image")
    axis.set_ylabel("TPR at 1% FPR")
    axis.set_xlim(0, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "quality_vs_verification.png", dpi=180)
    plt.close(fig)

    lines = [
        "# WAVES-Core and StirMark-Compatible Benchmark",
        "",
        f"Samples per method: {rows[0]['num_samples']}",
        "",
        "StirMark note: the track uses published StirMark 4 profile values in Python. "
        "It is not a binary-identical StirMark run.",
        "",
        "## Clean Baseline",
        "",
        "| Method | AUC | TPR@0.1%FPR | TPR@1%FPR | ID Top-1 | Bit Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = lookup[(method, "clean")]
        lines.append(
            f"| {method} | {row['auc']:.3f} | {row['tpr_0_1fpr']:.3f} | "
            f"{row['tpr_1fpr']:.3f} | {row['id_top1']:.3f} | {row['bit_acc']:.3f} |"
        )

    lines.extend(["", "## Attack-Family Means", ""])
    lines.extend(
        [
            "| Method | Suite | AUC | TPR@1%FPR | ID Top-1 | Bit Acc | SSIM |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        for suite in ["waves_core", "stirmark_compatible"]:
            subset = [row for row in rows if row["method"] == method and row["suite"] == suite]
            if not subset:
                continue
            lines.append(
                f"| {method} | {suite} | "
                f"{np.mean([r['auc'] for r in subset]):.3f} | "
                f"{np.mean([r['tpr_1fpr'] for r in subset]):.3f} | "
                f"{np.mean([r['id_top1'] for r in subset]):.3f} | "
                f"{np.mean([r['bit_acc'] for r in subset]):.3f} | "
                f"{np.mean([r['ssim'] for r in subset]):.3f} |"
            )
    (result_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    result_dir = project_root / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args)

    first_method_dir = project_root / args.input_dir / args.dataset_id / args.methods[0]
    clean_dir = first_method_dir / "img_pil"
    available = sorted(list_image_indices(clean_dir))
    for method in args.methods:
        method_wm = project_root / args.input_dir / args.dataset_id / method / "img_pil_wm"
        available = sorted(set(available) & list_image_indices(method_wm))
    selected = available[args.start_index : args.start_index + args.num_samples]
    if len(selected) != args.num_samples:
        raise RuntimeError(f"Requested {args.num_samples} images but found {len(selected)}")

    print(f"Methods: {args.methods}")
    print(f"Conditions: {len(conditions)}")
    print(f"Indices: {selected[0]}..{selected[-1]} ({len(selected)})")
    (result_dir / "conditions.json").write_text(
        json.dumps([asdict(condition) for condition in conditions], indent=2)
    )

    utils.set_random_seed(args.seed)
    pipe = load_pipeline(args.model_id)
    clean_latents = build_clean_cache(
        pipe, clean_dir, selected, conditions, args, result_dir
    )

    rows = []
    for method in args.methods:
        rows.extend(
            evaluate_method(
                pipe,
                method,
                selected,
                conditions,
                clean_latents,
                args,
                project_root,
                result_dir,
            )
        )
    write_combined_outputs(rows, result_dir, args)
    print(f"Results written to {result_dir}")


if __name__ == "__main__":
    main()
