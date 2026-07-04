from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKSPACE_ROOT = ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from attack.util import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    load_imagenet_categories,
    load_model,
    save_explanation_rgba,
    save_hot_contribution_map,
    save_rgb_image,
)


DEFAULT_INPUT_JSON = ROOT / "localization" / "transfer_failed_100.json"
DEFAULT_OUTPUT_DIR = ROOT / "localized" / "grid_point_game"
EXPLAIN_METHOD = "simple-gradient"
# INPUT_FORMAT = "spatial-rgb"
INPUT_FORMAT = "model-transform"
IG_STEPS = 100
SG_SAMPLES = 32
SG_NOISE_STD = 0.1
SMOOTH = 0

IMAGENET_CATEGORIES = load_imagenet_categories()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run localisation point-game style scoring on 2x2 grids built from transfer_failed samples. "
            "Each sample uses itemrefs: source_adv + target_pred_class + two random_class refs."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Path to transfer_failed JSON (contains samples with itemrefs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bcos", "torchvision", "bcosify"],
        required=True,
        help="Target model backend.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Target model name.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path for model loading.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--explain-method",
        type=str,
        choices=["bcos-explain", "integrated-gradients", "simple-gradient", "smoothgrad"],
        default=EXPLAIN_METHOD,
        help="Attribution method.",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["spatial-rgb", "model-transform"],
        default=INPUT_FORMAT,
        help="Image loading mode before inverse_transform.",
    )
    parser.add_argument("--ig-steps", type=int, default=IG_STEPS)
    parser.add_argument("--sg-samples", type=int, default=SG_SAMPLES)
    parser.add_argument("--sg-noise-std", type=float, default=SG_NOISE_STD)
    parser.add_argument("--smooth", type=int, default=SMOOTH)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed tag for output folder. If omitted, will use input_json meta.seed when available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    return parser.parse_args()


def category_name(index: int) -> str:
    if 0 <= index < len(IMAGENET_CATEGORIES):
        return IMAGENET_CATEGORIES[index]
    return str(index)


def load_input_rgb(image_path: Path, transform_cls, device: torch.device, input_format: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if input_format == "spatial-rgb":
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return rgb_tensor.to(device)
    return transform_cls.spatial_transform(image)[None].to(device)


def load_input_rgb_raw(image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return rgb_tensor.to(device)


def get_single_prediction(model: torch.nn.Module, transform_cls, rgb_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(transform_cls.inverse_transform(rgb_tensor))
    return int(logits.argmax(dim=1).item())


def make_multi_image_grid(rgb_tiles: list[torch.Tensor]) -> torch.Tensor:
    if len(rgb_tiles) != 4:
        raise ValueError("Expected exactly 4 tiles for 2x2 grid.")
    h0, w0 = rgb_tiles[0].shape[-2:]
    for idx, tile in enumerate(rgb_tiles):
        hi, wi = tile.shape[-2:]
        if (hi, wi) != (h0, w0):
            raise ValueError(
                f"All tile sizes must match. Tile 0 is {(h0, w0)}, tile {idx} is {(hi, wi)}."
            )
    row1 = torch.cat([rgb_tiles[0], rgb_tiles[1]], dim=3)
    row2 = torch.cat([rgb_tiles[2], rgb_tiles[3]], dim=3)
    return torch.cat([row1, row2], dim=2)


def extract_cell_map_from_grid(contribution_map: torch.Tensor, cell_index: int, single_shape: int) -> torch.Tensor:
    row = cell_index // 2
    col = cell_index % 2
    y0, y1 = row * single_shape, (row + 1) * single_shape
    x0, x1 = col * single_shape, (col + 1) * single_shape
    return contribution_map[:, :, y0:y1, x0:x1]


def compute_grid_scores(contribution_maps: torch.Tensor, single_shape: int) -> torch.Tensor:
    positive_maps = contribution_maps.clamp(min=0)
    pooled = (
        F.avg_pool2d(positive_maps, single_shape, stride=single_shape)
        .permute(0, 1, 3, 2)
        .reshape(positive_maps.shape[0], -1)
    )
    totals = pooled.sum(1, keepdim=True)
    normalized = torch.where(totals * pooled > 0, pooled / totals, torch.zeros_like(pooled))
    cell_indices = torch.arange(normalized.shape[0], device=normalized.device)
    return normalized[cell_indices, cell_indices]


def unpack_explain_output(explain_result):
    if isinstance(explain_result, tuple):
        if not explain_result:
            raise ValueError("model.explain returned an empty tuple")
        explain_result = explain_result[0]
    if not isinstance(explain_result, dict):
        raise TypeError(f"Unsupported explain output type: {type(explain_result)}")
    return explain_result


def compute_bcos_contribution(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    explain_result = model.explain(model_input.detach().clone().requires_grad_(True), idx=target_class)
    expl_out = unpack_explain_output(explain_result)
    contribution_map = expl_out["contribution_map"]
    if contribution_map.ndim == 3:
        contribution_map = contribution_map.unsqueeze(1)
    return contribution_map, expl_out.get("explanation"), expl_out.get("dynamic_linear_weights")


def compute_integrated_gradients_contribution(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
    ig_steps: int,
) -> tuple[torch.Tensor, None, None]:
    if ig_steps <= 0:
        raise ValueError("ig_steps must be > 0")
    baseline = torch.zeros_like(model_input)
    total_grads = torch.zeros_like(model_input)
    for step_idx in range(1, ig_steps + 1):
        alpha = float(step_idx) / float(ig_steps)
        interpolated = baseline + alpha * (model_input - baseline)
        interpolated.requires_grad_(True)
        logits = model(interpolated)
        target_logit = logits[:, target_class].sum()
        grads = torch.autograd.grad(target_logit, interpolated, retain_graph=False)[0]
        total_grads = total_grads + grads.detach()
    avg_grads = total_grads / float(ig_steps)
    attributions = (model_input - baseline) * avg_grads
    return attributions.sum(dim=1, keepdim=True), None, None


def compute_simple_gradient_contribution(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
) -> tuple[torch.Tensor, None, None]:
    model_input_for_grad = model_input.detach().clone().requires_grad_(True)
    logits = model(model_input_for_grad)
    target_logit = logits[:, target_class].sum()
    grads = torch.autograd.grad(target_logit, model_input_for_grad, retain_graph=False)[0]
    return grads.sum(dim=1, keepdim=True), None, None


def compute_smoothgrad_contribution(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
    sg_samples: int,
    sg_noise_std: float,
) -> tuple[torch.Tensor, None, None]:
    if sg_samples <= 0:
        raise ValueError("sg_samples must be > 0")
    if sg_noise_std < 0:
        raise ValueError("sg_noise_std must be >= 0")
    total_grads = torch.zeros_like(model_input)
    for _ in range(sg_samples):
        noisy_input = model_input.detach().clone()
        if sg_noise_std > 0:
            noisy_input = noisy_input + torch.randn_like(model_input) * sg_noise_std
        noisy_input.requires_grad_(True)
        logits = model(noisy_input)
        target_logit = logits[:, target_class].sum()
        grads = torch.autograd.grad(target_logit, noisy_input, retain_graph=False)[0]
        total_grads = total_grads + grads.detach()
    avg_grads = total_grads / float(sg_samples)
    attributions = model_input * avg_grads
    return attributions.sum(dim=1, keepdim=True), None, None


def compute_contribution_by_method(
    explain_method: str,
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
    ig_steps: int,
    sg_samples: int,
    sg_noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if explain_method == "bcos-explain":
        return compute_bcos_contribution(model, model_input, target_class)
    if explain_method == "integrated-gradients":
        return compute_integrated_gradients_contribution(model, model_input, target_class, ig_steps)
    if explain_method == "simple-gradient":
        return compute_simple_gradient_contribution(model, model_input, target_class)
    return compute_smoothgrad_contribution(model, model_input, target_class, sg_samples, sg_noise_std)


def resolve_clean_from_adv(adv_path: Path) -> Path:
    candidates = [
        adv_path.with_name("clean.png"),
        adv_path.with_name("clean_rgb.png"),
        Path(str(adv_path).replace("adv.png", "clean.png")),
        Path(str(adv_path).replace("adv_rgb.png", "clean_rgb.png")),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find clean image paired with source adv image: {adv_path}")


def extract_grid_paths_from_itemrefs(sample: dict) -> tuple[list[Path], list[Path], dict]:
    itemrefs = sample.get("itemrefs", [])
    if not isinstance(itemrefs, list):
        raise ValueError("sample.itemrefs must be a list")

    source_adv = None
    others: list[dict] = []
    for item in itemrefs:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        img_path = item.get("img_path")
        if not isinstance(img_path, str) or not img_path:
            continue
        if role == "source_adv" and source_adv is None:
            source_adv = Path(img_path)
        else:
            others.append(item)

    if source_adv is None:
        raise ValueError("itemrefs must contain one role=source_adv")
    if len(others) < 3:
        raise ValueError("itemrefs must contain at least 3 non-source items")

    picked = others[:3]
    adv_paths = [source_adv] + [Path(str(x["img_path"])) for x in picked]
    clean_source = resolve_clean_from_adv(source_adv)
    clean_paths = [clean_source] + [Path(str(x["img_path"])) for x in picked]

    info = {
        "source_adv_path": str(source_adv),
        "source_clean_path": str(clean_source),
        "other_items": picked,
    }
    return adv_paths, clean_paths, info


def run_grid_localization(
    model: torch.nn.Module,
    transform_cls,
    image_paths: list[Path],
    output_dir: Path,
    prefix: str,
    explain_method: str,
    input_format: str,
    ig_steps: int,
    sg_samples: int,
    sg_noise_std: float,
    smooth: int,
    device: torch.device,
) -> dict:
    if len(image_paths) != 4:
        raise ValueError("Expected 4 image paths")
    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(f"Image does not exist: {p}")

    rgb_tiles: list[torch.Tensor] = []
    single_predictions: list[int] = []
    # Rule: first tile is source image (adv/clean) -> keep raw; other 3 are references -> apply spatial transform.
    preprocess_modes: list[str] = []
    for tile_index, image_path in enumerate(image_paths):
        if tile_index == 0:
            rgb = load_input_rgb_raw(image_path, device)
            preprocess_modes.append("raw")
        else:
            rgb = load_input_rgb(image_path, transform_cls, device, "model-transform")
            preprocess_modes.append("spatial")
        rgb_tiles.append(rgb)
        single_predictions.append(get_single_prediction(model, transform_cls, rgb))

    grid_rgb = make_multi_image_grid(rgb_tiles)
    single_shape = rgb_tiles[0].shape[-1]
    model_grid_input = transform_cls.inverse_transform(grid_rgb).detach().clone()

    with torch.no_grad():
        grid_logits = model(model_grid_input)
    grid_prediction = int(grid_logits.argmax(dim=1).item())

    cell_contribution_maps: list[torch.Tensor] = []
    cell_records: list[dict] = []

    grid_dir = output_dir / prefix
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_rgb_path = grid_dir / "grid_rgb.png"
    save_rgb_image(grid_rgb, grid_rgb_path)

    for cell_index, target_class in enumerate(single_predictions):
        contribution_map, explanation, dynamic_linear_weights = compute_contribution_by_method(
            explain_method=explain_method,
            model=model,
            model_input=model_grid_input,
            target_class=target_class,
            ig_steps=ig_steps,
            sg_samples=sg_samples,
            sg_noise_std=sg_noise_std,
        )
        cell_map = extract_cell_map_from_grid(contribution_map, cell_index, single_shape)
        cell_contribution_maps.append(contribution_map)

        base_name = f"cell-{cell_index}_target-{target_class}_pred-{single_predictions[cell_index]}"
        grid_map_hot = grid_dir / f"{base_name}_grid_hot.png"
        cell_map_hot = grid_dir / f"{base_name}_cell_hot.png"
        grid_map_tensor = grid_dir / f"{base_name}_grid.pt"
        cell_map_tensor = grid_dir / f"{base_name}_cell.pt"

        save_hot_contribution_map(contribution_map.squeeze(0).squeeze(0), grid_map_hot)
        save_hot_contribution_map(cell_map.squeeze(0).squeeze(0), cell_map_hot)
        torch.save(contribution_map.detach().cpu(), grid_map_tensor)
        torch.save(cell_map.detach().cpu(), cell_map_tensor)

        expl_path = None
        if explanation is not None:
            expl_path = grid_dir / f"{base_name}_explanation.png"
            save_explanation_rgba(explanation, expl_path)

        dyn_path = None
        if dynamic_linear_weights is not None:
            dyn_path = grid_dir / f"{base_name}_dynamic_linear_weights.pt"
            torch.save(dynamic_linear_weights.detach().cpu(), dyn_path)

        cell_records.append(
            {
                "cell_index": cell_index,
                "target_class": int(target_class),
                "target_class_name": category_name(int(target_class)),
                "grid_saliency_hot": str(grid_map_hot),
                "cell_saliency_hot": str(cell_map_hot),
                "grid_saliency_tensor": str(grid_map_tensor),
                "cell_saliency_tensor": str(cell_map_tensor),
                "explanation": str(expl_path) if expl_path is not None else None,
                "dynamic_linear_weights": str(dyn_path) if dyn_path is not None else None,
            }
        )

    contribution_maps_tensor = torch.cat(cell_contribution_maps, dim=0)
    if smooth > 0:
        contribution_maps_tensor = F.avg_pool2d(
            contribution_maps_tensor,
            smooth,
            stride=1,
            padding=(smooth - 1) // 2,
        )

    localization_scores = compute_grid_scores(contribution_maps_tensor, single_shape)

    return {
        "grid_dir": str(grid_dir),
        "grid_rgb": str(grid_rgb_path),
        "images": [str(p) for p in image_paths],
        "single_predictions": [int(x) for x in single_predictions],
        "single_prediction_names": [category_name(int(x)) for x in single_predictions],
        "grid_prediction": int(grid_prediction),
        "grid_prediction_name": category_name(int(grid_prediction)),
        "preprocess_modes": preprocess_modes,
        "localization_scores": localization_scores.detach().cpu().tolist(),
        "mean_localization_score": float(localization_scores.mean().item()),
        "cells": cell_records,
    }


def sanitize_name(raw: str) -> str:
    text = raw.strip()
    if not text:
        return "unknown"
    kept = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            kept.append(ch)
        else:
            kept.append("_")
    return "".join(kept)


def main() -> None:
    args = parse_args()
    if not args.input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    if args.explain_method in {"integrated-gradients", "simple-gradient", "smoothgrad"} and args.model_type != "torchvision":
        raise ValueError(
            "integrated-gradients/simple-gradient/smoothgrad currently require --model-type torchvision"
        )

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    if not isinstance(samples, list) or not samples:
        raise ValueError("Input JSON must contain non-empty 'samples' list")

    payload_meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    json_seed = payload_meta.get("seed") if isinstance(payload_meta, dict) else None
    run_seed = args.seed if args.seed is not None else json_seed

    device = torch.device(args.device)
    out_root = args.output_dir / f"{sanitize_name(args.model_type)}_{sanitize_name(args.model_name)}"
    if run_seed is not None:
        out_root = out_root / f"seed_{int(run_seed)}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.max_samples is not None:
        samples = samples[: max(0, int(args.max_samples))]

    model, resolved_checkpoint_path = load_model(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        return_checkpoint_path=True,
    )
    transform_cls = model.transform

    records: list[dict] = []
    failed: list[dict] = []

    for sample_index, sample in tqdm(enumerate(samples)):
        img_name = str(sample.get("img_name", f"sample_{sample_index:04d}"))
        sample_name = f"sample_{sample_index:04d}_{sanitize_name(img_name)}"
        sample_out_dir = out_root / sample_name

        try:
            adv_paths, clean_paths, grid_info = extract_grid_paths_from_itemrefs(sample)
            adv_result = run_grid_localization(
                model=model,
                transform_cls=transform_cls,
                image_paths=adv_paths,
                output_dir=sample_out_dir,
                prefix="adv_grid",
                explain_method=args.explain_method,
                input_format=args.input_format,
                ig_steps=args.ig_steps,
                sg_samples=args.sg_samples,
                sg_noise_std=args.sg_noise_std,
                smooth=args.smooth,
                device=device,
            )
            clean_result = run_grid_localization(
                model=model,
                transform_cls=transform_cls,
                image_paths=clean_paths,
                output_dir=sample_out_dir,
                prefix="clean_grid",
                explain_method=args.explain_method,
                input_format=args.input_format,
                ig_steps=args.ig_steps,
                sg_samples=args.sg_samples,
                sg_noise_std=args.sg_noise_std,
                smooth=args.smooth,
                device=device,
            )

            mean_adv = float(adv_result["mean_localization_score"])
            mean_clean = float(clean_result["mean_localization_score"])

            records.append(
                {
                    "sample_index": sample_index,
                    "img_name": img_name,
                    "source_pred": sample.get("source_pred"),
                    "target_pred": sample.get("target_pred"),
                    "grid_info": grid_info,
                    "adv": adv_result,
                    "clean": clean_result,
                    "compare": {
                        "mean_adv": mean_adv,
                        "mean_clean": mean_clean,
                        "delta_adv_minus_clean": mean_adv - mean_clean,
                    },
                }
            )

            print(
                f"[{sample_index + 1}/{len(samples)}] {img_name} | "
                f"adv={mean_adv:.6f} clean={mean_clean:.6f} delta={mean_adv - mean_clean:.6f}"
            )
        except Exception as exc:
            failed.append(
                {
                    "sample_index": sample_index,
                    "img_name": img_name,
                    "error": str(exc),
                }
            )
            print(f"[{sample_index + 1}/{len(samples)}] {img_name} | ERROR: {exc}")

    summary = {
        "input_json": str(args.input_json),
        "output_root": str(out_root),
        "seed": int(run_seed) if run_seed is not None else None,
        "model_type": args.model_type,
        "model_name": args.model_name,
        "checkpoint": str(resolved_checkpoint_path),
        "explain_method": args.explain_method,
        "input_format": args.input_format,
        "ig_steps": args.ig_steps,
        "sg_samples": args.sg_samples,
        "sg_noise_std": args.sg_noise_std,
        "smooth": args.smooth,
        "num_samples_total": len(samples),
        "num_samples_success": len(records),
        "num_samples_failed": len(failed),
        "details": records,
        "failed_samples": failed,
    }

    summary_path = out_root / "localisation_grid_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved summary: {summary_path}")
    print(f"Completed: success={len(records)} failed={len(failed)}")


if __name__ == "__main__":
    main()
