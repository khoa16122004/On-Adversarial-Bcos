from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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

from attack.util import (
    DEFAULT_MODEL_TYPE,
    DEFAULT_CHECKPOINT_DIR,
    load_model,
    load_imagenet_categories,
    save_explanation_rgba,
    save_hot_contribution_map,
    save_rgb_image,
)


IMAGE_PATHS = [
    Path(r"D:\BCOS_ATTACK\attack_result\bcos\resnet50\epsilon_0p05\local_black_swan_black-swan_o100_tuntargeted\adv_rgb.png"),
    Path(r"D:\BCOS_ATTACK\attack_result\torchvision\resnet50\epsilon_0p05\local_ostrich_ostrich_o9_tuntargeted\clean_rgb.png"),
    Path(r"D:\BCOS_ATTACK\attack_result\torchvision\resnet50\epsilon_0p05\local_peacock_peacock_o84_tuntargeted\clean_rgb.png"),
    Path(r"D:\BCOS_ATTACK\attack_result\torchvision\resnet50\epsilon_0p05\local_pelican_pelican_o144_tuntargeted\clean_rgb.png"),
    



]
# IMAGE_PATHS = [
#     Path(r"D:\BCOS_ATTACK\test_imgs\pgd_logit_attack\resnet50\epsilon_0p05\local_tench_tench_o0_t100\adv_rgb.png"),
#     Path(r"D:\BCOS_ATTACK\test_imgs\pgd_logit_attack\resnet50\epsilon_0p05\local_tench_tench_o0_t281\adv_rgb.png"),
#     Path(r"D:\BCOS_ATTACK\test_imgs\pgd_logit_attack\resnet50\epsilon_0p05\local_tench_tench_o0_t500\adv_rgb.png"),
#     Path(r"D:\BCOS_ATTACK\test_imgs\pgd_logit_attack\resnet50\epsilon_0p05\local_tench_tench_o0_t100\clean_rgb.png"),
# ]
# EXPLAIN_CLASS_SPECS = ["pred", "pred", "pred", "pred"]
# IMAGE_PATHS = [
#     Path(r"D:\BCOS_ATTACK\B-cos-v2\test_imgs\pgd_rgb_attack\pred-class-0_adv_rgb.png"),
#     Path(r"D:\BCOS_ATTACK\B-cos-v2\test_imgs\pgd_rgb_attack\pred-class-100_adv_rgb.png"),
#     Path(r"D:\BCOS_ATTACK\B-cos-v2\test_imgs\pgd_rgb_attack\pred-class-281_clean_rgb.png"),
#     Path(r"D:\BCOS_ATTACK\B-cos-v2\test_imgs\pgd_rgb_attack\pred-class-500_adv_rgb.png"),
# ]
# EXPLAIN_CLASS_SPECS = ["9", "84", "85", "86"]
EXPLAIN_CLASS_SPECS = ["pred", "pred", "pred", "pred"]
EXPLAIN_METHOD = "simple-gradient"
MODEL_TYPE = "torchvision"
MODEL_NAME = "resnet50"
CHECKPOINT_PATH = None
CHECKPOINT_DIR = DEFAULT_CHECKPOINT_DIR
SMOOTH = 0
INPUT_FORMAT = "spatial-rgb"
IG_STEPS = 100
SG_SAMPLES = 32
SG_NOISE_STD = 0.1
TOP5_SALIENCY_K = 5
OUTPUT_DIR = ROOT / "test_imgs" / "localisation_grid_demo"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_CATEGORIES = load_imagenet_categories()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localisation demo with B-cos explain or Integrated Gradients.")
    parser.add_argument(
        "--explain-method",
        type=str,
        choices=["bcos-explain", "integrated-gradients", "simple-gradient", "smoothgrad"],
        default=EXPLAIN_METHOD,
        help="Attribution method to run.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bcos", "torchvision"],
        default=MODEL_TYPE,
        help="Model backend to use.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Model name from selected backend.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Optional checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Checkpoint directory for B-cos models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTH,
        help="Optional smoothing kernel size over contribution maps.",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["spatial-rgb", "model-transform"],
        default=INPUT_FORMAT,
        help="Input loading mode.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=IG_STEPS,
        help="Number of integration steps for integrated gradients.",
    )
    parser.add_argument(
        "--sg-samples",
        type=int,
        default=SG_SAMPLES,
        help="Number of noisy samples for SmoothGrad.",
    )
    parser.add_argument(
        "--sg-noise-std",
        type=float,
        default=SG_NOISE_STD,
        help="Gaussian noise std for SmoothGrad.",
    )
    parser.add_argument(
        "--top5-saliency-k",
        type=int,
        default=TOP5_SALIENCY_K,
        help="Number of top adversarial classes used for per-image top-k saliency export.",
    )
    parser.add_argument(
        "--target-specs",
        type=str,
        default=",".join(EXPLAIN_CLASS_SPECS),
        help="Comma-separated target specs per cell. Use 'pred' or class index.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string for torch.",
    )
    return parser.parse_args()


def category_name(index: int) -> str:
    if 0 <= index < len(IMAGENET_CATEGORIES):
        return IMAGENET_CATEGORIES[index]
    return str(index)


def get_single_prediction(
    model: torch.nn.Module,
    transform_cls,
    rgb_tensor: torch.Tensor,
) -> int:
    with torch.no_grad():
        logits = model(transform_cls.inverse_transform(rgb_tensor))
    return int(logits.argmax(dim=1).item())


def compute_grid_scores(
    contribution_maps: torch.Tensor,
    single_shape: int,
) -> torch.Tensor:
    positive_maps = contribution_maps.clamp(min=0)
    pooled = (
        F.avg_pool2d(positive_maps, single_shape, stride=single_shape)
        .permute(0, 1, 3, 2)
        .reshape(positive_maps.shape[0], -1)
    )
    totals = pooled.sum(1, keepdim=True)
    normalized = torch.where(totals * pooled > 0, pooled / totals, torch.zeros_like(pooled))
    cell_indices = torch.arange(normalized.shape[0])
    return normalized[cell_indices, cell_indices]


def load_input_rgb(image_path: Path, transform_cls, device: torch.device, input_format: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if input_format == "spatial-rgb":
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return rgb_tensor.to(device)
    return transform_cls.spatial_transform(image)[None].to(device)


def resolve_target_classes(
    explain_class_specs: list[str],
    single_predictions: list[int],
) -> list[int]:
    if len(explain_class_specs) != len(single_predictions):
        raise ValueError(
            f"Expected {len(single_predictions)} explain class specs, got {len(explain_class_specs)}"
        )

    target_classes: list[int] = []
    for cell_index, spec in enumerate(explain_class_specs):
        normalized_spec = spec.strip().lower()
        if normalized_spec == "pred":
            target_classes.append(single_predictions[cell_index])
            continue
        target_classes.append(int(spec))
    return target_classes


def resolve_grid_extra_target_classes(
    extra_specs: list[str],
    grid_prediction: int,
) -> list[int]:
    target_classes: list[int] = []
    for spec in extra_specs:
        normalized_spec = spec.strip().lower()
        if normalized_spec == "pred":
            target_classes.append(int(grid_prediction))
            continue
        target_classes.append(int(spec))
    return target_classes


def make_multi_image_grid(rgb_tiles: list[torch.Tensor]) -> torch.Tensor:
    if len(rgb_tiles) != 4:
        raise ValueError("This demo expects exactly 4 tiles to form a 2x2 grid.")
    row1 = torch.cat([rgb_tiles[0], rgb_tiles[1]], dim=3)
    row2 = torch.cat([rgb_tiles[2], rgb_tiles[3]], dim=3)
    return torch.cat([row1, row2], dim=2)


def extract_cell_map_from_grid(
    contribution_map: torch.Tensor,
    cell_index: int,
    single_shape: int,
) -> torch.Tensor:
    if contribution_map.ndim != 4:
        raise ValueError("contribution_map must have shape [B, 1, H, W].")
    if contribution_map.shape[0] != 1:
        raise ValueError("Expected batch size 1 for contribution map.")
    if contribution_map.shape[2] != 2 * single_shape or contribution_map.shape[3] != 2 * single_shape:
        raise ValueError("Unexpected contribution map spatial shape for 2x2 grid.")

    row = cell_index // 2
    col = cell_index % 2
    y0, y1 = row * single_shape, (row + 1) * single_shape
    x0, x1 = col * single_shape, (col + 1) * single_shape
    return contribution_map[:, :, y0:y1, x0:x1]


def unpack_explain_output(explain_result):
    """Support both explain(...) -> dict and explain(...) -> (dict, logits)."""
    if isinstance(explain_result, tuple):
        if not explain_result:
            raise ValueError("model.explain returned an empty tuple.")
        explain_result = explain_result[0]
    if not isinstance(explain_result, dict):
        raise TypeError(f"Unsupported explain output type: {type(explain_result)}")
    return explain_result


def parse_class_value(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        first_token = cleaned.split(",")[0].strip()
        return int(first_token)
    if isinstance(value, list) and value:
        return parse_class_value(value[0])
    return None


def resolve_saliency_reference_classes(
    image_path: Path,
    fallback_original_class: int,
    fallback_attack_class: int,
) -> dict[str, int]:
    metadata_path = image_path.parent / "attack_metadata.json"
    if not metadata_path.exists():
        return {
            "original": int(fallback_original_class),
            "attack": int(fallback_attack_class),
        }

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "original": int(fallback_original_class),
            "attack": int(fallback_attack_class),
        }

    original_class = parse_class_value(metadata.get("original_class"))
    attack_class = parse_class_value(metadata.get("target_class"))
    if attack_class is None:
        attack_class = parse_class_value(metadata.get("adversarial_class"))

    if original_class is None:
        original_class = int(fallback_original_class)
    if attack_class is None:
        attack_class = int(fallback_attack_class)

    return {
        "original": int(original_class),
        "attack": int(attack_class),
    }


def resolve_pair_clean_image_path(image_path: Path) -> Path | None:
    if image_path.name == "clean_rgb.png" and image_path.exists():
        return image_path
    candidate = image_path.parent / "clean_rgb.png"
    if candidate.exists():
        return candidate
    return None


def resolve_adv_topk_classes(
    image_path: Path,
    fallback_class: int,
    k: int,
) -> tuple[list[int], str]:
    metadata_path = image_path.parent / "attack_metadata.json"
    if not metadata_path.exists():
        return [int(fallback_class)], "fallback_target"

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return [int(fallback_class)], "fallback_target"

    top5_root = metadata.get("top5_predictions")
    if not isinstance(top5_root, dict):
        return [int(fallback_class)], "fallback_target"
    adversarial_block = top5_root.get("adversarial")
    if not isinstance(adversarial_block, dict):
        return [int(fallback_class)], "fallback_target"

    selected_key = "top_probs"
    top_items = adversarial_block.get(selected_key)
    if not isinstance(top_items, list) or not top_items:
        selected_key = "top_logits"
        top_items = adversarial_block.get(selected_key)
    if not isinstance(top_items, list) or not top_items:
        return [int(fallback_class)], "fallback_target"

    top_classes: list[int] = []
    for item in top_items:
        if not isinstance(item, dict):
            continue
        class_id = item.get("class_id")
        if class_id is None:
            continue
        try:
            top_classes.append(int(class_id))
        except Exception:
            continue

    if not top_classes:
        return [int(fallback_class)], "fallback_target"

    return top_classes[: max(1, int(k))], selected_key


def compute_bcos_contribution(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    explain_result = model.explain(
        model_input.detach().clone().requires_grad_(True),
        idx=target_class,
    )
    expl_out = unpack_explain_output(explain_result)
    contribution_map = expl_out["contribution_map"]
    if contribution_map.ndim == 3:
        contribution_map = contribution_map.unsqueeze(1)
    explanation = expl_out.get("explanation")
    dynamic_linear_weights = expl_out.get("dynamic_linear_weights")
    return contribution_map, explanation, dynamic_linear_weights


def compute_integrated_gradients_contribution(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
    ig_steps: int,
) -> tuple[torch.Tensor, None, None]:
    if ig_steps <= 0:
        raise ValueError("ig_steps must be > 0.")

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
    contribution_map = attributions.sum(dim=1, keepdim=True)
    return contribution_map, None, None


def compute_simple_gradient_contribution(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
) -> tuple[torch.Tensor, None, None]:
    model_input_for_grad = model_input.detach().clone().requires_grad_(True)
    logits = model(model_input_for_grad)
    target_logit = logits[:, target_class].sum()
    grads = torch.autograd.grad(target_logit, model_input_for_grad, retain_graph=False)[0]
    contribution_map = grads.sum(dim=1, keepdim=True)
    return contribution_map, None, None


def compute_smoothgrad_contribution(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_class: int,
    sg_samples: int,
    sg_noise_std: float,
) -> tuple[torch.Tensor, None, None]:
    if sg_samples <= 0:
        raise ValueError("sg_samples must be > 0.")
    if sg_noise_std < 0:
        raise ValueError("sg_noise_std must be >= 0.")

    total_grads = torch.zeros_like(model_input)

    for _ in range(sg_samples):
        if sg_noise_std == 0:
            noisy_input = model_input.detach().clone()
        else:
            noisy_input = model_input.detach().clone() + torch.randn_like(model_input) * sg_noise_std
        noisy_input.requires_grad_(True)
        logits = model(noisy_input)
        target_logit = logits[:, target_class].sum()
        grads = torch.autograd.grad(target_logit, noisy_input, retain_graph=False)[0]
        total_grads = total_grads + grads.detach()

    avg_grads = total_grads / float(sg_samples)
    attributions = model_input * avg_grads
    contribution_map = attributions.sum(dim=1, keepdim=True)
    return contribution_map, None, None


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
        return compute_bcos_contribution(
            model=model,
            model_input=model_input,
            target_class=target_class,
        )
    if explain_method == "integrated-gradients":
        return compute_integrated_gradients_contribution(
            model=model,
            model_input=model_input,
            target_class=target_class,
            ig_steps=ig_steps,
        )
    if explain_method == "simple-gradient":
        return compute_simple_gradient_contribution(
            model=model,
            model_input=model_input,
            target_class=target_class,
        )
    return compute_smoothgrad_contribution(
        model=model,
        model_input=model_input,
        target_class=target_class,
        sg_samples=sg_samples,
        sg_noise_std=sg_noise_std,
    )


def main() -> None:
    args = parse_args()
    if args.explain_method in {"integrated-gradients", "simple-gradient", "smoothgrad"} and args.model_type != "torchvision":
        raise ValueError(
            "--explain-method integrated-gradients, simple-gradient, or smoothgrad requires --model-type torchvision."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_output_dir = output_dir / "grid_outputs"
    single_target_output_dir = output_dir / "single_target_outputs"
    top5_adv_output_dir = output_dir / "top5_saliency" / "adv"
    top5_clean_output_dir = output_dir / "top5_saliency" / "clean"
    grid_output_dir.mkdir(parents=True, exist_ok=True)
    single_target_output_dir.mkdir(parents=True, exist_ok=True)
    top5_adv_output_dir.mkdir(parents=True, exist_ok=True)
    top5_clean_output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    explain_class_specs = [token.strip() for token in args.target_specs.split(",") if token.strip()]
    if len(explain_class_specs) < len(IMAGE_PATHS):
        raise ValueError(
            f"Expected at least {len(IMAGE_PATHS)} target specs but got {len(explain_class_specs)} from --target-specs."
        )

    for image_path in IMAGE_PATHS:
        print(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

    model, resolved_checkpoint_path = load_model(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        return_checkpoint_path=True,
    )
    transform_cls = model.transform

    rgb_tiles: list[torch.Tensor] = []
    single_predictions: list[int] = []
    for image_path in IMAGE_PATHS:
        rgb_tensor = load_input_rgb(image_path, transform_cls, device, args.input_format)
        rgb_tiles.append(rgb_tensor)
        single_predictions.append(get_single_prediction(model, transform_cls, rgb_tensor))

    cell_target_specs = explain_class_specs[: len(IMAGE_PATHS)]
    extra_target_specs = explain_class_specs[len(IMAGE_PATHS):]

    cell_target_classes = resolve_target_classes(cell_target_specs, single_predictions)
    grid_rgb = make_multi_image_grid(rgb_tiles)
    single_shape = rgb_tiles[0].shape[-1]
    model_grid_input = transform_cls.inverse_transform(grid_rgb).detach().clone()
    with torch.no_grad():
        grid_logits = model(model_grid_input)
    grid_prediction = int(grid_logits.argmax(dim=1).item())
    extra_target_classes = resolve_grid_extra_target_classes(extra_target_specs, grid_prediction)
    target_classes = cell_target_classes + extra_target_classes

    explanation_records = []
    grid_only_records = []
    cell_contribution_maps = []
    for target_index, target_class in enumerate(target_classes):
        contribution_map, explanation, dynamic_linear_weights = compute_contribution_by_method(
            explain_method=args.explain_method,
            model=model,
            model_input=model_grid_input,
            target_class=target_class,
            ig_steps=args.ig_steps,
            sg_samples=args.sg_samples,
            sg_noise_std=args.sg_noise_std,
        )
        if target_index < len(IMAGE_PATHS):
            cell_contribution_maps.append(contribution_map)

        if target_index < len(IMAGE_PATHS):
            file_prefix = (
                f"cell-{target_index}_map-class-{target_class}_pred-class-{single_predictions[target_index]}"
            )
        else:
            file_prefix = f"grid-extra-{target_index - len(IMAGE_PATHS)}_map-class-{target_class}"

        save_hot_contribution_map(
            contribution_map.squeeze(0).squeeze(0),
            grid_output_dir / f"{file_prefix}_contribution_map_hot.png",
        )

        if target_index < len(IMAGE_PATHS):
            cell_map = extract_cell_map_from_grid(
                contribution_map=contribution_map,
                cell_index=target_index,
                single_shape=single_shape,
            )
            save_hot_contribution_map(
                cell_map.squeeze(0).squeeze(0),
                grid_output_dir / f"{file_prefix}_contribution_map_cell_hot.png",
            )

        if explanation is not None:
            save_explanation_rgba(
                explanation,
                grid_output_dir / f"{file_prefix}_explanation.png",
            )
        torch.save(
            contribution_map.detach().cpu(),
            grid_output_dir / f"{file_prefix}_contribution_map.pt",
        )
        if target_index < len(IMAGE_PATHS):
            torch.save(
                cell_map.detach().cpu(),
                grid_output_dir / f"{file_prefix}_contribution_map_cell.pt",
            )

        if target_index >= len(IMAGE_PATHS):
            grid_only_records.append(
                {
                    "target_index": target_index,
                    "target_class": int(target_class),
                    "target_class_name": category_name(int(target_class)),
                    "target_spec": explain_class_specs[target_index],
                    "grid_contribution_map": str(grid_output_dir / f"{file_prefix}_contribution_map.pt"),
                    "grid_contribution_map_hot": str(grid_output_dir / f"{file_prefix}_contribution_map_hot.png"),
                }
            )
            if dynamic_linear_weights is not None:
                torch.save(
                    dynamic_linear_weights.detach().cpu(),
                    grid_output_dir / f"{file_prefix}_dynamic_linear_weights.pt",
                )
            continue

        single_model_input = transform_cls.inverse_transform(rgb_tiles[target_index]).detach().clone()
        single_contribution_map, single_explanation, single_dynamic_linear_weights = compute_contribution_by_method(
            explain_method=args.explain_method,
            model=model,
            model_input=single_model_input,
            target_class=target_class,
            ig_steps=args.ig_steps,
            sg_samples=args.sg_samples,
            sg_noise_std=args.sg_noise_std,
        )
        sample_single_output_dir = single_target_output_dir / f"cell-{target_index}_target-class-{target_class}"
        sample_single_output_dir.mkdir(parents=True, exist_ok=True)

        single_target_png_path = sample_single_output_dir / f"{file_prefix}_single_target_saliency_hot.png"
        single_target_tensor_path = sample_single_output_dir / f"{file_prefix}_single_target_saliency.pt"
        save_hot_contribution_map(single_contribution_map.squeeze(0).squeeze(0), single_target_png_path)
        torch.save(single_contribution_map.detach().cpu(), single_target_tensor_path)

        single_target_paths: dict[str, str] = {
            "saliency_png": str(single_target_png_path),
            "saliency_tensor": str(single_target_tensor_path),
        }
        if single_explanation is not None:
            single_explanation_path = sample_single_output_dir / f"{file_prefix}_single_target_explanation.png"
            save_explanation_rgba(single_explanation, single_explanation_path)
            single_target_paths["explanation"] = str(single_explanation_path)
        if single_dynamic_linear_weights is not None:
            single_dynamic_path = sample_single_output_dir / f"{file_prefix}_single_target_dynamic_linear_weights.pt"
            torch.save(single_dynamic_linear_weights.detach().cpu(), single_dynamic_path)
            single_target_paths["dynamic_linear_weights"] = str(single_dynamic_path)

        if dynamic_linear_weights is not None:
            torch.save(
                dynamic_linear_weights.detach().cpu(),
                grid_output_dir / f"{file_prefix}_dynamic_linear_weights.pt",
            )

        adv_topk_classes, adv_topk_source = resolve_adv_topk_classes(
            image_path=IMAGE_PATHS[target_index],
            fallback_class=target_class,
            k=args.top5_saliency_k,
        )
        clean_pair_path = resolve_pair_clean_image_path(IMAGE_PATHS[target_index])
        clean_pair_input = None
        if clean_pair_path is not None:
            clean_pair_rgb = load_input_rgb(clean_pair_path, transform_cls, device, args.input_format)
            clean_pair_input = transform_cls.inverse_transform(clean_pair_rgb).detach().clone()

        sample_top5_adv_dir = top5_adv_output_dir / f"cell-{target_index}"
        sample_top5_clean_dir = top5_clean_output_dir / f"cell-{target_index}"
        sample_top5_adv_dir.mkdir(parents=True, exist_ok=True)
        sample_top5_clean_dir.mkdir(parents=True, exist_ok=True)

        top5_adv_paths: list[dict[str, str | int]] = []
        top5_clean_paths: list[dict[str, str | int]] = []
        for rank_idx, class_id in enumerate(adv_topk_classes, start=1):
            top5_adv_map, top5_adv_expl, top5_adv_dyn = compute_contribution_by_method(
                explain_method=args.explain_method,
                model=model,
                model_input=single_model_input,
                target_class=class_id,
                ig_steps=args.ig_steps,
                sg_samples=args.sg_samples,
                sg_noise_std=args.sg_noise_std,
            )
            adv_rank_prefix = f"cell-{target_index}_rank-{rank_idx}_class-{class_id}"
            adv_png_path = sample_top5_adv_dir / f"{adv_rank_prefix}_saliency_hot.png"
            adv_tensor_path = sample_top5_adv_dir / f"{adv_rank_prefix}_saliency.pt"
            save_hot_contribution_map(top5_adv_map.squeeze(0).squeeze(0), adv_png_path)
            torch.save(top5_adv_map.detach().cpu(), adv_tensor_path)

            adv_record: dict[str, str | int] = {
                "rank": rank_idx,
                "class_id": int(class_id),
                "class_name": category_name(int(class_id)),
                "saliency_png": str(adv_png_path),
                "saliency_tensor": str(adv_tensor_path),
            }
            if top5_adv_expl is not None:
                adv_expl_path = sample_top5_adv_dir / f"{adv_rank_prefix}_explanation.png"
                save_explanation_rgba(top5_adv_expl, adv_expl_path)
                adv_record["explanation"] = str(adv_expl_path)
            if top5_adv_dyn is not None:
                adv_dyn_path = sample_top5_adv_dir / f"{adv_rank_prefix}_dynamic_linear_weights.pt"
                torch.save(top5_adv_dyn.detach().cpu(), adv_dyn_path)
                adv_record["dynamic_linear_weights"] = str(adv_dyn_path)
            top5_adv_paths.append(adv_record)

            if clean_pair_input is not None:
                top5_clean_map, top5_clean_expl, top5_clean_dyn = compute_contribution_by_method(
                    explain_method=args.explain_method,
                    model=model,
                    model_input=clean_pair_input,
                    target_class=class_id,
                    ig_steps=args.ig_steps,
                    sg_samples=args.sg_samples,
                    sg_noise_std=args.sg_noise_std,
                )
                clean_rank_prefix = f"cell-{target_index}_rank-{rank_idx}_class-{class_id}"
                clean_png_path = sample_top5_clean_dir / f"{clean_rank_prefix}_saliency_hot.png"
                clean_tensor_path = sample_top5_clean_dir / f"{clean_rank_prefix}_saliency.pt"
                save_hot_contribution_map(top5_clean_map.squeeze(0).squeeze(0), clean_png_path)
                torch.save(top5_clean_map.detach().cpu(), clean_tensor_path)

                clean_record: dict[str, str | int] = {
                    "rank": rank_idx,
                    "class_id": int(class_id),
                    "class_name": category_name(int(class_id)),
                    "saliency_png": str(clean_png_path),
                    "saliency_tensor": str(clean_tensor_path),
                }
                if top5_clean_expl is not None:
                    clean_expl_path = sample_top5_clean_dir / f"{clean_rank_prefix}_explanation.png"
                    save_explanation_rgba(top5_clean_expl, clean_expl_path)
                    clean_record["explanation"] = str(clean_expl_path)
                if top5_clean_dyn is not None:
                    clean_dyn_path = sample_top5_clean_dir / f"{clean_rank_prefix}_dynamic_linear_weights.pt"
                    torch.save(top5_clean_dyn.detach().cpu(), clean_dyn_path)
                    clean_record["dynamic_linear_weights"] = str(clean_dyn_path)
                top5_clean_paths.append(clean_record)

        explanation_records.append(
            {
                "cell_index": target_index,
                "image_path": str(IMAGE_PATHS[target_index]),
                "target_class": target_class,
                "target_class_name": category_name(target_class),
                "target_spec": explain_class_specs[target_index],
                "single_prediction": single_predictions[target_index],
                "single_prediction_name": category_name(single_predictions[target_index]),
                "single_target_class": target_class,
                "single_target_class_name": category_name(target_class),
                "single_target_paths": single_target_paths,
                "adv_topk_classes_for_saliency": adv_topk_classes,
                "adv_topk_source": adv_topk_source,
                "clean_pair_image_path": str(clean_pair_path) if clean_pair_path is not None else None,
                "top5_adv_paths": top5_adv_paths,
                "top5_clean_paths": top5_clean_paths,
            }
        )

    contribution_maps_tensor = torch.cat(cell_contribution_maps, dim=0)
    if args.smooth > 0:
        contribution_maps_tensor = F.avg_pool2d(
            contribution_maps_tensor,
            args.smooth,
            stride=1,
            padding=(args.smooth - 1) // 2,
        )

    localization_scores = compute_grid_scores(contribution_maps_tensor, single_shape)
    grid_rgb_path = grid_output_dir / "grid_rgb.png"
    save_rgb_image(grid_rgb, grid_rgb_path)

    summary = {
        "checkpoint": str(resolved_checkpoint_path),
        "model_type": args.model_type,
        "model_name": args.model_name,
        "explain_method": args.explain_method,
        "grid_output_dir": str(grid_output_dir),
        "single_target_output_dir": str(single_target_output_dir),
        "top5_adv_output_dir": str(top5_adv_output_dir),
        "top5_clean_output_dir": str(top5_clean_output_dir),
        "images": [str(image_path) for image_path in IMAGE_PATHS],
        "target_specs": explain_class_specs,
        "cell_target_specs": cell_target_specs,
        "extra_target_specs": extra_target_specs,
        "single_predictions": single_predictions,
        "single_prediction_names": [category_name(index) for index in single_predictions],
        "target_classes": target_classes,
        "cell_target_classes": cell_target_classes,
        "extra_target_classes": extra_target_classes,
        "target_class_names": [category_name(index) for index in target_classes],
        "grid_prediction": grid_prediction,
        "grid_prediction_name": category_name(grid_prediction),
        "ig_steps": args.ig_steps,
        "sg_samples": args.sg_samples,
        "sg_noise_std": args.sg_noise_std,
        "top5_saliency_k": args.top5_saliency_k,
        "smooth": args.smooth,
        "input_format": args.input_format,
        "localization_scores": localization_scores.detach().cpu().tolist(),
        "mean_localization_score": float(localization_scores.mean().item()),
        "details": explanation_records,
        "grid_extra_details": grid_only_records,
    }
    summary_path = output_dir / "localisation_grid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved grid outputs to: {grid_output_dir}")
    print(f"Saved per-image single-target outputs to: {single_target_output_dir}")
    print(f"Saved top-k adv saliency outputs to: {top5_adv_output_dir}")
    print(f"Saved top-k clean saliency outputs to: {top5_clean_output_dir}")
    print(f"Saved grid RGB to: {grid_rgb_path}")
    for cell_index, score in enumerate(summary["localization_scores"]):
        print(
            f"Cell {cell_index}: target={cell_target_classes[cell_index]} "
            f"({category_name(cell_target_classes[cell_index])}), "
            f"target_spec={cell_target_specs[cell_index]}, "
            f"pred={single_predictions[cell_index]} "
            f"({category_name(single_predictions[cell_index])}), "
            f"localization_score={score:.6f}"
        )
    print(f"Mean localization score: {summary['mean_localization_score']:.6f}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
