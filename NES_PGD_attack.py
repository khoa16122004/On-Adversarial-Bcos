from __future__ import annotations

import argparse
import json
from pathlib import Path
import tqdm
import torch
import torch.nn.functional as F
from PIL import Image

from attack.util import (
    load_imagenet_categories,
    load_model,
    save_perturbation_image,
    save_rgb_image,
)
from script.const import CHECKPOINT_DIR, CLASSIFICATION_RESULT_DIR, PROJECT_ROOT


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _format_epsilon_dir(epsilon: float) -> str:
    eps_str = format(epsilon, ".4f").rstrip("0").rstrip(".")
    return f"epsilon_{eps_str}"


def _load_attack_samples(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("attack_1k.json must be a dict: {class_id: [img_path]}")

    normalized: dict[str, list[str]] = {}
    for class_id, items in data.items():
        if not isinstance(items, list):
            raise ValueError(f"class '{class_id}' must map to a list")
        normalized[str(class_id)] = [str(item) for item in items]
    return normalized


def _model_forward(model: torch.nn.Module, clean_or_adv_rgb: torch.Tensor) -> torch.Tensor:
    transform = getattr(model, "transform", None)
    if transform is not None and hasattr(transform, "inverse_transform"):
        model_input = transform.inverse_transform(clean_or_adv_rgb)
    else:
        model_input = clean_or_adv_rgb
    return model(model_input)


def _model_forward_batched(
    model: torch.nn.Module,
    inputs_rgb: torch.Tensor,
    eval_batch_size: int,
) -> torch.Tensor:
    if eval_batch_size <= 0 or inputs_rgb.shape[0] <= eval_batch_size:
        return _model_forward(model, inputs_rgb)

    logits_chunks: list[torch.Tensor] = []
    for start in range(0, inputs_rgb.shape[0], eval_batch_size):
        logits_chunks.append(_model_forward(model, inputs_rgb[start : start + eval_batch_size]))
    return torch.cat(logits_chunks, dim=0)


def _estimate_nes_gradient_and_logits(
    model: torch.nn.Module,
    adv_rgb: torch.Tensor,
    original_classes: torch.Tensor,
    nes_m: int,
    nes_sigma: float,
    nes_eval_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if nes_m <= 0:
        raise ValueError("nes_m must be > 0")
    if nes_sigma <= 0:
        raise ValueError("nes_sigma must be > 0")

    batch_size, channels, height, width = adv_rgb.shape

    with torch.no_grad():
        noise = torch.randn(
            (nes_m, batch_size, channels, height, width),
            device=adv_rgb.device,
            dtype=adv_rgb.dtype,
        )
        adv_expanded = adv_rgb.unsqueeze(0)

        x_plus = (adv_expanded + nes_sigma * noise).clamp(0.0, 1.0)
        x_minus = (adv_expanded - nes_sigma * noise).clamp(0.0, 1.0)

        x_plus_flat = x_plus.reshape(nes_m * batch_size, channels, height, width)
        x_minus_flat = x_minus.reshape(nes_m * batch_size, channels, height, width)
        x_all = torch.cat([x_plus_flat, x_minus_flat], dim=0)

        logits_all = _model_forward_batched(model, x_all, eval_batch_size=nes_eval_batch_size)

        logits_plus_flat = logits_all[: nes_m * batch_size]
        logits_minus_flat = logits_all[nes_m * batch_size :]

        targets_rep = original_classes.repeat(nes_m)
        losses_plus = F.cross_entropy(logits_plus_flat, targets_rep, reduction="none").view(nes_m, batch_size)
        losses_minus = F.cross_entropy(logits_minus_flat, targets_rep, reduction="none").view(nes_m, batch_size)

        coeff = ((losses_plus - losses_minus) / (2.0 * nes_sigma)).view(nes_m, batch_size, 1, 1, 1)
        grad_est = (coeff * noise).mean(dim=0)

        logits_plus = logits_plus_flat.view(nes_m, batch_size, -1)
        logits_minus = logits_minus_flat.view(nes_m, batch_size, -1)
        surrogate_logits = 0.5 * (logits_plus + logits_minus).mean(dim=0)

    return grad_est, surrogate_logits


def _run_nes_pgd_untargeted_batch(
    model: torch.nn.Module,
    clean_rgb: torch.Tensor,
    original_classes: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float,
    nes_m: int,
    nes_sigma: float,
    nes_eval_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[dict[str, float | int | str | None]]]]:
    batch_size = clean_rgb.shape[0]
    adv_rgb = clean_rgb.detach().clone()
    success_steps = torch.full((batch_size,), -1, device=clean_rgb.device, dtype=torch.long)
    history_by_sample: list[list[dict[str, float | int | str | None]]] = [[] for _ in range(batch_size)]

    for step in range(steps):
        grad_est, surrogate_logits = _estimate_nes_gradient_and_logits(
            model=model,
            adv_rgb=adv_rgb,
            original_classes=original_classes,
            nes_m=nes_m,
            nes_sigma=nes_sigma,
            nes_eval_batch_size=nes_eval_batch_size,
        )
        losses = F.cross_entropy(surrogate_logits, original_classes, reduction="none")
        probs = F.softmax(surrogate_logits, dim=1)
        pred_before_update = surrogate_logits.argmax(dim=1)

        if step > 0:
            newly_success = (success_steps == -1) & (pred_before_update != original_classes)
            success_steps[newly_success] = step

        grad_sign = grad_est.sign()
        updated = adv_rgb + step_size * grad_sign

        perturbation = (updated - clean_rgb).clamp(-epsilon, epsilon)
        adv_rgb = (clean_rgb + perturbation).clamp(0.0, 1.0).detach()

        for idx in range(batch_size):
            cls = int(original_classes[idx].item())
            history_by_sample[idx].append(
                {
                    "step": step + 1,
                    "loss": float(losses[idx].item()),
                    "prob_original_class": float(probs[idx, cls].item()),
                    "logit_original_class": float(surrogate_logits[idx, cls].item()),
                    "original_class": cls,
                    "pred_class": int(pred_before_update[idx].item()),
                    "loss_type": "crossentropy",
                    "target_class": None,
                }
            )

    with torch.no_grad():
        final_logits = _model_forward(model, adv_rgb)
        final_preds = final_logits.argmax(dim=1)

    final_success = (success_steps == -1) & (final_preds != original_classes)
    success_steps[final_success] = steps

    return adv_rgb.detach(), final_preds, success_steps.detach(), history_by_sample


def run_attack(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    model = load_model(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        # checkpoint=checkpoint,
        # checkpoint_dir=Path(args.checkpoint_dir),
    )
    model.eval()

    categories: list[str] = []
    try:
        categories = load_imagenet_categories()
    except Exception:
        categories = []

    samples = _load_attack_samples(Path(args.attack_json))
    output_root = Path(args.output_root) / args.model_type / args.model_name / "NES_PGD"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(samples)} classes from {args.attack_json}")
    print(f"Model: {args.model_type}/{args.model_name}")
    print(f"Output root: {output_root}")

    records: list[tuple[int, Path]] = []
    for class_id_str, image_list in samples.items():
        if not image_list:
            print(f"[skip] class {class_id_str}: empty image list")
            continue
        image_path = Path(image_list[0])
        class_id = int(class_id_str)
        if not image_path.exists():
            print(f"[skip] class {class_id}: file not found: {image_path}")
            continue
        records.append((class_id, image_path))

    if not records:
        print("No valid samples to attack.")
        return

    for epsilon in args.epsilons:
        step_size = args.step_size if args.step_size is not None else (2.5 * float(epsilon) / args.steps)
        eps_dir = output_root / _format_epsilon_dir(float(epsilon))
        eps_dir.mkdir(parents=True, exist_ok=True)

        for start in tqdm.tqdm(range(0, len(records), args.batch_size), desc=f"epsilon={epsilon}"):
            batch_records = records[start : start + args.batch_size]

            clean_tensors: list[torch.Tensor] = []
            class_ids: list[int] = []
            image_paths: list[Path] = []
            image_names: list[str] = []

            for class_id, image_path in batch_records:
                pil_image = Image.open(image_path).convert("RGB")
                clean_tensor = model.transform.spatial_transform(pil_image)
                clean_tensors.append(clean_tensor)
                class_ids.append(class_id)
                image_paths.append(image_path)
                image_names.append(image_path.stem)

            clean_rgb = torch.stack(clean_tensors, dim=0).to(device)
            class_tensor = torch.tensor(class_ids, device=device, dtype=torch.long)

            with torch.no_grad():
                logits_clean = _model_forward(model, clean_rgb)
                pred_clean = logits_clean.argmax(dim=1)

            adv_rgb, final_preds, success_steps, history_batch = _run_nes_pgd_untargeted_batch(
                model=model,
                clean_rgb=clean_rgb,
                original_classes=class_tensor,
                epsilon=float(epsilon),
                steps=args.steps,
                step_size=float(step_size),
                nes_m=int(args.nes_m),
                nes_sigma=float(args.nes_sigma),
                nes_eval_batch_size=int(args.nes_eval_batch_size),
            )

            for idx in range(len(batch_records)):
                class_id = class_ids[idx]
                image_path = image_paths[idx]
                image_name = image_names[idx]
                clean_rgb_i = clean_rgb[idx : idx + 1]
                adv_rgb_i = adv_rgb[idx : idx + 1]
                perturbation_i = adv_rgb_i - clean_rgb_i
                history = history_batch[idx]
                final_pred = int(final_preds[idx].item())
                success_step = int(success_steps[idx].item())

                sample_out_dir = eps_dir / image_name
                sample_out_dir.mkdir(parents=True, exist_ok=True)

                adv_png_path = sample_out_dir / "adv.png"
                clean_png_path = sample_out_dir / "clean.png"
                perturb_png_path = sample_out_dir / "perturbation.png"
                adv_tensor_path = sample_out_dir / "adv.pt"
                metadata_path = sample_out_dir / "metadata.json"
                history_txt_path = sample_out_dir / "history.txt"
                history_json_path = sample_out_dir / "history.json"

                save_rgb_image(adv_rgb_i, adv_png_path)
                save_rgb_image(clean_rgb_i, clean_png_path)
                save_perturbation_image(perturbation_i, perturb_png_path, epsilon=float(epsilon))
                torch.save(adv_rgb_i.detach().cpu(), adv_tensor_path)

                metadata = {
                    "image_path": str(image_path),
                    "class_id": class_id,
                    "class_name": categories[class_id] if categories and 0 <= class_id < len(categories) else None,
                    "model_type": args.model_type,
                    "model_name": args.model_name,
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "attack": "NES_PGD",
                    "gradient_estimator": "NES",
                    "nes_m": int(args.nes_m),
                    "nes_sigma": float(args.nes_sigma),
                    "epsilon": float(epsilon),
                    "steps": int(args.steps),
                    "step_size": float(step_size),
                    "targeted": False,
                    "loss": "cross_entropy",
                    "clean_pred": int(pred_clean[idx].item()),
                    "final_pred": final_pred,
                    "success": bool(success_step != -1),
                    "success_step": success_step,
                    "history_len": len(history),
                    "output_files": {
                        "adv_png": str(adv_png_path),
                        "clean_png": str(clean_png_path),
                        "perturbation_png": str(perturb_png_path),
                        "adv_tensor": str(adv_tensor_path),
                        "history_txt": str(history_txt_path),
                        "history_json": str(history_json_path),
                    },
                }

                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                with history_txt_path.open("w", encoding="utf-8") as f:
                    for row in history:
                        f.write(f"{row['loss']}\n")

                with history_json_path.open("w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NES+PGD attack over samples in attack_1k.json")
    parser.add_argument("--model-type", type=str, default="bcosify", choices=["torchvision", "bcos", "bcosify"])
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, default=str(Path(CHECKPOINT_DIR)))
    parser.add_argument(
        "--attack-json",
        type=str,
        default=str(Path(CLASSIFICATION_RESULT_DIR) / "attack_1k.json"),
        help="Input json containing one sample per class",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(PROJECT_ROOT) / "attack_result"),
        help="Root output folder",
    )
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.03, 0.05, 0.1, 0.2])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for attack samples")
    parser.add_argument("--nes-m", type=int, default=50, help="Number of Gaussian directions for NES")
    parser.add_argument("--nes-sigma", type=float, default=1.0 / 255.0, help="NES smoothing parameter")
    parser.add_argument(
        "--nes-eval-batch-size",
        type=int,
        default=256,
        help="Chunk size for batched NES forward evaluations",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="If omitted, uses 2.5 * epsilon / steps for each epsilon",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0...")
    return parser.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
