from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from attack.PGD import PGDAttack
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


def run_attack(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    model = load_model(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        checkpoint=checkpoint,
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    model.eval()

    categories: list[str] = []
    try:
        categories = load_imagenet_categories()
    except Exception:
        categories = []

    samples = _load_attack_samples(Path(args.attack_json))
    output_root = Path(args.output_root) / args.model_type / args.model_name / "PGD"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(samples)} classes from {args.attack_json}")
    print(f"Model: {args.model_type}/{args.model_name}")
    print(f"Output root: {output_root}")

    for class_id_str, image_list in samples.items():
        if not image_list:
            print(f"[skip] class {class_id_str}: empty image list")
            continue

        class_id = int(class_id_str)
        image_path = Path(image_list[0])
        if not image_path.exists():
            print(f"[skip] class {class_id}: file not found: {image_path}")
            continue

        pil_image = Image.open(image_path).convert("RGB")
        clean_rgb = model.transform.spatial_transform(pil_image).unsqueeze(0).to(device)
        image_name = image_path.stem

        with torch.no_grad():
            logits_clean = model(model.transform.inverse_transform(clean_rgb))
            pred_clean = int(logits_clean.argmax(dim=1).item())

        for epsilon in args.epsilons:
            step_size = args.step_size if args.step_size is not None else (2.5 * float(epsilon) / args.steps)

            attacker = PGDAttack(model=model, epsilon=float(epsilon))
            adv_rgb, final_pred, success_step, history = attacker.solve(
                clean_rgb=clean_rgb,
                original_class=class_id,
                step_size=float(step_size),
                steps=args.steps,
                target_class=None,
                loss_fn=None,
            )

            perturbation = adv_rgb - clean_rgb

            sample_out_dir = output_root / _format_epsilon_dir(float(epsilon)) / image_name
            sample_out_dir.mkdir(parents=True, exist_ok=True)

            adv_png_path = sample_out_dir / "adv.png"
            clean_png_path = sample_out_dir / "clean.png"
            perturb_png_path = sample_out_dir / "perturbation.png"
            adv_tensor_path = sample_out_dir / "adv.pt"
            metadata_path = sample_out_dir / "metadata.json"
            history_txt_path = sample_out_dir / "history.txt"
            history_json_path = sample_out_dir / "history.json"

            save_rgb_image(adv_rgb, adv_png_path)
            save_rgb_image(clean_rgb, clean_png_path)
            save_perturbation_image(perturbation, perturb_png_path, epsilon=float(epsilon))
            torch.save(adv_rgb.detach().cpu(), adv_tensor_path)

            metadata = {
                "image_path": str(image_path),
                "class_id": class_id,
                "class_name": categories[class_id] if categories and 0 <= class_id < len(categories) else None,
                "model_type": args.model_type,
                "model_name": args.model_name,
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "attack": "PGD",
                "epsilon": float(epsilon),
                "steps": int(args.steps),
                "step_size": float(step_size),
                "targeted": False,
                "loss": "cross_entropy",
                "clean_pred": pred_clean,
                "final_pred": int(final_pred),
                "success": bool(success_step != -1),
                "success_step": int(success_step),
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

            print(
                f"[done] class={class_id} img={image_name} eps={epsilon} "
                f"clean_pred={pred_clean} final_pred={final_pred} success_step={success_step}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PGD attack over samples in attack_1k.json")
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