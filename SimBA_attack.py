from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import tqdm
from PIL import Image

from attack.SimBaAttack import SimBAAttack
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
	output_root = Path(args.output_root) / args.model_type / args.model_name / "SimBA"
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
		eps_dir = output_root / _format_epsilon_dir(float(epsilon))
		eps_dir.mkdir(parents=True, exist_ok=True)

		attacker = SimBAAttack(
			model=model,
			epsilon=float(epsilon),
			steps=int(args.steps),
			order=args.order,
			freq_dims=int(args.freq_dims),
			stride=int(args.stride),
			pixel_attack=bool(args.pixel_attack),
			linf_bound=float(args.linf_bound),
			image_size=int(args.image_size),
		)

		for class_id, image_path in tqdm.tqdm(records, desc=f"epsilon={epsilon}"):
			pil_image = Image.open(image_path).convert("RGB")
			clean_rgb = model.transform.spatial_transform(pil_image).unsqueeze(0).to(device)

			with torch.no_grad():
				clean_logits = model(
					model.transform.inverse_transform(clean_rgb)
					if hasattr(model.transform, "inverse_transform")
					else clean_rgb
				)
				clean_pred = int(clean_logits.argmax(dim=1).item())

			(
				adv_rgb,
				final_pred,
				success_step,
				queries,
				history,
				_,
				_,
				final_best_score,
			) = attacker.solve(
				clean_rgb=clean_rgb,
				original_class=class_id,
				target_class=None,
				targeted=False,
				log_every=1,
				stop_on_success=args.stop_on_success,
				device=device,
			)

			clean_rgb_cpu = clean_rgb.detach().cpu()
			adv_rgb_cpu = adv_rgb.detach().cpu()
			perturbation = adv_rgb_cpu - clean_rgb_cpu

			image_name = image_path.stem
			sample_out_dir = eps_dir / image_name
			sample_out_dir.mkdir(parents=True, exist_ok=True)

			adv_png_path = sample_out_dir / "adv.png"
			clean_png_path = sample_out_dir / "clean.png"
			perturb_png_path = sample_out_dir / "perturbation.png"
			adv_tensor_path = sample_out_dir / "adv.pt"
			metadata_path = sample_out_dir / "metadata.json"
			history_txt_path = sample_out_dir / "history.txt"
			history_json_path = sample_out_dir / "history.json"

			save_rgb_image(adv_rgb_cpu, adv_png_path)
			save_rgb_image(clean_rgb_cpu, clean_png_path)
			save_perturbation_image(perturbation, perturb_png_path, epsilon=float(epsilon))
			torch.save(adv_rgb_cpu, adv_tensor_path)

			metadata = {
				"image_path": str(image_path),
				"class_id": int(class_id),
				"class_name": categories[class_id] if categories and 0 <= class_id < len(categories) else None,
				"model_type": args.model_type,
				"model_name": args.model_name,
				"checkpoint": str(checkpoint) if checkpoint is not None else None,
				"attack": "SimBA",
				"epsilon": float(epsilon),
				"steps": int(args.steps),
				"order": args.order,
				"freq_dims": int(args.freq_dims),
				"stride": int(args.stride),
				"pixel_attack": bool(args.pixel_attack),
				"linf_bound": float(args.linf_bound),
				"image_size": int(args.image_size),
				"targeted": False,
				"objective": "crossentropy",
				"clean_pred": clean_pred,
				"final_pred": int(final_pred),
				"success": bool(success_step != -1),
				"success_step": int(success_step),
				"queries": int(queries),
				"history_len": len(history),
				"final_best_score": float(final_best_score),
				"stop_on_success": bool(args.stop_on_success),
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
					# Keep same convention as PGD: one scalar per line.
					f.write(f"{row['loss']}\n")

			with history_json_path.open("w", encoding="utf-8") as f:
				json.dump(history, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run SimBA attack over samples in attack_1k.json")
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
	parser.add_argument("--order", type=str, default="rand", choices=["rand", "diag", "strided", "block"])
	parser.add_argument("--freq-dims", type=int, default=14)
	parser.add_argument("--stride", type=int, default=7)
	parser.add_argument("--pixel-attack", action="store_true")
	parser.add_argument("--linf-bound", type=float, default=0.0)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument(
		"--stop-on-success",
		action="store_true",
		help="Stop per sample right after first success (default: keep running all steps to log full history)",
	)
	parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0...")
	return parser.parse_args()


if __name__ == "__main__":
	run_attack(parse_args())
