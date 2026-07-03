import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from attack.util import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_MODEL_TYPE,
    load_model,
)

from dataloader import get_imagenet_dataloader


def _to_serializable(obj: dict[int, list[str]]) -> dict[str, list[str]]:
    return {str(class_id): paths for class_id, paths in sorted(obj.items())}


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    model = load_model(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )
    model.eval()

    if not hasattr(model, "transform") or not hasattr(model.transform, "spatial_transform"):
        raise ValueError("Loaded model must provide transform.spatial_transform for dataloader preprocessing.")

    loader, _ = get_imagenet_dataloader(
        img_dir=args.img_dir,
        annotations_file=args.annotations_file,
        batch_size=args.batch_size,
        transform=model.transform.spatial_transform,
        num_workers=args.num_workers,
        shuffle=False,
    )

    total_correct = 0
    total_samples = 0
    correct_by_class: dict[int, list[str]] = {}

    for images, class_ids, class_names_batch, img_paths, _ in tqdm(loader):
        _ = class_names_batch
        images = images.to(device, non_blocking=True)
        class_ids = torch.as_tensor(class_ids, device=device, dtype=torch.long)
        model_input = model.transform.inverse_transform(images)

        with torch.no_grad():
            logits = model(model_input)
            predictions = logits.argmax(dim=1)

        matches = predictions.eq(class_ids)
        total_correct += int(matches.sum().item())
        total_samples += int(class_ids.numel())

        for sample_idx, is_match in enumerate(matches.tolist()):
            if not bool(is_match):
                continue
            class_id = int(class_ids[sample_idx].item())
            correct_by_class.setdefault(class_id, []).append(str(img_paths[sample_idx]))

    top1_acc = float(total_correct / total_samples) if total_samples > 0 else 0.0

    print(f"Top-1 accuracy: {top1_acc:.4f}")
    print(f"Number of correct samples: {total_correct}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_tag = "custom" if args.checkpoint is not None else "default"
    output_path = out_dir / f"{args.model_type}_{args.model_name}_{checkpoint_tag}_imagenet_correct_paths.json"
    summary_path = out_dir / f"{args.model_type}_{args.model_name}_{checkpoint_tag}_summary.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(correct_by_class), f, ensure_ascii=False, indent=2)

    summary = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "checkpoint_dir": str(args.checkpoint_dir),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "top1_acc": top1_acc,
        "num_classes_with_correct": len(correct_by_class),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify ImageNet-style folder data and save correct paths by class id.")

    parser.add_argument("--img-dir", type=str, required=True, help="Root directory of class folders with images.")
    parser.add_argument("--annotations-file", type=str, required=True, help="JSON mapping folder -> [class_id, class_name].")

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bcos", "torchvision", "bcosify"],
        default=DEFAULT_MODEL_TYPE,
        help="Model backend to use.",
    )
    parser.add_argument("--model-name", type=str, default="resnet50", help="Model name to load.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory used when --checkpoint is not provided.",
    )

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataloader.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--output-dir", type=Path, default=Path("attack_result") / "classify", help="Output directory.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use: auto, cpu, cuda, cuda:0...",
    )

    args = parser.parse_args()
    main(args)