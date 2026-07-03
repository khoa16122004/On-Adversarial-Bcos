from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import tqdm

from attack.util import load_model
from script.const import CHECKPOINT_DIR, PROJECT_ROOT


ALL_MODELS: list[tuple[str, str]] = [
    ("torchvision", "resnet50"),
    ("torchvision", "densenet121"),
    ("torchvision", "vit_b_16"),
    ("bcos", "resnet50"),
    ("bcos", "densenet121"),
    ("bcos", "simple_vit_b_patch16_224"),
    ("bcosify", "resnet50"),
    ("bcosify", "densenet121"),
    ("bcosify", "simple_vit_b_patch16_224"),
]


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _source_attack_root(base_attack_root: Path, model_type: str, model_name: str) -> Path:
    return base_attack_root / model_type / model_name / "PGD"


def _parse_target_list(raw: str | None, source: tuple[str, str]) -> list[tuple[str, str]]:
    if raw is None or raw.strip().lower() == "all":
        return [m for m in ALL_MODELS if m != source]

    parsed: list[tuple[str, str]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid target token '{token}'. Use format model_type:model_name,model_type:model_name"
            )
        parsed.append((parts[0], parts[1]))
    return parsed


def _load_checkpoint_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Checkpoint override file must be a dict: {'model_type:model_name': '/path/to/ckpt'}")
    return {str(k): str(v) for k, v in data.items()}


def _load_adv_records(epsilon_dir: Path) -> list[dict]:
    records: list[dict] = []
    for image_dir in sorted(p for p in epsilon_dir.iterdir() if p.is_dir()):
        adv_path = image_dir / "adv.pt"
        metadata_path = image_dir / "metadata.json"
        if not adv_path.exists() or not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        records.append(
            {
                "image_name": image_dir.name,
                "adv_path": adv_path,
                "class_id": int(meta["class_id"]),
                "source_clean_pred": int(meta.get("clean_pred", meta["class_id"])),
            }
        )
    return records


def _predict_batch(model: torch.nn.Module, rgb_batch: torch.Tensor) -> torch.Tensor:
    transform = getattr(model, "transform", None)
    if transform is not None and hasattr(transform, "inverse_transform"):
        model_input = transform.inverse_transform(rgb_batch)
    else:
        model_input = rgb_batch
    with torch.no_grad():
        logits = model(model_input)
    return logits.argmax(dim=1)


def _evaluate_transfer(
    model: torch.nn.Module,
    device: torch.device,
    records: list[dict],
    batch_size: int,
) -> dict:
    total = len(records)
    success_vs_true = 0
    success_vs_source_clean = 0
    details: list[dict] = []

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        rgb_batch = []
        for item in batch:
            tensor = torch.load(item["adv_path"], map_location="cpu")
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            rgb_batch.append(tensor.squeeze(0))
        rgb_batch_t = torch.stack(rgb_batch, dim=0).to(device)
        preds = _predict_batch(model, rgb_batch_t).detach().cpu().tolist()

        for item, pred in zip(batch, preds):
            class_id = int(item["class_id"])
            source_clean_pred = int(item["source_clean_pred"])
            succ_true = int(pred != class_id)
            succ_src = int(pred != source_clean_pred)
            success_vs_true += succ_true
            success_vs_source_clean += succ_src
            details.append(
                {
                    "image_name": item["image_name"],
                    "class_id": class_id,
                    "source_clean_pred": source_clean_pred,
                    "target_adv_pred": int(pred),
                    "success_vs_true": bool(succ_true),
                    "success_vs_source_clean": bool(succ_src),
                }
            )

    asr_true = (success_vs_true / total) if total > 0 else 0.0
    asr_source_clean = (success_vs_source_clean / total) if total > 0 else 0.0
    return {
        "total": total,
        "success_vs_true": success_vs_true,
        "success_vs_source_clean": success_vs_source_clean,
        "asr_vs_true": asr_true,
        "asr_vs_source_clean": asr_source_clean,
        "details": details,
    }


def run_transfer(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)

    source = (args.source_model_type, args.source_model_name)
    targets = _parse_target_list(args.targets, source)
    checkpoint_overrides = _load_checkpoint_overrides(
        Path(args.checkpoint_override_json) if args.checkpoint_override_json else None
    )

    attack_root = _source_attack_root(Path(args.attack_root), args.source_model_type, args.source_model_name)
    if not attack_root.exists():
        raise FileNotFoundError(f"Source PGD folder not found: {attack_root}")

    epsilon_dirs = sorted(p for p in attack_root.iterdir() if p.is_dir() and p.name.startswith("epsilon_"))
    if args.epsilons:
        wanted = {f"epsilon_{format(e, '.4f').rstrip('0').rstrip('.')}" for e in args.epsilons}
        epsilon_dirs = [p for p in epsilon_dirs if p.name in wanted]
    if not epsilon_dirs:
        raise FileNotFoundError(f"No epsilon directories found under {attack_root}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    full_report: dict = {
        "source": {"model_type": args.source_model_type, "model_name": args.source_model_name},
        "device": str(device),
        "results": {},
    }

    for target_type, target_name in targets:
        target_key = f"{target_type}:{target_name}"
        print(f"[target] {target_key}")

        override_ckpt = checkpoint_overrides.get(target_key)
        model = load_model(
            model_type=target_type,
            model_name=target_name,
            device=device,
            checkpoint=Path(override_ckpt) if override_ckpt else None,
            checkpoint_dir=Path(args.checkpoint_dir),
        )
        model.eval()

        full_report["results"][target_key] = {}
        for eps_dir in tqdm.tqdm(epsilon_dirs, desc=f"eval {target_key}"):
            epsilon_name = eps_dir.name
            records = _load_adv_records(eps_dir)
            metrics = _evaluate_transfer(
                model=model,
                device=device,
                records=records,
                batch_size=args.batch_size,
            )

            full_report["results"][target_key][epsilon_name] = {
                "metrics": {
                    "total": metrics["total"],
                    "success_vs_true": metrics["success_vs_true"],
                    "success_vs_source_clean": metrics["success_vs_source_clean"],
                    "asr_vs_true": metrics["asr_vs_true"],
                    "asr_vs_source_clean": metrics["asr_vs_source_clean"],
                },
                "details": metrics["details"],
            }

            summary_rows.append(
                {
                    "source_model_type": args.source_model_type,
                    "source_model_name": args.source_model_name,
                    "target_model_type": target_type,
                    "target_model_name": target_name,
                    "epsilon": epsilon_name,
                    "total": metrics["total"],
                    "success_vs_true": metrics["success_vs_true"],
                    "asr_vs_true": metrics["asr_vs_true"],
                    "success_vs_source_clean": metrics["success_vs_source_clean"],
                    "asr_vs_source_clean": metrics["asr_vs_source_clean"],
                }
            )

    report_json = out_root / (
        f"transfer_{args.source_model_type}_{args.source_model_name}.json"
    )
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    report_csv = out_root / (
        f"transfer_{args.source_model_type}_{args.source_model_name}_summary.csv"
    )
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_model_type",
                "source_model_name",
                "target_model_type",
                "target_model_name",
                "epsilon",
                "total",
                "success_vs_true",
                "asr_vs_true",
                "success_vs_source_clean",
                "asr_vs_source_clean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved report: {report_json}")
    print(f"Saved summary: {report_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate transfer attack from source PGD outputs to target models")
    parser.add_argument("--source-model-type", type=str, required=True, choices=["torchvision", "bcos", "bcosify"])
    parser.add_argument("--source-model-name", type=str, required=True)
    parser.add_argument(
        "--attack-root",
        type=str,
        default=str(Path(PROJECT_ROOT) / "attack_result"),
        help="Root folder containing PGD attack outputs",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(PROJECT_ROOT) / "transfer_result"),
        help="Output folder for transfer reports",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="all",
        help="'all' or comma-separated list like torchvision:resnet50,bcos:resnet50",
    )
    parser.add_argument(
        "--checkpoint-override-json",
        type=str,
        default=None,
        help="Optional JSON mapping 'model_type:model_name' -> checkpoint_path",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(Path(CHECKPOINT_DIR)))
    parser.add_argument("--epsilons", type=float, nargs="*", default=None, help="Optional epsilon list filter")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0...")
    return parser.parse_args()


if __name__ == "__main__":
    run_transfer(parse_args())