from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attack.util import DEFAULT_CHECKPOINT_DIR, load_model, save_explanation_rgba
from script.const import ANNOTATIONS_FILE, IMAGENET_TRAIN_DATA, IMAGENET_VAL_DATA
from script.dataloader import get_imagenet_dataloader

from trades import trades_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRADES adversarial training on ImageNet-style folders for torchvision/bcos/bcosify models."
    )
    parser.add_argument("--model-type", type=str, choices=["torchvision", "bcos", "bcosify"], required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional starting checkpoint path.")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Initialize model randomly without loading pretrained/default/checkpoint weights.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory used by bcos/bcosify when --checkpoint is not provided.",
    )

    parser.add_argument("--train-dir", type=str, default=IMAGENET_TRAIN_DATA)
    parser.add_argument("--val-dir", type=str, default=IMAGENET_VAL_DATA)
    parser.add_argument("--annotations-file", type=str, default=ANNOTATIONS_FILE)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--disable-dataloader-cache",
        action="store_true",
        help="Disable filesystem cache for image index metadata.",
    )
    parser.add_argument(
        "--dataloader-cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for dataloader index metadata.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Number of prefetched batches per worker (only when num-workers > 0).",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent dataloader workers across epochs.",
    )

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--epsilon", type=float, default=4.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=1.0 / 255.0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--distance", type=str, choices=["l_inf", "l_2"], default="l_inf")
    parser.add_argument(
        "--train-objective",
        type=str,
        choices=["trades", "clean"],
        default="trades",
        help="trades: supervised + KL robust term, clean: supervised loss only (no KL/PGD branch).",
    )
    parser.add_argument(
        "--supervised-loss",
        type=str,
        choices=["auto", "ce", "bce", "bce_uniform"],
        default="auto",
        help="Supervised classification loss. auto: ce for torchvision, bce_uniform for bcos/bcosify.",
    )
    parser.add_argument(
        "--bce-off-label",
        type=float,
        default=None,
        help="Off-label target value used when supervised-loss is bce_uniform. None means 1/num_classes.",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--debug-grad-every-steps",
        type=int,
        default=0,
        help="If >0, log gradient/update diagnostics every N global steps.",
    )
    parser.add_argument(
        "--val-every-steps",
        type=int,
        default=100,
        help="Run validation once every N training iterations (0 to disable).",
    )
    parser.add_argument(
        "--explain-every-steps",
        type=int,
        default=100,
        help="Save explanation snapshots once every N training iterations (0 to disable).",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints") / "trades")
    parser.add_argument(
        "--visualize-json",
        type=Path,
        default=Path("visualize_during_train.json"),
        help="JSON mapping class_id -> [image_path] used for saving explanation snapshots.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _to_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _args_to_jsonable_dict(args: argparse.Namespace) -> dict[str, object]:
    return {k: _to_jsonable(v) for k, v in vars(args).items()}


def _compute_supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervised_loss: str,
    bce_off_label: float | None,
    reduction: str = "mean",
) -> torch.Tensor:
    if supervised_loss == "ce":
        return F.cross_entropy(logits, targets, reduction=reduction)

    num_classes = logits.shape[-1]
    bce_targets = F.one_hot(targets, num_classes=num_classes).to(dtype=logits.dtype)
    if supervised_loss == "bce_uniform":
        off_label = bce_off_label if bce_off_label is not None else 1.0 / float(num_classes)
        bce_targets = bce_targets.clamp(min=off_label)
    return F.binary_cross_entropy_with_logits(logits, bce_targets, reduction=reduction)


def _resolve_supervised_loss(args: argparse.Namespace) -> str:
    if args.supervised_loss != "auto":
        return args.supervised_loss
    if args.model_type == "torchvision":
        return "ce"
    return "bce_uniform"


def _compute_grad_l2_norm(model: torch.nn.Module) -> float:
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float(torch.sum(grad * grad).item())
    return total_sq ** 0.5


def _load_visualize_records(path: Path) -> list[tuple[int, Path]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("visualize json must be a dict: {class_id: [image_path, ...]}")

    records: list[tuple[int, Path]] = []
    for class_id_raw, image_list in payload.items():
        if not isinstance(image_list, list) or not image_list:
            continue
        class_id = int(class_id_raw)
        image_path = Path(str(image_list[0]))
        if image_path.exists():
            records.append((class_id, image_path))
    return records


def _unpack_explain_output(explain_result: object) -> dict:
    if isinstance(explain_result, tuple):
        if not explain_result:
            raise ValueError("model.explain returned empty tuple")
        explain_result = explain_result[0]
    if not isinstance(explain_result, dict):
        raise TypeError(f"Unsupported explain output type: {type(explain_result)}")
    return explain_result


def generate_explanations_for_epoch(
    model: torch.nn.Module,
    device: torch.device,
    records: list[tuple[int, Path]],
    out_root: Path,
    epoch_tag: str,
) -> dict[str, int]:
    if not records:
        return {"saved": 0, "skipped": 0, "errors": 0}
    if not hasattr(model, "explain"):
        return {"saved": 0, "skipped": len(records), "errors": 0}

    stats = {"saved": 0, "skipped": 0, "errors": 0}
    epoch_dir = out_root / epoch_tag
    epoch_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    for class_id, image_path in records:
        sample_name = image_path.stem
        sample_dir = epoch_dir / f"class_{class_id}" / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            pil_image = Image.open(image_path).convert("RGB")
            clean_rgb = model.transform.spatial_transform(pil_image).unsqueeze(0).to(device)
            model_input = model.transform.inverse_transform(clean_rgb)
            explain_result = model.explain(model_input.detach().clone().requires_grad_(True), idx=class_id)
            explain_out = _unpack_explain_output(explain_result)
            explanation = explain_out.get("explanation")
            if explanation is None:
                stats["skipped"] += 1
                continue
            save_explanation_rgba(explanation, sample_dir / "explanation.png")
            stats["saved"] += 1
        except Exception:
            stats["errors"] += 1
    if was_training:
        model.train()
    return stats


def run_eval(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    supervised_loss: str,
    bce_off_label: float | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for images, class_ids, _, _, _ in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            targets = torch.as_tensor(class_ids, device=device, dtype=torch.long)

            logits = model(model.transform.inverse_transform(images))
            loss = _compute_supervised_loss(
                logits,
                targets,
                supervised_loss=supervised_loss,
                bce_off_label=bce_off_label,
                reduction="sum",
            )

            preds = logits.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total_loss += float(loss.item())
            total_seen += int(targets.numel())

    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    if was_training:
        model.train()
    return {"loss": avg_loss, "acc": acc, "samples": float(total_seen)}


def save_best_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    best_epoch: int,
    best_train_loss: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{args.model_type}_{args.model_name}_trades_best_train_loss.pth"
    serialized_args = _args_to_jsonable_dict(args)
    payload = {
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "model_type": args.model_type,
        "model_name": args.model_name,
        "args": serialized_args,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }
    torch.save(payload, best_path)
    return best_path


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    supervised_loss = _resolve_supervised_loss(args)

    model = load_model(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        checkpoint=args.checkpoint,
        # checkpoint_dir=args.checkpoint_dir,
        # from_scratch=args.from_scratch,
    )
    model.train()

    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params}")

    monitor_param: torch.nn.Parameter | None = next(
        (p for p in model.parameters() if p.requires_grad),
        None,
    )
    if monitor_param is None:
        raise ValueError("Model has no trainable parameters (all requires_grad=False).")

    if not hasattr(model, "transform") or not hasattr(model.transform, "spatial_transform"):
        raise ValueError("Loaded model must expose transform.spatial_transform.")

    # train_loader, _ = get_imagenet_dataloader(
    #     img_dir=args.train_dir,
    #     annotations_file=args.annotations_file,
    #     batch_size=args.batch_size,
    #     transform=model.transform.spatial_transform,
    #     num_workers=args.num_workers,
    #     shuffle=True,
    # )
    train_loader, _ = get_imagenet_dataloader( # for test
        img_dir=args.train_dir,
        annotations_file=args.annotations_file,
        batch_size=args.batch_size,
        transform=model.transform.spatial_transform,
        num_workers=args.num_workers,
        shuffle=True,
        use_cache=not args.disable_dataloader_cache,
        cache_dir=args.dataloader_cache_dir,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader, _ = get_imagenet_dataloader(
        img_dir=args.val_dir,
        annotations_file=args.annotations_file,
        batch_size=args.val_batch_size,
        transform=model.transform.spatial_transform,
        num_workers=args.num_workers,
        shuffle=False,
        use_cache=not args.disable_dataloader_cache,
        cache_dir=args.dataloader_cache_dir,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float]] = []
    best_train_loss = float("inf")
    best_epoch = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serialized_args = _args_to_jsonable_dict(args)
    config_path = args.output_dir / f"{args.model_type}_{args.model_name}_trades_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_args, f, ensure_ascii=False, indent=2)

    epoch_log_path = args.output_dir / f"{args.model_type}_{args.model_name}_trades_epoch_log.jsonl"
    with epoch_log_path.open("w", encoding="utf-8") as f:
        f.write("")

    iter_log_path = args.output_dir / f"{args.model_type}_{args.model_name}_trades_iter_log.jsonl"
    with iter_log_path.open("w", encoding="utf-8") as f:
        f.write("")

    val_step_log_path = args.output_dir / f"{args.model_type}_{args.model_name}_trades_val_step_log.jsonl"
    with val_step_log_path.open("w", encoding="utf-8") as f:
        f.write("")

    enable_explanation_saving = args.model_type != "torchvision"
    explain_log_path: Path | None = None
    explain_out_root: Path | None = None
    visualize_records: list[tuple[int, Path]] = []
    if enable_explanation_saving:
        explain_log_path = args.output_dir / f"{args.model_type}_{args.model_name}_trades_explain_log.jsonl"
        with explain_log_path.open("w", encoding="utf-8") as f:
            f.write("")
        visualize_records = _load_visualize_records(args.visualize_json)
        explain_out_root = args.output_dir / f"{args.model_type}_{args.model_name}_explanations"
        explain_out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Start TRADES training")
    print(f"Model: {args.model_type}/{args.model_name}")
    print(f"From scratch: {args.from_scratch}")
    print(f"Train objective: {args.train_objective}")
    print(f"Supervised loss: {supervised_loss}")
    if supervised_loss == "bce_uniform":
        if args.bce_off_label is None:
            print("BCE off-label: auto (1/num_classes)")
        else:
            print(f"BCE off-label: {args.bce_off_label}")
    print(f"Train dir: {args.train_dir}")
    print(f"Val dir: {args.val_dir}")
    print(f"Dataloader cache: {not args.disable_dataloader_cache}")
    print(f"Dataloader cache dir: {args.dataloader_cache_dir}")
    print(f"Dataloader persistent workers: {not args.no_persistent_workers}")
    print(f"Dataloader prefetch factor: {args.prefetch_factor}")
    print(f"Iter log: {iter_log_path}")
    print(f"Val every steps: {args.val_every_steps}")
    print(f"Explain every steps: {args.explain_every_steps}")
    if enable_explanation_saving:
        print(f"Explain json: {args.visualize_json}")
        print(f"Explain samples: {len(visualize_records)}")
    else:
        print("Explain saving: disabled for torchvision")
    print("=" * 70)

    if enable_explanation_saving and explain_log_path is not None and explain_out_root is not None:
        start_explain_stats = generate_explanations_for_epoch(
            model=model,
            device=device,
            records=visualize_records,
            out_root=explain_out_root,
            epoch_tag="epoch_000_start",
        )
        with explain_log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": 0,
                        "tag": "epoch_000_start",
                        "saved": int(start_explain_stats["saved"]),
                        "skipped": int(start_explain_stats["skipped"]),
                        "errors": int(start_explain_stats["errors"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch_idx, (images, class_ids, _, _, _) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            targets = torch.as_tensor(class_ids, device=device, dtype=torch.long)

            loss = trades_loss(
                model=model,
                x_natural=images,
                y=targets,
                optimizer=optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
                distance=args.distance,
                natural_loss=supervised_loss,
                bce_off_label=args.bce_off_label,
                use_robust_loss=args.train_objective == "trades",
                preprocess=model.transform.inverse_transform,
                clip_min=0.0,
                clip_max=1.0,
            )
            loss.backward()

            debug_grad = args.debug_grad_every_steps > 0 and global_step % args.debug_grad_every_steps == 0
            grad_l2_norm = 0.0
            monitor_grad_l2 = 0.0
            monitor_update_l2 = 0.0
            monitor_before = None
            if debug_grad:
                grad_l2_norm = _compute_grad_l2_norm(model)
                if monitor_param.grad is not None:
                    monitor_grad_l2 = float(monitor_param.grad.detach().norm().item())
                monitor_before = monitor_param.detach().clone()

            optimizer.step()

            batch_size = int(targets.numel())
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            global_step += 1

            iter_record = {
                "epoch": int(epoch),
                "iter": int(batch_idx),
                "global_step": int(global_step),
                "loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "batch_size": int(batch_size),
            }
            if debug_grad and monitor_before is not None:
                monitor_update_l2 = float((monitor_param.detach() - monitor_before).norm().item())
                iter_record["grad_l2_norm"] = grad_l2_norm
                iter_record["monitor_grad_l2"] = monitor_grad_l2
                iter_record["monitor_update_l2"] = monitor_update_l2
                iter_record["grad_is_finite"] = bool(torch.isfinite(torch.tensor(grad_l2_norm)))
                if grad_l2_norm <= 0.0 or monitor_update_l2 <= 0.0:
                    print(
                        "[debug-grad] potential stalled update "
                        f"step={global_step} grad_l2={grad_l2_norm:.6e} monitor_update_l2={monitor_update_l2:.6e}"
                    )

            with iter_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(iter_record, ensure_ascii=False) + "\n")

            if args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                val_step_stats = run_eval(
                    model=model,
                    loader=val_loader,
                    device=device,
                    supervised_loss=supervised_loss,
                    bce_off_label=args.bce_off_label,
                )
                val_step_record = {
                    "epoch": int(epoch),
                    "iter": int(batch_idx),
                    "global_step": int(global_step),
                    "val_loss": float(val_step_stats["loss"]),
                    "val_acc": float(val_step_stats["acc"]),
                    "val_samples": int(val_step_stats["samples"]),
                }
                with val_step_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(val_step_record, ensure_ascii=False) + "\n")
                progress.set_postfix(
                    {
                        "loss": f"{running_loss / max(seen, 1):.4f}",
                        "val_acc": f"{val_step_stats['acc']:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
                    }
                )

            if (
                enable_explanation_saving
                and explain_log_path is not None
                and explain_out_root is not None
                and args.explain_every_steps > 0
                and global_step % args.explain_every_steps == 0
            ):
                explain_step_stats = generate_explanations_for_epoch(
                    model=model,
                    device=device,
                    records=visualize_records,
                    out_root=explain_out_root,
                    epoch_tag=f"step_{global_step:07d}",
                )
                with explain_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "epoch": int(epoch),
                                "iter": int(batch_idx),
                                "global_step": int(global_step),
                                "tag": f"step_{global_step:07d}",
                                "saved": int(explain_step_stats["saved"]),
                                "skipped": int(explain_step_stats["skipped"]),
                                "errors": int(explain_step_stats["errors"]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if batch_idx % args.log_interval == 0:
                progress.set_postfix(
                    {
                        "loss": f"{running_loss / max(seen, 1):.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
                    }
                )

        scheduler.step()

        train_avg_loss = running_loss / max(seen, 1)
        val_stats = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            supervised_loss=supervised_loss,
            bce_off_label=args.bce_off_label,
        )
        record = {
            "epoch": int(epoch),
            "train_loss": train_avg_loss,
            "train_samples": int(seen),
            "val_loss": float(val_stats["loss"]),
            "val_acc": float(val_stats["acc"]),
            "val_samples": int(val_stats["samples"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        with epoch_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[epoch {epoch}] train_loss={train_avg_loss:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}"
        )

        if train_avg_loss < best_train_loss:
            best_train_loss = train_avg_loss
            best_epoch = epoch
            best_path = save_best_checkpoint(
                out_dir=args.output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                history=history,
                best_epoch=best_epoch,
                best_train_loss=best_train_loss,
            )
            print(f"Saved best checkpoint: {best_path} (train_loss={best_train_loss:.4f})")

        if enable_explanation_saving and explain_log_path is not None and explain_out_root is not None:
            explain_stats = generate_explanations_for_epoch(
                model=model,
                device=device,
                records=visualize_records,
                out_root=explain_out_root,
                epoch_tag=f"epoch_{epoch:03d}",
            )
            with explain_log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": int(epoch),
                            "tag": f"epoch_{epoch:03d}",
                            "saved": int(explain_stats["saved"]),
                            "skipped": int(explain_stats["skipped"]),
                            "errors": int(explain_stats["errors"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / f"{args.model_type}_{args.model_name}_trades_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Saved history: {history_path}")
    print(f"Saved config: {config_path}")
    print(f"Saved epoch log: {epoch_log_path}")
    print(f"Saved iter log: {iter_log_path}")
    print(f"Saved val-step log: {val_step_log_path}")
    if enable_explanation_saving and explain_log_path is not None and explain_out_root is not None:
        print(f"Saved explain log: {explain_log_path}")
        print(f"Saved explanations dir: {explain_out_root}")
    print(f"Best epoch: {best_epoch} | Best train_loss: {best_train_loss:.4f}")


if __name__ == "__main__":
    main()
