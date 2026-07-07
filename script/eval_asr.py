from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from const import PROJECT_ROOT


def _collect_metadata_files(attack_root: Path, attack_method: str) -> list[Path]:
    return sorted(attack_root.glob(f"*/*/{attack_method}/epsilon_*/*/metadata.json"))


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def eval_asr(attack_root: Path, attack_method: str) -> tuple[list[dict], dict]:
    files = _collect_metadata_files(attack_root, attack_method)
    if not files:
        raise FileNotFoundError(f"No metadata.json found under: {attack_root}")

    grouped: dict[tuple[str, str, str], dict] = defaultdict(lambda: {
        "total": 0,
        "success": 0,
        "success_steps": [],
        "step_sizes": [],
    })

    for metadata_path in files:
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        model_type = str(meta.get("model_type", "unknown"))
        model_name = str(meta.get("model_name", "unknown"))
        epsilon = f"{_safe_float(meta.get('epsilon', 0.0)):.6g}"

        key = (model_type, model_name, epsilon)
        bucket = grouped[key]
        bucket["total"] += 1

        is_success = bool(meta.get("success", False))
        if is_success:
            bucket["success"] += 1
            success_step = int(meta.get("success_step", -1))
            if success_step > 0:
                bucket["success_steps"].append(success_step)

        step_size = _safe_float(meta.get("step_size", 0.0), 0.0)
        if step_size > 0:
            bucket["step_sizes"].append(step_size)

    rows: list[dict] = []
    total_all = 0
    success_all = 0

    for (model_type, model_name, epsilon), bucket in sorted(grouped.items()):
        total = int(bucket["total"])
        success = int(bucket["success"])
        asr = (success / total) if total > 0 else 0.0

        success_steps = bucket["success_steps"]
        step_sizes = bucket["step_sizes"]

        rows.append(
            {
                "model_type": model_type,
                "model_name": model_name,
                "epsilon": epsilon,
                "total": total,
                "success": success,
                "asr": asr,
                "mean_success_step": (sum(success_steps) / len(success_steps)) if success_steps else None,
                "mean_step_size": (sum(step_sizes) / len(step_sizes)) if step_sizes else None,
            }
        )

        total_all += total
        success_all += success

    overall = {
        "total": total_all,
        "success": success_all,
        "asr": (success_all / total_all) if total_all > 0 else 0.0,
        "num_groups": len(rows),
    }

    return rows, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PGD attack ASR from attack_result metadata files.")
    parser.add_argument(
        "--attack-root",
        type=Path,
        default=Path(PROJECT_ROOT) / "attack_result",
        help="Root directory containing attack outputs.",
    )
    parser.add_argument(
        "--attack-method",
        type=str,
        default="PGD",
        help="Attack method to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(PROJECT_ROOT) / "eval_result",
        help="Directory to save ASR reports.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="asr",
        help="Output filename prefix.",
    )
    args = parser.parse_args()

    rows, overall = eval_asr(args.attack_root, args.attack_method)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.prefix}_summary.json"
    csv_path = args.output_dir / f"{args.prefix}_summary.csv"

    report = {
        "attack_root": str(args.attack_root),
        "overall": overall,
        "groups": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "model_name",
                "epsilon",
                "total",
                "success",
                "asr",
                "mean_success_step",
                "mean_step_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Overall ASR: {overall['asr']:.4f} ({overall['success']}/{overall['total']})")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
