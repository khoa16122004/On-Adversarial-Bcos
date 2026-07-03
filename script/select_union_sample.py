from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from const import CLASSIFICATION_RESULT_DIR


def _load_correct_paths(json_path: Path) -> dict[str, set[str]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid format in {json_path}: expected object/dict.")
    loaded: dict[str, set[str]] = {}
    for class_id, paths in data.items():
        if not isinstance(paths, list):
            raise ValueError(f"Invalid format in {json_path}: class '{class_id}' must map to a list.")
        loaded[str(class_id)] = set(str(p) for p in paths)
    return loaded


def _intersect_across_models(
    per_model: list[dict[str, set[str]]],
) -> tuple[dict[str, list[str]], list[str]]:
    if not per_model:
        return {}, []

    common_class_ids = set(per_model[0].keys())
    for model_dict in per_model[1:]:
        common_class_ids &= set(model_dict.keys())

    result: dict[str, list[str]] = {}
    fallback_class_ids: list[str] = []
    for class_id in sorted(common_class_ids, key=int):
        common_paths = set(per_model[0][class_id])
        for model_dict in per_model[1:]:
            common_paths &= model_dict[class_id]

        if common_paths:
            result[class_id] = sorted(common_paths)
            continue

        # Fallback: choose the sample that appears in the most models for this class.
        candidate_counter: Counter[str] = Counter()
        for model_dict in per_model:
            candidate_counter.update(model_dict[class_id])

        if not candidate_counter:
            result[class_id] = []
            continue

        max_count = max(candidate_counter.values())
        top_candidates = sorted(path for path, count in candidate_counter.items() if count == max_count)
        result[class_id] = [top_candidates[0]]
        fallback_class_ids.append(class_id)

    return result, fallback_class_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Intersect correctly classified samples across all *_imagenet_correct_paths.json files. "
            "Only samples present in every model file are kept."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(CLASSIFICATION_RESULT_DIR),
        help="Root folder containing classification result JSON files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_imagenet_correct_paths.json",
        help="Glob pattern used to discover model result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CLASSIFICATION_RESULT_DIR) / "correct_selection.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--drop-empty-classes",
        action="store_true",
        help="Drop classes whose intersection is empty.",
    )

    args = parser.parse_args()

    input_root = args.input_root
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    result_files = sorted(input_root.rglob(args.pattern))
    if not result_files:
        raise FileNotFoundError(
            f"No files found under {input_root} matching pattern '{args.pattern}'."
        )

    per_model = [_load_correct_paths(path) for path in result_files]
    merged, fallback_class_ids = _intersect_across_models(per_model)

    if args.drop_empty_classes:
        merged = {k: v for k, v in merged.items() if len(v) > 0}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    non_empty = sum(1 for v in merged.values() if len(v) > 0)
    total_samples = sum(len(v) for v in merged.values())
    empty_class_ids = [k for k, v in merged.items() if len(v) == 0]

    print(f"Found {len(result_files)} model result files.")
    print(f"Saved: {args.output}")
    print(f"Classes in output: {len(merged)} (non-empty: {non_empty})")
    print(f"Total selected samples: {total_samples}")
    print(f"Empty classes: {len(empty_class_ids)}")
    print(f"Fallback classes (most frequent sample): {len(fallback_class_ids)}")
    if empty_class_ids:
        print("Empty class ids:", ",".join(empty_class_ids))
    if fallback_class_ids:
        print("Fallback class ids:", ",".join(fallback_class_ids))


if __name__ == "__main__":
    main()