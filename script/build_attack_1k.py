from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from const import CLASSIFICATION_RESULT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build attack_1k.json from correct_selection.json by sampling 1 path per class. "
            "Output keeps the same dict structure, each class has a list with exactly one item."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(CLASSIFICATION_RESULT_DIR) / "correct_selection.json",
        help="Path to correct_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CLASSIFICATION_RESULT_DIR) / "attack_1k.json",
        help="Path to output attack_1k.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object: {class_id: [paths, ...]}")

    rng = random.Random(args.seed)
    output: dict[str, list[str]] = {}

    for class_id, candidates in data.items():
        if not isinstance(candidates, list):
            raise ValueError(f"Class '{class_id}' must map to a list of paths.")
        if not candidates:
            raise ValueError(
                f"Class '{class_id}' has empty list. Please fix correct_selection.json first."
            )
        chosen = rng.choice(candidates)
        output[str(class_id)] = [str(chosen)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Input classes: {len(data)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
