from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from const import ANNOTATIONS_FILE, IMAGENET_VAL_DATA, PROJECT_ROOT
from dataloader import ImageNet


def _sanitize_slug(text: str) -> str:
    cleaned = text.strip().replace(":", "_").replace("/", "_").replace("\\", "_")
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in cleaned)


def _parse_target_pair(target_key: str) -> tuple[str, str]:
    if ":" in target_key:
        target_type, target_name = target_key.split(":", 1)
        return target_type.strip(), target_name.strip()
    return "unknown", target_key.strip()


def _build_transfer_tag(source_type: str, source_name: str, target_key: str) -> str:
    target_type, target_name = _parse_target_pair(target_key)
    src = f"{_sanitize_slug(source_type)}_{_sanitize_slug(source_name)}"
    tgt = f"{_sanitize_slug(target_type)}_{_sanitize_slug(target_name)}"
    return f"from_{src}__to__{tgt}"


def _resolve_output_path(requested_output: Path, meta: dict[str, object]) -> Path:
    source_type = str(meta.get("source_model_type", "unknown"))
    source_name = str(meta.get("source_model_name", "unknown"))
    target_key = str(meta.get("target", "unknown"))
    transfer_tag = _build_transfer_tag(source_type, source_name, target_key)

    # If output is a directory-like path, create a descriptive filename inside it.
    if requested_output.suffix.lower() != ".json":
        return requested_output / f"transfer_failed_{transfer_tag}.json"

    # Keep explicit descriptive names as-is.
    if "__to__" in requested_output.stem and "from_" in requested_output.stem:
        return requested_output

    # For generic names, append transfer tag before extension.
    return requested_output.with_name(f"{requested_output.stem}__{transfer_tag}.json")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in: {path}")
    return data


def _resolve_epsilon_key(report: dict, epsilon: str | None) -> str:
    results = report.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("Transfer report does not contain 'results'.")

    first_target = next(iter(results.values()))
    if not isinstance(first_target, dict) or not first_target:
        raise ValueError("Transfer report does not contain epsilon groups under 'results'.")

    available = sorted(first_target.keys())
    if epsilon is None:
        if len(available) != 1:
            raise ValueError(
                "Multiple epsilons found. Please set --epsilon. "
                f"Available: {', '.join(available)}"
            )
        return available[0]

    eps_raw = epsilon.strip()
    eps_key = eps_raw if eps_raw.startswith("epsilon_") else f"epsilon_{eps_raw}"
    if eps_key not in first_target:
        raise ValueError(
            f"Epsilon '{eps_key}' not found in transfer report. "
            f"Available: {', '.join(available)}"
        )
    return eps_key


def _find_metadata_path(
    attack_root: Path,
    source_model_type: str,
    source_model_name: str,
    epsilon_key: str,
    image_name: str,
) -> Path:
    candidates = [
        attack_root / source_model_type / source_model_name / "PGD" / epsilon_key / image_name / "metadata.json",
        attack_root / source_model_name / "PGD" / epsilon_key / image_name / "metadata.json",
        attack_root / "PGD" / epsilon_key / image_name / "metadata.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "metadata.json not found for image "
        f"'{image_name}'. Checked: {', '.join(str(p) for p in candidates)}"
    )


def _sample_two_random_classes(rng: random.Random, exclude: set[int]) -> list[int]:
    pool = [i for i in range(1000) if i not in exclude]
    if len(pool) < 2:
        raise ValueError("Not enough classes available to sample two random classes.")
    return rng.sample(pool, 2)


def _build_class_id_to_images(
    imagenet_val_dir: Path,
    annotations_file: Path,
) -> tuple[dict[int, list[str]], dict[int, str], dict[int, str]]:
    dataset = ImageNet(
        img_dir=str(imagenet_val_dir),
        annotations_file=str(annotations_file),
        transform=None,
    )

    class_id_to_images: dict[int, list[str]] = {}
    class_id_to_folder: dict[int, str] = {}
    for img_path, class_id, _class_name, _folder_name in dataset.samples:
        class_id_to_images.setdefault(int(class_id), []).append(str(img_path))
        class_id_to_folder.setdefault(int(class_id), str(_folder_name))

    if not class_id_to_images:
        raise ValueError(
            "No ImageNet validation images found. "
            f"Check --imagenet-val-dir: {imagenet_val_dir} and --annotations-file: {annotations_file}"
        )

    class_id_to_name = dict(dataset.class_id_to_name)
    return class_id_to_images, class_id_to_folder, class_id_to_name


def _sample_two_random_refs(
    rng: random.Random,
    class_id_to_images: dict[int, list[str]],
    class_id_to_folder: dict[int, str],
    class_id_to_name: dict[int, str],
    exclude: set[int],
) -> tuple[list[int], list[dict[str, object]]]:
    available_classes = [
        class_id for class_id, images in class_id_to_images.items() if class_id not in exclude and len(images) > 0
    ]
    if len(available_classes) < 2:
        raise ValueError("Not enough available classes (with images) to sample two random references.")

    class_ids = rng.sample(available_classes, 2)
    refs = [
        {
            "role": "random_class",
            "class_id": class_id,
            "class_folder": class_id_to_folder.get(class_id, ""),
            "class_name": class_id_to_name.get(class_id, ""),
            "img_path": rng.choice(class_id_to_images[class_id]),
        }
        for class_id in class_ids
    ]
    return class_ids, refs


def _sample_one_ref_by_class_id(
    rng: random.Random,
    class_id: int,
    class_id_to_images: dict[int, list[str]],
    class_id_to_folder: dict[int, str],
    class_id_to_name: dict[int, str],
    role: str,
) -> dict[str, object]:
    images = class_id_to_images.get(class_id, [])
    if not images:
        raise ValueError(f"No images found for class_id={class_id}.")

    return {
        "role": role,
        "class_id": class_id,
        "class_folder": class_id_to_folder.get(class_id, ""),
        "class_name": class_id_to_name.get(class_id, ""),
        "img_path": rng.choice(images),
    }


def _resolve_source_adv_png(metadata_path: Path, source_meta: dict) -> str:
    raw = source_meta.get("adv_png")
    if isinstance(raw, str) and raw:
        raw_path = Path(raw)
        if raw_path.exists():
            return str(raw_path)

    candidates = [
        metadata_path.parent / "adv.png",
        metadata_path.parent / "adv.jpg",
        metadata_path.parent / "adv.jpeg",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Cannot find source adversarial image near metadata: {metadata_path}"
    )


def build_failure_samples(
    transfer_json: Path,
    attack_root: Path,
    imagenet_val_dir: Path,
    annotations_file: Path,
    epsilon: str | None,
    sample_size: int,
    seed: int,
) -> tuple[list[dict], dict[str, object]]:
    report = _load_json(transfer_json)

    source_info = report.get("source", {})
    if not isinstance(source_info, dict):
        raise ValueError("Transfer report has invalid 'source' format.")

    source_model_type = str(source_info.get("model_type", ""))
    source_model_name = str(source_info.get("model_name", ""))
    if not source_model_type or not source_model_name:
        raise ValueError("Transfer report missing source model info.")

    results = report.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("Transfer report has no target results.")

    target_key = next(iter(results.keys()))
    target_block = results[target_key]
    if not isinstance(target_block, dict):
        raise ValueError("Invalid target block format in transfer report.")

    epsilon_key = _resolve_epsilon_key(report, epsilon)
    eps_block = target_block.get(epsilon_key, {})
    if not isinstance(eps_block, dict):
        raise ValueError(f"Invalid epsilon block: {epsilon_key}")

    details = eps_block.get("details", [])
    if not isinstance(details, list):
        raise ValueError("Invalid 'details' format in transfer report.")

    failed = [d for d in details if not bool(d.get("success_vs_source_clean", False))]
    if not failed:
        raise ValueError("No failed transfer samples found for selected epsilon.")

    if sample_size > len(failed):
        raise ValueError(
            f"Requested sample_size={sample_size}, but only {len(failed)} failed samples available."
        )

    rng = random.Random(seed)
    chosen = rng.sample(failed, sample_size)
    class_id_to_images, class_id_to_folder, class_id_to_name = _build_class_id_to_images(
        imagenet_val_dir=imagenet_val_dir,
        annotations_file=annotations_file,
    )

    output: list[dict] = []
    for item in chosen:
        image_name = str(item["image_name"])
        target_pred = int(item["target_adv_pred"])

        metadata_path = _find_metadata_path(
            attack_root=attack_root,
            source_model_type=source_model_type,
            source_model_name=source_model_name,
            epsilon_key=epsilon_key,
            image_name=image_name,
        )
        source_meta = _load_json(metadata_path)

        if "final_pred" in source_meta:
            source_pred = int(source_meta["final_pred"])
        elif "clean_pred" in source_meta:
            source_pred = int(source_meta["clean_pred"])
        else:
            raise KeyError(
                f"Missing 'final_pred'/'clean_pred' in {metadata_path} for image {image_name}"
            )

        source_adv_png = _resolve_source_adv_png(metadata_path=metadata_path, source_meta=source_meta)

        target_ref = _sample_one_ref_by_class_id(
            rng=rng,
            class_id=target_pred,
            class_id_to_images=class_id_to_images,
            class_id_to_folder=class_id_to_folder,
            class_id_to_name=class_id_to_name,
            role="target_pred_class",
        )

        random_classes, random_refs = _sample_two_random_refs(
            rng=rng,
            class_id_to_images=class_id_to_images,
            class_id_to_folder=class_id_to_folder,
            class_id_to_name=class_id_to_name,
            exclude={source_pred, target_pred},
        )

        output.append(
            {
                "img_name": image_name,
                "source_pred": source_pred,
                "target_pred": target_pred,
                "two_random_classes": random_classes,
                "itemrefs": [
                    {
                        "role": "source_adv",
                        "img_path": source_adv_png,
                    },
                    target_ref,
                    *random_refs,
                ],
            }
        )

    meta = {
        "transfer_json": str(transfer_json),
        "attack_root": str(attack_root),
        "source_model_type": source_model_type,
        "source_model_name": source_model_name,
        "target": target_key,
        "transfer_tag": _build_transfer_tag(source_model_type, source_model_name, target_key),
        "epsilon": epsilon_key,
        "sample_size": sample_size,
        "seed": seed,
        "num_failed_available": len(failed),
    }
    return output, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select failed transfer samples, recover source prediction from source attack metadata, "
            "and attach two random ImageNet classes."
        )
    )
    parser.add_argument("--transfer-json", type=Path, required=True, help="Path to transfer report JSON")
    parser.add_argument(
        "--attack-root",
        type=Path,
        default=Path(PROJECT_ROOT) / "attack_result",
        help="Root of source attack outputs (contains model folders or PGD folders)",
    )
    parser.add_argument(
        "--epsilon",
        type=str,
        default="0.03",
        help="Epsilon key, e.g. 0.03 or epsilon_0.03. If omitted, report must contain exactly one epsilon.",
    )
    parser.add_argument(
        "--imagenet-val-dir",
        type=Path,
        default=Path(IMAGENET_VAL_DATA),
        help="ImageNet val folder, containing class subfolders.",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=Path(ANNOTATIONS_FILE),
        help="Annotation JSON mapping class folder -> [label_id, class_name].",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of failed samples to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(PROJECT_ROOT) / "localization" / "transfer_failed_100.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    records, meta = build_failure_samples(
        transfer_json=args.transfer_json,
        attack_root=args.attack_root,
        imagenet_val_dir=args.imagenet_val_dir,
        annotations_file=args.annotations_file,
        epsilon=args.epsilon,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    resolved_output = _resolve_output_path(args.output, meta)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "samples": records}, f, ensure_ascii=False, indent=2)

    print(f"Saved: {resolved_output}")
    print(f"Selected {len(records)} failed transfer samples")
    print(f"Source: {meta['source_model_type']}:{meta['source_model_name']}")
    print(f"Target: {meta['target']} | Epsilon: {meta['epsilon']}")


if __name__ == "__main__":
    main()