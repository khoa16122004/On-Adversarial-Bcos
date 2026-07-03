from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(WORKSPACE_ROOT))

from attack.util import DEFAULT_CHECKPOINT_DIR, list_bcos_models


REPO = "B-cos/B-cos-v2"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download pretrained B-cos checkpoints and save them locally.",
	)
	parser.add_argument(
		"--model",
		nargs="+",
		default=["resnet50"],
		help="One or more B-cos model names to download. Defaults to resnet50.",
	)
	parser.add_argument(
		"--all",
		action="store_true",
		help="Download all models returned by bcos.models.pretrained.list_available().",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_CHECKPOINT_DIR,
		help="Directory to store downloaded checkpoints.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite an existing checkpoint file if it already exists.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	available_models = set(list_bcos_models())
	model_names = sorted(available_models) if args.all else args.model

	args.output_dir.mkdir(parents=True, exist_ok=True)

	for model_name in model_names:
		if model_name not in available_models:
			raise ValueError(
				f"Unknown B-cos model '{model_name}'. Available models: {', '.join(sorted(available_models))}"
			)

		output_path = args.output_dir / f"{model_name}.pth"
		if output_path.exists() and not args.force:
			print(f"Skipping {model_name}: {output_path} already exists")
			continue

		print(f"Downloading {model_name} from {REPO}...")
		model = torch.hub.load(REPO, model_name, pretrained=True, trust_repo=True)
		torch.save(model.state_dict(), output_path)
		print(f"Saved {model_name} to {output_path}")


if __name__ == "__main__":
	main()