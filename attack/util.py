from __future__ import annotations

import importlib
import importlib.util
import pathlib
import pickle
import sys
import warnings
from pathlib import Path, PosixPath, WindowsPath

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

from script.const import CHECKPOINT_DIR


BCOS_ROOT = Path(__file__).resolve().parents[1] / "B-cos-v2"
BCOSIFICATION_ROOT = Path(__file__).resolve().parents[1] / "B-cosification"
DEFAULT_CHECKPOINT_DIR = Path(CHECKPOINT_DIR) / "bcos-v2"
DEFAULT_BCOSIFY_CHECKPOINT_DIR = Path(CHECKPOINT_DIR) / "bcosify"
DEFAULT_MODEL_TYPE = "bcos"
_ACTIVE_BCOS_SOURCE: str | None = None


def parse_target_classes(raw: str | None) -> list[int]:
	if raw is None:
		return []
	tokens = [token.strip() for token in raw.split(",") if token.strip()]
	return [int(token) for token in tokens]


def resolve_target_class_for_sample(target_classes: list[int], sample_index: int) -> int | None:
	if not target_classes:
		return None
	return int(target_classes[sample_index % len(target_classes)])


def load_simba_utils_module(root: Path | None = None):
	if root is None:
		root = Path(__file__).resolve().parents[1]
	module_path = root / "simple-blackbox-attack" / "utils.py"
	if not module_path.exists():
		# Fallback to local SimBA helpers when external repo is not present.
		from attack import simba_fallback_utils

		warnings.warn(
			f"Missing external SimBA utils at {module_path}. "
			"Using attack.simba_fallback_utils instead.",
			RuntimeWarning,
		)
		return simba_fallback_utils

	spec = importlib.util.spec_from_file_location("simba_utils_local", module_path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Could not load SimBA utils module from: {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


class ImageNetRGBTransform:
	"""Split ImageNet preprocessing into RGB-space and normalization-space transforms."""

	def __init__(
		self,
		image_size: int = 224,
		resize_size: int = 256,
		mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
		std: tuple[float, float, float] = (0.229, 0.224, 0.225),
	) -> None:
		self.spatial_transform = tv_transforms.Compose(
			[
				tv_transforms.Resize(resize_size, interpolation=tv_transforms.InterpolationMode.BILINEAR),
				tv_transforms.CenterCrop(image_size),
				tv_transforms.ToTensor(),
			]
		)
		mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
		std_tensor = torch.tensor(std).view(1, 3, 1, 1)

		def _normalize(rgb_tensor: torch.Tensor) -> torch.Tensor:
			if rgb_tensor.ndim == 3:
				rgb_tensor_batched = rgb_tensor.unsqueeze(0)
				result = (rgb_tensor_batched - mean_tensor.to(rgb_tensor.device)) / std_tensor.to(rgb_tensor.device)
				return result.squeeze(0)
			if rgb_tensor.ndim == 4:
				return (rgb_tensor - mean_tensor.to(rgb_tensor.device)) / std_tensor.to(rgb_tensor.device)
			raise ValueError("Expected tensor shape [C,H,W] or [B,C,H,W].")

		self.inverse_transform = _normalize


class BcosifyImageTransform:
	"""ImageNet transform with add-inverse channels for B-cosify models."""

	def __init__(
		self,
		image_size: int = 224,
		resize_size: int = 256,
	) -> None:
		self.spatial_transform = tv_transforms.Compose(
			[
				tv_transforms.Resize(resize_size, interpolation=tv_transforms.InterpolationMode.BILINEAR),
				tv_transforms.CenterCrop(image_size),
				tv_transforms.ToTensor(),
			]
		)

		def _add_inverse(rgb_tensor: torch.Tensor) -> torch.Tensor:
			if rgb_tensor.ndim == 3:
				rgb_tensor = rgb_tensor.unsqueeze(0)
			if rgb_tensor.ndim != 4 or rgb_tensor.shape[1] != 3:
				raise ValueError("Expected RGB tensor shape [B,3,H,W] or [3,H,W].")
			return torch.cat([rgb_tensor, 1 - rgb_tensor], dim=1)

		self.inverse_transform = _add_inverse


def _ensure_bcos_root_on_path() -> None:
	global _ACTIVE_BCOS_SOURCE
	if _ACTIVE_BCOS_SOURCE is None:
		_ACTIVE_BCOS_SOURCE = "bcos-v2"
	elif _ACTIVE_BCOS_SOURCE != "bcos-v2":
		raise RuntimeError(
			"Cannot switch bcos backend source within the same process (current: "
			f"{_ACTIVE_BCOS_SOURCE}, requested: bcos-v2). Please restart the app/kernel."
		)
	if str(BCOS_ROOT) not in sys.path:
		sys.path.insert(0, str(BCOS_ROOT))


def _ensure_bcosification_root_on_path() -> None:
	global _ACTIVE_BCOS_SOURCE
	if _ACTIVE_BCOS_SOURCE is None:
		_ACTIVE_BCOS_SOURCE = "bcosification"
	elif _ACTIVE_BCOS_SOURCE != "bcosification":
		raise RuntimeError(
			"Cannot switch bcos backend source within the same process (current: "
			f"{_ACTIVE_BCOS_SOURCE}, requested: bcosification). Please restart the app/kernel."
		)
	if str(BCOSIFICATION_ROOT) not in sys.path:
		sys.path.insert(0, str(BCOSIFICATION_ROOT))

def _get_pretrained_module():
	_ensure_bcos_root_on_path()
	return importlib.import_module("bcos.models.pretrained")


def _get_bcosification_imagenet_model_factory_module():
	_ensure_bcosification_root_on_path()
	return importlib.import_module("bcos.experiments.ImageNet.bcosification.model")


def _get_bcosification_vit_model_factory_module():
	_ensure_bcosification_root_on_path()
	return importlib.import_module("bcos.experiments.ImageNet.vit_bcosification.model")


def load_imagenet_categories() -> list[str]:
	# Use torchvision categories to avoid importing bcos backend modules at startup.
	return list(tv_models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


def list_bcos_models() -> list[str]:
	pretrained_module = _get_pretrained_module()
	return sorted(pretrained_module.list_available())


def list_torchvision_models() -> list[str]:
	if hasattr(tv_models, "list_models"):
		return sorted(tv_models.list_models())
	return sorted(
		model_name
		for model_name in dir(tv_models)
		if model_name.islower() and callable(getattr(tv_models, model_name))
	)


def _normalize_model_key(value: str) -> str:
	return "".join(char for char in value.lower() if char.isalnum())


def _resolve_model_name(name_like: str) -> str:
	available_models = list_bcos_models()
	normalized_input = _normalize_model_key(name_like)
	for model_name in available_models:
		if normalized_input == _normalize_model_key(model_name):
			return model_name
	raise ValueError(
		f"Unknown B-cos model '{name_like}'. Available models: {', '.join(available_models)}"
	)


def _resolve_torchvision_model_name(name_like: str) -> str:
	available_models = list_torchvision_models()
	normalized_input = _normalize_model_key(name_like)
	for model_name in available_models:
		if normalized_input == _normalize_model_key(model_name):
			return model_name
	raise ValueError(
		f"Unknown torchvision model '{name_like}'. Available models: {', '.join(available_models)}"
	)


def resolve_bcos_checkpoint_path(
	model_name: str,
	checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
	if not checkpoint_dir.exists():
		raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

	resolved_model_name = _resolve_model_name(model_name)
	normalized_model_name = _normalize_model_key(resolved_model_name)
	candidates: list[tuple[int, Path]] = []

	for checkpoint_path in checkpoint_dir.glob("*.pth"):
		stem = checkpoint_path.stem
		stem_prefix = stem.split("-", 1)[0]
		if stem == resolved_model_name:
			candidates.append((0, checkpoint_path))
		elif stem_prefix == resolved_model_name:
			candidates.append((1, checkpoint_path))
		elif _normalize_model_key(stem) == normalized_model_name:
			candidates.append((2, checkpoint_path))
		elif _normalize_model_key(stem_prefix) == normalized_model_name:
			candidates.append((3, checkpoint_path))

	if not candidates:
		raise FileNotFoundError(
			f"No checkpoint matching model '{resolved_model_name}' was found in {checkpoint_dir}"
		)

	_, best_match = sorted(candidates, key=lambda item: (item[0], len(item[1].name), item[1].name))[0]
	return best_match


def resolve_bcosify_checkpoint_path(
	model_name: str,
	checkpoint_dir: Path = DEFAULT_BCOSIFY_CHECKPOINT_DIR,
) -> Path:
	checkpoint_dir = Path(checkpoint_dir)
	if not checkpoint_dir.exists():
		raise FileNotFoundError(f"B-cosify checkpoint directory does not exist: {checkpoint_dir}")

	normalized_model_name = _normalize_model_key(model_name)
	candidates: list[tuple[int, Path]] = []

	for pattern in ("*.pth", "*.pt", "*.ckpt"):
		for checkpoint_path in checkpoint_dir.glob(pattern):
			stem = checkpoint_path.stem
			if stem.endswith(".ckpt"):
				stem = Path(stem).stem
			stem_norm = _normalize_model_key(stem)
			if stem_norm == normalized_model_name:
				candidates.append((0, checkpoint_path))
			elif normalized_model_name in stem_norm:
				candidates.append((1, checkpoint_path))

	if not candidates:
		raise FileNotFoundError(
			f"No B-cosify checkpoint matching model '{model_name}' was found in {checkpoint_dir}"
		)

	_, best_match = sorted(candidates, key=lambda item: (item[0], len(item[1].name), item[1].name))[0]
	return best_match


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
	def _load_with_posixpath_compat(*, weights_only: bool):
		# Some checkpoints serialized on Linux contain PosixPath objects that
		# cannot be instantiated on Windows. Retry with a temporary compatibility map.
		if sys.platform != "win32":
			return torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)

		try:
			return torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
		except pathlib.UnsupportedOperation as path_exc:
			if "PosixPath" not in str(path_exc):
				raise
			original_posix_path = pathlib.PosixPath
			pathlib.PosixPath = pathlib.PurePosixPath  # type: ignore[assignment]
			try:
				return torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
			finally:
				pathlib.PosixPath = original_posix_path

	try:
		state_dict = _load_with_posixpath_compat(weights_only=True)
	except pickle.UnpicklingError as exc:
		message = str(exc)
		retry_error: Exception | None = None

		# PyTorch >=2.6 defaults to weights_only=True and may reject serialized path classes.
		if "Unsupported global" in message or "pathlib.PosixPath" in message:
			try:
				if hasattr(torch.serialization, "safe_globals"):
					with torch.serialization.safe_globals([PosixPath, WindowsPath]):
						state_dict = _load_with_posixpath_compat(weights_only=True)
				elif hasattr(torch.serialization, "add_safe_globals"):
					torch.serialization.add_safe_globals([PosixPath, WindowsPath])
					state_dict = _load_with_posixpath_compat(weights_only=True)
				else:
					retry_error = exc
			except Exception as inner_exc:  # nosec - trusted/local checkpoint fallback handled below
				retry_error = inner_exc
		else:
			retry_error = exc

		if retry_error is not None:
			warnings.warn(
				"weights_only checkpoint load failed; retrying with weights_only=False. "
				"Only use trusted checkpoints. "
				f"Original error: {retry_error}",
				RuntimeWarning,
			)
			state_dict = _load_with_posixpath_compat(weights_only=False)
	if isinstance(state_dict, dict) and "state_dict" in state_dict:
		state_dict = state_dict["state_dict"]
	return state_dict


def _adapt_state_dict_for_model(
	state_dict: dict[str, torch.Tensor],
	model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
	model_keys = set(model.state_dict().keys())
	if all(key in model_keys for key in state_dict.keys()):
		return state_dict

	stripped = {
		key.replace("model.", "", 1): value
		for key, value in state_dict.items()
		if key.startswith("model.")
	}
	if stripped and all(key in model_keys for key in stripped.keys()):
		return stripped

	return state_dict


def _build_bcosify_model_config(model_name: str) -> dict:
	normalized = _normalize_model_key(model_name)
	if normalized in {"resnet18", "resnet_18"}:
		return {
			"is_bcos": True,
			"name": "resnet18",
			"bcos_args": {
				"b": 2,
				"max_out": 1,
			},
			"last_layer_name": "fc",
			"weights": None,
			"bcosify_args": {
				"fix_b": True,
				"use_bias": False,
				"norm_layer": "BnUncV2",
				"manual_optim": False,
				"gap": True,
				"act_layer": True,
			},
			"standard_changes": {
				"maxpool": nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
			},
		}
	if normalized in {"resnet50", "resnet_50"}:
		return {
			"is_bcos": True,
			"name": "resnet50",
			"bcos_args": {
				"b": 2,
				"max_out": 1,
			},
			"last_layer_name": "fc",
			"weights": None,
			"bcosify_args": {
				"fix_b": True,
				"use_bias": False,
				"norm_layer": "BnUncV2",
				"manual_optim": False,
				"gap": True,
				"act_layer": True,
			},
			"standard_changes": {
				"maxpool": nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
			},
		}
	if normalized in {"densenet121", "densenet_121"}:
		return {
			"is_bcos": True,
			"name": "densenet121",
			"bcos_args": {
				"b": 2,
				"max_out": 1,
			},
			"last_layer_name": "classifier",
			"weights": None,
			"bcosify_args": {
				"fix_b": True,
				"use_bias": False,
				"norm_layer": "BnUncV2",
				"manual_optim": False,
				"gap": True,
				"act_layer": True,
			},
			"standard_changes": {
				"features[3]": nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
			},
		}

	raise ValueError("bcosify model_name must be one of: resnet18, resnet50, densenet121")


def _resolve_bcosify_vit_arch(model_name_like: str) -> str | None:
	normalized = _normalize_model_key(model_name_like)
	vit_arches = [
		"simple_vit_ti_patch16_224",
		"simple_vit_s_patch16_224",
		"simple_vit_b_patch16_224",
		"simple_vit_l_patch16_224",
		"vitc_ti_patch1_14",
		"vitc_s_patch1_14",
		"vitc_b_patch1_14",
		"vitc_l_patch1_14",
	]
	for arch in vit_arches:
		if _normalize_model_key(arch) in normalized:
			return arch
	return None


def _build_bcosify_vit_model_config(model_name_like: str, vit_arch: str) -> dict:
	normalized = _normalize_model_key(model_name_like)
	return {
		"is_bcos": True,
		"name": vit_arch,
		"weights": "pretrained",
		"bcos_args": {
			"b": 2,
			"max_out": 1,
		},
		"args": {
			"num_classes": 1000,
			"gap_reorder": "gapreorder" in normalized,
		},
		"bcosify_args": {
			"fix_b": True,
			"use_bias": "usebias" in normalized,
		},
		"logit_layer": True,
		"act_layer": "nogelu" not in normalized,
	}


def load_bcosify_model(
	model_name_or_checkpoint: str | Path,
	device: torch.device,
	checkpoint_dir: Path = DEFAULT_BCOSIFY_CHECKPOINT_DIR,
	return_checkpoint_path: bool = False,
) -> torch.nn.Module | tuple[torch.nn.Module, Path]:
	model_name_hint = str(model_name_or_checkpoint)

	if isinstance(model_name_or_checkpoint, Path):
		checkpoint_path = model_name_or_checkpoint
		model_name_hint = checkpoint_path.stem
		resolved_model_name = _normalize_model_key(model_name_hint)
	else:
		candidate_path = Path(model_name_or_checkpoint)
		is_checkpoint_like = candidate_path.suffix in {".pth", ".pt", ".ckpt"} or candidate_path.is_absolute()
		if is_checkpoint_like:
			checkpoint_path = candidate_path
			model_name_hint = checkpoint_path.stem
			resolved_model_name = _normalize_model_key(model_name_hint)
		else:
			checkpoint_path = resolve_bcosify_checkpoint_path(model_name_or_checkpoint, checkpoint_dir)
			model_name_hint = str(model_name_or_checkpoint)
			resolved_model_name = _normalize_model_key(model_name_or_checkpoint)

	vit_arch = _resolve_bcosify_vit_arch(model_name_hint)
	if vit_arch is not None:
		model_factory_module = _get_bcosification_vit_model_factory_module()
		model_config = _build_bcosify_vit_model_config(model_name_hint, vit_arch)
	elif "resnet18" in resolved_model_name:
		model_factory_module = _get_bcosification_imagenet_model_factory_module()
		model_config = _build_bcosify_model_config("resnet18")
	elif "resnet50" in resolved_model_name:
		model_factory_module = _get_bcosification_imagenet_model_factory_module()
		model_config = _build_bcosify_model_config("resnet50")
	elif "densenet121" in resolved_model_name:
		model_factory_module = _get_bcosification_imagenet_model_factory_module()
		model_config = _build_bcosify_model_config("densenet121")
	else:
		raise ValueError(
			"bcosify model_name must be one of: resnet18, resnet50, densenet121, "
			"or a supported ViT arch (simple_vit_*, vitc_*)."
		)

	model = model_factory_module.get_model(model_config)

	state_dict = _load_state_dict(checkpoint_path)
	adapted_state_dict = _adapt_state_dict_for_model(state_dict, model)
	model.load_state_dict(adapted_state_dict, strict=False)

	model.transform = BcosifyImageTransform()
	model = model.to(device)
	model.eval()

	if return_checkpoint_path:
		return model, checkpoint_path
	return model


def load_bcos_model(
	model_name_or_checkpoint: str | Path,
	device: torch.device,
	checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
	return_checkpoint_path: bool = False,
) -> torch.nn.Module | tuple[torch.nn.Module, Path]:
	pretrained_module = _get_pretrained_module()

	if isinstance(model_name_or_checkpoint, Path):
		checkpoint_path = model_name_or_checkpoint
		resolved_model_name = _resolve_model_name(checkpoint_path.stem.split("-", 1)[0])
	else:
		candidate_path = Path(model_name_or_checkpoint)
		is_checkpoint_like = candidate_path.suffix in {".pth", ".pt"} or candidate_path.is_absolute()
		if is_checkpoint_like:
			checkpoint_path = candidate_path
			resolved_model_name = _resolve_model_name(checkpoint_path.stem.split("-", 1)[0])
		else:
			resolved_model_name = _resolve_model_name(model_name_or_checkpoint)
			checkpoint_path = resolve_bcos_checkpoint_path(resolved_model_name, checkpoint_dir)

	model_factory = getattr(pretrained_module, resolved_model_name)
	model = model_factory(pretrained=False)
	model.load_state_dict(_load_state_dict(checkpoint_path))
	model = model.to(device)
	model.eval()

	if return_checkpoint_path:
		return model, checkpoint_path
	return model


def load_torchvision_model(
	model_name: str,
	device: torch.device,
	checkpoint_path: Path | None = None,
	return_checkpoint_path: bool = False,
) -> torch.nn.Module | tuple[torch.nn.Module, Path | None]:
	resolved_model_name = _resolve_torchvision_model_name(model_name)
	model_factory = getattr(tv_models, resolved_model_name)

	if checkpoint_path is None:
		weights_enum = tv_models.get_model_weights(model_factory)
		weights = weights_enum.DEFAULT
		model = model_factory(weights=weights)
	else:
		model = model_factory(weights=None)
		model.load_state_dict(_load_state_dict(checkpoint_path))

	model.transform = ImageNetRGBTransform()
	model = model.to(device)
	model.eval()

	if return_checkpoint_path:
		return model, checkpoint_path
	return model


def load_model(
	model_type: str,
	model_name: str,
	device: torch.device,
	checkpoint: Path | None = None,
	checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
	return_checkpoint_path: bool = False,
) -> torch.nn.Module | tuple[torch.nn.Module, Path | None]:
	model_type = model_type.lower()
	if model_type == "bcos":
		model_source: str | Path = checkpoint if checkpoint is not None else model_name
		return load_bcos_model(
			model_source,
			device=device,
			checkpoint_dir=checkpoint_dir,
			return_checkpoint_path=return_checkpoint_path,
		)
	if model_type == "torchvision":
		return load_torchvision_model(
			model_name=model_name,
			device=device,
			checkpoint_path=checkpoint,
			return_checkpoint_path=return_checkpoint_path,
		)
	if model_type == "bcosify":
		model_source: str | Path = checkpoint if checkpoint is not None else model_name
		bcosify_checkpoint_dir = checkpoint_dir
		if checkpoint is None and checkpoint_dir == DEFAULT_CHECKPOINT_DIR:
			bcosify_checkpoint_dir = DEFAULT_BCOSIFY_CHECKPOINT_DIR
		return load_bcosify_model(
			model_source,
			device=device,
			checkpoint_dir=bcosify_checkpoint_dir,
			return_checkpoint_path=return_checkpoint_path,
		)
	raise ValueError("model_type must be one of: 'bcos', 'torchvision', 'bcosify'.")


def save_rgb_image(rgb_tensor: torch.Tensor, output_path: Path) -> None:
	if rgb_tensor.ndim == 4:
		rgb_tensor = rgb_tensor.squeeze(0)
	rgb_np = rgb_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
	plt.imsave(output_path, rgb_np)


def save_perturbation_image(
	perturbation: torch.Tensor,
	output_path: Path,
	epsilon: float,
) -> None:
	if perturbation.ndim == 4:
		perturbation = perturbation.squeeze(0)
	if epsilon > 0:
		perturbation = perturbation / epsilon
	perturbation_np = perturbation.detach().cpu().clamp(-1, 1)
	perturbation_np = ((perturbation_np + 1) / 2).permute(1, 2, 0).numpy()
	plt.imsave(output_path, perturbation_np)


def save_hot_contribution_map(
	contribution_map: torch.Tensor,
	output_path: Path,
	cmap: str = "hot",
) -> None:
	positive_map = contribution_map.detach().cpu().clamp(min=0)
	if positive_map.max().item() > 0:
		positive_map = positive_map / positive_map.max()
	plt.imsave(output_path, positive_map.numpy(), cmap=cmap, vmin=0.0, vmax=1.0)


def save_explanation_rgba(explanation: torch.Tensor | object, output_path: Path) -> None:
	explanation_tensor = torch.as_tensor(explanation).detach().cpu().clamp(0, 1)
	Image.fromarray((explanation_tensor.numpy() * 255).astype("uint8"), mode="RGBA").save(
		output_path
	)


def take_important_region(
    explanation: torch.Tensor,
    keep_ratio: float = 0.5,
) -> torch.Tensor:
    
    # chỉ giữ contribution dương
    importance_map = explanation[0].detach().clamp(min=0)
    # importance_map = explanation[0].detach()
    flat = importance_map.flatten()

    if flat.numel() == 0:
        raise ValueError("Empty explanation.")

    total = flat.sum()

    if total <= 0:
        return torch.zeros_like(
            importance_map,
            dtype=torch.bool
        ).unsqueeze(0)

    sorted_vals, sorted_idx = torch.sort(
        flat,
        descending=True
    )

    cumulative = torch.cumsum(
        sorted_vals,
        dim=0
    )

    threshold = total * keep_ratio

    selected = (
        torch.searchsorted(
            cumulative,
            threshold
        ).item()
        + 1
    )

    mask = torch.zeros_like(
        flat,
        dtype=torch.bool
    )

    mask[
        sorted_idx[:selected]
    ] = True

    return mask.reshape_as(
        importance_map
    ).unsqueeze(0)