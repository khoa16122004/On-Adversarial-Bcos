from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TargetSpec = int | str | Sequence[int]


class PGDAttack:
    """
    PGD attack with crossentropy objective only:
    - untargeted maximizes CE on original class.
    - targeted minimizes CE on target class(es).
    """
    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        self.model = model
        self.epsilon = float(epsilon)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.transform = getattr(model, "transform", None)
        self.history: list[dict[str, Any]] = []

    def _predict(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        model_input = rgb_tensor
        if self.transform is not None and hasattr(self.transform, "inverse_transform"):
            model_input = self.transform.inverse_transform(rgb_tensor)
        return self.model(model_input)

    @staticmethod
    def _validate_inputs(clean_rgb: torch.Tensor, step_size: float, steps: int) -> None:
        if clean_rgb.ndim != 4:
            raise ValueError("clean_rgb must have shape [B, 3, H, W].")
        if clean_rgb.shape[0] != 1:
            raise ValueError("PGD currently expects batch size 1.")
        if clean_rgb.shape[1] != 3:
            raise ValueError("clean_rgb must be an RGB tensor with shape [B, 3, H, W].")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

    def _compute_loss(
        self,
        logits: torch.Tensor,
        original_class: int,
        target_classes: list[int],
        loss_fn: LossFn,
    ) -> torch.Tensor:
        if not target_classes:
            targets = torch.tensor([original_class], device=logits.device, dtype=torch.long)
            return loss_fn(logits, targets)

        if len(target_classes) == 1:
            targets = torch.tensor(target_classes, device=logits.device, dtype=torch.long)
            return loss_fn(logits, targets)

        target_distribution = torch.zeros_like(logits)
        target_weight = 1.0 / len(target_classes)
        target_distribution[:, target_classes] = target_weight
        return self._soft_target_cross_entropy(logits, target_distribution)

    def _resolve_target_classes(self, target_class: TargetSpec | None) -> list[int]:
        if target_class is None:
            return []
        if isinstance(target_class, int):
            return [target_class]
        if isinstance(target_class, str):
            parsed_targets = [part.strip() for part in target_class.split(",") if part.strip()]
            if not parsed_targets:
                raise ValueError("target_class string must contain at least one class id.")
            return [int(part) for part in parsed_targets]
        resolved_targets = [int(class_id) for class_id in target_class]
        if not resolved_targets:
            raise ValueError("target_class sequence must contain at least one class id.")
        return resolved_targets

    @staticmethod
    def _soft_target_cross_entropy(logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        return -(target_distribution * log_probs).sum(dim=1).mean()

    @staticmethod
    def _is_success(
        pred_class: int,
        original_class: int,
        target_classes: list[int],
        is_targeted: bool,
    ) -> bool:
        if not is_targeted:
            return pred_class != original_class
        return pred_class in target_classes

    @staticmethod
    def _build_history_row(
        step: int,
        loss: torch.Tensor,
        logits: torch.Tensor,
        original_class: int,
        target_classes: list[int],
        is_targeted: bool,
    ) -> dict[str, Any]:
        return {
            "step": step + 1,
            "loss": float(loss.item()),
            "prob_original_class": float(F.softmax(logits.detach(), dim=1)[0, original_class].item()),
            "logit_original_class": float(logits[0, original_class].detach().item()),
            "original_class": original_class,
            "pred_class": int(logits.argmax(dim=1).item()),
            "loss_type": "crossentropy",
            "target_class": target_classes if is_targeted else None,
        }

    def solve(
        self,
        clean_rgb: torch.Tensor,
        original_class: int,
        step_size: float,
        steps: int,
        target_class: TargetSpec | None = None,
        loss_fn: LossFn | None = None,
    ) -> tuple[torch.Tensor, int, int, list[dict[str, Any]]]:
        self._validate_inputs(clean_rgb=clean_rgb, step_size=step_size, steps=steps)

        resolved_target_classes = self._resolve_target_classes(target_class)
        is_targeted = len(resolved_target_classes) > 0
        loss_fn = loss_fn or F.cross_entropy
        self.model.eval()

        adv_rgb = clean_rgb.detach().clone()
        success_step = -1
        final_pred = int(original_class)
        direction = -1.0 if is_targeted else 1.0

        history: list[dict[str, Any]] = []

        for step in tqdm(range(steps), desc="PGD Attack", unit="step"):
            adv_rgb.requires_grad_(True)
            logits = self._predict(adv_rgb)

            loss = self._compute_loss(
                logits=logits,
                original_class=original_class,
                target_classes=resolved_target_classes,
                loss_fn=loss_fn,
            )

            history.append(
                self._build_history_row(
                    step=step,
                    loss=loss,
                    logits=logits,
                    original_class=original_class,
                    target_classes=resolved_target_classes,
                    is_targeted=is_targeted,
                )
            )

            gradient_sign = torch.autograd.grad(loss, adv_rgb)[0].sign()
            updated = adv_rgb.detach() + direction * step_size * gradient_sign

            perturbation = (updated - clean_rgb).clamp(-self.epsilon, self.epsilon)
            adv_rgb = (clean_rgb + perturbation).clamp(self.clamp_min, self.clamp_max)

            with torch.no_grad():
                updated_logits = self._predict(adv_rgb)
                final_pred = int(updated_logits.argmax(dim=1).item())

            if success_step == -1 and self._is_success(
                pred_class=final_pred,
                original_class=original_class,
                target_classes=resolved_target_classes,
                is_targeted=is_targeted,
            ):
                success_step = step + 1

        self.history = history
        return adv_rgb.detach(), final_pred, success_step, history


