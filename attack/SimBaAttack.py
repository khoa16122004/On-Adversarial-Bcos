from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from attack.util import load_simba_utils_module

class SimBAAttack:
    """SimBA black-box attack with crossentropy objective only."""

    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float,
        steps: int,
        order: str = "rand",
        freq_dims: int = 14,
        stride: int = 7,
        pixel_attack: bool = False,
        linf_bound: float = 0.0,
        image_size: int = 224,
        simba_utils: Any | None = None,
    ) -> None:
        self.model = model
        self.transform = getattr(model, "transform", None)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.order = str(order)
        self.freq_dims = int(freq_dims)
        self.stride = int(stride)
        self.pixel_attack = bool(pixel_attack)
        self.linf_bound = float(linf_bound)
        self.image_size = int(image_size)
        self.simba_utils = simba_utils or load_simba_utils_module()

    def _predict_logits(self, rgb_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        model_input = rgb_tensor.to(device)
        if self.transform is not None and hasattr(self.transform, "inverse_transform"):
            model_input = self.transform.inverse_transform(model_input)

        with torch.no_grad():
            logits = self.model(model_input)
        return logits[0].detach().cpu()

    @staticmethod
    def _score_crossentropy(
        logits: torch.Tensor,
        original_class: int,
        target_class: int | None,
        targeted: bool,
    ) -> float:
        logits_2d = logits.detach().cpu().view(1, -1)
        if targeted:
            if target_class is None:
                raise ValueError("target_class is required for targeted attack.")
            target = torch.tensor([int(target_class)], dtype=torch.long)
            return float(-F.cross_entropy(logits_2d, target).item())

        original = torch.tensor([int(original_class)], dtype=torch.long)
        return float(F.cross_entropy(logits_2d, original).item())

    @staticmethod
    def _expand_vector(x: torch.Tensor, size: int, image_size: int) -> torch.Tensor:
        z = torch.zeros(1, 3, image_size, image_size, dtype=x.dtype)
        z[:, :, :size, :size] = x.view(1, 3, size, size)
        return z

    def _get_indices(self) -> tuple[torch.Tensor, int]:
        if self.order == "rand":
            n_dims = 3 * self.freq_dims * self.freq_dims
            indices = torch.randperm(n_dims)[: self.steps]
            return indices.long(), int(self.freq_dims)

        if self.order == "diag":
            indices = self.simba_utils.diagonal_order(self.image_size, 3)[: self.steps]
        elif self.order == "strided":
            indices = self.simba_utils.block_order(
                self.image_size,
                3,
                initial_size=self.freq_dims,
                stride=self.stride,
            )[: self.steps]
        else:
            indices = self.simba_utils.block_order(self.image_size, 3)[: self.steps]

        return indices.long(), int(self.image_size)

    def solve(
        self,
        clean_rgb: torch.Tensor,
        original_class: int,
        target_class: int | None = None,
        targeted: bool = False,
        log_every: int = 100,
        stop_on_success: bool = True,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, int, int, int, list[dict[str, Any]], torch.Tensor, torch.Tensor, float]:
        if clean_rgb.ndim != 4 or clean_rgb.shape[0] != 1 or clean_rgb.shape[1] != 3:
            raise ValueError("clean_rgb must have shape [1, 3, H, W].")
        if targeted and target_class is None:
            raise ValueError("target_class is required when targeted=True.")
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if log_every <= 0:
            raise ValueError("log_every must be positive.")

        if device is None:
            device = clean_rgb.device

        indices, expand_dims = self._get_indices()
        n_dims = 3 * expand_dims * expand_dims

        if self.pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: self.simba_utils.block_idct(z, block_size=self.image_size, linf_bound=self.linf_bound)

        x_vec = torch.zeros(n_dims, dtype=clean_rgb.dtype)
        clean_rgb_cpu = clean_rgb.detach().cpu().clone()

        clean_logits = self._predict_logits(clean_rgb_cpu, device)
        best_rgb = clean_rgb_cpu.clone()
        best_logits = clean_logits.clone()
        best_pred = int(best_logits.argmax().item())
        best_score = self._score_crossentropy(
            logits=best_logits,
            original_class=original_class,
            target_class=target_class,
            targeted=targeted,
        )

        queries = 1
        success_step = -1
        history: list[dict[str, Any]] = []

        def is_success(pred: int) -> bool:
            if targeted:
                return target_class is not None and pred == int(target_class)
            return pred != int(original_class)

        if is_success(best_pred):
            success_step = 0

        max_steps = min(int(self.steps), int(indices.numel()))
        for step_idx in range(max_steps):
            if stop_on_success and success_step != -1:
                break

            dim = int(indices[step_idx].item())
            diff = torch.zeros(n_dims, dtype=clean_rgb.dtype)
            diff[dim] = self.epsilon

            candidates: list[tuple[float, torch.Tensor, torch.Tensor, int, str]] = []
            for direction, sign in (("left", -1.0), ("right", 1.0)):
                candidate_vec = x_vec + sign * diff
                perturbation = trans(self._expand_vector(candidate_vec, expand_dims, self.image_size))
                candidate_rgb = (clean_rgb_cpu + perturbation).clamp(0, 1)
                candidate_logits = self._predict_logits(candidate_rgb, device)
                candidate_pred = int(candidate_logits.argmax().item())
                candidate_score = self._score_crossentropy(
                    logits=candidate_logits,
                    original_class=original_class,
                    target_class=target_class,
                    targeted=targeted,
                )
                candidates.append((candidate_score, candidate_vec, candidate_logits, candidate_pred, direction))
                queries += 1

            candidates.sort(key=lambda item: item[0], reverse=True)
            chosen_score, chosen_vec, chosen_logits, chosen_pred, chosen_direction = candidates[0]

            if chosen_score > best_score:
                x_vec = chosen_vec
                best_logits = chosen_logits
                best_pred = chosen_pred
                best_score = chosen_score
                perturbation = trans(self._expand_vector(x_vec, expand_dims, self.image_size))
                best_rgb = (clean_rgb_cpu + perturbation).clamp(0, 1)

            if is_success(best_pred):
                success_step = step_idx + 1

            if (step_idx + 1) % max(1, int(log_every)) == 0 or step_idx == 0 or step_idx + 1 == max_steps:
                history.append(
                    {
                        "step": int(step_idx + 1),
                        "score": float(best_score),
                        "loss": float(best_score),
                        "pred_class": int(best_pred),
                        "queries": int(queries),
                        "chosen_direction": chosen_direction,
                        "objective": "crossentropy",
                    }
                )

        adv_rgb = best_rgb.clone()
        adv_logits = self._predict_logits(adv_rgb, device)
        adv_class = int(adv_logits.argmax().item())
        return adv_rgb, adv_class, success_step, queries, history, clean_logits, adv_logits, float(best_score)
