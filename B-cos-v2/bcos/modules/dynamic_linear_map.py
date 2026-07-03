from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class DynamicLinearMapOutput:
    """
    Structured output for local affine map around x.

    Shapes:
    - logits_at_x: [B, K]
    - class_indices: [K]
    - w: [B, K, D]
    - b: [B, K]
    """

    logits_at_x: torch.Tensor
    class_indices: torch.Tensor
    w: torch.Tensor
    b: torch.Tensor


class DynamicLinearMapNetwork(nn.Module):
    """
    Differentiable extractor for local affine map rows around input x.

    For selected classes k:
        f_k(z) ~= <W_k(x), z> + b_k(x)
    where W_k(x) is the Jacobian row at anchor x.
    """

    def __init__(
        self,
        model: nn.Module,
        bcos_transform: type | None = None,
        class_indices: Sequence[int] | None = None,
        use_explanation_mode: bool = False,
        create_graph: bool = False,
        detach_output: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.bcos_transform = bcos_transform
        self.class_indices = None if class_indices is None else tuple(int(i) for i in class_indices)
        self.use_explanation_mode = use_explanation_mode
        self.create_graph = create_graph
        self.detach_output = detach_output

    @staticmethod
    def _resolve_indices(logits: torch.Tensor, class_indices: Sequence[int] | None) -> list[int]:
        if class_indices is None:
            return list(range(logits.shape[1]))
        return [int(idx) for idx in class_indices]

    def _prepare_model_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.bcos_transform is None:
            return x
        return self.bcos_transform(x)

    def forward(self, x: torch.Tensor) -> DynamicLinearMapOutput:
        if x.ndim < 2:
            raise ValueError("Expected batched input with shape [B, ...].")

        x_var = x
        if not x_var.requires_grad:
            x_var = x_var.requires_grad_(True)

        mode_ctx = (
            self.model.explanation_mode()
            if self.use_explanation_mode and hasattr(self.model, "explanation_mode")
            else nullcontext()
        )

        with torch.enable_grad(), mode_ctx:
            logits = self.model(self._prepare_model_input(x_var))
            selected_indices = self._resolve_indices(logits, self.class_indices)
            idx_tensor = torch.as_tensor(selected_indices, device=logits.device, dtype=torch.long)

            jacobian_rows = []
            for i, class_idx in enumerate(selected_indices):
                keep_graph = self.create_graph or (i < len(selected_indices) - 1)
                grad_i = torch.autograd.grad(
                    logits[:, class_idx].sum(),
                    x_var,
                    retain_graph=keep_graph,
                    create_graph=self.create_graph,
                )[0]
                jacobian_rows.append(grad_i.flatten(start_dim=1))

            w = torch.stack(jacobian_rows, dim=1)
            logits_at_x = logits.index_select(dim=1, index=idx_tensor)
            x_flat = x_var.flatten(start_dim=1)
            b = logits_at_x - (w * x_flat.unsqueeze(1)).sum(dim=-1)

            if self.detach_output:
                w = w.detach()
                b = b.detach()
                logits_at_x = logits_at_x.detach()

        return DynamicLinearMapOutput(
            logits_at_x=logits_at_x,
            class_indices=idx_tensor,
            w=w,
            b=b,
        )

    @staticmethod
    def apply_local_affine(output: DynamicLinearMapOutput, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.flatten(start_dim=1)
        return (output.w * x_flat.unsqueeze(1)).sum(dim=-1) + output.b


class W1ToLNetwork(nn.Module):
    """
    Exact Jacobian map network: x -> W_{1->L}(x).

    Output shape: [B, K, D]
    - B: batch size
    - K: number of selected class logits
    - D: flattened input dimension
    """

    def __init__(
        self,
        model: nn.Module,
        bcos_transform: type | None = None,
        class_indices: Sequence[int] | None = None,
        use_explanation_mode: bool = False,
        create_graph: bool = False,
    ) -> None:
        super().__init__()
        self.map_network = DynamicLinearMapNetwork(
            model=model,
            bcos_transform=bcos_transform,
            class_indices=class_indices,
            use_explanation_mode=use_explanation_mode,
            create_graph=create_graph,
            detach_output=False,
        )

    @staticmethod
    def _resolve_indices(logits: torch.Tensor, class_indices: Sequence[int] | None) -> list[int]:
        if class_indices is None:
            return list(range(logits.shape[1]))
        return [int(idx) for idx in class_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map_network(x).w


def build_dynamic_linear_map_network(
    model: nn.Module,
    bcos_transform: type | None = None,
    class_indices: Iterable[int] | None = None,
    use_explanation_mode: bool = False,
    create_graph: bool = False,
    detach_output: bool = False,
) -> DynamicLinearMapNetwork:
    indices = None if class_indices is None else tuple(int(idx) for idx in class_indices)
    return DynamicLinearMapNetwork(
        model=model,
        bcos_transform=bcos_transform,
        class_indices=indices,
        use_explanation_mode=use_explanation_mode,
        create_graph=create_graph,
        detach_output=detach_output,
    )


def build_w1_to_l_network(
    model: nn.Module,
    bcos_transform: type | None = None,
    class_indices: Iterable[int] | None = None,
    use_explanation_mode: bool = False,
    create_graph: bool = False,
) -> W1ToLNetwork:
    indices = None if class_indices is None else tuple(int(idx) for idx in class_indices)
    return W1ToLNetwork(
        model=model,
        bcos_transform=bcos_transform,
        class_indices=indices,
        use_explanation_mode=use_explanation_mode,
        create_graph=create_graph,
    )
