"""Pair-reference graph refinement used by the public EventRefine model."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ContextCalibratedAffinityRefinement(nn.Module):
    """One fixed feature-space graph update with a pair-specific reference score.

    Candidate identities are determined by deterministic spatial-cell/local-order
    slots. Each observed edge score is compared with the reference score computed
    for that same edge. Only the positive difference is used.
    """

    def __init__(
        self,
        c_in: int = 64,
        c_graph: int = 16,
        k: int = 8,
        *,
        mode: str = "spatial",
        operator: str = "pair_reference",
    ) -> None:
        super().__init__()
        if mode != "spatial":
            raise ValueError("EventRefine fixes mode='spatial'")
        if operator != "pair_reference":
            raise ValueError("EventRefine fixes operator='pair_reference'")
        if int(k) != 8:
            raise ValueError("EventRefine fixes k=8")
        self.c_in = int(c_in)
        self.c_graph = int(c_graph)
        self.k = int(k)
        self.mode = mode
        self.operator = operator
        self.down = nn.Linear(c_in, c_graph, bias=False)
        self.aff_mlp = nn.Sequential(
            nn.Linear(2 * c_graph + 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.ref_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.up = nn.Linear(c_graph, c_in, bias=False)
        nn.init.zeros_(self.up.weight)

    def get_neighbors(self, cell_ids: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
        """Return eight slots in stable spatial-cell order.

        Boundary indices are clamped, so duplicates and self-edges are
        intentional. Slots are not filtered by distance or cell membership.
        """
        order = torch.argsort(cell_ids, stable=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(n, device=device)
        offsets = [-4, -3, -2, -1, 1, 2, 3, 4]
        return torch.stack(
            [order[(inverse + offset).clamp(0, n - 1)] for offset in offsets],
            dim=-1,
        )

    def forward(
        self,
        h: torch.Tensor,
        cell_ids: torch.Tensor,
        xy: torch.Tensor,
        t: torch.Tensor,
        reference_descriptor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if h.ndim != 2 or xy.shape != (h.shape[0], 2) or t.shape != (h.shape[0],):
            raise ValueError("graph inputs are not aligned")
        if cell_ids.shape != (h.shape[0],) or reference_descriptor.shape != (h.shape[0],):
            raise ValueError("graph cell/reference inputs are not aligned")
        n = int(h.shape[0])
        if n == 0:
            empty = h.new_empty((0, self.k))
            indices = torch.empty((0, self.k), dtype=torch.long, device=h.device)
            return h, empty, empty, indices

        neighbor_idx = self.get_neighbors(cell_ids, n, h.device)
        z = self.down(h)
        z_i = z.unsqueeze(1)
        z_j = z[neighbor_idx]
        xy_i = xy.unsqueeze(1)
        xy_j = xy[neighbor_idx]
        dt = (t[neighbor_idx] - t.unsqueeze(1)).unsqueeze(-1)
        distance = ((xy_j - xy_i).norm(dim=-1, keepdim=True) / 50.0).clamp(0.0, 5.0)
        dt_norm = (dt.abs() / 50.0).clamp(0.0, 5.0)
        geometry = torch.cat([distance, dt_norm], dim=-1)
        descriptor_product = (
            reference_descriptor.unsqueeze(1).unsqueeze(-1)
            * reference_descriptor[neighbor_idx].unsqueeze(-1)
        )
        observed_input = torch.cat([(z_i - z_j).abs(), z_i * z_j, geometry], dim=-1)
        reference_input = torch.cat([geometry, descriptor_product], dim=-1)

        device_type = h.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            observed = torch.sigmoid(self.aff_mlp(observed_input))
            reference = torch.sigmoid(self.ref_mlp(reference_input))
            # Compare each edge with its own reference, including boundary
            # duplicates and self-edges in the fixed candidate slots.
            weights = F.relu(observed - reference)
            coefficient = weights.to(dtype=z.dtype)
            numerator = (coefficient * (z_j - z_i)).sum(dim=1)
            denominator = 1.0 + coefficient.abs().sum(dim=1)
            refined = h + self.up(numerator / denominator)

        return refined, observed.squeeze(-1), reference.squeeze(-1), neighbor_idx


__all__ = ["ContextCalibratedAffinityRefinement"]

