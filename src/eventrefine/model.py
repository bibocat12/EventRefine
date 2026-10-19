"""Fixed public EventRefine model.

This release exposes one architecture: sensor/local-attention encoding followed
by feature-space graph refinement with one global reference threshold. Ablation
switches and experiment-selection logic are intentionally not part of this
public model class.
"""
from __future__ import annotations

import torch
from torch import nn

from .models import graph_refinement as graph
from .models import sensor_backbone as backbone


class EventRefineModel(nn.Module):
    """Event-wise EV-UAV segmentation with global-reference refinement."""

    def __init__(self, *, c_stem: int = 64, c_graph: int = 16, k: int = 8) -> None:
        super().__init__()
        if (int(c_stem), int(c_graph), int(k)) != (64, 16, 8):
            raise ValueError("the public EventRefine release fixes c_stem=64, c_graph=16, k=8")
        self.c_stem = int(c_stem)
        self.c_graph = int(c_graph)
        self.k = int(k)
        self.encoder_context = "local_attention"
        self.graph_operator = "clipped_global"
        self.refinement_mode = "feature_graph"
        self.base = backbone.SensorEventBackbone(
            c_stem=self.c_stem, encoder_context=self.encoder_context
        )
        # Keep the validated checkpoint namespace (`cc_mar`) so released
        # checkpoints can be loaded strictly without a compatibility alias.
        self.cc_mar = graph.ContextCalibratedAffinityRefinement(
            c_in=self.c_stem,
            c_graph=self.c_graph,
            k=self.k,
            mode="spatial",
            operator=self.graph_operator,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return final logits, pre-refinement logits, and graph diagnostics."""
        h0_point = self.base.stem(batch["coords_point"], batch["feat_point"])
        h0_s = backbone.place(h0_point, batch["row_S"], batch["rows_S"])
        h_s = self.base.local_S(h0_s, batch["vr_S"])[batch["row_S"]]

        velocity = self.base.velocity(h_s.detach())
        motion_confidence = self.base.motion_conf(h_s)
        xy = torch.stack([batch["x"], batch["y"]], dim=1)
        v_route = velocity.detach()
        x_warp = (batch["x"] - v_route[:, 0] * batch["dt"]).clamp(0, backbone.W_IMG - 1)
        y_warp = (batch["y"] - v_route[:, 1] * batch["dt"]).clamp(0, backbone.H_IMG - 1)
        row_t, rows_t, _ = backbone.layout_of(x_warp, y_warp, batch["tau_raw"])
        coords_t = torch.stack(
            [2 * x_warp / backbone.W_IMG - 1,
             2 * y_warp / backbone.H_IMG - 1,
             batch["tau_n"]],
            dim=1,
        )
        h0_point_t = self.base.stem(coords_t, batch["feat_point"])
        h0_t = backbone.place(h0_point_t, row_t, rows_t)
        h_t = self.base.local_T(h0_t, backbone.vrows(row_t, rows_t))[row_t]
        alpha_gate = torch.sigmoid(
            self.base.fuse(torch.cat([h_s, h_t, (h_s - h_t).abs()], dim=1))
        )
        alpha = motion_confidence.unsqueeze(1) * alpha_gate
        h_fused = (1.0 - alpha) * h_s + alpha * h_t

        # EventRefine selects the sensor representation. The motion branch is
        # still computed by the checkpoint-compatible backbone and its cost is
        # included in runtime measurements.
        h_in = h_s
        pre_logits = self.base.pre(torch.cat([h0_point, h_in], dim=1))
        before_graph_logits = self.base.head(torch.cat([h0_point, h_in], dim=1))

        cell_ids = (
            (batch["x"] / backbone.S_FINE).long().clamp(0, backbone.NX_FINE - 1)
            * backbone.NY_FINE
            + (batch["y"] / backbone.S_FINE).long().clamp(0, backbone.NY_FINE - 1)
        )
        polarity = batch["feat_point"][:, 3]
        refined, observed, reference, neighbor_idx = self.cc_mar(
            h_in,
            cell_ids,
            xy,
            batch["tau_raw"],
            polarity,
        )
        final_logits = self.base.head(torch.cat([h0_point, refined], dim=1))
        stats = {
            "observed_affinity": observed,
            "global_reference": reference,
            "neighbor_idx": neighbor_idx,
            "before_graph_logits": before_graph_logits,
            "encoder_features": h_in,
            "refined_features": refined,
            "h0_point": h0_point,
            "velocity": velocity,
            "motion_confidence": motion_confidence,
            "motion_features": h_t,
            "motion_computed": torch.ones((), dtype=torch.bool, device=final_logits.device),
            "sensor_motion_fusion": h_fused,
        }
        return final_logits, pre_logits, stats


__all__ = ["EventRefineModel"]
