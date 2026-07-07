import copy

import torch

from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)


class DiffusionUnetBSplineImagePolicy(DiffusionUnetHybridImagePolicy):
    """Diffusion UNet image policy that predicts B-spline action chunks.

    The task shape meta keeps describing the regular action vector dimensions
    because datasets, normalizers, and rollout decoders reason about those
    physical dimensions. This adapter adds one model channel for the knot column
    and returns the full predicted horizon as a B-spline parameter chunk.
    """

    def __init__(self, shape_meta: dict, bspline_degree=3, **kwargs):
        bspline_shape_meta = copy.deepcopy(shape_meta)
        action_shape = bspline_shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        bspline_shape_meta["action"]["shape"] = [int(action_shape[0]) + 1]
        super().__init__(shape_meta=bspline_shape_meta, **kwargs)
        self.regular_action_dim = int(action_shape[0])
        self.bspline_degree = int(bspline_degree)

    def select_action(
            self,
            action_pred: torch.Tensor,
            n_obs_steps: int,
            ) -> torch.Tensor:
        del n_obs_steps
        return action_pred
