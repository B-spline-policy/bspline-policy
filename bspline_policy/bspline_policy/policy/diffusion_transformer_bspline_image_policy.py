import copy

from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)


class DiffusionTransformerBSplineImagePolicy(DiffusionTransformerHybridImagePolicy):
    """Transformer image policy that predicts B-spline action chunks.

    Datasets keep shape_meta action dims at the physical control-point count.
    This adapter adds one model channel for the knot column.
    """

    def __init__(self, shape_meta: dict, bspline_degree=3, **kwargs):
        bspline_shape_meta = copy.deepcopy(shape_meta)
        action_shape = bspline_shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        bspline_shape_meta["action"]["shape"] = [int(action_shape[0]) + 1]
        super().__init__(shape_meta=bspline_shape_meta, **kwargs)
        self.regular_action_dim = int(action_shape[0])
        self.bspline_degree = int(bspline_degree)

    def predict_action(self, obs_dict):
        result = super().predict_action(obs_dict)
        result["action"] = result["action_pred"]
        return result
