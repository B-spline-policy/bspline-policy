"""Lightweight B-spline knot encoding helpers."""

import torch


def encode_relative_knots(action_data, degree: int = 3):
    """Encode knot values as first valid knot plus adjacent differences."""
    result = action_data.clone() if torch.is_tensor(action_data) else action_data.copy()
    knots = result[..., 0]
    original_knots = knots.clone() if torch.is_tensor(knots) else knots.copy()

    knots[..., 0] = original_knots[..., degree]
    knots[..., 1:] = original_knots[..., 1:] - original_knots[..., :-1]
    return result


def decode_relative_knots(action_data, degree: int = 3):
    """Decode the representation produced by encode_relative_knots."""
    result = action_data.clone() if torch.is_tensor(action_data) else action_data.copy()
    encoded = result[..., 0].clone() if torch.is_tensor(result) else result[..., 0].copy()
    knots = result[..., 0]
    n_knots = knots.shape[-1]

    knots[..., degree] = encoded[..., 0]
    for knot_idx in range(degree - 1, -1, -1):
        knots[..., knot_idx] = knots[..., knot_idx + 1] - encoded[..., knot_idx + 1]
    for knot_idx in range(degree + 1, n_knots):
        knots[..., knot_idx] = knots[..., knot_idx - 1] + encoded[..., knot_idx]

    return result
