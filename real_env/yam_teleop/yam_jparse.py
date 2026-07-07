"""
JAX implementation of the J-PARSE algorithm and velocity IK controller.

J-PARSE (Jacobian-based Projection Algorithm for Resolving Singularities
Effectively) provides singularity-aware inverse kinematics by computing a
modified pseudo-inverse that handles singular configurations smoothly.

Reference: https://github.com/chungmin99/jparse

Vendored for tidybot2 from raiden/raiden/robot/_jparse.py (originally pyroki PR #85).
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
from jax.typing import ArrayLike


def compute_jacobian(
    robot: pk.Robot,
    cfg: ArrayLike,
    target_link_index: int,
    position_only: bool = True,
) -> jnp.ndarray:
    """Compute geometric Jacobian via autodiff on pyroki FK."""
    cfg = jnp.asarray(cfg)

    if position_only:
        jacobian = jax.jacfwd(
            lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
        )(cfg)[target_link_index]
    else:

        def get_pos_and_R_flat(q: jax.Array) -> jnp.ndarray:
            poses = robot.forward_kinematics(q)
            pose = jaxlie.SE3(poses[target_link_index])
            return jnp.concatenate(
                [
                    pose.translation(),
                    pose.rotation().as_matrix().reshape(-1),
                ]
            )

        J_combined = jax.jacfwd(get_pos_and_R_flat)(cfg)
        J_pos = J_combined[:3, :]
        dR_flat_dq = J_combined[3:, :]

        n_joints = cfg.shape[-1]
        dR_dq = dR_flat_dq.reshape(3, 3, n_joints)
        R_mat = (
            jaxlie.SE3(robot.forward_kinematics(cfg)[target_link_index])
            .rotation()
            .as_matrix()
        )
        omega_skew = jnp.einsum("acj,bc->abj", dR_dq, R_mat)
        J_ang = jnp.stack(
            [
                omega_skew[2, 1],
                omega_skew[0, 2],
                omega_skew[1, 0],
            ]
        )

        jacobian = jnp.vstack([J_pos, J_ang])

    return jacobian


def jparse_pseudoinverse(
    jacobian: ArrayLike,
    gamma: float = 0.1,
    singular_direction_gain_position: float = 1.0,
    singular_direction_gain_angular: float = 1.0,
    position_dimensions: int | None = None,
    angular_dimensions: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    J = jnp.asarray(jacobian)
    m, n = J.shape

    if position_dimensions is None and angular_dimensions is None:
        pos_dims = m
        ang_dims = 0
    else:
        if position_dimensions is None or angular_dimensions is None:
            raise ValueError(
                "Both position_dimensions and angular_dimensions must be provided."
            )
        if position_dimensions + angular_dimensions != m:
            raise ValueError(
                "position_dimensions + angular_dimensions must equal Jacobian row count."
            )
        pos_dims = position_dimensions
        ang_dims = angular_dimensions

    U, S, Vt = jnp.linalg.svd(J, full_matrices=True)
    k = S.shape[0]

    sigma_max = jnp.max(S)
    threshold = gamma * sigma_max

    non_singular = S > threshold

    S_safety = jnp.where(non_singular, S, threshold)

    S_proj = jnp.where(non_singular, S, 0.0)

    U_k = U[:, :k]
    Vt_k = Vt[:k, :]

    J_safety = U_k * S_safety[None, :] @ Vt_k
    J_proj = U_k * S_proj[None, :] @ Vt_k

    J_safety_pinv = jnp.linalg.pinv(J_safety)
    J_proj_pinv = jnp.linalg.pinv(J_proj)

    phi = jnp.where(non_singular, 0.0, S / (sigma_max * gamma))
    singular_gains = jnp.concatenate(
        [
            jnp.full((pos_dims,), singular_direction_gain_position),
            jnp.full((ang_dims,), singular_direction_gain_angular),
        ]
    )
    Kp = jnp.diag(singular_gains)
    Phi_singular = (U_k * phi[None, :]) @ U_k.T @ Kp

    J_parse = J_safety_pinv @ J_proj @ J_proj_pinv + J_safety_pinv @ Phi_singular

    nullspace = jnp.eye(n) - J_safety_pinv @ J_safety

    return J_parse, nullspace


def pinv(jacobian: ArrayLike) -> jnp.ndarray:
    return jnp.linalg.pinv(jnp.asarray(jacobian))


def damped_least_squares(
    jacobian: ArrayLike,
    damping: float = 0.05,
) -> jnp.ndarray:
    J = jnp.asarray(jacobian)
    n = J.shape[1]
    return jnp.linalg.inv(J.T @ J + damping**2 * jnp.eye(n)) @ J.T


def manipulability_measure(jacobian: ArrayLike) -> jnp.ndarray:
    J = jnp.asarray(jacobian)
    return jnp.sqrt(jnp.linalg.det(J @ J.T))


def inverse_condition_number(jacobian: ArrayLike) -> jnp.ndarray:
    J = jnp.asarray(jacobian)
    S = jnp.linalg.svd(J, compute_uv=False)
    return jnp.min(S) / jnp.max(S)


def jparse_step(
    robot: pk.Robot,
    cfg: ArrayLike,
    target_link_index: int,
    target_position: ArrayLike,
    target_wxyz: ArrayLike | None = None,
    *,
    method: Literal["jparse", "pinv", "dls"] = "jparse",
    gamma: float = 0.05,
    singular_direction_gain_position: float = 1.0,
    singular_direction_gain_angular: float = 1.0,
    position_gain: float = 5.0,
    orientation_gain: float = 1.0,
    nullspace_gain: float = 0.5,
    max_joint_velocity: float = 2.0,
    dls_damping: float = 0.05,
    dt: float = 0.01,
    home_cfg: ArrayLike | None = None,
) -> tuple[np.ndarray, dict]:
    cfg = jnp.asarray(cfg)
    target_position = jnp.asarray(target_position)
    position_only = target_wxyz is None

    poses = robot.forward_kinematics(cfg)
    target_pose = jaxlie.SE3(poses[target_link_index])
    current_pos = target_pose.translation()

    pos_error = target_position - current_pos
    pos_error_mag = jnp.linalg.norm(pos_error)

    omega_error = jnp.zeros(3)
    if position_only:
        v_des = position_gain * pos_error
    else:
        assert target_wxyz is not None
        tw = jnp.asarray(target_wxyz)
        tw = tw / jnp.linalg.norm(tw)

        current_wxyz = target_pose.rotation().wxyz
        current_wxyz = current_wxyz / jnp.linalg.norm(current_wxyz)

        tw = jnp.asarray(jnp.where(jnp.dot(tw, current_wxyz) < 0, -tw, tw))

        q_current = jaxlie.SO3(current_wxyz)
        q_target = jaxlie.SO3(tw)
        omega_error = (q_target @ q_current.inverse()).log()

        omega_mag = jnp.linalg.norm(omega_error)
        max_omega = 1.0
        omega_error = jnp.asarray(
            jnp.where(
                omega_mag > max_omega, omega_error * max_omega / omega_mag, omega_error
            )
        )

        v_des = jnp.concatenate(
            [
                position_gain * pos_error,
                orientation_gain * omega_error,
            ]
        )

    jacobian = compute_jacobian(
        robot, cfg, target_link_index, position_only=position_only
    )

    if method == "jparse":
        J_inv, N = jparse_pseudoinverse(
            jacobian,
            gamma=gamma,
            singular_direction_gain_position=singular_direction_gain_position,
            singular_direction_gain_angular=singular_direction_gain_angular,
            position_dimensions=3,
            angular_dimensions=0 if position_only else 3,
        )
    elif method == "pinv":
        J_inv = pinv(jacobian)
        N = jnp.eye(jacobian.shape[1]) - J_inv @ jacobian
    else:
        J_inv = damped_least_squares(jacobian, dls_damping)
        N = jnp.eye(jacobian.shape[1]) - J_inv @ jacobian

    dq = J_inv @ v_des

    if nullspace_gain > 0:
        if home_cfg is None:
            lower = robot.joints.lower_limits
            upper = robot.joints.upper_limits
            home = (lower + upper) / 2.0
        else:
            home = jnp.asarray(home_cfg)
        dq = dq + N @ (-nullspace_gain * (cfg - home))

    max_joint_vel = jnp.max(jnp.abs(dq))

    scale = jnp.where(
        jnp.max(jnp.abs(dq)) > max_joint_velocity,
        max_joint_velocity / jnp.max(jnp.abs(dq)),
        1.0,
    )
    dq = dq * scale

    new_cfg = cfg + dq * dt

    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    new_cfg = jnp.clip(new_cfg, lower, upper)

    info = {
        "position_error": pos_error_mag,
        "orientation_error": jnp.linalg.norm(omega_error),
        "max_joint_vel": max_joint_vel,
        "jacobian": jacobian,
        "manipulability": manipulability_measure(jacobian),
        "inverse_condition_number": inverse_condition_number(jacobian),
    }

    return new_cfg, info
