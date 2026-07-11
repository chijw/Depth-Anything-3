"""Synthetic parity tests for Lyra-2's robust Umeyama fallback.

The small reference helpers below are copied from Lyra-2's vendored DA3 commit
1ed6cb8.  Keeping the oracle in this submodule makes the test independently
runnable without requiring a sibling Lyra-2 checkout.
"""

from __future__ import annotations

import numpy as np
import pytest
from evo.core.trajectory import PosePath3D

from depth_anything_3.utils.pose_align import (
    _umeyama_sim3_from_paths,
    _umeyama_sim3_from_paths_robust,
    align_poses_umeyama,
)


def _official_add_tiny_pose_offsets(poses: np.ndarray, offset_scale: float = 1e-6) -> np.ndarray:
    poses = poses.copy()
    for i in range(len(poses)):
        poses[i, :3, 3] += np.array(
            [
                offset_scale * np.sin(2 * np.pi * i * 0.07),
                offset_scale * np.cos(2 * np.pi * i * 0.11),
                offset_scale * np.sin(2 * np.pi * i * 0.13),
            ]
        )
    return poses


def _official_direct_umeyama(pose_ref: np.ndarray, pose_est: np.ndarray):
    path_ref = PosePath3D(poses_se3=pose_ref.copy())
    path_est = PosePath3D(poses_se3=pose_est.copy())
    rotation, translation, scale = path_est.align(path_ref, correct_scale=True)
    return rotation, translation, scale, np.stack(path_est.poses_se3)


def _official_apply_sim3(
    poses: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
) -> np.ndarray:
    aligned = poses.copy()
    aligned[:, :3, :3] = rotation @ poses[:, :3, :3]
    aligned[:, :3, 3] = (rotation @ (scale * poses[:, :3, 3].T)).T + translation
    return aligned


def _official_robust_umeyama(pose_ref: np.ndarray, pose_est: np.ndarray):
    try:
        return _official_direct_umeyama(pose_ref, pose_est)
    except Exception:
        ref_offset = _official_add_tiny_pose_offsets(pose_ref)
        est_offset = _official_add_tiny_pose_offsets(pose_est)
        rotation, translation, scale, _ = _official_direct_umeyama(ref_offset, est_offset)
        aligned = _official_apply_sim3(pose_est, rotation, translation, scale)
        return rotation, translation, scale, aligned


def _poses(points: np.ndarray) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float64), (len(points), 1, 1))
    poses[:, :3, 3] = points
    return poses


def _assert_alignment_equal(actual, expected) -> None:
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual[3], expected[3], rtol=0.0, atol=1e-12)


def test_nondegenerate_alignment_matches_lyra2_official_reference() -> None:
    ref_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.5, -0.5, 2.0],
            [-0.5, 1.0, 1.5],
        ],
        dtype=np.float64,
    )
    est_points = 2.5 * ref_points + np.array([3.0, -2.0, 0.75])
    pose_ref = _poses(ref_points)
    pose_est = _poses(est_points)

    actual = _umeyama_sim3_from_paths_robust(pose_ref, pose_est)
    expected = _official_robust_umeyama(pose_ref, pose_est)
    _assert_alignment_equal(actual, expected)

    # The public extrinsics API must route through the same robust implementation.
    ext_ref = np.linalg.inv(pose_ref)
    ext_est = np.linalg.inv(pose_est)
    public = align_poses_umeyama(ext_ref, ext_est, return_aligned=True)
    np.testing.assert_allclose(public[0], expected[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(public[1], expected[1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(public[2], expected[2], rtol=0.0, atol=1e-12)


def test_pure_forward_alignment_uses_official_fallback_and_stays_finite() -> None:
    frame_count = 8
    frame_ids = np.arange(frame_count, dtype=np.float64)
    pose_ref = _poses(np.stack([np.zeros(frame_count), np.zeros(frame_count), frame_ids], axis=1))
    pose_est = _poses(
        np.stack(
            [
                np.zeros(frame_count),
                np.zeros(frame_count),
                2.0 * frame_ids + 3.0,
            ],
            axis=1,
        )
    )

    # Prove this fixture really exercises the retry rather than the normal path.
    with pytest.raises(Exception):
        _umeyama_sim3_from_paths(pose_ref, pose_est)

    actual = _umeyama_sim3_from_paths_robust(pose_ref, pose_est)
    expected = _official_robust_umeyama(pose_ref, pose_est)
    _assert_alignment_equal(actual, expected)
    for value in actual:
        assert np.isfinite(value).all()

    ext_ref = np.linalg.inv(pose_ref)
    ext_est = np.linalg.inv(pose_est)
    for ransac in (False, True):
        rotation, translation, scale, aligned = align_poses_umeyama(
            ext_ref,
            ext_est,
            return_aligned=True,
            ransac=ransac,
            random_state=0,
        )
        assert np.isfinite(rotation).all()
        assert np.isfinite(translation).all()
        assert np.isfinite(scale).all()
        assert np.isfinite(aligned).all()
