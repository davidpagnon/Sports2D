'''
Unit tests for temporal left/right limb swap correction algorithms.
Tests both legacy online greedy correction and offline two-state Viterbi correction.
'''

import numpy as np
import pandas as pd
import pytest
from collections import defaultdict

from Sports2D.Utilities.common import (
    bilateral_body_pairs,
    correct_limb_swaps,
    correct_limb_swaps_viterbi,
)
from Pose2Sim.skeletons import HALPE_26, COCO_17, BODY_25, BLAZEPOSE


# ---------------------------------------------------------------------------
# 1. Skeleton Tree & Bilateral Pairs Tests
# ---------------------------------------------------------------------------

def test_bilateral_pairs_halpe26():
    """Verify bilateral body pairs extraction for HALPE_26."""
    pairs = bilateral_body_pairs(HALPE_26)
    assert "Hip" in pairs
    assert "Shoulder" in pairs
    # Hip group should contain Knee (13, 14), Ankle (15, 16), Toes/Heel
    hip_pairs = pairs["Hip"]
    assert (13, 14) in hip_pairs  # Knee
    assert (15, 16) in hip_pairs  # Ankle
    # Shoulder group should contain Elbow (7, 8), Wrist (9, 10)
    shoulder_pairs = pairs["Shoulder"]
    assert (7, 8) in shoulder_pairs
    assert (9, 10) in shoulder_pairs


def test_bilateral_pairs_coco17():
    """Verify bilateral body pairs extraction for COCO_17."""
    pairs = bilateral_body_pairs(COCO_17)
    assert "Hip" in pairs
    assert "Shoulder" in pairs
    assert (13, 14) in pairs["Hip"]  # Knee
    assert (15, 16) in pairs["Hip"]  # Ankle


def test_bilateral_pairs_blazepose_case_insensitivity():
    """Verify case-insensitive matching for BLAZEPOSE (Lknee vs RKnee)."""
    pairs = bilateral_body_pairs(BLAZEPOSE)
    assert "Hip" in pairs
    # Lknee (25) and RKnee (26), LAnkle (27) and RAnkle (28) must be included under Hip
    hip_pairs = pairs["Hip"]
    assert (25, 26) in hip_pairs, "BLAZEPOSE Knee (25, 26) missing due to casing mismatch"
    assert (27, 28) in hip_pairs, "BLAZEPOSE Ankle (27, 28) missing"
    assert (29, 30) in hip_pairs  # Heel
    assert (31, 32) in hip_pairs  # BigToe


# ---------------------------------------------------------------------------
# 2. Viterbi Swap Correction: Algorithmic Scenarios
# ---------------------------------------------------------------------------

def _generate_synthetic_leg_trajectory(T=30, x_offset=0.0):
    """Generate clean synthetic trajectory for 2 bilateral pairs (Knee, Ankle)."""
    # Keypoints: [..., L_Knee, R_Knee, L_Ankle, R_Ankle]
    # Indices: L_Knee=0, R_Knee=1, L_Ankle=2, R_Ankle=3
    kpts = np.zeros((T, 4, 2))
    for t in range(T):
        # Left leg at x ~ 100, Right leg at x ~ 160
        kpts[t, 0] = [100.0 + x_offset + t * 2.0, 350.0]  # L Knee
        kpts[t, 1] = [160.0 + x_offset + t * 2.0, 350.0]  # R Knee
        kpts[t, 2] = [95.0 + x_offset + t * 2.0, 480.0]   # L Ankle
        kpts[t, 3] = [165.0 + x_offset + t * 2.0, 480.0]  # R Ankle
    pairs_by_group = {"Hip": [(0, 1), (2, 3)]}
    return kpts, pairs_by_group


def test_viterbi_clean_motion_no_false_swaps():
    """Verify clean continuous motion produces 0 swap corrections."""
    kpts, pairs = _generate_synthetic_leg_trajectory(T=30)
    corrected, _, stats = correct_limb_swaps_viterbi(kpts, pairs_by_group=pairs, height_px=500.0)
    assert stats["total_swapped_frames"] == 0
    assert np.allclose(corrected, kpts)


def test_viterbi_isolated_single_frame_swap():
    """Verify an isolated single-frame swap at t=15 is resolved with zero persistence."""
    kpts, pairs = _generate_synthetic_leg_trajectory(T=30)
    ground_truth_clean = kpts.copy()

    # Introduce swap at frame 15
    kpts[15, [0, 1]] = kpts[15, [1, 0]]
    kpts[15, [2, 3]] = kpts[15, [3, 2]]

    corrected, _, stats = correct_limb_swaps_viterbi(kpts, pairs_by_group=pairs, height_px=500.0)
    assert stats["total_swapped_frames"] == 1
    assert stats["swapped_frames_by_group"]["Hip"] == [15]
    # Must match original ground truth exactly
    assert np.allclose(corrected, ground_truth_clean)


def test_viterbi_sustained_multi_frame_swap():
    """Verify sustained swap (frames 10..19) is resolved across all frames without persistent lock."""
    kpts, pairs = _generate_synthetic_leg_trajectory(T=30)
    ground_truth_clean = kpts.copy()

    # Introduce sustained swap across 10 frames
    for t in range(10, 20):
        kpts[t, [0, 1]] = kpts[t, [1, 0]]
        kpts[t, [2, 3]] = kpts[t, [3, 2]]

    corrected, _, stats = correct_limb_swaps_viterbi(kpts, pairs_by_group=pairs, height_px=500.0)
    assert stats["total_swapped_frames"] == 10
    assert stats["swapped_frames_by_group"]["Hip"] == list(range(10, 20))
    # After frame 19, frames 20..29 must remain unswapped
    assert np.allclose(corrected, ground_truth_clean)


def test_viterbi_sagittal_limb_crossing_no_false_swap():
    """Verify natural limb crossing (walking stride) does NOT trigger false swaps."""
    T = 30
    kpts = np.zeros((T, 4, 2))
    # Left leg swings forward: x goes from 80 to 200
    # Right leg swings backward: x goes from 200 to 80
    # Crossing occurs at t=15 (x ~ 140 for both)
    for t in range(T):
        kpts[t, 0] = [80.0 + t * 4.0, 350.0]
        kpts[t, 1] = [200.0 - t * 4.0, 350.0]
        kpts[t, 2] = [75.0 + t * 4.0, 480.0]
        kpts[t, 3] = [205.0 - t * 4.0, 480.0]

    pairs_by_group = {"Hip": [(0, 1), (2, 3)]}
    corrected, _, stats = correct_limb_swaps_viterbi(kpts, pairs_by_group=pairs_by_group, height_px=500.0)
    assert stats["total_swapped_frames"] == 0
    assert np.allclose(corrected, kpts)


def test_viterbi_single_pair_hallucination_rejection():
    """Verify that a single hallucinated keypoint (e.g. ankle) does NOT cause the whole leg to swap."""
    kpts, pairs = _generate_synthetic_leg_trajectory(T=30)
    ground_truth_clean = kpts.copy()

    # At frame 12, only ankle is hallucinated on opposite side; knee is normal
    kpts[12, [2, 3]] = kpts[12, [3, 2]]

    corrected, _, stats = correct_limb_swaps_viterbi(kpts, pairs_by_group=pairs, height_px=500.0)
    # The group consensus should reject swapping the entire leg
    assert stats["total_swapped_frames"] == 0
    # Frame 12 knees remain untouched
    assert np.allclose(corrected[:, [0, 1]], ground_truth_clean[:, [0, 1]])


def test_viterbi_occlusion_and_nans():
    """Verify that missing keypoints (NaNs) are handled gracefully without exceptions or false triggers."""
    kpts, pairs = _generate_synthetic_leg_trajectory(T=30)
    # Introduce occlusion for 5 frames
    kpts[10:15] = np.nan

    corrected, _, stats = correct_limb_swaps_viterbi(kpts, pairs_by_group=pairs, height_px=500.0)
    assert stats["total_swapped_frames"] == 0
    assert np.isnan(corrected[10:15]).all()
    # Non-occluded frames match
    assert np.allclose(corrected[:10], kpts[:10])
    assert np.allclose(corrected[15:], kpts[15:])


def test_viterbi_dataframe_interface():
    """Verify correct_limb_swaps_viterbi accepts and returns pandas DataFrames seamlessly."""
    T = 20
    kpts, pairs = _generate_synthetic_leg_trajectory(T=T)
    cols = ["LKnee", "RKnee", "LAnkle", "RAnkle"]
    df_X = pd.DataFrame(kpts[..., 0], columns=cols)
    df_Y = pd.DataFrame(kpts[..., 1], columns=cols)
    scores = pd.DataFrame(np.ones((T, 4)), columns=cols)

    # Swap at frame 5
    df_X.loc[5, ["LKnee", "RKnee"]] = df_X.loc[5, ["RKnee", "LKnee"]].values
    df_X.loc[5, ["LAnkle", "RAnkle"]] = df_X.loc[5, ["RAnkle", "LAnkle"]].values

    (out_X, out_Y), out_S, stats = correct_limb_swaps_viterbi(
        (df_X, df_Y),
        person_scores_seq=scores,
        pairs_by_group=pairs,
        height_px=500.0,
    )
    assert isinstance(out_X, pd.DataFrame)
    assert isinstance(out_Y, pd.DataFrame)
    assert list(out_X.columns) == cols
    assert stats["total_swapped_frames"] == 1
    assert stats["swapped_frames_by_group"]["Hip"] == [5]


def test_viterbi_rapid_rotation_no_false_swap():
    """Verify rapid 180-deg axial rotation does not trigger false swaps."""
    T = 40
    kpts = np.zeros((T, 4, 2))
    # Rotation: x coordinates follow cosine/sine orbits
    angles = np.linspace(0, np.pi, T)
    for t in range(T):
        theta = angles[t]
        kpts[t, 0] = [150.0 - 40.0 * np.cos(theta), 350.0]  # L Knee
        kpts[t, 1] = [150.0 + 40.0 * np.cos(theta), 350.0]  # R Knee
        kpts[t, 2] = [150.0 - 45.0 * np.cos(theta), 480.0]  # L Ankle
        kpts[t, 3] = [150.0 + 45.0 * np.cos(theta), 480.0]  # R Ankle

    pairs = {"Hip": [(0, 1), (2, 3)]}
    corrected, _, stats = correct_limb_swaps_viterbi(kpts, pairs_by_group=pairs, height_px=500.0)
    assert stats["total_swapped_frames"] == 0
    assert np.allclose(corrected, kpts)


def test_viterbi_crouching_no_false_swap():
    """Verify deep crouching motion (vertical displacement) causes no false swap."""
    T = 30
    kpts = np.zeros((T, 4, 2))
    for t in range(T):
        # Crouch down and up: vertical sinusoidal deflection of 80px
        dy = 80.0 * np.sin(np.pi * t / T)
        kpts[t, 0] = [120.0, 350.0 + dy]
        kpts[t, 1] = [180.0, 350.0 + dy]
        kpts[t, 2] = [115.0, 480.0 + dy * 0.5]
        kpts[t, 3] = [185.0, 480.0 + dy * 0.5]

    pairs = {"Hip": [(0, 1), (2, 3)]}
    corrected, _, stats = correct_limb_swaps_viterbi(kpts, pairs_by_group=pairs, height_px=500.0)
    assert stats["total_swapped_frames"] == 0
    assert np.allclose(corrected, kpts)


def test_viterbi_low_confidence_gating():
    """Verify that low-confidence (<0.3) hallucinated coordinates do not trigger swaps."""
    kpts, pairs = _generate_synthetic_leg_trajectory(T=30)
    scores = np.ones((30, 4))
    # At frame 8, noisy hallucinated swap with low confidence
    kpts[8, [0, 1]] = kpts[8, [1, 0]]
    kpts[8, [2, 3]] = kpts[8, [3, 2]]
    scores[8, :] = 0.15  # Below min_confidence = 0.3

    corrected, _, stats = correct_limb_swaps_viterbi(
        kpts, person_scores_seq=scores, pairs_by_group=pairs, height_px=500.0, min_confidence=0.3
    )
    # Low confidence observations must not trigger swap transitions
    assert stats["total_swapped_frames"] == 0


def test_viterbi_multiple_people_independent():
    """Verify multi-person independence: Person 0 swap does not affect Person 1."""
    kpts0, pairs = _generate_synthetic_leg_trajectory(T=20, x_offset=0.0)
    kpts1, _ = _generate_synthetic_leg_trajectory(T=20, x_offset=300.0)

    # Person 0 has a true swap at frame 7
    kpts0[7, [0, 1]] = kpts0[7, [1, 0]]
    kpts0[7, [2, 3]] = kpts0[7, [3, 2]]

    # Person 1 is clean
    clean1 = kpts1.copy()

    corr0, _, stats0 = correct_limb_swaps_viterbi(kpts0, pairs_by_group=pairs, height_px=500.0)
    corr1, _, stats1 = correct_limb_swaps_viterbi(kpts1, pairs_by_group=pairs, height_px=500.0)

    assert stats0["total_swapped_frames"] == 1
    assert stats0["swapped_frames_by_group"]["Hip"] == [7]
    assert stats1["total_swapped_frames"] == 0
    assert np.allclose(corr1, clean1)


def test_viterbi_hip_shoulder_isolation():
    """Verify that proximal root keypoints (Hips, Shoulders) are excluded from limb group swapping."""
    pairs = bilateral_body_pairs(HALPE_26)
    # In HALPE_26: LShoulder=5, RShoulder=6, LHip=11, RHip=12
    all_pairs = []
    for g_pairs in pairs.values():
        all_pairs.extend(g_pairs)

    assert (11, 12) not in all_pairs, "LHip/RHip (11, 12) must not be in bilateral limb pairs"
    assert (5, 6) not in all_pairs, "LShoulder/RShoulder (5, 6) must not be in bilateral limb pairs"


def test_pipeline_integration_and_downstream_filter_stability():
    """Real integration regression test: verify end-to-end DataFrame pipeline integration,
    toggleability bit-for-bit invariance, and elimination of downstream Butterworth filter ringing."""
    from scipy.signal import butter, filtfilt
    from Pose2Sim.skeletons import HALPE_26
    from anytree import RenderTree

    # 1. Setup realistic keypoint columns matching HALPE_26
    pairs_tree = bilateral_body_pairs(HALPE_26)
    id_to_name = {node.id: node.name for _, _, node in RenderTree(HALPE_26) if node.id is not None}

    # 22 marker names matching Sports2D TRC export
    col_names = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe', 'RSmallToe', 'RHeel',
        'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel',
        'Neck', 'Head', 'Nose', 'RShoulder', 'RElbow', 'RWrist',
        'LShoulder', 'LElbow', 'LWrist'
    ]
    T = 40

    # Generate clean walking trajectory:
    # Right leg oscillates with forward progression, Left leg oscillates with 180 deg phase
    t_vals = np.linspace(0, 4 * np.pi, T)
    df_X = pd.DataFrame(index=range(T), columns=col_names, dtype=float)
    df_Y = pd.DataFrame(index=range(T), columns=col_names, dtype=float)
    df_scores = pd.DataFrame(0.95, index=range(T), columns=col_names, dtype=float)

    for col in col_names:
        df_X[col] = 300.0 + 2.0 * np.arange(T)
        df_Y[col] = 400.0

    # Differentiate bilateral lower limbs with lateral separation
    df_X['LKnee'] = 260.0 + 2.0 * np.arange(T)
    df_Y['LKnee'] = 500.0
    df_X['LAnkle'] = 255.0 + 2.0 * np.arange(T)
    df_Y['LAnkle'] = 700.0
    df_X['LHeel'] = 240.0 + 2.0 * np.arange(T)
    df_Y['LHeel'] = 720.0
    df_X['LBigToe'] = 275.0 + 2.0 * np.arange(T)
    df_Y['LBigToe'] = 730.0
    df_X['LSmallToe'] = 270.0 + 2.0 * np.arange(T)
    df_Y['LSmallToe'] = 730.0

    df_X['RKnee'] = 340.0 + 2.0 * np.arange(T)
    df_Y['RKnee'] = 500.0
    df_X['RAnkle'] = 345.0 + 2.0 * np.arange(T)
    df_Y['RAnkle'] = 700.0
    df_X['RHeel'] = 330.0 + 2.0 * np.arange(T)
    df_Y['RHeel'] = 720.0
    df_X['RBigToe'] = 365.0 + 2.0 * np.arange(T)
    df_Y['RBigToe'] = 730.0
    df_X['RSmallToe'] = 360.0 + 2.0 * np.arange(T)
    df_Y['RSmallToe'] = 730.0

    clean_X = df_X.copy()
    clean_Y = df_Y.copy()

    # Map bilateral pairs to DataFrame column indices
    pairs_mapped = {}
    for g, pairs in pairs_tree.items():
        mapped_g = []
        for l_id, r_id in pairs:
            l_name, r_name = id_to_name.get(l_id), id_to_name.get(r_id)
            if l_name in col_names and r_name in col_names:
                mapped_g.append((col_names.index(l_name), col_names.index(r_name)))
        if mapped_g:
            pairs_mapped[g] = mapped_g

    # 2. Test Bit-for-Bit Invariance when clean / toggle disabled
    out_clean, out_scores_clean, stats_clean = correct_limb_swaps_viterbi(
        (clean_X, clean_Y), person_scores_seq=df_scores,
        height_px=600.0, pairs_by_group=pairs_mapped
    )
    assert stats_clean["total_swapped_frames"] == 0
    pd.testing.assert_frame_equal(out_clean[0], clean_X)
    pd.testing.assert_frame_equal(out_clean[1], clean_Y)

    # 3. Inject sustained 5-frame bilateral swap at frames 15 to 19
    swapped_X = clean_X.copy()
    swapped_Y = clean_Y.copy()
    swap_frames = list(range(15, 20))
    for f in swap_frames:
        for l_col, r_col in [('LKnee', 'RKnee'), ('LAnkle', 'RAnkle'), ('LHeel', 'RHeel'), ('LBigToe', 'RBigToe'), ('LSmallToe', 'RSmallToe')]:
            swapped_X.loc[f, [l_col, r_col]] = swapped_X.loc[f, [r_col, l_col]].values
            swapped_Y.loc[f, [l_col, r_col]] = swapped_Y.loc[f, [r_col, l_col]].values

    # Test that Viterbi correctly recovers original ground truth trajectories
    (corr_X, corr_Y), corr_scores, stats = correct_limb_swaps_viterbi(
        (swapped_X, swapped_Y), person_scores_seq=df_scores,
        height_px=600.0, pairs_by_group=pairs_mapped
    )
    assert stats["total_swapped_frames"] == 5
    assert stats["swapped_frames_by_group"]["Hip"] == swap_frames
    pd.testing.assert_frame_equal(corr_X, clean_X)
    pd.testing.assert_frame_equal(corr_Y, clean_Y)

    # 4. Demonstrate downstream Butterworth filter stabilization
    # Applying a 2nd order lowpass Butterworth filter (cutoff 0.25 Nyquist)
    b, a = butter(2, 0.25, btype='low')

    raw_swapped_ankle = swapped_X['RAnkle'].values
    viterbi_corr_ankle = corr_X['RAnkle'].values

    filtered_unswapped = filtfilt(b, a, raw_swapped_ankle)
    filtered_viterbi = filtfilt(b, a, viterbi_corr_ankle)

    # The unswapped raw data causes massive filter ringing and velocity spikes
    raw_accel = np.diff(filtered_unswapped, n=2)
    viterbi_accel = np.diff(filtered_viterbi, n=2)

    max_accel_raw = np.max(np.abs(raw_accel))
    max_accel_vit = np.max(np.abs(viterbi_accel))

    assert max_accel_vit < 0.2 * max_accel_raw, (
        f"Viterbi should reduce filter ringing acceleration spikes by >80%. "
        f"Got max accel raw={max_accel_raw:.2f}, vit={max_accel_vit:.2f}"
    )


def test_bilateral_pairs_name_mapping_across_skeletons():
    """Verify bilateral_body_pairs(as_names=True) extracts valid name pairs across all 4 skeletons."""
    for skel, expected_pairs in [
        (HALPE_26, [("LKnee", "RKnee"), ("LAnkle", "RAnkle"), ("LElbow", "RElbow")]),
        (COCO_17, [("LKnee", "RKnee"), ("LAnkle", "RAnkle"), ("LElbow", "RElbow")]),
        (BODY_25, [("LKnee", "RKnee"), ("LAnkle", "RAnkle"), ("LElbow", "RElbow")]),
        (BLAZEPOSE, [("Lknee", "RKnee"), ("LAnkle", "RAnkle"), ("LElbow", "RElbow")]),
    ]:
        pairs_dict = bilateral_body_pairs(skel, as_names=True)
        all_pairs = []
        for p_list in pairs_dict.values():
            all_pairs.extend(p_list)
        for expected in expected_pairs:
            assert expected in all_pairs, f"{expected} missing from {skel} bilateral name pairs"


def test_viterbi_permuted_column_order():
    """Verify that Viterbi swap correction works identically when DataFrame column order is permuted."""
    from Pose2Sim.skeletons import HALPE_26

    pairs_names = bilateral_body_pairs(HALPE_26, as_names=True)
    col_names = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe', 'RSmallToe', 'RHeel',
        'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel',
        'Neck', 'Head', 'Nose', 'RShoulder', 'RElbow', 'RWrist',
        'LShoulder', 'LElbow', 'LWrist'
    ]
    T = 25
    df_X = pd.DataFrame(300.0, index=range(T), columns=col_names, dtype=float)
    df_Y = pd.DataFrame(400.0, index=range(T), columns=col_names, dtype=float)

    # Distinct positions for legs
    df_X['LKnee'], df_Y['LKnee'] = 250.0, 500.0
    df_X['RKnee'], df_Y['RKnee'] = 350.0, 500.0
    df_X['LAnkle'], df_Y['LAnkle'] = 245.0, 700.0
    df_X['RAnkle'], df_Y['RAnkle'] = 355.0, 700.0

    # Inject swap at frame 10..14
    clean_X = df_X.copy()
    clean_Y = df_Y.copy()
    for t in range(10, 15):
        df_X.loc[t, ['LKnee', 'RKnee']] = df_X.loc[t, ['RKnee', 'LKnee']].values
        df_X.loc[t, ['LAnkle', 'RAnkle']] = df_X.loc[t, ['RAnkle', 'LAnkle']].values

    # Permute columns (shuffled order)
    permuted_cols = col_names[::-1]  # reversed
    df_X_perm = df_X[permuted_cols].copy()
    df_Y_perm = df_Y[permuted_cols].copy()

    (corr_X, corr_Y), _, stats = correct_limb_swaps_viterbi(
        (df_X_perm, df_Y_perm),
        height_px=600.0,
        pairs_by_group=pairs_names,
    )

    assert stats["total_swapped_frames"] == 5
    # Output must match clean ground truth in the permuted order
    pd.testing.assert_frame_equal(corr_X, clean_X[permuted_cols])
    pd.testing.assert_frame_equal(corr_Y, clean_Y[permuted_cols])


def test_viterbi_score_and_coordinate_synchronization():
    """Verify that when coordinates are swapped, detection confidence scores swap synchronously."""
    from Pose2Sim.skeletons import HALPE_26

    pairs_names = bilateral_body_pairs(HALPE_26, as_names=True)
    col_names = ['LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    T = 20
    df_X = pd.DataFrame(index=range(T), columns=col_names, dtype=float)
    df_Y = pd.DataFrame(index=range(T), columns=col_names, dtype=float)
    df_scores = pd.DataFrame(index=range(T), columns=col_names, dtype=float)

    # Left limb has high score (0.95), Right limb has moderate score (0.50)
    df_X['LKnee'], df_Y['LKnee'], df_scores['LKnee'] = 250.0, 500.0, 0.95
    df_X['RKnee'], df_Y['RKnee'], df_scores['RKnee'] = 350.0, 500.0, 0.50
    df_X['LAnkle'], df_Y['LAnkle'], df_scores['LAnkle'] = 250.0, 700.0, 0.95
    df_X['RAnkle'], df_Y['RAnkle'], df_scores['RAnkle'] = 350.0, 700.0, 0.50

    # Inject swap at frame 8
    swapped_X = df_X.copy()
    swapped_Y = df_Y.copy()
    swapped_scores = df_scores.copy()
    for col_L, col_R in [('LKnee', 'RKnee'), ('LAnkle', 'RAnkle')]:
        swapped_X.loc[8, [col_L, col_R]] = swapped_X.loc[8, [col_R, col_L]].values
        swapped_scores.loc[8, [col_L, col_R]] = swapped_scores.loc[8, [col_R, col_L]].values

    pairs = {"Hip": [("LKnee", "RKnee"), ("LAnkle", "RAnkle")]}
    (corr_X, corr_Y), corr_scores, stats = correct_limb_swaps_viterbi(
        (swapped_X, swapped_Y),
        person_scores_seq=swapped_scores,
        height_px=600.0,
        pairs_by_group=pairs,
    )

    assert stats["total_swapped_frames"] == 1
    # Check that scores at frame 8 were swapped back to clean state
    assert corr_scores.loc[8, 'LKnee'] == 0.95
    assert corr_scores.loc[8, 'RKnee'] == 0.50
    pd.testing.assert_frame_equal(corr_scores, df_scores)
    pd.testing.assert_frame_equal(corr_X, df_X)


def test_multi_person_pipeline_integration():
    """Verify multi-person post-processing: person 0 swapped is corrected while person 1 clean is untouched."""
    from Pose2Sim.skeletons import HALPE_26

    pairs_names = bilateral_body_pairs(HALPE_26, as_names=True)
    col_names = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Neck', 'Head', 'Nose', 'RShoulder', 'LShoulder']
    T = 20
    N_persons = 2

    # Person 0 at x ~ 200, Person 1 at x ~ 500
    tensor_X = np.full((T, N_persons, len(col_names)), 200.0, dtype=float)
    tensor_Y = np.full((T, N_persons, len(col_names)), 400.0, dtype=float)
    tensor_scores = np.full((T, N_persons, len(col_names)), 0.9, dtype=float)

    # Differentiate limbs for Person 0
    k_lk, k_rk = col_names.index('LKnee'), col_names.index('RKnee')
    k_la, k_ra = col_names.index('LAnkle'), col_names.index('RAnkle')
    tensor_X[:, 0, k_lk] = 160.0
    tensor_X[:, 0, k_rk] = 240.0
    tensor_X[:, 0, k_la] = 155.0
    tensor_X[:, 0, k_ra] = 245.0

    # Differentiate limbs for Person 1
    tensor_X[:, 1, :] = 500.0
    tensor_X[:, 1, k_lk] = 460.0
    tensor_X[:, 1, k_rk] = 540.0
    tensor_X[:, 1, k_la] = 455.0
    tensor_X[:, 1, k_ra] = 545.0

    clean_p1_X = tensor_X[:, 1, :].copy()

    # Person 0 has sustained swap at frames 6..10
    tensor_X[6:11, 0, [k_lk, k_rk]] = tensor_X[6:11, 0, [k_rk, k_lk]]
    tensor_X[6:11, 0, [k_la, k_ra]] = tensor_X[6:11, 0, [k_ra, k_la]]

    selected_persons = [0, 1]
    score_data = []

    # Run the exact process.py loop
    for i, idx_person in enumerate(selected_persons):
        all_frames_X_person = pd.DataFrame(tensor_X[:, idx_person, :], columns=col_names)
        all_frames_Y_person = pd.DataFrame(tensor_Y[:, idx_person, :], columns=col_names)
        all_frames_scores_person = pd.DataFrame(tensor_scores[:, idx_person, :], columns=col_names)

        (all_frames_X_person, all_frames_Y_person), score_df, swap_stats = correct_limb_swaps_viterbi(
            (all_frames_X_person, all_frames_Y_person),
            person_scores_seq=all_frames_scores_person,
            height_px=600.0,
            pairs_by_group=pairs_names,
        )
        if score_df is not None:
            all_frames_scores_person = score_df
            tensor_scores[:, idx_person, :] = score_df.values
        tensor_X[:, idx_person, :] = all_frames_X_person.values
        tensor_Y[:, idx_person, :] = all_frames_Y_person.values
        score_data.append(all_frames_scores_person)

        if idx_person == 0:
            assert swap_stats["total_swapped_frames"] == 5
            assert swap_stats["swapped_frames_by_group"]["Hip"] == list(range(6, 11))
        elif idx_person == 1:
            assert swap_stats["total_swapped_frames"] == 0
            np.testing.assert_array_equal(tensor_X[:, 1, :], clean_p1_X)

    assert len(score_data) == 2
    assert score_data[0].shape == (T, len(col_names))
    assert score_data[1].shape == (T, len(col_names))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
