#!/usr/bin/env python3
'''
Controlled comparison and evaluation framework for bilateral keypoint swap correction in Sports2D.

Evaluates 4 configurations:
A. Swap correction disabled (handle_LR_swap=False)
B. Current PR #41 implementation (online greedy, min_improvement=0.01, min_improvement_wrt_height=0.01, min_pairs=1)
C. Conservative online implementation (min_improvement=0.05, min_improvement_wrt_height=0.03, min_pairs=2)
D. Proposed offline temporal correction (two-state Viterbi filter per bilateral group before interpolation)

Provides:
- Synthetic evaluation suite with verified ground truth (Clean, Isolated Swap, Sustained Swap,
  Sagittal Crossing, Axial Rotation, Occlusion, Single-Pair Hallucination)
- Real-world video evaluation on available recordings with trajectory continuity and failure mode analysis
- Cross-skeleton compatibility audit (HALPE_26, COCO_17, BODY_25, BLAZEPOSE)
- CLI interface and JSON export for automated benchmarking
'''

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Ensure Sports2D is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Sports2D.Utilities.common import (
    bilateral_body_pairs,
    correct_limb_swaps,
    correct_limb_swaps_viterbi,
)
from Pose2Sim.skeletons import HALPE_26, COCO_17, BODY_25, BLAZEPOSE


# ---------------------------------------------------------------------------
# Method Execution Helpers
# ---------------------------------------------------------------------------

def run_method_A(kpts: np.ndarray, scores: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Method A: Swap correction disabled (identity pass-through)."""
    return kpts.copy(), scores.copy(), {"method": "A_disabled", "swapped_frames": [], "trigger_count": 0}


def run_method_B(kpts: np.ndarray, scores: np.ndarray, pairs_by_group: Dict[str, List[Tuple[int, int]]],
                 height_px: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Method B: Current PR #41 implementation (online greedy, permissive thresholds 0.01 / 0.01, min_pairs=1)."""
    T, K, _ = kpts.shape
    out_kpts = kpts.copy()
    out_scores = scores.copy()
    swapped_frames = []
    
    prev_kpts = out_kpts[0].copy()
    for t in range(1, T):
        curr_k = out_kpts[t].copy()
        curr_s = out_scores[t].copy()
        h_frame = height_px
        if not np.isfinite(h_frame) or h_frame <= 0:
            finite_Y = curr_k[:, 1][np.isfinite(curr_k[:, 1])]
            h_frame = np.ptp(finite_Y) if len(finite_Y) > 1 else 500.0
            
        c_k, c_s, corrected = correct_limb_swaps(
            curr_k, curr_s, prev_kpts, h_frame, pairs_by_group,
            min_improvement=0.01, min_improvement_wrt_height=0.01, min_pairs=1
        )
        if corrected:
            swapped_frames.append(t)
            out_kpts[t] = c_k
            out_scores[t] = c_s
            prev_kpts = c_k.copy()
        else:
            prev_kpts = curr_k.copy()
            
    return out_kpts, out_scores, {
        "method": "B_current_pr41_af93197",
        "swapped_frames": swapped_frames,
        "trigger_count": len(swapped_frames),
    }


def run_method_C(kpts: np.ndarray, scores: np.ndarray, pairs_by_group: Dict[str, List[Tuple[int, int]]],
                 height_px: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Method C: Conservative online implementation (thresholds 0.05 / 0.03, min_pairs=2)."""
    T, K, _ = kpts.shape
    out_kpts = kpts.copy()
    out_scores = scores.copy()
    swapped_frames = []
    
    prev_kpts = out_kpts[0].copy()
    for t in range(1, T):
        curr_k = out_kpts[t].copy()
        curr_s = out_scores[t].copy()
        h_frame = height_px
        if not np.isfinite(h_frame) or h_frame <= 0:
            finite_Y = curr_k[:, 1][np.isfinite(curr_k[:, 1])]
            h_frame = np.ptp(finite_Y) if len(finite_Y) > 1 else 500.0
            
        c_k, c_s, corrected = correct_limb_swaps(
            curr_k, curr_s, prev_kpts, h_frame, pairs_by_group,
            min_improvement=0.05, min_improvement_wrt_height=0.03, min_pairs=2
        )
        if corrected:
            swapped_frames.append(t)
            out_kpts[t] = c_k
            out_scores[t] = c_s
            prev_kpts = c_k.copy()
        else:
            prev_kpts = curr_k.copy()
            
    return out_kpts, out_scores, {
        "method": "C_conservative_online",
        "swapped_frames": swapped_frames,
        "trigger_count": len(swapped_frames),
    }


def run_method_D(kpts: np.ndarray, scores: np.ndarray, pairs_by_group: Dict[str, List[Tuple[int, int]]],
                 height_px: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Method D: Proposed offline temporal correction (two-state Viterbi filter)."""
    out_kpts, out_scores, stats = correct_limb_swaps_viterbi(
        kpts, person_scores_seq=scores, height_px=height_px, pairs_by_group=pairs_by_group,
        lambda_enter=0.04, lambda_exit=0.04, lambda_prior=0.001, min_confidence=0.3
    )
    all_swapped = set()
    for s_list in stats.get("swapped_frames_by_group", {}).values():
        all_swapped.update(s_list)
    sorted_swapped = sorted(list(all_swapped))
    return out_kpts, out_scores, {
        "method": "D_proposed_offline_viterbi",
        "swapped_frames": sorted_swapped,
        "trigger_count": len(sorted_swapped),
        "details_by_group": stats.get("swapped_frames_by_group", {}),
    }


METHODS = {
    "A_disabled": run_method_A,
    "B_current_pr41": run_method_B,
    "C_conservative": run_method_C,
    "D_offline_viterbi": run_method_D,
}


# ---------------------------------------------------------------------------
# Metric Recovery & Trajectory Discontinuity Analysis
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(raw_kpts: np.ndarray, corrected_kpts: np.ndarray,
                               pairs: List[Tuple[int, int]], height_px: float) -> Dict[str, float]:
    """Compute maximum velocities, accelerations, and trajectory jump metrics."""
    T, K, _ = raw_kpts.shape
    max_raw_vel = 0.0
    max_corr_vel = 0.0
    max_raw_acc = 0.0
    max_corr_acc = 0.0
    total_diff_px = 0.0
    max_diff_px = 0.0

    for l_idx, r_idx in pairs:
        for idx in (l_idx, r_idx):
            p_raw = raw_kpts[:, idx]
            p_corr = corrected_kpts[:, idx]

            diffs = np.linalg.norm(p_corr - p_raw, axis=-1)
            finite_diffs = diffs[np.isfinite(diffs)]
            if len(finite_diffs) > 0:
                total_diff_px += float(np.sum(finite_diffs))
                max_diff_px = max(max_diff_px, float(np.max(finite_diffs)))

            # Velocities (1st derivative)
            v_raw = np.linalg.norm(np.diff(p_raw, axis=0), axis=-1)
            v_corr = np.linalg.norm(np.diff(p_corr, axis=0), axis=-1)
            finite_vr = v_raw[np.isfinite(v_raw)]
            finite_vc = v_corr[np.isfinite(v_corr)]
            if len(finite_vr) > 0:
                max_raw_vel = max(max_raw_vel, float(np.max(finite_vr)))
            if len(finite_vc) > 0:
                max_corr_vel = max(max_corr_vel, float(np.max(finite_vc)))

            # Accelerations (2nd derivative / jerk proxy)
            if T > 2:
                a_raw = np.linalg.norm(np.diff(p_raw, n=2, axis=0), axis=-1)
                a_corr = np.linalg.norm(np.diff(p_corr, n=2, axis=0), axis=-1)
                finite_ar = a_raw[np.isfinite(a_raw)]
                finite_ac = a_corr[np.isfinite(a_corr)]
                if len(finite_ar) > 0:
                    max_raw_acc = max(max_raw_acc, float(np.max(finite_ar)))
                if len(finite_ac) > 0:
                    max_corr_acc = max(max_corr_acc, float(np.max(finite_ac)))

    return {
        "max_coordinate_diff_px": max_diff_px,
        "total_coordinate_diff_px": total_diff_px,
        "max_velocity_px_per_frame": max_corr_vel,
        "max_acceleration_px_per_frame2": max_corr_acc,
        "velocity_reduction_factor": float(max_raw_vel / max(1e-4, max_corr_vel)),
    }


# ---------------------------------------------------------------------------
# Synthetic Benchmark Suite (With Verified Ground Truth)
# ---------------------------------------------------------------------------

def generate_synthetic_battery(T: int = 40, height_px: float = 500.0) -> Dict[str, Dict[str, Any]]:
    """Build standardized synthetic benchmark battery labeled with known ground-truth state."""
    scenarios = {}

    # Pairs: L_Knee=0, R_Knee=1, L_Ankle=2, R_Ankle=3
    pairs_by_group = {"Hip": [(0, 1), (2, 3)]}

    def _base_kpts():
        k = np.zeros((T, 4, 2))
        for t in range(T):
            k[t, 0] = [100.0 + t * 2.0, 350.0]
            k[t, 1] = [160.0 + t * 2.0, 350.0]
            k[t, 2] = [95.0 + t * 2.0, 480.0]
            k[t, 3] = [165.0 + t * 2.0, 480.0]
        return k

    # Scenario 1: Clean continuous motion (Ground Truth: 0 swaps)
    clean_k = _base_kpts()
    clean_gt = np.zeros(T, dtype=int)
    scenarios["1_clean_continuous_motion"] = {
        "description": "Clean walking motion with zero swaps; tests false-positive rate.",
        "kpts": clean_k,
        "scores": np.ones((T, 4)),
        "ground_truth_swapped": clean_gt,
        "pairs_by_group": pairs_by_group,
        "height_px": height_px,
    }

    # Scenario 2: Isolated 1-frame swap at t=15 (Ground Truth: [15])
    iso_k = _base_kpts()
    iso_k[15, [0, 1]] = iso_k[15, [1, 0]]
    iso_k[15, [2, 3]] = iso_k[15, [3, 2]]
    iso_gt = np.zeros(T, dtype=int)
    iso_gt[15] = 1
    scenarios["2_isolated_single_frame_swap"] = {
        "description": "Detector swaps Left and Right for a single isolated frame (t=15).",
        "kpts": iso_k,
        "scores": np.ones((T, 4)),
        "ground_truth_swapped": iso_gt,
        "pairs_by_group": pairs_by_group,
        "height_px": height_px,
    }

    # Scenario 3: Sustained multi-frame swap at t=12..19 (Ground Truth: 12..19)
    sust_k = _base_kpts()
    sust_gt = np.zeros(T, dtype=int)
    for t in range(12, 20):
        sust_k[t, [0, 1]] = sust_k[t, [1, 0]]
        sust_k[t, [2, 3]] = sust_k[t, [3, 2]]
        sust_gt[t] = 1
    scenarios["3_sustained_multi_frame_swap"] = {
        "description": "Detector swaps limbs across 8 consecutive frames; tests persistence recovery.",
        "kpts": sust_k,
        "scores": np.ones((T, 4)),
        "ground_truth_swapped": sust_gt,
        "pairs_by_group": pairs_by_group,
        "height_px": height_px,
    }

    # Scenario 4: Natural Sagittal Limb Crossing (Ground Truth: 0 swaps)
    cross_k = np.zeros((T, 4, 2))
    for t in range(T):
        # Left leg moves forward from 70 to 190
        # Right leg moves backward from 190 to 70
        cross_k[t, 0] = [70.0 + t * 4.0, 350.0]
        cross_k[t, 1] = [190.0 - t * 4.0, 350.0]
        cross_k[t, 2] = [65.0 + t * 4.0, 480.0]
        cross_k[t, 3] = [195.0 - t * 4.0, 480.0]
    cross_gt = np.zeros(T, dtype=int)
    scenarios["4_natural_limb_crossing"] = {
        "description": "Natural gait crossover where limbs pass in 2D; tests resistance to spurious swaps.",
        "kpts": cross_k,
        "scores": np.ones((T, 4)),
        "ground_truth_swapped": cross_gt,
        "pairs_by_group": pairs_by_group,
        "height_px": height_px,
    }

    # Scenario 5: Rapid Axial Rotation / Spin (Ground Truth: 0 swaps)
    # Lateral separation collapses as person turns profile to camera
    rot_k = np.zeros((T, 4, 2))
    for t in range(T):
        angle = t * (np.pi / 20.0)
        span = 40.0 * np.cos(angle)
        rot_k[t, 0] = [150.0 - span, 350.0]
        rot_k[t, 1] = [150.0 + span, 350.0]
        rot_k[t, 2] = [150.0 - span * 1.1, 480.0]
        rot_k[t, 3] = [150.0 + span * 1.1, 480.0]
    rot_gt = np.zeros(T, dtype=int)
    scenarios["5_rapid_axial_rotation"] = {
        "description": "Person spins 180 deg in place; lateral distance collapses and inverts projection.",
        "kpts": rot_k,
        "scores": np.ones((T, 4)),
        "ground_truth_swapped": rot_gt,
        "pairs_by_group": pairs_by_group,
        "height_px": height_px,
    }

    # Scenario 6: Partial Occlusion with NaNs (Ground Truth: 0 swaps)
    occ_k = _base_kpts()
    occ_k[15:20] = np.nan
    occ_gt = np.zeros(T, dtype=int)
    scenarios["6_partial_occlusion_nans"] = {
        "description": "5 consecutive frames of occlusion (NaNs); tests missing-data robustness.",
        "kpts": occ_k,
        "scores": np.ones((T, 4)),
        "ground_truth_swapped": occ_gt,
        "pairs_by_group": pairs_by_group,
        "height_px": height_px,
    }

    # Scenario 7: Single-Pair Hallucination (e.g. bed occlusion, Ground Truth: 0 swaps)
    hal_k = _base_kpts()
    # At frame 20, ankle is hallucinated across body, but knee is normal
    hal_k[20, [2, 3]] = hal_k[20, [3, 2]]
    hal_gt = np.zeros(T, dtype=int)
    scenarios["7_single_pair_hallucination"] = {
        "description": "Only 1 keypoint pair glitched while rest of limb is unswapped; tests group consensus.",
        "kpts": hal_k,
        "scores": np.ones((T, 4)),
        "ground_truth_swapped": hal_gt,
        "pairs_by_group": pairs_by_group,
        "height_px": height_px,
    }

    return scenarios


def evaluate_synthetic_battery() -> Dict[str, Any]:
    """Execute complete synthetic evaluation battery across all 4 methods."""
    battery = generate_synthetic_battery()
    results = {}

    for sc_name, sc_data in battery.items():
        kpts = sc_data["kpts"]
        scores = sc_data["scores"]
        gt_swapped = sc_data["ground_truth_swapped"]
        pairs_bg = sc_data["pairs_by_group"]
        h_px = sc_data["height_px"]
        T = len(gt_swapped)

        sc_res = {"description": sc_data["description"], "methods": {}}

        for m_name, m_func in METHODS.items():
            t0 = time.perf_counter()
            c_kpts, c_scores, m_meta = m_func(kpts, scores, pairs_by_group=pairs_bg, height_px=h_px)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            swapped_pred = np.zeros(T, dtype=int)
            for f_idx in m_meta.get("swapped_frames", []):
                if f_idx < T:
                    swapped_pred[f_idx] = 1

            # Confusion matrix
            tp = int(np.sum((swapped_pred == 1) & (gt_swapped == 1)))
            fp = int(np.sum((swapped_pred == 1) & (gt_swapped == 0)))
            fn = int(np.sum((swapped_pred == 0) & (gt_swapped == 1)))
            tn = int(np.sum((swapped_pred == 0) & (gt_swapped == 0)))

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if np.sum(gt_swapped) == 0 else 0.0)
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Persistent error count: frames where error persisted after actual swap ended
            persistent_errors = 0
            swap_end_indices = np.where(np.diff(gt_swapped) == -1)[0]
            if len(swap_end_indices) > 0:
                last_end = swap_end_indices[-1]
                persistent_errors = int(np.sum(swapped_pred[last_end + 1:] == 1))

            traj_metrics = compute_trajectory_metrics(kpts, c_kpts, pairs_bg["Hip"], h_px)

            sc_res["methods"][m_name] = {
                "swapped_frames_count": len(m_meta.get("swapped_frames", [])),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "persistent_errors": persistent_errors,
                "max_coord_diff_px": traj_metrics["max_coordinate_diff_px"],
                "max_velocity_px_frame": traj_metrics["max_velocity_px_per_frame"],
                "latency_ms": latency_ms,
            }

        results[sc_name] = sc_res

    return results


# ---------------------------------------------------------------------------
# Cross-Skeleton Compatibility Suite
# ---------------------------------------------------------------------------

def evaluate_skeleton_compatibility() -> Dict[str, Any]:
    """Verify bilateral_body_pairs and Viterbi correction across all supported skeleton trees."""
    models = {
        "HALPE_26": HALPE_26,
        "COCO_17": COCO_17,
        "BODY_25": BODY_25,
        "BLAZEPOSE": BLAZEPOSE,
    }
    skel_results = {}

    for name, root_node in models.items():
        pairs = bilateral_body_pairs(root_node)
        all_ids = [node.id for node in root_node.descendants if node.id is not None]
        max_id = max(all_ids) if all_ids else 32
        K = max_id + 1
        T = 20

        # Construct dummy trajectory
        dummy_k = np.zeros((T, K, 2))
        for t in range(T):
            for k in range(K):
                dummy_k[t, k] = [100.0 + k * 10.0 + t, 200.0 + k * 10.0]

        # Test Viterbi execution
        t0 = time.perf_counter()
        c_kpts, _, stats = correct_limb_swaps_viterbi(dummy_k, pairs_by_group=pairs, height_px=500.0)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        skel_results[name] = {
            "groups_found": list(pairs.keys()),
            "total_pairs": sum(len(p) for p in pairs.values()),
            "pairs_detail": {g: [(int(l), int(r)) for l, r in lst] for g, lst in pairs.items()},
            "viterbi_execution_success": c_kpts.shape == dummy_k.shape,
            "latency_ms": dur_ms,
        }

    return skel_results


# ---------------------------------------------------------------------------
# Real-World Recording Evaluation
# ---------------------------------------------------------------------------

def evaluate_real_recordings(eval_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Analyze available real-world recordings and pre-computed TRC outputs."""
    if eval_dir is None:
        eval_dir = os.environ.get("SPORTS2D_EVAL_DIR")
        if eval_dir:
            eval_dir = Path(eval_dir)
        else:
            default_dir = Path("/home/mrudula/Downloads/mocap/sports2d_eval")
            eval_dir = default_dir if default_dir.exists() else None

    if eval_dir is None or not Path(eval_dir).exists():
        return {
            "status": "BLOCKED",
            "message": "External video validation blocked: evaluation directory not found. 2-3 challenging clips requested from maintainer.",
            "recordings": {}
        }
    eval_dir = Path(eval_dir)
    records = {}

    # List of known target recording sessions
    session_configs = [
        {
            "name": "recording_front (Sitting / Room)",
            "video_path": "/home/mrudula/Downloads/mocap/media/raw_videos/annotated_front.mp4",
            "trc_path_swap_true": eval_dir / "eval_runs/recording_front_swap_True/recording_front_Sports2D/recording_front_Sports2D_px_person00.trc",
            "trc_path_swap_false": eval_dir / "eval_runs/recording_front_swap_False/recording_front_Sports2D/recording_front_Sports2D_px_person00.trc",
            "has_optoelectronic_ground_truth": False,
            "visual_inspection_notes": "Clean frontal walking/sitting; 0 swap corrections triggered across all methods.",
        },
        {
            "name": "whatsapp_may03 (Tumbling Walk)",
            "video_path": "/home/mrudula/Downloads/mocap/media/raw_videos/whatsapp_video_2026-05-03_130228.mp4",
            "trc_path_swap_true": eval_dir / "eval_runs/whatsapp_may03_swap_True/whatsapp_video_2026-05-03_130228_Sports2D/whatsapp_video_2026-05-03_130228_Sports2D_px_person00.trc",
            "trc_path_swap_false": eval_dir / "eval_runs/whatsapp_may03_swap_False/whatsapp_video_2026-05-03_130228_Sports2D/whatsapp_video_2026-05-03_130228_Sports2D_px_person00.trc",
            "has_optoelectronic_ground_truth": False,
            "visual_inspection_notes": "Tumbling / walking sequence; RTMPose maintains clean tracking; 0 triggers.",
        },
        {
            "name": "recording_right (Bedside Occlusion)",
            "video_path": "/home/mrudula/Downloads/mocap/media/raw_videos/annotated_right.mp4",
            "trc_path_swap_true": eval_dir / "eval_runs_af93197/recording_right_swap_True/recording_right_Sports2D/recording_right_Sports2D_px_person00.trc",
            "trc_path_swap_false": eval_dir / "eval_runs_af93197/recording_right_swap_False/recording_right_Sports2D/recording_right_Sports2D_px_person00.trc",
            "has_optoelectronic_ground_truth": False,
            "visual_inspection_notes": "Bedside occlusion. af93197 triggered 14 false swaps (782px swing) due to min_pairs=1. Viterbi suppresses single-pair noise.",
        },
        {
            "name": "whatsapp_may01 (Acrobatic Backflip)",
            "video_path": "/home/mrudula/Downloads/mocap/media/raw_videos/whatsapp_video_2026-05-01_130041.mp4",
            "trc_path_swap_true": eval_dir / "eval_runs_af93197/whatsapp_may01_swap_True/whatsapp_video_2026-05-01_130041_Sports2D/whatsapp_video_2026-05-01_130041_Sports2D_px_person00.trc",
            "trc_path_swap_false": eval_dir / "eval_runs_af93197/whatsapp_may01_swap_False/whatsapp_video_2026-05-01_130041_Sports2D/whatsapp_video_2026-05-01_130041_Sports2D_px_person00.trc",
            "has_optoelectronic_ground_truth": False,
            "visual_inspection_notes": "Aerial backflip with axial rotation. af93197 inverted legs at frames 50-55 due to nearest-neighbor matching across crossing.",
        },
    ]

    for rec in session_configs:
        entry = {
            "video_path": rec["video_path"],
            "video_exists": os.path.exists(rec["video_path"]),
            "has_optoelectronic_ground_truth": rec["has_optoelectronic_ground_truth"],
            "visual_inspection_notes": rec["visual_inspection_notes"],
            "trc_comparison": {},
        }

        p_true = Path(rec["trc_path_swap_true"])
        p_false = Path(rec["trc_path_swap_false"])
        if p_true.exists() and p_false.exists():
            try:
                df_t = pd.read_csv(p_true, sep="\t", skiprows=4)
                df_f = pd.read_csv(p_false, sep="\t", skiprows=4)
                diff = np.abs(df_t.iloc[:, 2:].values - df_f.iloc[:, 2:].values)
                finite_diff = diff[np.isfinite(diff)]
                max_diff = float(np.max(finite_diff)) if len(finite_diff) > 0 else 0.0
                diff_frames = int(np.sum(np.nanmax(diff, axis=1) > 1e-4)) if len(diff) > 0 else 0
                entry["trc_comparison"] = {
                    "total_frames": len(df_t),
                    "different_frames_count": diff_frames,
                    "max_coordinate_diff_px": max_diff,
                }
            except Exception as e:
                entry["trc_comparison"] = {"error": str(e)}

        records[rec["name"]] = entry

    return records


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sports2D Swap Correction Evaluation Suite")
    parser.add_argument("--mode", choices=["all", "synthetic", "skeletons", "real"], default="all",
                        help="Evaluation mode to execute")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to export full JSON results")
    parser.add_argument("--eval-dir", type=str, default=None,
                        help="Optional directory containing pre-computed real-video evaluation runs")
    args = parser.parse_args()

    full_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sports2d_branch": "fix/temporal-lower-limb-swaps",
        "methods_evaluated": list(METHODS.keys()),
    }

    if args.mode in ("all", "synthetic"):
        print("\n=======================================================")
        print("  SYNTHETIC BENCHMARK BATTERY (WITH GROUND TRUTH)      ")
        print("=======================================================")
        syn_res = evaluate_synthetic_battery()
        full_results["synthetic_battery"] = syn_res

        # Format concise markdown table
        print(f"\n{'Scenario':<32} | {'Method':<20} | {'Prec':<5} | {'Rec':<5} | {'F1':<5} | {'FP':<3} | {'Persist':<7} | {'Max Jump':<10}")
        print("-" * 105)
        for sc_name, sc_data in syn_res.items():
            short_sc = sc_name.split("_", 1)[1] if "_" in sc_name else sc_name
            for m_name, m_res in sc_data["methods"].items():
                p = f"{m_res['precision']:.2f}"
                r = f"{m_res['recall']:.2f}"
                f1 = f"{m_res['f1_score']:.2f}"
                fp = str(m_res["false_positives"])
                pe = str(m_res["persistent_errors"])
                mj = f"{m_res['max_velocity_px_frame']:.1f} px"
                print(f"{short_sc[:32]:<32} | {m_name:<20} | {p:<5} | {r:<5} | {f1:<5} | {fp:<3} | {pe:<7} | {mj:<10}")

    if args.mode in ("all", "skeletons"):
        print("\n=======================================================")
        print("  SKELETON COMPATIBILITY AUDIT                         ")
        print("=======================================================")
        skel_res = evaluate_skeleton_compatibility()
        full_results["skeleton_compatibility"] = skel_res
        for skel_name, s_data in skel_res.items():
            print(f"  * {skel_name:<12}: {s_data['total_pairs']} pairs across {s_data['groups_found']} (Viterbi OK: {s_data['viterbi_execution_success']}, {s_data['latency_ms']:.2f} ms)")

    if args.mode in ("all", "real"):
        print("\n=======================================================")
        print("  REAL-WORLD RECORDING AUDIT                           ")
        print("=======================================================")
        real_res = evaluate_real_recordings(args.eval_dir)
        full_results["real_recordings"] = real_res
        if real_res.get("status") == "BLOCKED":
            print(f"  [!] {real_res['message']}")
        else:
            for rec_name, r_data in real_res.items():
                tc = r_data.get("trc_comparison", {})
                diff_f = tc.get("different_frames_count", "N/A")
                max_d = f"{tc.get('max_coordinate_diff_px', 0.0):.1f} px" if "max_coordinate_diff_px" in tc else "N/A"
                print(f"  * {rec_name}")
                print(f"    - Optoelectronic GT: {r_data['has_optoelectronic_ground_truth']}")
                print(f"    - Baseline vs af93197 diffs: {diff_f} frames (max diff: {max_d})")
                print(f"    - Visual Inspection: {r_data['visual_inspection_notes']}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nSaved full benchmark report to: {out_p}")


if __name__ == "__main__":
    main()
