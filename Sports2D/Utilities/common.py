#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Common classes and functions                 ##
    ##################################################
    
    - A class for displaying several matplotlib figures in tabs.
    - A function for interpolating sequences with missing data. 
    It does not interpolate sequences of more than N contiguous missing data.

'''


## INIT
from importlib.metadata import version
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
import av


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = version("sports2d")
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## CONSTANTS
marker_Z_positions = {'right':
                        {"RHip": 0.105, "RKnee": 0.0886, "RAnkle": 0.0972, "RBigToe":0.0766, "RHeel":0.0883, "RSmallToe": 0.1200, 
                        "RShoulder": 0.2016, "RElbow": 0.1613, "RWrist": 0.120, "RThumb": 0.1625, "RIndex": 0.1735, "RPinky": 0.1740, "REye": 0.0311, 
                        "LHip": -0.105, "LKnee": -0.0886, "LAnkle": -0.0972, "LBigToe": -0.0766, "LHeel": -0.0883, "LSmallToe": -0.1200, 
                        "LShoulder": -0.2016, "LElbow": -0.1613, "LWrist": -0.120, "LThumb": -0.1625, "LIndex": -0.1735, "LPinky": -0.1740, "LEye": -0.0311, 
                        "Hip": 0.0, "Neck": 0.0, "Head":0.0, "Nose": 0.0},
                    'left':
                        {"RHip": -0.105, "RKnee": -0.0886, "RAnkle": -0.0972, "RBigToe": -0.0766, "RHeel": -0.0883, "RSmallToe": -0.1200, 
                        "RShoulder": -0.2016, "RElbow": -0.1613, "RWrist": -0.120, "RThumb": -0.1625, "RIndex": -0.1735, "RPinky": -0.1740, "REye": -0.0311, 
                        "LHip": 0.105, "LKnee": 0.0886, "LAnkle": 0.0972, "LBigToe":0.0766, "LHeel":0.0883, "LSmallToe": 0.1200, 
                        "LShoulder": 0.2016, "LElbow": 0.1613, "LWrist": 0.120, "LThumb": 0.1625, "LIndex": 0.1735, "LPinky": 0.1740, "LEye": 0.0311, 
                        "Hip": 0.0, "Neck": 0.0, "Head":0.0, "Nose": 0.0},
                    'front': # original knee:0.0179
                        {"RHip": 0.0301, "RKnee": 0.129, "RAnkle": 0.0230, "RBigToe": 0.2179, "RHeel": -0.0119, "RSmallToe": 0.1804, 
                        "RShoulder": -0.01275, "RElbow": 0.0702, "RWrist": 0.1076, "RThumb": 0.0106, "RIndex": -0.0004, "RPinky": -0.0009, "REye": 0.0702, 
                        "LHip": 0.0301, "LKnee": 0.129, "LAnkle": 0.0230, "LBigToe": 0.2179, "LHeel": -0.0119, "LSmallToe": 0.1804, 
                        "LShoulder": -0.01275, "LElbow": 0.0702, "LWrist": 0.1076, "LThumb": 0.0106, "LIndex": -0.0004, "LPinky": -0.0009, "LEye": 0.0702, 
                        "Hip": 0.0301, "Neck": 0.0008, "Head": 0.0655, "Nose": 0.1076},
                    'back':
                        {"RHip": -0.0301, "RKnee": -0.129, "RAnkle": -0.0230, "RBigToe": -0.2179, "RHeel": 0.0119, "RSmallToe": -0.1804, 
                        "RShoulder": 0.01275, "RElbow": 0.0702, "RWrist": -1076.0002, "RThumb": -0.0106, "RIndex": 0.0004, "RPinky": 0.0009, "REye": -0.0702, 
                        "LHip": -0.0301, "LKnee": -0.129, "LAnkle": -0.0230, "LBigToe": -0.2179, "LHeel": 0.0119, "LSmallToe": -0.1804, 
                        "LShoulder": 0.01275, "LElbow": 0.0702, "LWrist": -0.1076, "LThumb": -0.0106, "LIndex": 0.0004, "LPinky": 0.0009, "LEye": -0.0702, 
                        "Hip": -0.0301, "Neck": -0.0008, "Head": -0.0655, "Nose": -0.1076},
            }


## FUNCTIONS
def to_dict(d):
    '''
    Convert a defaultdict to a dict.
    '''

    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    return d


def make_homogeneous(list_of_arrays):
    '''
    Make a list of arrays (or a list of lists) homogeneous by padding with nans

    Example: foo = [[array([nan, 656.02643776]), array([nan, nan])],
                    [array([1, 2, 3]), array([1, 2])]]
    becomes foo_updated = array([[[nan, 656.02643776, nan], [nan, nan, nan]],
                                [[1., 2., 3.], [1., 2., nan]]])
    Or foo = [[1, 2, 3], [1, 2], [3, 4, 5]]
    becomes foo_updated = array([[1., 2., 3.], [1., 2., nan], [3., 4., 5.]])

    INPUTS:
    - list_of_arrays: list of arrays or list of lists

    OUTPUT:
    - np.array(list_of_arrays): numpy array of padded arrays
    '''
    
    def get_max_shape(list_of_arrays):
        '''
        Recursively determine the maximum shape of a list of arrays.
        '''
        if isinstance(list_of_arrays[0], list):
            # Maximum length at the current level plus the max shape at the next level
            return [max(len(arr) for arr in list_of_arrays)] + get_max_shape(
                [item for sublist in list_of_arrays for item in sublist])
        else:
            # Determine the maximum shape across all list_of_arrays at this level
            non_empty = [arr for arr in list_of_arrays if arr.size > 0]
            if not non_empty: 
                return [len(list_of_arrays)]
            max_ndim = max(arr.ndim for arr in non_empty)
            return [len(list_of_arrays)] + [
                        max(arr.shape[i] for arr in list_of_arrays if arr.size > 0) 
                        for i in range(max_ndim)]

    def pad_with_nans(list_of_arrays, target_shape):
        '''
        Recursively pad list_of_arrays with nans to match the target shape.
        '''
        if isinstance(list_of_arrays, np.ndarray):        
            # Pad the current array to the target shape
            if list_of_arrays.size == 0:
                return np.full(target_shape, np.nan)
            for dim_index in range(0, len(target_shape)):
                if dim_index == len(list_of_arrays.shape) or dim_index > len(list_of_arrays.shape):
                    list_of_arrays = np.expand_dims(list_of_arrays, 0)
            pad_width = []        
            for dim_index in range(0, len(target_shape)):
                max_dim = target_shape[dim_index]
                curr_dim = list_of_arrays.shape[dim_index]
                pad_width.append((0, max_dim - curr_dim))
            return np.pad(list_of_arrays.astype(float), pad_width, constant_values=np.nan)
        # Recursively pad each array in the list
        return [pad_with_nans(array, target_shape[1:]) for array in list_of_arrays]

    # Pad all missing dimensions of arrays with nans
    list_of_arrays = [np.array(arr, dtype=float) if not isinstance(arr, np.ndarray) else arr for arr in list_of_arrays]
    max_shape = get_max_shape(list_of_arrays)
    list_of_arrays = pad_with_nans(list_of_arrays, max_shape)

    return np.array(list_of_arrays)


def get_start_time_ffmpeg(video_path):
    '''
    Get the start time of a video using av (PyAV).
    '''

    try:
        with av.open(str(video_path)) as container:
            # container.start_time is in AV_TIME_BASE units (microseconds)
            if container.start_time is not None:
                return container.start_time / 1_000_000
            return 0.0
    except Exception as e:
        logging.warning(f"Could not determine video start time. Starting time set to 0.0. Error: {e}")
        return 0.0


def resample_video(vid_output_path, desired_framerate=30, fps=240):
    '''
    Resample video to the desired fps using av (PyAV).
    '''

    vid_output_path = Path(vid_output_path)
    new_vid_path = vid_output_path.parent / Path(vid_output_path.stem + '_2' + vid_output_path.suffix)
    pts_factor = fps / desired_framerate

    with av.open(str(vid_output_path)) as in_container:
        in_stream = in_container.streams.video[0]
        codec_name = in_stream.codec_context.name
        pix_fmt = in_stream.pix_fmt or 'yuv420p'

        with av.open(str(new_vid_path), mode='w') as out_container:
            out_stream = out_container.add_stream(codec_name, rate=desired_framerate)
            out_stream.width = in_stream.width
            out_stream.height = in_stream.height
            out_stream.pix_fmt = pix_fmt

            for frame in in_container.decode(video=0):
                if frame.pts is not None:
                    frame.pts = int(frame.pts * pts_factor)
                frame.dts = None
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)
            for packet in out_stream.encode():
                out_container.mux(packet)

    vid_output_path.unlink()
    new_vid_path.rename(vid_output_path)


def write_calibration(calib_params, toml_path):
    '''
    Write calibration file from calibration parameters
    '''
    
    S, D, N, K, R, T, P = calib_params
    with open(toml_path, 'w+') as cal_f:
        for c in range(len(S)):
            cam_str = f'[{N[c]}]\n'
            name_str = f'name = "{N[c]}"\n'
            size_str = f'size = {S[c]} \n'
            mat_str = f'matrix = {K[c]} \n'
            dist_str = f'distortions = {D[c]} \n' 
            rot_str = f'rotation = {R[c]} \n'
            tran_str = f'translation = {T[c]} \n'
            fish_str = f'fisheye = false\n\n'
            cal_f.write(cam_str + name_str + size_str + mat_str + dist_str + rot_str + tran_str + fish_str)
        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)


def bilateral_body_pairs(skeleton_model):
    '''
    Retrieve matching left/right body keypoint IDs from a Pose2Sim skeleton
    tree, grouped by lower- and upper-body
    Face and hand keypoints are ignored.

    OUTPUT:
    - groups: dict of root keypoint names (keys) with list of child keypoint id pairs (values)
              eg {"Hip": [(13, 14), (15, 16), (17, 20), (18, 21), (19, 22)], # Hip, Knee, Ankle, BigToe, Heel, SmallToe
                  "Shoulder": [(7, 8), (9, 10)]}                               # Shoulder, Elbow, Wrist
    '''

    nodes = (skeleton_model,) + tuple(skeleton_model.descendants)
    nodes_by_name = {node.name: node for node in nodes if node.id is not None}

    groups = defaultdict(list)
    for name, left_node in nodes_by_name.items():
        if not name.startswith('L'):
            continue
        right_node = nodes_by_name.get(f'R{name[1:]}')
        if right_node is None:
            continue

        # Exclude finger and face keypoints
        left_ancestors = {ancestor.name for ancestor in left_node.ancestors}
        right_ancestors = {ancestor.name for ancestor in right_node.ancestors}
        ancestors = left_ancestors | right_ancestors
        if ancestors.intersection({'Head', 'Nose'}):
            continue
        if any(ancestor.endswith('Wrist') for ancestor in ancestors):
            continue

        # Climb while the parent and its mirror are both valid bilateral keypoints and stop at the first non-L/R ancestor
        current = left_node
        while current.parent is not None:
            parent_name = current.parent.name
            if parent_name[:1] == 'L':
                mirror = 'R' + parent_name[1:]
            elif parent_name[:1] == 'R':
                mirror = 'L' + parent_name[1:]
            else:
                break
            if parent_name not in nodes_by_name or mirror not in nodes_by_name:
                break
            current = current.parent
        group_name = (current.name[1:] if current.name[:1] in ('L', 'R') else current.name)

        # Skip most proximal keypoints (Hip, Shoulder) that rarely swap
        if current is left_node:
            continue

        groups[group_name].append((left_node.id, right_node.id))

    return dict(groups)


def correct_limb_swaps(person_keypoints, person_scores, previous_keypoints, height_px, pairs_by_group, 
                       min_improvement=0.05, min_improvement_wrt_height=0.03, min_pairs=1):
    '''
    Correct clear temporal left/right limb assignment swaps.

    A swap is only triggered if swapping all the points of a group (upper or lower body) largely 
    reduces the frame-by-frame displacement (>min_improvement), by an amount that is not negligible
    compared to the person's height (> min_improvement_wrt_height), and with at least min_pairs valid 
    pairs of keypoints.
    Once triggered, the swaps are applied to individual pairs, only if the swapped keypoint is closer
    to the previous frame. 

    INPUTS:
    - person_keypoints: (K, 2) array of current keypoint coordinates
    - person_scores: (K,) array of current confidence scores
    - previous_keypoints: (K, 2) array for the same person in the previous frame
    - height_px: detected person height in pixels
    - pairs_by_group:  dict of root keypoint names (keys) with list of child keypoint id pairs (values)
                       eg {"Hip": [(13, 14), (15, 16), (17, 20), (18, 21), (19, 22)], # Knee, Ankle, BigToe, Heel, SmallToe
                           "Shoulder": [(7, 8), (9, 10)]}                             # Elbow, Wrist
    - min_improvement: minimum swapped/original improvement per group
    - min_improvement_wrt_height: minimum improvement relative to height_px, per group
    - min_pairs: minimum number of valid bilateral pairs required per group

    OUTPUTS:
    - keypoints: corrected keypoint coordinates
    - scores: corrected confidence scores
    - corrected: whether at least one pair was swapped
    '''

    keypoints = np.asarray(person_keypoints).copy()
    scores = np.asarray(person_scores).copy()
    previous_keypoints = np.asarray(previous_keypoints)

    if keypoints.ndim != 2 or previous_keypoints.shape != keypoints.shape:
        return keypoints, scores, False
    if not np.isfinite(height_px) or height_px <= 0:
        return keypoints, scores, False

    # Swap in this frame? Evaluate frame-by-frame displacement within lower and upper body
    corrected = False
    for pairs in pairs_by_group.values():
        pair_data = []
        original_cost, swapped_cost = 0.0, 0.0
        for left_idx, right_idx in pairs:
            if max(left_idx, right_idx) >= len(keypoints):
                continue
            current_left, current_right = keypoints[left_idx], keypoints[right_idx]
            previous_left, previous_right = previous_keypoints[left_idx], previous_keypoints[right_idx]
            if not all(np.isfinite(point).all() for point in (current_left, current_right, previous_left, previous_right)):
                continue
            pair_original_cost = (np.linalg.norm(current_left - previous_left) + np.linalg.norm(current_right - previous_right))
            pair_swapped_cost = (np.linalg.norm(current_left - previous_right) + np.linalg.norm(current_right - previous_left))
            original_cost += pair_original_cost
            swapped_cost += pair_swapped_cost
            pair_data.append((left_idx, right_idx, pair_original_cost, pair_swapped_cost))
        if len(pair_data) < min_pairs or original_cost <= 0:
            continue
        mean_improvement_wrt_height = (original_cost - swapped_cost) / len(pair_data) / height_px
        relative_improvement = (original_cost - swapped_cost) / original_cost
        detected_swap = (relative_improvement >= min_improvement and mean_improvement_wrt_height >= min_improvement_wrt_height)
        if not detected_swap:
            continue

        # If detected_swap, only swap individual pairs whose displacement decreases # by more than 5% ?
        for left_idx, right_idx, pair_original_cost, pair_swapped_cost in pair_data:
            if pair_swapped_cost < pair_original_cost: # > pair_original_cost * .95 ?
                keypoints[[left_idx, right_idx]] = keypoints[[right_idx, left_idx]]
                scores[[left_idx, right_idx]] = scores[[right_idx, left_idx]]
                corrected = True

    return keypoints, scores, corrected

