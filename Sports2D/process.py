#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##############################################################
    ## Compute pose and angles from video or webcam input       ##
    ##############################################################
    
    Detects 2D joint centers from a video or a webcam with RTMLib.
    Computes selected joint and segment angles. 
    Optionally saves processed image files and video file.
    Optionally saves processed poses as a TRC file, and angles as a MOT file (OpenSim compatible).

    This scripts:
    - loads skeleton information
    - reads stream from a video or a webcam
    - sets up the RTMLib pose tracker from RTMlib with specified parameters
    - detects poses within the selected time range
    - tracks people so that their IDs are consistent across frames
    - retrieves the keypoints with high enough confidence, and only keeps the persons with enough high-confidence keypoints
    - computes joint and segment angles, and flips those on the left/right side them if the respective foot is pointing to the left
    - draws bounding boxes around each person with their IDs
    - draws joint and segment angles on the body, and writes the values either near the joint/segment, or on the upper-left of the image with a progress bar
    - draws the skeleton and the keypoints, with a green to red color scale to account for their confidence
    - optionally show processed images, saves them, or saves them as a video
    - interpolates missing pose and angle sequences if gaps are not too large
    - filters them with the selected filter and parameters
    - optionally plots pose and angle data before and after processing for comparison
    - optionally saves poses for each person as a trc file, and angles as a mot file
        
    /!\ Warning /!\
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal or frontal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
        
    INPUTS:
    - a video or a webcam
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one trc file of joint coordinates per detected person
    - one mot file of joint angles per detected person
    - image files, video
    - a logs.txt file 
'''    


## INIT
from pathlib import Path
import sys
import logging
import json
import ast
from functools import partial
from datetime import datetime
import itertools as it
from tqdm import tqdm
from anytree import RenderTree, PreOrderIter

import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Custom

from Sports2D.Utilities import filter
from Sports2D.Utilities.common import *
from Sports2D.Utilities.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def setup_webcam(webcam_id, save_vid, vid_output_path, input_size):
    '''
    Set up webcam capture with OpenCV.

    INPUTS:
    - webcam_id: int. The ID of the webcam to capture from
    - input_size: tuple. The size of the input frame (width, height)

    OUTPUTS:
    - cap: cv2.VideoCapture. The webcam capture object
    - out_vid: cv2.VideoWriter. The video writer object
    - cam_width: int. The actual width of the webcam frame
    - cam_height: int. The actual height of the webcam frame
    - fps: int. The frame rate of the webcam
    '''

    #, cv2.CAP_DSHOW launches faster but only works for windows and esc key does not work
    cap = cv2.VideoCapture(webcam_id) 
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open webcam #{webcam_id}. Make sure that your webcam is available and has the right 'webcam_id' (check in your Config.toml file).")

    # set width and height to closest available for the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_size[1])
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    if cam_width != input_size[0] or cam_height != input_size[1]:
        logging.warning(f"Warning: Your webcam does not support {input_size[0]}x{input_size[1]} resolution. Resolution set to the closest supported one: {cam_width}x{cam_height}.")
    
    out_vid = None
    if save_vid:
        # fourcc MJPG produces very large files but is faster. If it is too slow, consider using it and then converting the video to h264
        # try:
        #     fourcc = cv2.VideoWriter_fourcc(*'avc1') # =h264. better compression and quality but may fail on some systems
        #     out_vid = cv2.VideoWriter(vid_output_path, fourcc, fps, (cam_width, cam_height))
        #     if not out_vid.isOpened():
        #         raise ValueError("Failed to open video writer with 'avc1' (h264)")
        # except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(vid_output_path, fourcc, fps, (cam_width, cam_height))
            # logging.info("Failed to open video writer with 'avc1' (h264). Using 'mp4v' instead.")

    return cap, out_vid, cam_width, cam_height, fps


def setup_video(video_file_path, save_vid, vid_output_path):
    '''
    Set up video capture with OpenCV.

    INPUTS:
    - video_file_path: Path. The path to the video file
    - save_vid: bool. Whether to save the video output
    - vid_output_path: Path. The path to save the video output

    OUTPUTS:
    - cap: cv2.VideoCapture. The video capture object
    - out_vid: cv2.VideoWriter. The video writer object
    - cam_width: int. The width of the video
    - cam_height: int. The height of the video
    - fps: int. The frame rate of the video
    '''
    
    if video_file_path.name == video_file_path.stem:
        raise ValueError("Please set video_input to 'webcam' or to a video file (with extension) in Config.toml")
    try:
        cap = cv2.VideoCapture(str(video_file_path.absolute()))
        if not cap.isOpened():
            raise
    except:
        raise NameError(f"{video_file_path} is not a video. Check video_dir and video_input in your Config.toml file.")
    
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_vid = None
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    if save_vid:
        # try:
        #     fourcc = cv2.VideoWriter_fourcc(*'avc1') # =h264. better compression and quality but may fail on some systems
        #     out_vid = cv2.VideoWriter(str(vid_output_path.absolute()), fourcc, fps, (cam_width, cam_height))
        #     if not out_vid.isOpened():
        #         raise ValueError("Failed to open video writer with 'avc1' (h264)")
        # except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(str(vid_output_path.absolute()), fourcc, fps, (cam_width, cam_height))
            # logging.info("Failed to open video writer with 'avc1' (h264). Using 'mp4v' instead.")
    
    return cap, out_vid, cam_width, cam_height, fps


def setup_backend_device(backend='auto', device='auto'):
    '''
    Set up the backend and device for the pose tracker based on the availability of hardware acceleration.
    TensorRT is not supported by RTMLib yet: https://github.com/Tau-J/rtmlib/issues/12

    If device and backend are not specified, they are automatically set up in the following order of priority:
    1. GPU with CUDA and ONNXRuntime backend (if CUDAExecutionProvider is available)
    2. GPU with ROCm and ONNXRuntime backend (if ROCMExecutionProvider is available, for AMD GPUs)
    3. GPU with MPS or CoreML and ONNXRuntime backend (for macOS systems)
    4. CPU with OpenVINO backend (default fallback)
    '''

    if device!='auto' and backend!='auto':
        device = device.lower()
        backend = backend.lower()

    if device=='auto' or backend=='auto':
        if device=='auto' and backend!='auto' or device!='auto' and backend=='auto':
            logging.warning(f"If you set device or backend to 'auto', you must set the other to 'auto' as well. Both device and backend will be determined automatically.")

        try:
            import torch
            import onnxruntime as ort
            if torch.cuda.is_available() == True and 'CUDAExecutionProvider' in ort.get_available_providers():
                device = 'cuda'
                backend = 'onnxruntime'
                logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
            elif torch.cuda.is_available() == True and 'ROCMExecutionProvider' in ort.get_available_providers():
                device = 'rocm'
                backend = 'onnxruntime'
                logging.info(f"\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
            else:
                raise 
        except:
            try:
                import onnxruntime as ort
                if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                    device = 'mps'
                    backend = 'onnxruntime'
                    logging.info(f"\nValid MPS installation found: using ONNXRuntime backend with GPU.")
                else:
                    raise
            except:
                device = 'cpu'
                backend = 'openvino'
                logging.info(f"\nNo valid CUDA installation found: using OpenVINO backend with CPU.")
        
    return backend, device


def setup_pose_tracker(ModelClass, det_frequency, mode, tracking, backend, device):
    '''
    Set up the RTMLib pose tracker with the appropriate model and backend.
    If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino

    INPUTS:
    - ModelClass: class. The RTMlib model class to use for pose detection (Body, BodyWithFeet, Wholebody)
    - det_frequency: int. The frequency of pose detection (every N frames)
    - mode: str. The mode of the pose tracker ('lightweight', 'balanced', 'performance')
    - tracking: bool. Whether to track persons across frames with RTMlib tracker
    - backend: str. The backend to use for pose detection (onnxruntime, openvino, opencv)
    - device: str. The device to use for pose detection (cpu, cuda, rocm, mps)

    OUTPUTS:
    - pose_tracker: PoseTracker. The initialized pose tracker object    
    '''

    backend, device = setup_backend_device(backend=backend, device=device)

    # Initialize the pose tracker with Halpe26 model
    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=False)
        
    return pose_tracker


def flip_left_right_direction(person_X, L_R_direction_idx, keypoints_names, keypoints_ids):
    '''
    Flip the points to the right or left for more consistent angle calculation 
    depending on which direction the person is facing

    INPUTS:
    - person_X: list of x coordinates
    - L_R_direction_idx: list of indices of the left toe, left heel, right toe, right heel
    - keypoints_names: list of keypoint names (see skeletons.py)
    - keypoints_ids: list of keypoint ids (see skeletons.py)

    OUTPUTS:
    - person_X_flipped: list of x coordinates after flipping
    '''

    Ltoe_idx, LHeel_idx, Rtoe_idx, RHeel_idx = L_R_direction_idx
    right_orientation = person_X[Rtoe_idx] - person_X[RHeel_idx]
    left_orientation = person_X[Ltoe_idx] - person_X[LHeel_idx]
    global_orientation = right_orientation + left_orientation
    
    person_X_flipped = person_X.copy()
    if left_orientation < 0:
        for k in keypoints_names:
            if k.startswith('L'):
                keypt_idx = keypoints_ids[keypoints_names.index(k)]
                person_X_flipped[keypt_idx] = person_X_flipped[keypt_idx] * -1
    if right_orientation < 0:
        for k in keypoints_names:
            if k.startswith('R'):
                keypt_idx = keypoints_ids[keypoints_names.index(k)]
                person_X_flipped[keypt_idx] = person_X_flipped[keypt_idx] * -1
    if global_orientation < 0:
        for k in keypoints_names:
            if not k.startswith('L') and not k.startswith('R'):
                keypt_idx = keypoints_ids[keypoints_names.index(k)]
                person_X_flipped[keypt_idx] = person_X_flipped[keypt_idx] * -1
    
    return person_X_flipped


def compute_angle(ang_name, person_X_flipped, person_Y, angle_dict, keypoints_ids, keypoints_names):
    '''
    Compute the angles from the 2D coordinates of the keypoints.
    Takes into account which side the participant is facing.
    Takes into account the offset and scaling of the angle from angle_dict.
    Requires points_to_angles function (see common.py)

    INPUTS:
    - ang_name: str. The name of the angle to compute
    - person_X_flipped: list of x coordinates after flipping if needed
    - person_Y: list of y coordinates
    - angle_dict: dict. The dictionary of angles to compute (name: [keypoints, type, offset, scaling])
    - keypoints_ids: list of keypoint ids (see skeletons.py)
    - keypoints_names: list of keypoint names (see skeletons.py)

    OUTPUTS:
    - ang: float. The computed angle
    '''

    ang_params = angle_dict.get(ang_name)
    if ang_params is not None:
        try:
            if ang_name in ['pelvis', 'trunk', 'shoulders']:
                angle_coords = [[np.abs(person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]]), person_Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0]]
            else:
                angle_coords = [[person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]], person_Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0]]
            ang = fixed_angles(angle_coords, ang_name)
        except:
            ang = np.nan    
    else:
        ang = np.nan
    
    return ang


def min_with_single_indices(L, T):
    '''
    Let L be a list (size s) with T associated tuple indices (size s).
    Select the smallest values of L, considering that 
    the next smallest value cannot have the same numbers 
    in the associated tuple as any of the previous ones.

    Example:
    L = [  20,   27,  51,    33,   43,   23,   37,   24,   4,   68,   84,    3  ]
    T = list(it.product(range(2),range(3)))
      = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]

    - 1st smallest value: 3 with tuple (2,3), index 11
    - 2nd smallest value when excluding indices (2,.) and (.,3), i.e. [(0,0),(0,1),(0,2),X,(1,0),(1,1),(1,2),X,X,X,X,X]:
    20 with tuple (0,0), index 0
    - 3rd smallest value when excluding [X,X,X,X,X,(1,1),(1,2),X,X,X,X,X]:
    23 with tuple (1,1), index 5
    
    INPUTS:
    - L: list (size s)
    - T: T associated tuple indices (size s)

    OUTPUTS: 
    - minL: list of smallest values of L, considering constraints on tuple indices
    - argminL: list of indices of smallest values of L (indices of best combinations)
    - T_minL: list of tuples associated with smallest values of L
    '''

    minL = [np.nanmin(L)]
    argminL = [np.nanargmin(L)]
    T_minL = [T[argminL[0]]]
    
    mask_tokeep = np.array([True for t in T])
    i=0
    while mask_tokeep.any()==True:
        mask_tokeep = mask_tokeep & np.array([t[0]!=T_minL[i][0] and t[1]!=T_minL[i][1] for t in T])
        if mask_tokeep.any()==True:
            indicesL_tokeep = np.where(mask_tokeep)[0]
            minL += [np.nanmin(np.array(L)[indicesL_tokeep]) if not np.isnan(np.array(L)[indicesL_tokeep]).all() else np.nan]
            argminL += [indicesL_tokeep[np.nanargmin(np.array(L)[indicesL_tokeep])] if not np.isnan(minL[-1]) else indicesL_tokeep[0]]
            T_minL += (T[argminL[i+1]],)
            i+=1
    
    return np.array(minL), np.array(argminL), np.array(T_minL)
    
    
def pad_shape(arr, target_len, fill_value=np.nan):
    '''
    Pads an array to the target length with specified fill values
    
    INPUTS:
    - arr: Input array to be padded.
    - target_len: The target length of the first dimension after padding.
    - fill_value: The value to use for padding (default: np.nan).
    
    OUTPUTS:
    - Padded array with shape (target_len, ...) matching the input dimensions.
    '''

    if len(arr) < target_len:
        pad_shape = (target_len - len(arr),) + arr.shape[1:]
        padding = np.full(pad_shape, fill_value)
        return np.concatenate((arr, padding))
    
    return arr


def sort_people_sports2d(keyptpre, keypt, scores=None):
    '''
    Associate persons across frames (Sports2D method)
    Persons' indices are sometimes swapped when changing frame
    A person is associated to another in the next frame when they are at a small distance
    
    N.B.: Requires min_with_single_indices and euclidian_distance function (see common.py)

    INPUTS:
    - keyptpre: (K, L, M) array of 2D coordinates for K persons in the previous frame, L keypoints, M 2D coordinates
    - keypt: idem keyptpre, for current frame
    - score: (K, L) array of confidence scores for K persons, L keypoints (optional) 
    
    OUTPUTS:
    - sorted_prev_keypoints: array with reordered persons with values of previous frame if current is empty
    - sorted_keypoints: array with reordered persons --> if scores is not None
    - sorted_scores: array with reordered scores     --> if scores is not None
    - associated_tuples: list of tuples with correspondences between persons across frames --> if scores is None (for Pose2Sim.triangulation())
    '''
    
    # Generate possible person correspondences across frames
    max_len = max(len(keyptpre), len(keypt))
    keyptpre = pad_shape(keyptpre, max_len, fill_value=np.nan)
    keypt = pad_shape(keypt, max_len, fill_value=np.nan)
    if scores is not None:
        scores = pad_shape(scores, max_len, fill_value=np.nan)
    
    # Compute distance between persons from one frame to another
    personsIDs_comb = sorted(list(it.product(range(len(keyptpre)), range(len(keypt)))))
    frame_by_frame_dist = [euclidean_distance(keyptpre[comb[0]],keypt[comb[1]]) for comb in personsIDs_comb]
    frame_by_frame_dist = np.mean(frame_by_frame_dist, axis=1)
    
    # Sort correspondences by distance
    _, _, associated_tuples = min_with_single_indices(frame_by_frame_dist, personsIDs_comb)
    
    # Associate points to same index across frames, nan if no correspondence
    sorted_keypoints = []
    for i in range(len(keyptpre)):
        id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
        if len(id_in_old) > 0:      sorted_keypoints += [keypt[id_in_old[0]]]
        else:                       sorted_keypoints += [keypt[i]]
    sorted_keypoints = np.array(sorted_keypoints)

    if scores is not None:
        sorted_scores = []
        for i in range(len(keyptpre)):
            id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
            if len(id_in_old) > 0:  sorted_scores += [scores[id_in_old[0]]]
            else:                   sorted_scores += [scores[i]]
        sorted_scores = np.array(sorted_scores)

    # Keep track of previous values even when missing for more than one frame
    sorted_prev_keypoints = np.where(np.isnan(sorted_keypoints) & ~np.isnan(keyptpre), keyptpre, sorted_keypoints)
    
    if scores is not None:
        return sorted_prev_keypoints, sorted_keypoints, sorted_scores
    else: # For Pose2Sim.triangulation()
        return sorted_keypoints, associated_tuples


def sort_people_rtmlib(pose_tracker, keypoints, scores):
    '''
    Associate persons across frames (RTMLib method)

    INPUTS:
    - pose_tracker: PoseTracker. The initialized RTMLib pose tracker object
    - keypoints: array of shape K, L, M with K the number of detected persons,
    L the number of detected keypoints, M their 2D coordinates
    - scores: array of shape K, L with K the number of detected persons,
    L the confidence of detected keypoints

    OUTPUT:
    - sorted_keypoints: array with reordered persons
    - sorted_scores: array with reordered scores
    '''
    
    try:
        desired_size = max(pose_tracker.track_ids_last_frame)+1
        sorted_keypoints = np.full((desired_size, keypoints.shape[1], 2), np.nan)
        sorted_keypoints[pose_tracker.track_ids_last_frame] = keypoints[:len(pose_tracker.track_ids_last_frame), :, :]
        sorted_scores = np.full((desired_size, scores.shape[1]), np.nan)
        sorted_scores[pose_tracker.track_ids_last_frame] = scores[:len(pose_tracker.track_ids_last_frame), :]
    except:
        sorted_keypoints, sorted_scores = keypoints, scores

    return sorted_keypoints, sorted_scores


def draw_dotted_line(img, start, direction, length, color=(0, 255, 0), gap=7, dot_length=3, thickness=thickness):
    '''
    Draw a dotted line with on a cv2 image

    INPUTS:
    - img: opencv image
    - start: np.array. The starting point of the line
    - direction: np.array. The direction of the line
    - length: int. The length of the line
    - color: tuple. The color of the line
    - gap: int. The distance between each dot
    - dot_length: int. The length of each dot
    - thickness: int. The thickness of the line

    OUTPUT:
    - img: image with the dotted line
    '''

    for i in range(0, length, gap):
        line_start = start + direction * i
        line_end = line_start + direction * dot_length
        cv2.line(img, tuple(line_start.astype(int)), tuple(line_end.astype(int)), color, thickness)


def draw_bounding_box(img, X, Y, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], fontSize=0.3, thickness=1):
    '''
    Draw bounding boxes and person ID around list of lists of X and Y coordinates.
    Bounding boxes have a different color for each person.
    
    INPUTS:
    - img: opencv image
    - X: list of list of x coordinates
    - Y: list of list of y coordinates
    - colors: list of colors to cycle through
    
    OUTPUT:
    - img: image with rectangles and person IDs
    '''
   
    color_cycle = it.cycle(colors)

    for i,(x,y) in enumerate(zip(X,Y)):
        color = next(color_cycle)
        if not np.isnan(x).all():
            x_min, y_min = np.nanmin(x).astype(int), np.nanmin(y).astype(int)
            x_max, y_max = np.nanmax(x).astype(int), np.nanmax(y).astype(int)
            if x_min < 0: x_min = 0
            if x_max > img.shape[1]: x_max = img.shape[1]
            if y_min < 0: y_min = 0
            if y_max > img.shape[0]: y_max = img.shape[0]

            # Draw rectangles
            cv2.rectangle(img, (x_min-25, y_min-25), (x_max+25, y_max+25), color, thickness) 
        
            # Write person ID
            cv2.putText(img, str(i), (x_min-30, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, fontSize+1, color, 2, cv2.LINE_AA) 
    
    return img


def draw_skel(img, X, Y, model, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
    '''
    Draws keypoints and skeleton for each person.
    Skeletons have a different color for each person.

    INPUTS:
    - img: opencv image
    - X: list of list of x coordinates
    - Y: list of list of y coordinates
    - model: skeleton model (from skeletons.py)
    - colors: list of colors to cycle through
    
    OUTPUT:
    - img: image with keypoints and skeleton
    '''
    
    # Get (unique) pairs between which to draw a line
    node_pairs = []
    for data_i in PreOrderIter(model.root, filter_=lambda node: node.is_leaf):
        node_branches = [node_i.id for node_i in data_i.path]
        node_pairs += [[node_branches[i],node_branches[i+1]] for i in range(len(node_branches)-1)]
    node_pairs = [list(x) for x in set(tuple(x) for x in node_pairs)]
    
    # Draw lines
    color_cycle = it.cycle(colors)
    for (x,y) in zip(X,Y):
        c = next(color_cycle)
        if not np.isnan(x).all():
            [cv2.line(img,
                (int(x[n[0]]), int(y[n[0]])), (int(x[n[1]]), int(y[n[1]])), c, thickness)
                for n in node_pairs
                if not None in n and not (np.isnan(x[n[0]]) or np.isnan(y[n[0]]) or np.isnan(x[n[1]]) or np.isnan(y[n[1]]))] # IF NOT NONE

    return img


def draw_keypts(img, X, Y, scores, cmap_str='RdYlGn'):
    '''
    Draws keypoints and skeleton for each person.
    Keypoints' colors depend on their score.

    INPUTS:
    - img: opencv image
    - X: list of list of x coordinates
    - Y: list of list of y coordinates
    - scores: list of list of scores
    - cmap_str: colormap name
    
    OUTPUT:
    - img: image with keypoints and skeleton
    '''
    
    scores = np.where(np.isnan(scores), 0, scores)
    # scores = (scores - 0.4) / (1-0.4) # to get a red color for scores lower than 0.4
    scores = np.where(scores>0.99, 0.99, scores)
    scores = np.where(scores<0, 0, scores)
    
    cmap = plt.get_cmap(cmap_str)
    for (x,y,s) in zip(X,Y,scores):
        c_k = np.array(cmap(s))[:,:-1]*255
        [cv2.circle(img, (int(x[i]), int(y[i])), thickness+4, c_k[i][::-1], -1)
            for i in range(len(x))
            if not (np.isnan(x[i]) or np.isnan(y[i]))]

    return img


def draw_angles(img, valid_X, valid_Y, valid_angles, valid_X_flipped, keypoints_ids, keypoints_names, angle_names, display_angle_values_on= ['body', 'list'], colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], fontSize=0.3, thickness=1):
    '''
    Draw angles on the image.
    Angles are displayed as a list on the image and/or on the body.

    INPUTS:
    - img: opencv image
    - valid_X: list of list of x coordinates
    - valid_Y: list of list of y coordinates
    - valid_angles: list of list of angles
    - valid_X_flipped: list of list of x coordinates after flipping if needed
    - keypoints_ids: list of keypoint ids (see skeletons.py)
    - keypoints_names: list of keypoint names (see skeletons.py)
    - angle_names: list of angle names
    - display_angle_values_on: list of str. 'body' and/or 'list'
    - colors: list of colors to cycle through

    OUTPUT:
    - img: image with angles
    '''

    color_cycle = it.cycle(colors)
    for person_id, (X,Y,angles, X_flipped) in enumerate(zip(valid_X, valid_Y, valid_angles, valid_X_flipped)):
        c = next(color_cycle)
        if not np.isnan(X).all():
            # person label
            if 'list' in display_angle_values_on:
                person_label_position = (int(10 + fontSize*150/0.3*person_id), int(fontSize*50))
                cv2.putText(img, f'person {person_id}', person_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (255,255,255), thickness+1, cv2.LINE_AA)
                cv2.putText(img, f'person {person_id}', person_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, c, thickness, cv2.LINE_AA)
            
            # angle lines, names and values
            ang_label_line = 1
            for k, ang in enumerate(angles):
                if not np.isnan(ang):
                    ang_name = angle_names[k]
                    ang_params = angle_dict.get(ang_name)
                    if ang_params is not None:
                        kpts = ang_params[0]
                        if not any(item not in keypoints_names+['Neck', 'Hip'] for item in kpts):
                            ang_coords = np.array([[X[keypoints_ids[keypoints_names.index(kpt)]], Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0] if kpt in keypoints_names])
                            X_flipped = np.append(X_flipped, X[len(X_flipped):])
                            X_flipped_coords = [X_flipped[keypoints_ids[keypoints_names.index(kpt)]] for kpt in ang_params[0] if kpt in keypoints_names]
                            flip = -1 if any(x_flipped < 0 for x_flipped in X_flipped_coords) else 1
                            flip = 1 if ang_name in ['pelvis', 'trunk', 'shoulders'] else flip
                            right_angle = True if ang_params[2]==90 else False
                            
                            # Draw angle
                            if len(ang_coords) == 2: # segment angle
                                app_point, vec = draw_segment_angle(img, ang_coords, flip)
                            else: # joint angle
                                app_point, vec1, vec2 = draw_joint_angle(img, ang_coords, flip, right_angle)
        
                            # Write angle on body
                            if 'body' in display_angle_values_on:
                                if len(ang_coords) == 2: # segment angle
                                    write_angle_on_body(img, ang, app_point, vec, np.array([1,0]), dist=20, color=(255,255,255), fontSize=fontSize, thickness=thickness)
                                else: # joint angle
                                    write_angle_on_body(img, ang, app_point, vec1, vec2, dist=40, color=(0,255,0), fontSize=fontSize, thickness=thickness)

                            # Write angle as a list on image with progress bar
                            if 'list' in display_angle_values_on:
                                if len(ang_coords) == 2: # segment angle
                                    ang_label_line = write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color = (255,255,255), fontSize=fontSize, thickness=thickness)
                                else:
                                    ang_label_line = write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color = (0,255,0), fontSize=fontSize, thickness=thickness)

    return img


def draw_segment_angle(img, ang_coords, flip):
    '''
    Draw a segment angle on the image.

    INPUTS:
    - img: opencv image
    - ang_coords: np.array. The 2D coordinates of the keypoints
    - flip: int. Whether the angle should be flipped

    OUTPUT:
    - app_point: np.array. The point where the angle is displayed
    - unit_segment_direction: np.array. The unit vector of the segment direction
    - img: image with the angle
    '''
    
    if not np.any(np.isnan(ang_coords)):
        app_point = np.int32(np.mean(ang_coords, axis=0))

        # segment line
        segment_direction = np.int32(ang_coords[0]) - np.int32(ang_coords[1])
        if (segment_direction==0).all():
            return app_point, np.array([0,0])
        unit_segment_direction = segment_direction/np.linalg.norm(segment_direction)
        cv2.line(img, app_point, np.int32(app_point+unit_segment_direction*20), (255,255,255), thickness)

        # horizontal line
        cv2.line(img, app_point, (np.int32(app_point[0])+flip*20, np.int32(app_point[1])), (255,255,255), thickness)

        return app_point, unit_segment_direction


def draw_joint_angle(img, ang_coords, flip, right_angle):
    '''
    Draw a joint angle on the image.

    INPUTS:
    - img: opencv image
    - ang_coords: np.array. The 2D coordinates of the keypoints
    - flip: int. Whether the angle should be flipped
    - right_angle: bool. Whether the angle should be offset by 90 degrees

    OUTPUT:
    - app_point: np.array. The point where the angle is displayed
    - unit_segment_direction: np.array. The unit vector of the segment direction
    - unit_parentsegment_direction: np.array. The unit vector of the parent segment direction
    - img: image with the angle
    '''
    
    if not np.any(np.isnan(ang_coords)):
        app_point = np.int32(ang_coords[1])
        
        segment_direction = np.int32(ang_coords[0] - ang_coords[1])
        parentsegment_direction = np.int32(ang_coords[-2] - ang_coords[-1])
        if (segment_direction==0).all() or (parentsegment_direction==0).all():
            return app_point, np.array([0,0]), np.array([0,0])
        
        if right_angle:
            segment_direction = np.array([-flip*segment_direction[1], flip*segment_direction[0]])
            segment_direction, parentsegment_direction = parentsegment_direction, segment_direction

        # segment line
        unit_segment_direction = segment_direction/np.linalg.norm(segment_direction)
        cv2.line(img, app_point, np.int32(app_point+unit_segment_direction*40), (0,255,0), thickness)
        
        # parent segment dotted line
        unit_parentsegment_direction = parentsegment_direction/np.linalg.norm(parentsegment_direction)
        draw_dotted_line(img, app_point, unit_parentsegment_direction, 40, color=(0, 255, 0), gap=7, dot_length=3, thickness=thickness)

        # arc
        start_angle = np.degrees(np.arctan2(unit_segment_direction[1], unit_segment_direction[0]))
        end_angle = np.degrees(np.arctan2(unit_parentsegment_direction[1], unit_parentsegment_direction[0]))
        if abs(end_angle - start_angle) > 180:
            if end_angle > start_angle: start_angle += 360
            else: end_angle += 360
        cv2.ellipse(img, app_point, (20, 20), 0, start_angle, end_angle, (0, 255, 0), thickness)

        return app_point, unit_segment_direction, unit_parentsegment_direction


def write_angle_on_body(img, ang, app_point, vec1, vec2, dist=40, color=(255,255,255), fontSize=0.3, thickness=1):
    '''
    Write the angle on the body.

    INPUTS:
    - img: opencv image
    - ang: float. The angle value to display
    - app_point: np.array. The point where the angle is displayed
    - vec1: np.array. The unit vector of the first segment
    - vec2: np.array. The unit vector of the second segment
    - dist: int. The distance from the origin where to write the angle
    - color: tuple. The color of the angle

    OUTPUT:
    - img: image with the angle
    '''

    vec_sum = vec1 + vec2
    if (vec_sum == 0.).all():
        return
    unit_vec_sum = vec_sum/np.linalg.norm(vec_sum)
    text_position = np.int32(app_point + unit_vec_sum*dist)
    cv2.putText(img, f'{ang:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, f'{ang:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)


def write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color=(255,255,255), fontSize=0.3, thickness=1):
    '''
    Write the angle as a list on the image with a progress bar.

    INPUTS:
    - img: opencv image
    - ang: float. The value of the angle to display
    - ang_name: str. The name of the angle
    - person_label_position: tuple. The position of the person label
    - ang_label_line: int. The line where to write the angle
    - color: tuple. The color of the angle

    OUTPUT:
    - ang_label_line: int. The updated line where to write the next angle
    - img: image with the angle
    '''
    
    if not np.any(np.isnan(ang)):
        # angle names and values
        ang_label_position = (person_label_position[0], person_label_position[1]+int((ang_label_line)*40*fontSize))
        ang_value_position = (ang_label_position[0]+int(250*fontSize), ang_label_position[1])
        cv2.putText(img, f'{ang_name}:', ang_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, f'{ang_name}:', ang_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)
        cv2.putText(img, f'{ang:.1f}', ang_value_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, f'{ang:.1f}', ang_value_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)
        
        # progress bar
        ang_percent = int(ang*50/180)
        y_crop, y_crop_end = ang_value_position[1] - int(35*fontSize), ang_value_position[1]
        x_crop, x_crop_end = ang_label_position[0]+int(300*fontSize), ang_label_position[0]+int(300*fontSize)+int(ang_percent*fontSize/0.3)
        if ang_percent < 0:
            x_crop, x_crop_end = x_crop_end, x_crop
        img_crop = img[y_crop:y_crop_end, x_crop:x_crop_end]
        if img_crop.size>0:
            white_rect = np.ones(img_crop.shape, dtype=np.uint8)*255
            alpha_rect = cv2.addWeighted(img_crop, 0.6, white_rect, 0.4, 1.0)
            img[y_crop:y_crop_end, x_crop:x_crop_end] = alpha_rect

        ang_label_line += 1
    
    return ang_label_line


def load_pose_file(Q_coords):
    '''
    Load 2D keypoints from a dataframe of XYZ coordinates

    INPUTS:
    - Q_coords: pd.DataFrame. The dataframe of XYZ coordinates

    OUTPUTS:
    - keypoints_all: np.array. The keypoints in the format (Nframes, 1, Nmarkers, 2)
    - scores_all: np.array. The scores in the format (Nframes, 1, Nmarkers)
    '''

    Z_cols = [3*i+2 for i in range(len(Q_coords.columns)//3)]
    Q_coords_xy = Q_coords.drop(Q_coords.columns[Z_cols], axis=1)
    kpt_number = len(Q_coords_xy.columns)//2

    # shape (Nframes, 2*Nmarkers) --> (Nframes, 1, Nmarkers, 2)
    keypoints_all = np.array(Q_coords_xy).reshape(len(Q_coords_xy), 1, kpt_number, 2)
    # shape (Nframes, 1, Nmarkers)
    scores_all = np.ones((len(Q_coords), 1, kpt_number))

    return keypoints_all, scores_all


def trc_data_from_XYZtime(X, Y, Z, time):
    '''
    Constructs trc_data from 3D coordinates and time.

    INPUTS:
    - X: pd.DataFrame. The x coordinates of the keypoints
    - Y: pd.DataFrame. The y coordinates of the keypoints
    - Z: pd.DataFrame. The z coordinates of the keypoints
    - time: pd.Series. The time series for the coordinates

    OUTPUT:
    - trc_data: pd.DataFrame. Dataframe of trc data
    '''

    trc_data = pd.concat([pd.concat([X.iloc[:,kpt], Y.iloc[:,kpt], Z.iloc[:,kpt]], axis=1) for kpt in range(len(X.columns))], axis=1)
    trc_data.insert(0, 't', time)

    return trc_data


def make_trc_with_trc_data(trc_data, trc_path, fps=30):
    '''
    Write a TRC file from a DataFrame of time and coordinates

    INPUTS:
    - trc_data: pd.DataFrame. The time and coordinates of the keypoints. 
                    The column names must be 't', 'kpt1', 'kpt1', 'kpt1', 'kpt2', 'kpt2', 'kpt2', ...
    - trc_path: Path. The path to the TRC file to save
    - fps: float. The framerate of the video

    OUTPUT:
    - None
    '''

    DataRate = CameraRate = OrigDataRate = fps
    NumFrames = len(trc_data)
    NumMarkers = (len(trc_data.columns)-1)//3
    keypoint_names = trc_data.columns[1::3]
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + str(trc_path), 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, 0, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoint_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoint_names))])]

    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        trc_data.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')


def make_mot_with_angles(angles, time, mot_path):
    '''
    Write a mot file from angles and time, compatible with OpenSim.

    INPUTS:
    - angles: pd.DataFrame. The angles to write
    - time: pd.Series. The time series for the angles
    - mot_path: str. The path where to save the mot file

    OUTPUT:
    - angles: pd.DataFrame. The data that has been written to the MOT file
    '''

    # Header
    nRows, nColumns = angles.shape
    angle_names = angles.columns
    header_mot = ['Coordinates', 
                  'version=1', 
                  f'{nRows=}',
                  f'{nColumns=}',
                  'inDegrees=yes',
                  '',
                  'Units are S.I. units (second, meters, Newtons, ...)',
                  "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).",
                  '',
                  'endheader',
                  'time\t' + '\t'.join(angle_names)]
                  
    # Write file
    angles.insert(0,'time',time)
    with open(mot_path, 'w') as mot_o:
        [mot_o.write(line+'\n') for line in header_mot]
        angles.to_csv(mot_o, sep='\t', index=False, header=None, lineterminator='\n')

    return angles


def pose_plots(trc_data_unfiltered, trc_data, person_id):
    '''
    Displays trc filtered and unfiltered data for comparison
    /!\ Often crashes on the third window...

    INPUTS:
    - trc_data_unfiltered: pd.DataFrame. The unfiltered trc data
    - trc_data: pd.DataFrame. The filtered trc data

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''
    
    mpl.use('qt5agg')
    mpl.rc('figure', max_open_warning=0)

    keypoints_names = trc_data.columns[1::3]
    
    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person'+ str(person_id) + ' coordinates') # Main title

    for id, keypoint in enumerate(keypoints_names):
        f = plt.figure()
        f.canvas.manager.window.setWindowTitle(keypoint + ' Plot')

        axX = plt.subplot(211)
        plt.plot(trc_data_unfiltered.iloc[:,0], trc_data_unfiltered.iloc[:,id*3+1], label='unfiltered')
        plt.plot(trc_data.iloc[:,0], trc_data.iloc[:,id*3+1], label='filtered')
        plt.setp(axX.get_xticklabels(), visible=False)
        axX.set_ylabel(keypoint+' X')
        plt.legend()

        axY = plt.subplot(212)
        plt.plot(trc_data_unfiltered.iloc[:,0], trc_data_unfiltered.iloc[:,id*3+2], label='unfiltered')
        plt.plot(trc_data.iloc[:,0], trc_data.iloc[:,id*3+2], label='filtered')
        axY.set_xlabel('Time (seconds)')
        axY.set_ylabel(keypoint+' Y')

        pw.addPlot(keypoint, f)
    
    pw.show()


def angle_plots(angle_data_unfiltered, angle_data, person_id):
    '''
    Displays angle filtered and unfiltered data for comparison
    /!\ Often crashes on the third window...

    INPUTS:
    - angle_data_unfiltered: pd.DataFrame. The unfiltered angle data
    - angle_data: pd.DataFrame. The filtered angle data

    OUTPUT:
    - matplotlib window with tabbed figures for each angle
    '''

    mpl.use('qt5agg')
    mpl.rc('figure', max_open_warning=0)

    angles_names = angle_data.columns[1:]

    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person'+ str(person_id) + ' angles') # Main title

    for id, angle in enumerate(angles_names):
        f = plt.figure()
        
        ax = plt.subplot(111)
        plt.plot(angle_data_unfiltered.iloc[:,0], angle_data_unfiltered.iloc[:,id+1], label='unfiltered')
        plt.plot(angle_data.iloc[:,0], angle_data.iloc[:,id+1], label='filtered')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(angle+' (°)')
        plt.legend()

        pw.addPlot(angle, f)

    pw.show()


def get_personID_with_highest_scores(all_frames_scores):
    '''
    Get the person ID with the highest scores

    INPUTS:
    - all_frames_scores: array of scores for all frames, all persons, all keypoints

    OUTPUT:
    - person_id: int. The person ID with the highest scores
    '''

    # Get the person with the highest scores over all frames and all keypoints
    person_id = np.argmax(np.nansum(np.nansum(all_frames_scores, axis=0), axis=1))

    return person_id


def compute_floor_line(trc_data, keypoint_names = ['LBigToe', 'RBigToe'], toe_speed_below = 1.0, tot_speed_above=2.0):
    '''
    Compute the floor line equation and angle 
    from the feet keypoints when they have zero speed.

    N.B.: Y coordinates point downwards

    INPUTS:
    - trc_data: pd.DataFrame. The trc data
    - keypoint_names: list of str. The names of the keypoints to use
    - toe_speed_below: float. The speed threshold (px/frame) below which the keypoints are considered as not moving

    OUTPUT:
    - angle: float. The angle of the floor line in radians
    - xy_origin: list. The origin of the floor line
    '''

    # Remove frames where the person is mostly not moving (outlier)
    av_speeds = np.nanmean([np.insert(np.linalg.norm(trc_data[kpt].diff(), axis=1)[1:],0,0) for kpt in trc_data.columns.unique()[1:]], axis=0)
    trc_data = trc_data[av_speeds>tot_speed_above]

    # Retrieve zero-speed coordinates for the foot
    low_speeds_X, low_speeds_Y = [], []
    for kpt in keypoint_names:
        speeds = np.linalg.norm(trc_data[kpt].diff(), axis=1)
        
        low_speed_frames = trc_data[speeds<toe_speed_below].index
        low_speeds_coords = trc_data[kpt].loc[low_speed_frames]
        low_speeds_coords = low_speeds_coords[low_speeds_coords!=0]

        low_speeds_X += low_speeds_coords.iloc[:,0].tolist()
        low_speeds_Y += low_speeds_coords.iloc[:,1].tolist()

    # Fit a line to the zero-speed coordinates
    floor_line = np.polyfit(low_speeds_X, low_speeds_Y, 1) # (slope, intercept)
    xy_origin = [0, floor_line[1]]

    # Compute the angle of the floor line in degrees
    angle = -np.arctan(floor_line[0])

    return angle, xy_origin


def convert_px_to_meters(Q_coords_kpt, person_height_m, height_px, cx, cy, floor_angle):
    '''
    Convert pixel coordinates to meters.

    INPUTS:
    - Q_coords_kpt: pd.DataFrame. The xyz coordinates of a keypoint in pixels, with z filled with zeros
    - person_height_m: float. The height of the person in meters
    - height_px: float. The height of the person in pixels
    - cx, cy: float. The origin of the image in pixels
    - floor_angle: float. The angle of the floor in radians

    OUTPUT:
    - Q_coords_kpt_m: pd.DataFrame. The XYZ coordinates of a keypoint in meters
    '''

    u = Q_coords_kpt.iloc[:,0]
    v = Q_coords_kpt.iloc[:,1]

    X = person_height_m / height_px * ((u-cx) + (v-cy)*np.sin(floor_angle))
    Y = - person_height_m / height_px * np.cos(floor_angle) * (v-cy - np.tan(floor_angle)*(u-cx))

    Q_coords_kpt_m = pd.DataFrame(np.array([X, Y, np.zeros_like(X)]).T, columns=Q_coords_kpt.columns)

    return Q_coords_kpt_m


def process_fun(config_dict, video_file, time_range, frame_rate, result_dir):
    '''
    Detect 2D joint centers from a video or a webcam with RTMLib.
    Compute selected joint and segment angles. 
    Optionally save processed image files and video file.
    Optionally save processed poses as a TRC file, and angles as a MOT file (OpenSim compatible).

    This scripts:
    - loads skeleton information
    - reads stream from a video or a webcam
    - sets up the RTMLib pose tracker from RTMlib with specified parameters
    - detects poses within the selected time range
    - tracks people so that their IDs are consistent across frames
    - retrieves the keypoints with high enough confidence, and only keeps the persons with enough high-confidence keypoints
    - computes joint and segment angles, and flips those on the left/right side them if the respective foot is pointing to the left
    - draws bounding boxes around each person with their IDs
    - draws joint and segment angles on the body, and writes the values either near the joint/segment, or on the upper-left of the image with a progress bar
    - draws the skeleton and the keypoints, with a green to red color scale to account for their confidence
    - optionally show processed images, saves them, or saves them as a video
    - interpolates missing pose and angle sequences if gaps are not too large
    - filters them with the selected filter and parameters
    - optionally plots pose and angle data before and after processing for comparison
    - optionally saves poses for each person as a trc file, and angles as a mot file
        
    /!\ Warning /!\d
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal or frontal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
        
    INPUTS:
    - a video or a webcam
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one trc file of joint coordinates per detected person
    - one mot file of joint angles per detected person
    - image files, video
    - a logs.txt file 
    '''
    
    # Base parameters
    video_dir = Path(config_dict.get('project').get('video_dir'))
    person_height_m = config_dict.get('project').get('person_height')
    # Pose from file
    load_trc = config_dict.get('project').get('load_trc')
    if load_trc == '': load_trc = None
    else: load_trc = Path(load_trc).resolve()
    compare = config_dict.get('project').get('compare')
    # Webcam settings
    webcam_id =  config_dict.get('project').get('webcam_id')
    input_size = config_dict.get('project').get('input_size')

    # Process settings
    multiperson = config_dict.get('process').get('multiperson')
    show_realtime_results = config_dict.get('process').get('show_realtime_results')
    save_vid = config_dict.get('process').get('save_vid')
    save_img = config_dict.get('process').get('save_img')
    save_pose = config_dict.get('process').get('save_pose')
    calculate_angles = config_dict.get('process').get('calculate_angles')
    save_angles = config_dict.get('process').get('save_angles')

    # Pose_advanced settings
    slowmo_factor = config_dict.get('pose').get('slowmo_factor')
    pose_model = config_dict.get('pose').get('pose_model')
    mode = config_dict.get('pose').get('mode')
    det_frequency = config_dict.get('pose').get('det_frequency')
    tracking_mode = config_dict.get('pose').get('tracking_mode')
    backend = config_dict.get('pose').get('backend')
    device = config_dict.get('pose').get('device')
    
    # Pixel to meters conversion
    to_meters = config_dict.get('px_to_meters_conversion').get('to_meters')
    save_calib = config_dict.get('px_to_meters_conversion').get('save_calib')
    # Calibration from file
    calib_file = config_dict.get('px_to_meters_conversion').get('calib_file')
    if calib_file == '': calib_file = None
    else: calib_file = Path(calib_file).resolve()
    # Calibration from person height
    calib_on_person_id = int(config_dict.get('px_to_meters_conversion').get('calib_on_person_id'))
    floor_angle = config_dict.get('px_to_meters_conversion').get('floor_angle') # 'auto' or float
    floor_angle = np.radians(float(floor_angle)) if floor_angle != 'auto' else floor_angle
    xy_origin = config_dict.get('px_to_meters_conversion').get('xy_origin') # ['auto'] or [x, y]    
    xy_origin = [float(o) for o in xy_origin] if xy_origin != ['auto'] else 'auto'

    fastest_frames_to_remove_percent = config_dict.get('px_to_meters_conversion').get('fastest_frames_to_remove_percent')
    large_hip_knee_angles = config_dict.get('px_to_meters_conversion').get('large_hip_knee_angles')
    trimmed_extrema_percent = config_dict.get('px_to_meters_conversion').get('trimmed_extrema_percent')
    close_to_zero_speed_px = config_dict.get('px_to_meters_conversion').get('close_to_zero_speed_px')

    keypoint_likelihood_threshold = config_dict.get('pose').get('keypoint_likelihood_threshold')
    average_likelihood_threshold = config_dict.get('pose').get('average_likelihood_threshold')
    keypoint_number_threshold = config_dict.get('pose').get('keypoint_number_threshold')

    # Angles advanced settings
    joint_angle_names = config_dict.get('angles').get('joint_angles')
    segment_angle_names = config_dict.get('angles').get('segment_angles')
    angle_names = joint_angle_names + segment_angle_names
    angle_names = [angle_name.lower() for angle_name in angle_names]
    display_angle_values_on = config_dict.get('angles').get('display_angle_values_on')
    fontSize = config_dict.get('angles').get('fontSize')
    thickness = 1 if fontSize < 0.8 else 2
    flip_left_right = config_dict.get('angles').get('flip_left_right')
    correct_segment_angles_with_floor_angle = config_dict.get('angles').get('correct_segment_angles_with_floor_angle')

    # Post-processing settings
    interpolate = config_dict.get('post-processing').get('interpolate')    
    interp_gap_smaller_than = config_dict.get('post-processing').get('interp_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('post-processing').get('fill_large_gaps_with')

    do_filter = config_dict.get('post-processing').get('filter')
    show_plots = config_dict.get('post-processing').get('show_graphs')
    filter_type = config_dict.get('post-processing').get('filter_type')
    butterworth_filter_order = config_dict.get('post-processing').get('butterworth').get('order')
    butterworth_filter_cutoff = config_dict.get('post-processing').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel = config_dict.get('post-processing').get('gaussian').get('sigma_kernel')
    loess_filter_kernel = config_dict.get('post-processing').get('loess').get('nb_values_used')
    median_filter_kernel = config_dict.get('post-processing').get('median').get('kernel_size')
    filter_options = [do_filter, filter_type,
                           butterworth_filter_order, butterworth_filter_cutoff, frame_rate,
                           gaussian_filter_kernel, loess_filter_kernel, median_filter_kernel]

    # Inverse kinematics settings
    do_ik = config_dict.get('inverse-kinematics').get('do_ik')
    osim_setup_path = config_dict.get('inverse-kinematics').get('osim_setup_path')
    person_orientations = config_dict.get('inverse-kinematics').get('person_orientation')

    # Create output directories
    if video_file == "webcam":
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f'webcam_{current_date}'
    else:
        video_file_path = video_dir / video_file
        video_file_stem = video_file.stem
        output_dir_name = f'{video_file_stem}_Sports2D'    
    output_dir = result_dir / output_dir_name
    img_output_dir = output_dir / f'{output_dir_name}_img'
    vid_output_path = output_dir / f'{output_dir_name}.mp4'
    pose_output_path = output_dir / f'{output_dir_name}_px.trc'
    pose_output_path_m = output_dir / f'{output_dir_name}_m.trc'
    angles_output_path = output_dir / f'{output_dir_name}_angles.mot'
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_img:
        img_output_dir.mkdir(parents=True, exist_ok=True)


    # Set up video capture
    if video_file == "webcam":
        cap, out_vid, cam_width, cam_height, fps = setup_webcam(webcam_id, save_vid, vid_output_path, input_size)
        frame_range = [0,sys.maxsize]
        frame_iterator = range(*frame_range)
        logging.warning('Webcam input: the framerate may vary. If results are filtered, Sports2D will use the average framerate as input.')
    else:
        cap, out_vid, cam_width, cam_height, fps = setup_video(video_file_path, save_vid, vid_output_path)
        fps *= slowmo_factor
        start_time = get_start_time_ffmpeg(video_file_path)
        frame_range = [int((time_range[0]-start_time) * frame_rate), int((time_range[1]-start_time) * frame_rate)] if time_range else [0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_iterator = tqdm(range(*frame_range)) # use a progress bar
    if show_realtime_results:
        cv2.namedWindow(f'{video_file} Sports2D', cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(f'{video_file} Sports2D', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN)

    # Select the appropriate model based on the model_type
    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        model_name = 'HALPE_26'
        ModelClass = BodyWithFeet # 26 keypoints(halpe26)
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY', 'WHOLE_BODY_WRIST'):
        model_name = 'COCO_133'
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_name = 'COCO_17'
        ModelClass = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation.")
    else:
        raise ValueError(f"Invalid model_type: {model_name}. Must be 'HALPE_26', 'COCO_133', or 'COCO_17'. Use another network (MMPose, DeepLabCut, OpenPose, AlphaPose, BlazePose...) and convert the output files if you need another model. See documentation.")
    pose_model_name = pose_model
    pose_model = eval(model_name)

    # Manually select the models if mode is a dictionary rather than 'lightweight', 'balanced', or 'performance'
    if not mode in ['lightweight', 'balanced', 'performance']:
        try:
            try:
                mode = ast.literal_eval(mode)
            except: # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
                mode = mode.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/',':/').replace('":"\\',':\\')
                mode = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', mode) # changes "[640", "640]" to [640,640]
                mode = json.loads(mode)
            det_class = mode.get('det_class')
            det = mode.get('det_model')
            det_input_size = mode.get('det_input_size')
            pose_class = mode.get('pose_class')
            pose = mode.get('pose_model')
            pose_input_size = mode.get('pose_input_size')

            ModelClass = partial(Custom,
                        det_class=det_class, det=det, det_input_size=det_input_size,
                        pose_class=pose_class, pose=pose, pose_input_size=pose_input_size,
                        backend=backend, device=device)
            
        except (json.JSONDecodeError, TypeError):
            logging.warning("\nInvalid mode. Must be 'lightweight', 'balanced', 'performance', or '''{dictionary}''' of parameters within triple quotes. Make sure input_sizes are within square brackets.")
            logging.warning('Using the default "balanced" mode.')
            mode = 'balanced'

    
    # Skip pose estimation or set it up:
    if load_trc:
        if not '_px' in str(load_trc): 
            logging.error(f'\n{load_trc} file needs to be in px, not in meters.')
        logging.info(f'\nUsing a pose file instead of running pose estimation and tracking: {load_trc}.')
        # Load pose file in px
        Q_coords, _, _, keypoints_names, _ = read_trc(load_trc)
        keypoints_ids = [i for i in range(len(keypoints_names))]
        keypoints_all, scores_all = load_pose_file(Q_coords)
        for pre, _, node in RenderTree(model_name):
            if node.name in keypoints_names:
                node.id = keypoints_names.index(node.name)
    
    else:
        # Retrieve keypoint names from model
        keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if node.id!=None]
        keypoints_names = [node.name for _, _, node in RenderTree(pose_model) if node.id!=None]

        tracking_rtmlib = True if (tracking_mode == 'rtmlib' and multiperson) else False
        pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, tracking_rtmlib, backend, device)
        logging.info(f'\nPose tracking set up for "{pose_model_name}" model.')
        logging.info(f'Mode: {mode}.\n')
        logging.info(f'Persons are detected every {det_frequency} frames and tracked inbetween. Multi-person is {"" if multiperson else "not "}selected.')
        logging.info(f"Parameters: {keypoint_likelihood_threshold=}, {average_likelihood_threshold=}, {keypoint_number_threshold=}")

    if flip_left_right:
        try:
            Ltoe_idx = keypoints_ids[keypoints_names.index('LBigToe')]
            LHeel_idx = keypoints_ids[keypoints_names.index('LHeel')]
            Rtoe_idx = keypoints_ids[keypoints_names.index('RBigToe')]
            RHeel_idx = keypoints_ids[keypoints_names.index('RHeel')]
            L_R_direction_idx = [Ltoe_idx, LHeel_idx, Rtoe_idx, RHeel_idx]
        except ValueError:
            logging.warning(f"Missing 'LBigToe', 'LHeel', 'RBigToe', 'RHeel' keypoints. flip_left_right will be set to False")
            flip_left_right = False

    if calculate_angles:
        for ang_name in angle_names:
            ang_params = angle_dict.get(ang_name)
            kpts = ang_params[0]
            if any(item not in keypoints_names+['Neck', 'Hip'] for item in kpts):
                logging.warning(f"Skipping {ang_name} angle computation because at least one of the following keypoints is not provided by the model: {ang_params[0]}.")


    # Process video or webcam feed
    logging.info(f"\nProcessing video stream...")
    # logging.info(f"{'Video, ' if save_vid else ''}{'Images, ' if save_img else ''}{'Pose, ' if save_pose else ''}{'Angles ' if save_angles else ''}{'and ' if save_angles or save_img or save_pose or save_vid else ''}Logs will be saved in {result_dir}.")
    all_frames_X, all_frames_Y, all_frames_scores, all_frames_angles = [], [], [], []
    frame_processing_times = []
    frame_count = 0
    while cap.isOpened():
        # Skip to the starting frame
        if frame_count < frame_range[0] and not load_trc:
            cap.read()
            frame_count += 1
            continue

        for frame_nb in frame_iterator:
            start_time = datetime.now()
            success, frame = cap.read()

            # If frame not grabbed
            if not success:
                logging.warning(f"Failed to grab frame {frame_count}.")
                if save_pose:
                    all_frames_X.append([])
                    all_frames_Y.append([])
                    all_frames_scores.append([])
                if save_angles:
                    all_frames_angles.append([])
                frame_count += 1
                continue
            else:
                cv2.putText(frame, f"Press 'q' to quit", (cam_width-int(400*fontSize), cam_height-20), cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (255,255,255), thickness+1, cv2.LINE_AA)
                cv2.putText(frame, f"Press 'q' to quit", (cam_width-int(400*fontSize), cam_height-20), cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (0,0,255), thickness, cv2.LINE_AA)
                frame_count += 1

            # Retrieve pose or Estimate pose and track people
            if load_trc: 
                if frame_nb >= len(keypoints_all):
                    break
                keypoints = keypoints_all[frame_nb]
                scores = scores_all[frame_nb]
            else: 
                # Detect poses
                keypoints, scores = pose_tracker(frame)
                # Track persons
                if tracking_rtmlib:
                    keypoints, scores = sort_people_rtmlib(pose_tracker, keypoints, scores)
                else:
                    if 'prev_keypoints' not in locals(): prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores)
                
            
            # Process coordinates and compute angles
            valid_X, valid_Y, valid_scores = [], [], []
            valid_X_flipped, valid_angles = [], []
            for person_idx in range(len(keypoints)):
                if load_trc:
                    person_X = keypoints[person_idx][:,0]
                    person_Y = keypoints[person_idx][:,1]
                    person_scores = scores[person_idx]
                else:
                    # Retrieve keypoints and scores for the person, remove low-confidence keypoints
                    person_X, person_Y = np.where(scores[person_idx][:, np.newaxis] < keypoint_likelihood_threshold, np.nan, keypoints[person_idx]).T
                    person_scores = np.where(scores[person_idx] < keypoint_likelihood_threshold, np.nan, scores[person_idx])

                    # Skip person if the fraction of valid detected keypoints is too low
                    enough_good_keypoints = len(person_scores[~np.isnan(person_scores)]) >= len(person_scores) * keypoint_number_threshold
                    scores_of_good_keypoints = person_scores[~np.isnan(person_scores)]
                    average_score_of_remaining_keypoints_is_enough = (np.nanmean(scores_of_good_keypoints) if len(scores_of_good_keypoints)>0 else 0) >= average_likelihood_threshold
                    if not enough_good_keypoints or not average_score_of_remaining_keypoints_is_enough:
                        person_X = np.full_like(person_X, np.nan)
                        person_Y = np.full_like(person_Y, np.nan)
                        person_scores = np.full_like(person_scores, np.nan)


                # Compute angles
                if calculate_angles:
                    # Check whether the person is looking to the left or right
                    if flip_left_right:
                        person_X_flipped = flip_left_right_direction(person_X, L_R_direction_idx, keypoints_names, keypoints_ids)
                    else:
                        person_X_flipped = person_X.copy()
                        
                    # Compute angles
                    person_angles = []
                    # Add Neck and Hip if not provided
                    new_keypoints_names, new_keypoints_ids = keypoints_names.copy(), keypoints_ids.copy()
                    for kpt in ['Neck', 'Hip']:
                        if kpt not in new_keypoints_names:
                            person_X_flipped, person_Y, person_scores = add_neck_hip_coords(kpt, person_X_flipped, person_Y, person_scores, new_keypoints_ids, new_keypoints_names)
                            person_X, _, _ = add_neck_hip_coords(kpt, person_X, person_Y, person_scores, new_keypoints_ids, new_keypoints_names)
                            new_keypoints_names.append(kpt)
                            new_keypoints_ids.append(len(person_X_flipped)-1)

                    for ang_name in angle_names:
                        ang_params = angle_dict.get(ang_name)
                        kpts = ang_params[0]
                        if not any(item not in new_keypoints_names for item in kpts):
                            ang = compute_angle(ang_name, person_X_flipped, person_Y, angle_dict, new_keypoints_ids, new_keypoints_names)
                        else:
                            ang = np.nan
                        person_angles.append(ang)
                    valid_angles.append(person_angles)
                    valid_X_flipped.append(person_X_flipped)
                valid_X.append(person_X)
                valid_Y.append(person_Y)
                valid_scores.append(person_scores)

            # Draw keypoints and skeleton
            if show_realtime_results or save_vid or save_img:
                img = frame.copy()
                img = draw_bounding_box(img, valid_X, valid_Y, colors=colors, fontSize=fontSize, thickness=thickness)
                img = draw_keypts(img, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
                img = draw_skel(img, valid_X, valid_Y, pose_model, colors=colors)
                if calculate_angles:
                    img = draw_angles(img, valid_X, valid_Y, valid_angles, valid_X_flipped, new_keypoints_ids, new_keypoints_names, angle_names, display_angle_values_on=display_angle_values_on, colors=colors, fontSize=fontSize, thickness=thickness)

                if show_realtime_results:
                    cv2.imshow(f'{video_file} Sports2D', img)
                    if (cv2.waitKey(1) & 0xFF) == ord('q') or (cv2.waitKey(1) & 0xFF) == 27:
                        break
                if save_vid:
                    out_vid.write(img)
                if save_img:
                    cv2.imwrite(str((img_output_dir / f'{output_dir_name}_{(frame_count-1):06d}.png')), img)

            all_frames_X.append(np.array(valid_X))
            all_frames_Y.append(np.array(valid_Y))
            all_frames_scores.append(np.array(valid_scores))
            
            if save_angles and calculate_angles:
                all_frames_angles.append(np.array(valid_angles))
            if video_file=='webcam' and save_vid:   # To adjust framerate of output video
                elapsed_time = (datetime.now() - start_time).total_seconds()
                frame_processing_times.append(elapsed_time)


        # End of the video is reached
        cap.release()
        logging.info(f"Video processing completed.")
        if save_vid:
            out_vid.release()
            if video_file == 'webcam':
                actual_framerate = len(frame_processing_times) / sum(frame_processing_times)
                logging.info(f"Rewriting webcam video based on the averate framerate {actual_framerate}.")
                resample_video(vid_output_path, fps, actual_framerate)
                fps = actual_framerate
            logging.info(f"Processed video saved to {vid_output_path.resolve()}.")
        if save_img:
            logging.info(f"Processed images saved to {img_output_dir.resolve()}.")
        if show_realtime_results:
            cv2.destroyAllWindows()
    

    # Post-processing: Interpolate, filter, and save pose and angles
    all_frames_X_homog = make_homogeneous(all_frames_X)
    all_frames_X_homog = all_frames_X_homog[...,keypoints_ids]
    all_frames_Y_homog = make_homogeneous(all_frames_Y)
    all_frames_Y_homog = all_frames_Y_homog[...,keypoints_ids]
    all_frames_Z_homog = pd.DataFrame(np.zeros_like(all_frames_X_homog)[:,0,:], columns=keypoints_names)
    all_frames_scores = make_homogeneous(all_frames_scores)

    frame_range = [0,frame_count] if video_file == 'webcam' else frame_range
    all_frames_time = pd.Series(np.linspace(frame_range[0]/fps, frame_range[1]/fps, frame_count+1), name='time')
    if not multiperson:
        calib_on_person_id = get_personID_with_highest_scores(all_frames_scores)
        detected_persons = [calib_on_person_id]
    else:
        detected_persons = range(all_frames_X_homog.shape[1])

    # Post-processing pose
    if save_pose:
        logging.info('\nPost-processing pose:')

        # Process pose for each person
        trc_data = []
        trc_data_unfiltered = []
        for i in detected_persons:
            pose_path_person = pose_output_path.parent / (pose_output_path.stem + f'_person{i:02d}.trc')
            all_frames_X_person = pd.DataFrame(all_frames_X_homog[:,i,:], columns=keypoints_names)
            all_frames_Y_person = pd.DataFrame(all_frames_Y_homog[:,i,:], columns=keypoints_names)

            # Delete person if less than 4 valid frames
            pose_nan_count = len(np.where(all_frames_X_person.sum(axis=1)==0)[0])
            if frame_count - pose_nan_count <= 4:
                trc_data_i = pd.DataFrame(0, index=all_frames_X_person.index, columns=np.array([[c]*3 for c in all_frames_X_person.columns]).flatten())
                trc_data_i.insert(0, 't', all_frames_time)
                trc_data.append(trc_data_i)
                trc_data_unfiltered_i = trc_data_i.copy()
                trc_data_unfiltered.append(trc_data_unfiltered_i)
                logging.info(f'- Person {i}: Less than 4 valid frames. Deleting person.')

            else:
                # Interpolate
                if not interpolate:
                    logging.info(f'- Person {i}: No interpolation.')
                    all_frames_X_person_interp = all_frames_X_person
                    all_frames_Y_person_interp = all_frames_Y_person
                else:
                    logging.info(f'- Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames. Large gaps filled with {fill_large_gaps_with}.')
                    all_frames_X_person_interp = all_frames_X_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])
                    all_frames_Y_person_interp = all_frames_Y_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])
                    if fill_large_gaps_with.lower() == 'last_value':
                        all_frames_X_person_interp = all_frames_X_person_interp.ffill(axis=0).bfill(axis=0)
                        all_frames_Y_person_interp = all_frames_Y_person_interp.ffill(axis=0).bfill(axis=0)
                    elif fill_large_gaps_with.lower() == 'zeros':
                        all_frames_X_person_interp.replace(np.nan, 0, inplace=True)
                        all_frames_Y_person_interp.replace(np.nan, 0, inplace=True)

                # Filter
                if not filter_options[0]:
                    logging.info(f'No filtering.')
                    all_frames_X_person_filt = all_frames_X_person_interp
                    all_frames_Y_person_filt = all_frames_Y_person_interp
                else:
                    filter_type = filter_options[1]
                    if filter_type == 'butterworth':
                        cutoff = filter_options[3]
                        if video_file == 'webcam':
                            if cutoff / (fps / 2) >= 1:
                                cutoff_old = cutoff
                                cutoff = fps/(2+0.001)
                                args = f'\n{cutoff_old:.1f} Hz cut-off framerate too large for a real-time framerate of {fps:.1f} Hz. Using a cut-off framerate of {cutoff:.1f} Hz instead.'
                                filter_options[3] = cutoff
                        args = f'Butterworth filter, {filter_options[2]}th order, {filter_options[3]} Hz.'
                        filter_options[4] = fps
                    if filter_type == 'gaussian':
                        args = f'Gaussian filter, Sigma kernel {filter_options[5]}.'
                    if filter_type == 'loess':
                        args = f'LOESS filter, window size of {filter_options[6]} frames.'
                    if filter_type == 'median':
                        args = f'Median filter, kernel of {filter_options[7]}.'
                    logging.info(f'Filtering with {args}')
                    all_frames_X_person_filt = all_frames_X_person_interp.apply(filter.filter1d, axis=0, args=filter_options)
                    all_frames_Y_person_filt = all_frames_Y_person_interp.apply(filter.filter1d, axis=0, args=filter_options)

                # Build TRC file
                trc_data_i = trc_data_from_XYZtime(all_frames_X_person_filt, all_frames_Y_person_filt, all_frames_Z_homog, all_frames_time)
                trc_data.append(trc_data_i)
                if not load_trc:
                    make_trc_with_trc_data(trc_data_i, str(pose_path_person), fps=fps)
                    logging.info(f'Pose in pixels saved to {pose_path_person.resolve()}.')

                # Plotting coordinates before and after interpolation and filtering
                trc_data_unfiltered_i = pd.concat([pd.concat([all_frames_X_person.iloc[:,kpt], all_frames_Y_person.iloc[:,kpt], all_frames_Z_homog.iloc[:,kpt]], axis=1) for kpt in range(len(all_frames_X_person.columns))], axis=1)
                trc_data_unfiltered_i.insert(0, 't', all_frames_time)
                trc_data_unfiltered.append(trc_data_unfiltered_i)
                if show_plots and not to_meters:
                    pose_plots(trc_data_unfiltered_i, trc_data_i, i)
            

        # Convert px to meters
        if to_meters:
            logging.info('\nConverting pose to meters:')
            if calib_on_person_id>=len(trc_data):
                logging.warning(f'Person #{calib_on_person_id} not detected in the video. Calibrating on person #0 instead.')
                calib_on_person_id = 0
            if calib_file:
                logging.info(f'Using calibration file to convert coordinates in meters: {calib_file}.')
                calib_params_dict = retrieve_calib_params(calib_file)
                # TODO

            else:
                # Compute calibration parameters
                if not multiperson: 
                    selected_person_id = calib_on_person_id
                    calib_on_person_id = 0
                height_px = compute_height(trc_data[calib_on_person_id].iloc[:,1:], keypoints_names,
                                            fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, close_to_zero_speed=close_to_zero_speed_px, large_hip_knee_angles=large_hip_knee_angles, trimmed_extrema_percent=trimmed_extrema_percent)

                if floor_angle == 'auto' or xy_origin == 'auto':
                    # estimated from the line formed by the toes when they are on the ground (where speed = 0)
                    try:
                        toe_speed_below = 1 # m/s (below which the foot is considered to be stationary)
                        px_per_m = height_px/person_height_m
                        toe_speed_below_px_frame = toe_speed_below * px_per_m / fps
                        try:
                            floor_angle_estim, xy_origin_estim = compute_floor_line(trc_data[calib_on_person_id], keypoint_names=['LBigToe', 'RBigToe'], toe_speed_below=toe_speed_below_px_frame)
                        except: # no feet points
                            floor_angle_estim, xy_origin_estim = compute_floor_line(trc_data[calib_on_person_id], keypoint_names=['LAnkle', 'RAnkle'], toe_speed_below=toe_speed_below_px_frame)
                            xy_origin_estim[0] = xy_origin_estim[0]-0.13
                            logging.warning(f'The RBigToe and LBigToe are missing from your model. Using ankles - 13 cm to compute the floor line.')
                    except:
                        floor_angle_estim = 0
                        xy_origin_estim = cam_width/2, cam_height/2
                        logging.warning(f'Could not estimate the floor angle and xy_origin. Make sure that the full body is visible. Using floor angle = 0° and xy_origin = [{cam_width/2}, {cam_height/2}].')
                if not floor_angle == 'auto':
                    floor_angle_estim = floor_angle
                if xy_origin == 'auto':
                    cx, cy = xy_origin_estim
                else:
                    cx, cy = xy_origin
                logging.info(f'Using height of person #{calib_on_person_id} ({person_height_m}m) to convert coordinates in meters. '
                             f'Floor angle: {np.degrees(floor_angle_estim) if not floor_angle=="auto" else f"auto (estimation: {round(np.degrees(floor_angle_estim),2)}°)"}, '
                             f'xy_origin: {xy_origin if not xy_origin=="auto" else f"auto (estimation: {[round(c) for c in xy_origin_estim]})"}.')

            # Coordinates in m
            for i in range(len(trc_data)):
                if not np.array(trc_data[i].iloc[:,1:] ==0).all():
                    trc_data_m_i = pd.concat([convert_px_to_meters(trc_data[i][kpt_name], person_height_m, height_px, cx, cy, -floor_angle_estim) for kpt_name in keypoints_names], axis=1)
                    trc_data_m_i.insert(0, 't', all_frames_time)
                    trc_data_unfiltered_m_i = pd.concat([convert_px_to_meters(trc_data_unfiltered[i][kpt_name], person_height_m, height_px, cx, cy, -floor_angle_estim) for kpt_name in keypoints_names], axis=1)
                    trc_data_unfiltered_m_i.insert(0, 't', all_frames_time)

                    if to_meters and show_plots:
                        pose_plots(trc_data_unfiltered_m_i, trc_data_m_i, i)
                    
                    # Write to trc file
                    idx_path = selected_person_id if not multiperson and not calib_file else i
                    pose_path_person_m_i = (pose_output_path.parent / (pose_output_path_m.stem + f'_person{idx_path:02d}.trc'))
                    make_trc_with_trc_data(trc_data_m_i, pose_path_person_m_i, fps=fps)
                    logging.info(f'Person {idx_path}: Pose in meters saved to {pose_path_person_m_i.resolve()}.')
                    
                
            







                # # plt.plot(trc_data_m.iloc[:,0], trc_data_m.iloc[:,1])
                # # plt.ylim([0,2])
                # # plt.show()



                # z = 3.0 # distance between the camera and the person. Required in the calibration file but simplified in the equations
                # f = height_px / person_height_m * z


                # # Name
                # N = [video_file]

                # # Size
                # S = [[cam_width, cam_height]]

                # # Distortions
                # D = [[0.0, 0.0, 0.0, 0.0]]
                        
                # # Camera matrix
                # K = [[[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]] # f and Z do not matter in 2D
                
                # # Rot, Trans
                # R = 
                # T = 

                # # Save calibration file

                # # Convert to meters
                # trc_data = 




                




    # Post-processing angles
    if save_angles and calculate_angles:
        logging.info('\nPost-processing angles:')
        all_frames_angles = make_homogeneous(all_frames_angles)
        
        # unwrap angles
        # all_frames_angles = np.unwrap(all_frames_angles, axis=0, period=180) # This give all nan values -> need to mask nans
        for i in range(all_frames_angles.shape[1]):  # for each person
            for j in range(all_frames_angles.shape[2]):  # for each angle
                valid_mask = ~np.isnan(all_frames_angles[:, i, j])
                all_frames_angles[valid_mask, i, j] = np.unwrap(all_frames_angles[valid_mask, i, j], period=180)

        # Process angles for each person
        for i in detected_persons:
            angles_path_person = angles_output_path.parent / (angles_output_path.stem + f'_person{i:02d}.mot')
            all_frames_angles_person = pd.DataFrame(all_frames_angles[:,i,:], columns=angle_names)
            
            # Delete person if less than 4 valid frames
            angle_nan_count = len(np.where(all_frames_angles_person.sum(axis=1)==0)[0])
            if frame_count - angle_nan_count <= 4:
                logging.info(f'- Person {i}: Less than 4 valid frames. Deleting person.')

            else:
                # Interpolate
                if not interpolate:
                    logging.info(f'- Person {i}: No interpolation.')
                    all_frames_angles_person_interp = all_frames_angles_person
                else:
                    logging.info(f'- Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames. Large gaps filled with {fill_large_gaps_with}.')
                    all_frames_angles_person_interp = all_frames_angles_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])
                    if fill_large_gaps_with == 'last_value':
                        all_frames_angles_person_interp = all_frames_angles_person_interp.ffill(axis=0).bfill(axis=0)
                    elif fill_large_gaps_with == 'zeros':
                        all_frames_angles_person_interp.replace(np.nan, 0, inplace=True)
                
                # Filter
                if not filter_options[0]:
                    logging.info(f'No filtering.')
                    all_frames_angles_person_filt = all_frames_angles_person_interp
                else:
                    filter_type = filter_options[1]
                    if filter_type == 'butterworth':
                        cutoff = filter_options[3]
                        if video_file == 'webcam':
                            if cutoff / (fps / 2) >= 1:
                                cutoff_old = cutoff
                                cutoff = fps/(2+0.001)
                                args = f'\n{cutoff_old:.1f} Hz cut-off framerate too large for a real-time framerate of {fps:.1f} Hz. Using a cut-off framerate of {cutoff:.1f} Hz instead.'
                                filter_options[3] = cutoff
                        args = f'Butterworth filter, {filter_options[2]}th order, {filter_options[3]} Hz.'
                        filter_options[4] = fps
                    if filter_type == 'gaussian':
                        args = f'Gaussian filter, Sigma kernel {filter_options[5]}.'
                    if filter_type == 'loess':
                        args = f'LOESS filter, window size of {filter_options[6]} frames.'
                    if filter_type == 'median':
                        args = f'Median filter, kernel of {filter_options[7]}.'
                    logging.info(f'Filtering with {args}')
                    all_frames_angles_person_filt = all_frames_angles_person_interp.apply(filter.filter1d, axis=0, args=filter_options)

                # Remove columns with all nan values
                all_frames_angles_person_filt.dropna(axis=1, how='all', inplace=True)
                all_frames_angles_person = all_frames_angles_person[all_frames_angles_person_filt.columns]

                # Add floor_angle_estim to segment angles
                if correct_segment_angles_with_floor_angle and to_meters: 
                    logging.info(f'Correcting segment angles by removing the {round(np.degrees(floor_angle_estim),2)}° floor angle.')
                    for ang_name in all_frames_angles_person_filt.columns:
                        if 'horizontal' in angle_dict[ang_name][1]:
                            all_frames_angles_person_filt[ang_name] -= np.degrees(floor_angle_estim)

                # Build mot file
                angle_data = make_mot_with_angles(all_frames_angles_person_filt, all_frames_time, str(angles_path_person))
                logging.info(f'Angles saved to {angles_path_person.resolve()}.')

                # Plotting angles before and after interpolation and filtering
                if show_plots:
                    all_frames_angles_person.insert(0, 't', all_frames_time)
                    angle_plots(all_frames_angles_person, angle_data, i) # i = current person
