#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##############################################################
    ## Compute pose and angles from video or webcam input       ##
    ##############################################################
    
    Read video or webcam input
    Detect joint centers with RTMPose
    Attribute them to the same person across frames
    Compute joint and segment angles
    Optionally interpolate missing data, filter them, and display figures
    Save image and video results, save pose as trc files, save angles as csv files
    
    Interpolates sequences of missing data if they are less than N frames long.
    Optionally filters results with Butterworth, gaussian, median, or loess filter.
    Optionally displays figures.

    If BlazePose is used, only one person can be detected.
    No interpolation nor filtering options available. Not plotting available.

    /!\ Warning /!\
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
   
    INPUTS:
    - a video
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one csv file of joint coordinates per detected person
    - optionally json directory, image directory, video
    - a logs.txt file 

'''    


## INIT
from pathlib import Path
import sys
import logging
from datetime import datetime
import itertools as it
from tqdm import tqdm
from anytree import RenderTree, PreOrderIter

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from rtmlib import PoseTracker, BodyWithFeet

from Sports2D.Utilities import filter
from Sports2D.Utilities.common import *
from Sports2D.Utilities.skeletons import *


## CONSTANTS
angle_dict = { 
    # joint angles
    'Right ankle': [['RKnee', 'RAnkle', 'RBigToe', 'RHeel'], 'dorsiflexion', 90, 1],
    'Left ankle': [['LKnee', 'LAnkle', 'LBigToe', 'LHeel'], 'dorsiflexion', 90, 1],
    'Right knee': [['RAnkle', 'RKnee', 'RHip'], 'flexion', -180, 1],
    'Left knee': [['LAnkle', 'LKnee', 'LHip'], 'flexion', -180, 1],
    'Right hip': [['RKnee', 'RHip', 'Hip', 'Neck'], 'flexion', 0, -1],
    'Left hip': [['LKnee', 'LHip', 'Hip', 'Neck'], 'flexion', 0, -1],
    # 'Lumbar': [['Neck', 'Hip', 'RHip', 'LHip'], 'flexion', -180, -1],
    # 'Neck': [['Head', 'Neck', 'RShoulder', 'LShoulder'], 'flexion', -180, -1],
    'Right shoulder': [['RElbow', 'RShoulder', 'Hip', 'Neck'], 'flexion', 0, -1],
    'Left shoulder': [['LElbow', 'LShoulder', 'Hip', 'Neck'], 'flexion', 0, -1],
    'Right elbow': [['RWrist', 'RElbow', 'RShoulder'], 'flexion', 180, -1],
    'Left elbow': [['LWrist', 'LElbow', 'LShoulder'], 'flexion', 180, -1],
    'Right wrist': [['RElbow', 'RWrist', 'RIndex'], 'flexion', -180, 1],
    'Left wrist': [['LElbow', 'LIndex', 'LWrist'], 'flexion', -180, 1],

    # segment angles
    'Right foot': [['RBigToe', 'RHeel'], 'horizontal', 0, -1],
    'Left foot': [['LBigToe', 'LHeel'], 'horizontal', 0, -1],
    'Right shank': [['RAnkle', 'RKnee'], 'horizontal', 0, -1],
    'Left shank': [['LAnkle', 'LKnee'], 'horizontal', 0, -1],
    'Right thigh': [['RKnee', 'RHip'], 'horizontal', 0, -1],
    'Left thigh': [['LKnee', 'LHip'], 'horizontal', 0, -1],
    'Pelvis': [['RHip', 'LHip'], 'horizontal', 0, -1],
    'Trunk': [['Neck', 'Hip'], 'horizontal', 0, -1],
    'Shoulders': [['RShoulder', 'LShoulder'], 'horizontal', 0, -1],
    'Head': [['Head', 'Neck'], 'horizontal', 0, -1],
    'Right arm': [['RElbow', 'RShoulder'], 'horizontal', 0, -1],
    'Left arm': [['LElbow', 'LShoulder'], 'horizontal', 0, -1],
    'Right forearm': [['RWrist', 'RElbow'], 'horizontal', 0, -1],
    'Left forearm': [['LWrist', 'LElbow'], 'horizontal', 0, -1],
    'Right hand': [['RIndex', 'RWrist'], 'horizontal', 0, -1],
    'Left hand': [['LIndex', 'LWrist'], 'horizontal', 0, -1]
    }

colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255),
            (125, 0, 0), (0, 125, 0), (0, 0, 125), (125, 125, 0), (125, 0, 125), (0, 125, 125), 
            (255, 125, 125), (125, 255, 125), (125, 125, 255), (255, 255, 125), (255, 125, 255), (125, 255, 255), (125, 125, 125),
            (255, 0, 125), (255, 125, 0), (0, 125, 255), (0, 255, 125), (125, 0, 255), (125, 255, 0), (0, 255, 0)]
thickness = 1
fontSize = 0.3


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
    - cam_width: int. The actual width of the webcam frame
    - cam_height: int. The actual height of the webcam frame
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
    if cam_width != input_size[0] or cam_height != input_size[1]:
        logging.warning(f"Warning: Your webcam does not support {input_size[0]}x{input_size[1]} resolution. Resolution set to the closest supported one: {cam_width}x{cam_height}.")
    
    out_vid = None
    if save_vid:
        fps = cap.get(cv2.CAP_PROP_FPS)
        # fourcc MJPG produces very large files but is faster. If it is too slow, consider using it and then converting the video to h264
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1') # =h264. better compression and quality but may fail on some systems
            out_vid = cv2.VideoWriter(vid_output_path, fourcc, fps, (cam_width, cam_height))
            if not out_vid.isOpened():
                raise ValueError("Failed to open video writer with 'avc1' (h264)")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(vid_output_path, fourcc, fps, (cam_width, cam_height))
            logging.info("Failed to open video writer with 'avc1' (h264). Using 'mp4v' instead.")

    return cap, out_vid, cam_width, cam_height, fps


def setup_video(video_file_path, save_vid, vid_output_path):
    '''
    '''
    
    if video_file_path.name == video_file_path.stem:
        raise ValueError("Please set video_input to 'webcam' or to a video file (with extension) in Config.toml")
    try:
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            raise
    except:
        raise NameError(f"{video_file_path} is not a video. Check video_dir and video_input in your Config.toml file.")
    
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_vid = None
    if save_vid:
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1') # =h264. better compression and quality but may fail on some systems
            out_vid = cv2.VideoWriter(vid_output_path, fourcc, fps, (cam_width, cam_height))
            if not out_vid.isOpened():
                raise ValueError("Failed to open video writer with 'avc1' (h264)")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(vid_output_path, fourcc, fps, (cam_width, cam_height))
            logging.info("Failed to open video writer with 'avc1' (h264). Using 'mp4v' instead.")
        
    return cap, out_vid


def setup_pose_tracker(det_frequency, mode, tracking):
    '''
    Set up the pose tracker with the appropriate model and backend.
    If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino

    INPUTS:
    - det_frequency: int. The frequency of pose detection (every N frames)
    - mode: str. The mode of the pose tracker ('lightweight', 'balanced', 'performance')

    OUTPUTS:
    - pose_tracker: PoseTracker. The initialized pose tracker object    
    '''

    # If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino
    try:
        import torch
        import onnxruntime as ort
        if torch.cuda.is_available() == True and 'CUDAExecutionProvider' in ort.get_available_providers():
            device = 'cuda'
            backend = 'onnxruntime'
            logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
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
    logging.info(f'Pose tracking set up for BodyWithFeet model in {mode} mode.\nPersons are detected every {det_frequency} frames and tracked inbetween. Multi-person is {"" if tracking else "not "}selected.')
    logging.info(f'Selected device: {device} with {backend} backend.')

    # Initialize the pose tracker with Halpe26 model
    pose_tracker = PoseTracker(
        BodyWithFeet,
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
    - person_X: list of x coordinates after flipping
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
    
    '''

    ang_params = angle_dict.get(ang_name)
    if ang_params is not None:
        angle_coords = [[person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]], person_Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0] if kpt in keypoints_names]
        ang = points2D_to_angles(angle_coords)
        ang += ang_params[2]
        ang *= ang_params[3]
        ang = ang-360 if ang>180 else ang
        ang = ang+360 if ang<-180 else ang
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
    
    
def sort_people_sports2d(keyptpre, keypt, scores):
    '''
    Associate persons across frames
    Persons' indices are sometimes swapped when changing frame
    A person is associated to another in the next frame when they are at a small distance
    
    N.B.: Requires min_with_single_indices and euclidian_distance function (see common.py)

    INPUTS:
    - keyptpre: array of shape K, L, M with K the number of detected persons,
    L the number of detected keypoints, M their 2D coordinates + confidence
    for the previous frame
    - keypt: idem keyptpre, for current frame
    - score: scores of detected keypoints
    
    OUTPUT:
    - keypt: array with reordered persons
    '''
    
    # Generate possible person correspondences across frames
    if len(keyptpre) < len(keypt):
        keyptpre = np.concatenate((keyptpre, np.full((len(keypt)-len(keyptpre), keypt.shape[1], 2), 0.)))
    if len(keypt) < len(keyptpre):
        keypt = np.concatenate((keypt, np.full((len(keyptpre)-len(keypt), keypt.shape[1], 2), 0.)))
        scores = np.concatenate((scores, np.full((len(keyptpre)-len(scores), scores.shape[1]), 0.)))
    personsIDs_comb = sorted(list(it.product(range(len(keyptpre)),range(len(keypt)))))
    
    # Compute distance between persons from one frame to another
    frame_by_frame_dist = []
    for comb in personsIDs_comb:
        frame_by_frame_dist += [euclidean_distance(keyptpre[comb[0]],keypt[comb[1]])]
    
    # Sort correspondences by distance
    _, _, associated_tuples = min_with_single_indices(frame_by_frame_dist, personsIDs_comb)
    
    # Associate points to same index across frames, nan if no correspondence
    sorted_keypoints, sorted_scores, personsIDs_sorted = [], [], []
    for i in range(len(keyptpre)):
        id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
        if len(id_in_old) > 0:
            personsIDs_sorted += id_in_old
            sorted_keypoints += [keypt[id_in_old[0]]]
            sorted_scores += [scores[id_in_old[0]]]
        else:
            personsIDs_sorted += [-1]
            sorted_keypoints += [keypt[i]]
            sorted_scores += [scores[i]]
    
    return np.array(sorted_keypoints), np.array(sorted_scores)


def sort_people_rtmlib(pose_tracker, keypoints, scores):
    '''
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
    '''
    for i in range(0, length, gap):
        line_start = start + direction * i
        line_end = line_start + direction * dot_length
        cv2.line(img, tuple(line_start.astype(int)), tuple(line_end.astype(int)), color, thickness)


def draw_bounding_box(img, X, Y, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
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
            cv2.putText(img, str(i), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA) 
    
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
                if not (np.isnan(x[n[0]]) or np.isnan(y[n[0]]) or np.isnan(x[n[1]]) or np.isnan(y[n[1]]))]

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
    - model: skeleton model (from skeletons.py)
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


def draw_angles(img, valid_X, valid_Y, valid_angles, valid_X_flipped, keypoints_ids, keypoints_names, angle_names, display_angle_values_on= ['body', 'list'], colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
    '''
    '''

    color_cycle = it.cycle(colors)
    for person_id, (X,Y,angles, X_flipped) in enumerate(zip(valid_X, valid_Y, valid_angles, valid_X_flipped)):
        c = next(color_cycle)
        if not np.isnan(X).all():
            # person label
            if 'list' in display_angle_values_on:
                person_label_position = (int(10 + fontSize*200/0.3*person_id), 15)
                cv2.putText(img, f'person {person_id}', person_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (255,255,255), thickness+1, cv2.LINE_AA)
                cv2.putText(img, f'person {person_id}', person_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, c, thickness, cv2.LINE_AA)
            
            # angle lines, names and values
            ang_label_line = 1
            for k, ang in enumerate(angles):
                if not np.isnan(ang):
                    ang_name = angle_names[k]
                    ang_params = angle_dict.get(ang_name)
                    if ang_params is not None:
                        ang_coords = np.array([[X[keypoints_ids[keypoints_names.index(kpt)]], Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0] if kpt in keypoints_names])
                        X_flipped_coords = [X_flipped[keypoints_ids[keypoints_names.index(kpt)]] for kpt in ang_params[0] if kpt in keypoints_names]
                        flip = -1 if any(x_flipped < 0 for x_flipped in X_flipped_coords) else 1
                        right_angle = True if ang_params[2]==90 else False
                        
                        # Draw angle
                        if 'body' in display_angle_values_on:
                            if len(ang_coords) == 2: # segment angle
                                app_point, vec = draw_segment_angle(img, ang_coords, flip)
                                write_angle_on_body(img, ang, app_point, vec, np.array([1,0]), dist=20, color=(255,255,255))
                            
                            else: # joint angle
                                app_point, vec1, vec2 = draw_joint_angle(img, ang_coords, flip, right_angle)
                                write_angle_on_body(img, ang, app_point, vec1, vec2, dist=40, color=(0,255,0))

                        # Write angle as a list on image with progress bar
                        if 'list' in display_angle_values_on:
                            if len(ang_coords) == 2: # segment angle
                                ang_label_line = write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color = (255,255,255))
                            else:
                                ang_label_line = write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color = (0,255,0))

    return img


def draw_segment_angle(img, ang_coords, flip):
    '''

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


def write_angle_on_body(img, ang, app_point, vec1, vec2, dist=40, color=(255,255,255)):
    '''
    '''

    vec_sum = vec1 + vec2
    if (vec_sum == 0.).all():
        return
    unit_vec_sum = vec_sum/np.linalg.norm(vec_sum)
    text_position = np.int32(app_point + unit_vec_sum*dist)
    cv2.putText(img, f'{ang:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, f'{ang:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)


def write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color=(255,255,255)):
    '''

    '''
    
    if not np.any(np.isnan(ang)):
        # angle names and values
        ang_label_position = (person_label_position[0], person_label_position[1]+ang_label_line*15)
        cv2.putText(img, f'{ang_name}:', ang_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, f'{ang_name}:', ang_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)
        cv2.putText(img, f'{ang:.1f}', (ang_label_position[0]+100, ang_label_position[1]), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, f'{ang:.1f}', (ang_label_position[0]+100, ang_label_position[1]), cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)
        
        # progress bar
        ang_percent = int(ang*50/180)
        y_crop, y_crop_end = 1+15*ang_label_line, 16+15*ang_label_line
        x_crop, x_crop_end = ang_label_position[0]+115, ang_label_position[0]+115+ang_percent
        if ang_percent < 0:
            x_crop, x_crop_end = x_crop_end, x_crop
        img_crop = img[y_crop:y_crop_end, x_crop:x_crop_end]
        if img_crop.size>0:
            white_rect = np.ones(img_crop.shape, dtype=np.uint8)*255
            alpha_rect = cv2.addWeighted(img_crop, 0.6, white_rect, 0.4, 1.0)
            img[y_crop:y_crop_end, x_crop:x_crop_end] = alpha_rect

        ang_label_line += 1
    
    return ang_label_line


def make_trc_with_XYZ(X, Y, Z, time, trc_path):
    '''
    '''
    
    #Header
    frame_rate = (len(X)-1)/(time.iloc[-1] - time.iloc[0])
    DataRate = CameraRate = OrigDataRate = frame_rate
    NumFrames = len(X)
    NumMarkers = len(X.columns)
    keypoint_names = X.columns
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_path, 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, 0, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoint_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoint_names))])]
    
    # Data
    Q = pd.concat([pd.concat([X.iloc[:,kpt], Y.iloc[:,kpt], Z.iloc[:,kpt]], axis=1) for kpt in range(len(X.columns))], axis=1)
    Q.insert(0, 't', time)

    # Write file
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path


def process_fun(config_dict, video_file, time_range, frame_rate, result_dir):
    '''
    Detect joint centers from a video with OpenPose or BlazePose.
    Save a 2D csv file per person, and optionally json files, image files, and video file.
    
    If OpenPose is used, multiple persons can be consistently detected across frames.
    Interpolates sequences of missing data if they are less than N frames long.
    Optionally filters results with Butterworth, gaussian, median, or loess filter.
    Optionally displays figures.

    If BlazePose is used, only one person can be detected.
    No interpolation nor filtering options available. Not plotting available.

    /!\ Warning /!\
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
        
    INPUTS:
    - a video or a webcam
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one csv file of joint coordinates per detected person
    - optionally json directory, image directory, video
    - a logs.txt file 
    '''
    
    # Base parameters
    video_dir = Path(config_dict.get('project').get('video_dir'))
    result_dir = Path(config_dict.get('project').get('result_dir'))
    webcam_id =  config_dict.get('project').get('webcam_id')
    input_size = config_dict.get('project').get('input_size')

    # Process settings
    tracking = config_dict.get('process').get('multiperson')
    show_realtime_results = config_dict.get('process').get('show_realtime_results')
    save_vid = config_dict.get('process').get('save_vid')
    save_img = config_dict.get('process').get('save_img')
    save_pose = config_dict.get('process').get('save_pose')
    save_angles = config_dict.get('process').get('save_angles')

    # Pose_advanced settings
    pose_model = config_dict.get('pose').get('pose_model')
    mode = config_dict.get('pose').get('mode')
    det_frequency = config_dict.get('pose').get('det_frequency')
    tracking_mode = config_dict.get('pose').get('tracking_mode')

    keypoint_likelihood_threshold = config_dict.get('pose').get('keypoint_likelihood_threshold')
    average_likelihood_threshold = config_dict.get('pose').get('average_likelihood_threshold')
    keypoint_number_threshold = config_dict.get('pose').get('keypoint_number_threshold')
    detection_time_threshold = config_dict.get('pose').get('detection_time_threshold')
    
    interp_gap_smaller_than = config_dict.get('pose').get('interp_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('pose').get('fill_large_gaps_with')

    do_filter_pose = config_dict.get('pose').get('filter')
    show_plots_pose = config_dict.get('pose').get('show_plots')
    filter_type_pose = config_dict.get('pose').get('filter_type')
    butterworth_filter_order_pose = config_dict.get('pose').get('butterworth').get('order')
    butterworth_filter_cutoff_pose = config_dict.get('pose').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel_pose = config_dict.get('pose').get('gaussian').get('sigma_kernel')
    loess_filter_kernel_pose = config_dict.get('pose').get('loess').get('nb_values_used')
    median_filter_kernel_pose = config_dict.get('pose').get('median').get('kernel_size')
    filter_options_pose = (do_filter_pose, filter_type_pose, show_plots_pose,
                           butterworth_filter_order_pose, butterworth_filter_cutoff_pose, frame_rate,
                           gaussian_filter_kernel_pose, loess_filter_kernel_pose, median_filter_kernel_pose)

    # Angles advanced settings
    joint_angle_names = config_dict.get('angles').get('joint_angles')
    segment_angle_names = config_dict.get('angles').get('segment_angles')
    angle_names = joint_angle_names + segment_angle_names
    display_angle_values_on = config_dict.get('angles').get('display_angle_values_on')
    flip_left_right = config_dict.get('angles').get('flip_left_right')
    
    do_filter_ang = config_dict.get('angles').get('filter')
    show_plots_ang = config_dict.get('angles').get('show_plots')
    filter_type_ang = config_dict.get('angles').get('filter_type')
    butterworth_filter_order_ang = config_dict.get('angles').get('butterworth').get('order')
    butterworth_filter_cutoff_ang = config_dict.get('angles').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel_ang = config_dict.get('angles').get('gaussian').get('sigma_kernel')
    loess_filter_kernel_ang = config_dict.get('angles').get('loess').get('nb_values_used')
    median_filter_kernel_ang = config_dict.get('angles').get('median').get('kernel_size')
    filter_options_ang = (do_filter_ang, filter_type_ang, show_plots_ang,
                          butterworth_filter_order_ang, butterworth_filter_cutoff_ang, frame_rate, 
                          gaussian_filter_kernel_ang, loess_filter_kernel_ang, median_filter_kernel_ang)


    # Create output directories
    if video_file == "webcam":
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f'webcam_{current_date} Sports2D'
    else:
        video_file_path = video_dir / video_file
        video_file_stem = video_file.stem
        output_dir_name = f'{video_file_stem}_Sports2D'    
    output_dir = result_dir / output_dir_name
    img_output_dir = output_dir / f'{output_dir_name}_img'
    vid_output_path = output_dir / f'{output_dir_name}_Sports2D.mp4'
    pose_output_path = output_dir / f'{output_dir_name}.trc'
    angles_output_path = output_dir / f'{output_dir_name}_angles.mot'

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_img:
        img_output_dir.mkdir(parents=True, exist_ok=True)


    # Retrieve keypoint names from model
    model = eval(pose_model)
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]

    Ltoe_idx = keypoints_ids[keypoints_names.index('LBigToe')]
    LHeel_idx = keypoints_ids[keypoints_names.index('LHeel')]
    Rtoe_idx = keypoints_ids[keypoints_names.index('RBigToe')]
    RHeel_idx = keypoints_ids[keypoints_names.index('RHeel')]
    L_R_direction_idx = [Ltoe_idx, LHeel_idx, Rtoe_idx, RHeel_idx]


    # Set up video capture
    if video_file == "webcam":
        cap, out_vid, cam_width, cam_height, fps = setup_webcam(webcam_id, save_vid, vid_output_path, input_size)
        frame_range = [0,sys.maxsize]
        frame_iterator = range(*frame_range)
        logging.warning('Webcam input: the framerate may vary. If results are filtered, Sports2D will use the average framerate as input.')
    else:
        cap, out_vid = setup_video(video_file_path, save_vid, vid_output_path)
        frame_range = [int(time_range[0] * frame_rate), int(time_range[1] * frame_rate)] if time_range else [0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_iterator = tqdm(range(*frame_range)) # use a progress bar
    if show_realtime_results:
        cv2.namedWindow(f'{video_file} Sports2D', cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(f'{video_file} Sports2D', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN)


    # Set up pose tracker
    tracking_rtmlib = True if (tracking_mode == 'rtmlib' and tracking) else False
    pose_tracker = setup_pose_tracker(det_frequency, mode, tracking_rtmlib)


    # Process video or webcam feed
    logging.info(f"{'Video, ' if save_vid else ''}{'Images, ' if save_img else ''}{'Pose, ' if save_pose else ''}{'Angles ' if save_angles else ''}{'and ' if save_angles or save_img or save_pose or save_vid else ''}Logs will be saved in {result_dir}.")
    
    logging.info(f"\nParameters: \n{f'{tracking_mode=}, ' if tracking else ''}{keypoint_likelihood_threshold=}, {average_likelihood_threshold=}, {keypoint_number_threshold=}")
    logging.info(f"Post-processing parameter: {detection_time_threshold=}, {interp_gap_smaller_than=}, {fill_large_gaps_with=}")
    logging.info(f"{filter_options_pose=}, {filter_options_ang=}")

    logging.info(f"\nProcessing video stream...")
    all_frames_X, all_frames_Y, all_frames_scores, all_frames_angles = [], [], [], []
    frame_processing_times = []
    frame_count = 0
    while cap.isOpened():
        if frame_count < frame_range[0]:
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
                continue
            else:
                if video_file == 'webcam':
                    cv2.putText(frame, f"Press 'q' to quit", (cam_width-150, cam_height-20), cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (255,255,255), thickness+1, cv2.LINE_AA)
                    cv2.putText(frame, f"Press 'q' to quit", (cam_width-150, cam_height-20), cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (0,0,255), thickness, cv2.LINE_AA)
            
            # Detect poses
            keypoints, scores = pose_tracker(frame)

            # Track persons
            if tracking: # multi-person
                if tracking_rtmlib:
                    keypoints, scores = sort_people_rtmlib(pose_tracker, keypoints, scores)
                else:
                    if 'prev_keypoints' not in locals(): prev_keypoints = keypoints
                    keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores)
                    prev_keypoints = keypoints
            else: # single person
                keypoints, scores = np.array([keypoints[0]]), np.array([scores[0]])
            
            
            # Process coordinates and compute angles
            valid_X, valid_Y, valid_scores = [], [], []
            valid_X_flipped, valid_angles = [], []
            for person_idx in range(len(keypoints)):
                # Retrieve keypoints and scores for the person, remove low-confidence keypoints
                person_X, person_Y = np.where(scores[person_idx][:, np.newaxis] < keypoint_likelihood_threshold, np.nan, keypoints[person_idx]).T
                person_scores = np.where(scores[person_idx] < keypoint_likelihood_threshold, np.nan, scores[person_idx])

                # Skip person if the fraction of valid detected keypoints is too low
                enough_good_keypoints = len(person_scores[~np.isnan(person_scores)]) >= len(person_scores) * keypoint_number_threshold
                average_score_of_remaining_keypoints_is_enough = np.nanmean(person_scores[~np.isnan(person_scores)]) >= average_likelihood_threshold
                if not enough_good_keypoints or not average_score_of_remaining_keypoints_is_enough:
                    person_X = np.full_like(person_X, np.nan)
                    person_Y = np.full_like(person_Y, np.nan)
                    person_scores = np.full_like(person_scores, np.nan)
                valid_X.append(person_X)
                valid_Y.append(person_Y)
                valid_scores.append(person_scores)

                # Check whether the person is looking to the left or right
                if flip_left_right:
                    person_X_flipped = flip_left_right_direction(person_X, L_R_direction_idx, keypoints_names, keypoints_ids)
                else:
                    person_X_flipped = person_X.copy()
                valid_X_flipped.append(person_X_flipped)
                    
                # Compute angles
                person_angles = []
                for ang_name in angle_names:
                    ang = compute_angle(ang_name, person_X_flipped, person_Y, angle_dict, keypoints_ids, keypoints_names)
                    person_angles.append(ang)
                valid_angles.append(person_angles)

            # Draw keypoints and skeleton
            if show_realtime_results or save_vid or save_img:
                img = frame.copy()
                img = draw_bounding_box(img, valid_X, valid_Y, colors=colors)
                img = draw_skel(img, valid_X, valid_Y, model, colors=colors)
                img = draw_angles(img, valid_X, valid_Y, valid_angles, valid_X_flipped, keypoints_ids, keypoints_names, angle_names, display_angle_values_on=display_angle_values_on, colors=colors)
                img = draw_keypts(img, valid_X, valid_Y, scores, cmap_str='RdYlGn')

                if show_realtime_results:
                    cv2.imshow(f'{video_file} Sports2D', img)
                    if (cv2.waitKey(1) & 0xFF) == ord('q') or (cv2.waitKey(1) & 0xFF) == 27:
                        break
                if save_vid:
                    out_vid.write(img)
                if save_img:
                    cv2.imwrite(str(img_output_dir / f'{output_dir_name}_{frame_count:06d}.png'), img)

            if save_pose:
                all_frames_X.append(valid_X)
                all_frames_Y.append(valid_Y)
                all_frames_scores.append(valid_scores)
            if save_angles:
                all_frames_angles.append(valid_angles)
            if video_file=='webcam' and save_vid:   # To adjust framerate of output video
                elapsed_time = (datetime.now() - start_time).total_seconds()
                frame_processing_times.append(elapsed_time)
            frame_count += 1

        cap.release()
        logging.info(f"Video processing completed.\n")
        if save_vid:
            out_vid.release()
            if video_file == 'webcam':
                actual_framerate = len(frame_processing_times) / sum(frame_processing_times)
                logging.info(f"Rewriting webcam video based on the averate framerate {actual_framerate}.")
                resample_video(vid_output_path, fps, actual_framerate)
            logging.info(f"--> Output video saved to {vid_output_path}.")
        if save_img:
            logging.info(f"--> Output images saved to {img_output_dir}.")
        if show_realtime_results:
            cv2.destroyAllWindows()
    

    # # Interpolate, filter, and save pose and angles
    # if save_pose:
    #     save_pose_to_trc(pose_output_path, all_frames_X, all_frames_Y, all_frames_scores, keypoints_names)

    #     keypoints_names_by_id = [name for id, name in sorted(zip(keypoints_ids, keypoints_names))]


    #     all_frames_X: same len for all frames (fill with nan if no person detected)
            
    #     all_frames_X = pd.DataFrame(np.array(all_frames_X), columns=keypoints_names_by_id)
    #     all_frames_Y = pd.DataFrame(np.array(all_frames_Y), columns=keypoints_names_by_id)

    #     # all_frames_Z = pd.DataFrame(np.zeros_like(all_frames_X), columns=keypoints_names_by_id)
    #     # all_frames_time = pd.Series(np.linspace(time_range[0], time_range[1], len(all_frames_X)), name='time')

    #     make_trc_with_XYZ(all_frames_X, all_frames_Y, all_frames_Z, all_frames_time, pose_output_path)

    #     .to_csv(pose_output_path, index=False, sep='\t')


        
        
    #     if filter_options_pose[0]:
    #         pass
    #         # logging.info(f"--> Filtering pose data with a {filter_options_pose[1]} filter.")
    #         # all_frames_X, all_frames_Y = filter_pose_data(all_frames_X, all_frames_Y, filter_options_pose)
    #         # save_pose_to_trc(pose_output_path, all_frames_X, all_frames_Y, all_frames_scores, keypoints_names, filtered=True)
    #     logging.info(f"--> Output pose saved to {pose_output_path}.")

    # if save_angles:
    #     save_angles_to_mot(angles_output_path, all_frames_angles, angle_names)
    #     logging.info(f"--> Output angles saved to {angles_output_path}.")

