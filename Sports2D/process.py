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
        
    ⚠ Warning ⚠ 
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
import copy
import shutil
import os
import re
import platform
from importlib.metadata import version
from datetime import datetime
import itertools as it
from tqdm import tqdm
from collections import defaultdict
from anytree import RenderTree

import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import patheffects

from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Hand, Custom
from rtmlib.tools.object_detection.post_processings import nms

from Sports2D.Utilities.common import *
from Pose2Sim.common import *
from Pose2Sim.skeletons import *
from Pose2Sim.calibration import toml_write
from Pose2Sim.triangulation import indices_of_first_last_non_nan_chunks
from Pose2Sim.personAssociation import *
from Pose2Sim.filtering import *

# Silence numpy "RuntimeWarning: Mean of empty slice"
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

# Not safe, but to be used until OpenMMLab/RTMlib's SSL certificates are updated
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


DEFAULT_MASS = 70
DEFAULT_HEIGHT = 1.7

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = version("sports2d")
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def setup_webcam(webcam_id, vid_output_path, input_size):
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    if cam_width != input_size[0] or cam_height != input_size[1]:
        logging.warning(f"Warning: Your webcam does not support {input_size[0]}x{input_size[1]} resolution. Resolution set to the closest supported one: {cam_width}x{cam_height}.")
    
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


def setup_video(video_file_path, vid_output_path, save_vid):
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


def setup_model_class_mode(pose_model, mode, config_dict={}):
    '''
    Set up the pose model class and mode for the pose tracker.
    '''

    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        model_name = 'HALPE_26'
        ModelClass = BodyWithFeet # 26 keypoints(halpe26)
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation in {mode} mode.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY', 'WHOLE_BODY_WRIST'):
        model_name = 'COCO_133'
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation in {mode} mode.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_name = 'COCO_17'
        ModelClass = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation in {mode} mode.")
    elif pose_model.upper() =='HAND':
        model_name = 'HAND_21'
        ModelClass = Hand
        logging.info(f"Using HAND_21 model for pose estimation in {mode} mode.")
    elif pose_model.upper() =='FACE':
        model_name = 'FACE_106'
        logging.info(f"Using FACE_106 model for pose estimation in {mode} mode.")
    elif pose_model.upper() =='ANIMAL':
        model_name = 'ANIMAL2D_17'
        logging.info(f"Using ANIMAL2D_17 model for pose estimation in {mode} mode.")
    else:
        model_name = pose_model.upper()
        logging.info(f"Using model {model_name} for pose estimation in {mode} mode.")
    try:
        pose_model = eval(model_name)
    except:
        try: # from Config.toml
            from anytree.importer import DictImporter
            model_name = pose_model.upper()
            pose_model = DictImporter().import_(config_dict.get('pose').get(pose_model)[0])
            if pose_model.id == 'None':
                pose_model.id = None
            logging.info(f"Using model {model_name} for pose estimation.")
        except:
            raise NameError(f'{pose_model} not found in skeletons.py nor in Config.toml')

    # Manually select the models if mode is a dictionary rather than 'lightweight', 'balanced', or 'performance'
    if not mode in ['lightweight', 'balanced', 'performance'] or 'ModelClass' not in locals():
        try:
            from functools import partial
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
                        pose_class=pose_class, pose=pose, pose_input_size=pose_input_size)
            logging.info(f"Using model {model_name} with the following custom parameters: {mode}.")

            if pose_class == 'RTMO' and model_name != 'COCO_17':
                logging.warning("RTMO currently only supports 'Body' pose_model. Switching to 'Body'.")
                pose_model = eval('COCO_17')
            
        except (json.JSONDecodeError, TypeError):
            logging.warning("Invalid mode. Must be 'lightweight', 'balanced', 'performance', or '''{dictionary}''' of parameters within triple quotes. Make sure input_sizes are within square brackets.")
            logging.warning('Using the default "balanced" mode.')
            mode = 'balanced'

    return pose_model, ModelClass, mode


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
                            if not np.any(np.isnan(ang_coords)):
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

    Z_cols = np.array([[3*i,3*i+1] for i in range(len(Q_coords.columns)//3)]).ravel()
    Q_coords_xy = Q_coords.iloc[:,Z_cols]
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

    columns_to_concat = []
    for kpt in range(len(X.columns)):
        columns_to_concat.extend([X.iloc[:,kpt], Y.iloc[:,kpt], Z.iloc[:,kpt]])
    trc_data = pd.concat([time] + columns_to_concat, axis=1)

    return trc_data


def make_trc_with_trc_data(trc_data, trc_path, fps=30):
    '''
    Write a TRC file from a DataFrame of time and coordinates

    INPUTS:
    - trc_data: pd.DataFrame. The time and coordinates of the keypoints. 
                    The column names must be 'time', 'kpt1', 'kpt1', 'kpt1', 'kpt2', 'kpt2', 'kpt2', ...
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
            'Frame#\tTime\t' + '\t\t\t'.join(keypoint_names) + '\t\t\t',
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


def pose_plots(trc_data_unfiltered, trc_data, person_id, show=True):
    '''
    Displays trc filtered and unfiltered data for comparison
    ⚠ Often crashes on the third window...

    INPUTS:
    - trc_data_unfiltered: pd.DataFrame. The unfiltered trc data
    - trc_data: pd.DataFrame. The filtered trc data
    - person_id: int. The ID of the person
    - show: bool. Whether to show the plots

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''

    os_name = platform.system()
    if os_name == 'Windows':
        mpl.use('qt5agg') # windows
    mpl.rc('figure', max_open_warning=0)

    keypoints_names = trc_data.columns[1::3]
    
    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person'+ str(person_id) + ' coordinates') # Main title

    for id, keypoint in enumerate(keypoints_names):
        f = plt.figure()
        if os_name == 'Windows':
            f.canvas.manager.window.setWindowTitle(keypoint + ' Plot') # windows
        elif os_name == 'Darwin':  # macOS
            f.canvas.manager.set_window_title(keypoint + ' Plot') # mac

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
    
    if show:
        pw.show()

    return pw


def angle_plots(angle_data_unfiltered, angle_data, person_id, show=True):
    '''
    Displays angle filtered and unfiltered data for comparison
    ⚠ Often crashes on the third window...

    INPUTS:
    - angle_data_unfiltered: pd.DataFrame. The unfiltered angle data
    - angle_data: pd.DataFrame. The filtered angle data

    OUTPUT:
    - matplotlib window with tabbed figures for each angle
    '''

    os_name = platform.system()
    if os_name == 'Windows':
        mpl.use('qt5agg') # windows
    mpl.rc('figure', max_open_warning=0)

    angles_names = angle_data.columns[1:]
    
    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person'+ str(person_id) + ' angles') # Main title

    for id, angle in enumerate(angles_names):
        f = plt.figure()
        if os_name == 'Windows':
            f.canvas.manager.window.setWindowTitle(angle + ' Plot') # windows
        elif os_name == 'Darwin':  # macOS
            f.canvas.manager.set_window_title(angle + ' Plot') # mac
        
        ax = plt.subplot(111)
        plt.plot(angle_data_unfiltered.iloc[:,0], angle_data_unfiltered.iloc[:,id+1], label='unfiltered')
        plt.plot(angle_data.iloc[:,0], angle_data.iloc[:,id+1], label='filtered')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(angle+' (°)')
        plt.legend()

        pw.addPlot(angle, f)

    if show:
        pw.show()

    return pw


def get_personIDs_with_highest_scores(all_frames_scores, nb_persons_to_detect):
    '''
    Get the person IDs with the highest scores

    INPUTS:
    - all_frames_scores: array of scores for all frames, all persons, all keypoints
    - nb_persons_to_detect: int or 'all'. The number of persons to detect

    OUTPUT:
    - selected_persons: list of int. The person IDs with the highest scores
    '''

    # Get the person with the highest scores over all frames and all keypoints
    score_means = np.nansum(np.nanmean(all_frames_scores, axis=0), axis=1)
    selected_persons = (-score_means).argsort()[:nb_persons_to_detect]
    
    return selected_persons


def get_personIDs_in_detection_order(nb_persons_to_detect, reverse=False):
    '''
    Get the person IDs in the order of detection

    INPUTS:
    - nb_persons_to_detect: int. The number of persons to detect
    - reverse: bool. Whether to reverse the order of detection

    OUTPUT:
    - selected_persons: list of int. The person IDs in the order of detection
    '''

    selected_persons = list(range(nb_persons_to_detect))
    if reverse:
        selected_persons = selected_persons[::-1]

    return selected_persons


def get_personIDs_with_largest_size(all_frames_X_homog, all_frames_Y_homog, nb_persons_to_detect, reverse=False, vertical=False):
    '''
    Get the person IDs with the largest size
    
    INPUTS:
    - all_frames_X_homog: shape (Nframes, Npersons, Nkpts)
    - all_frames_Y_homog: shape (Nframes, Npersons, Nkpts)
    - nb_persons_to_detect: int. The number of persons to detect
    - reverse: bool. Whether to reverse the order of detection from smallest to largest size
    - vertical: bool. Whether to compute the size in the vertical direction only

    OUTPUT:
    - selected_persons: list of int. The person IDs with the largest size
    '''

    # average size over all keypoints (axis=2) and all frames (axis=0) for each person (axis=1)
    y_sizes = np.array([np.nanmean(np.nanmax(all_frames_Y_homog, axis=2) - np.nanmin(all_frames_Y_homog, axis=2), axis=0)][0])
    if vertical:
        sizes = y_sizes
    else:
        x_sizes = np.array([np.nanmean(np.nanmax(all_frames_X_homog, axis=2) - np.nanmin(all_frames_X_homog, axis=2), axis=0)][0])
        sizes = np.sqrt(x_sizes**2 + y_sizes**2)

    if not reverse: # greatest to smallest size
        sizes = -sizes

    selected_persons = sizes.argsort()[:nb_persons_to_detect]
    
    return selected_persons


def get_personIDs_with_greatest_displacement(all_frames_X_homog, all_frames_Y_homog, nb_persons_to_detect, reverse=False, horizontal=True):
    '''
    Get the person IDs with the greatest displacement
    
    INPUTS:
    - all_frames_X_homog: shape (Nframes, Npersons, Nkpts) 
    - all_frames_Y_homog: shape (Nframes, Npersons, Nkpts) 
    - nb_persons_to_detect: int. The number of persons to detect
    - reverse: bool. Whether to reverse the order of detection from smallest to greatest displacement
    - horizontal: bool. Whether to compute the displacement in the horizontal direction

    OUTPUT:
    - selected_persons: list of int. The person IDs with the greatest displacement
    '''
    
    # Average position over all keypoints to shape (Npersons, Nframes, Ndims)
    mean_pos_X_kpts = np.nanmean(all_frames_X_homog, axis=2)
    
    # Compute sum of distances from one frame to the next
    if horizontal:
        max_dist_traveled = abs(np.nansum(np.diff(mean_pos_X_kpts, axis=0), axis=0))
    else:
        mean_pos_Y_kpts = np.nanmean(all_frames_Y_homog, axis=2)
        pos_XY = np.stack((mean_pos_X_kpts.T, mean_pos_Y_kpts.T), axis=-1)
        max_dist_traveled = np.nansum([euclidean_distance(m,p) for (m,p) in zip(pos_XY[:,1:,:], pos_XY[:,:-1,:])], axis=1)
    max_dist_traveled = np.where(np.isinf(max_dist_traveled), 0, max_dist_traveled)

    if not reverse: # greatest to smallest displacement
        max_dist_traveled = -max_dist_traveled
    
    selected_persons = (max_dist_traveled).argsort()[:nb_persons_to_detect]
    
    return selected_persons


def get_personIDs_on_click(video_file_path, frame_range, all_frames_X_homog, all_frames_Y_homog):
    '''
    Get the person IDs on click in the image

    INPUTS:
    - video_file_path: path to video file
    - frame_range: tuple (start_frame, end_frame)
    - all_frames_X_homog: shape (Nframes, Npersons, Nkpts)
    - all_frames_Y_homog: shape (Nframes, Npersons, Nkpts)

    OUTPUT:
    - selected_persons: list of int. The person IDs selected by the user
    '''

    # Reorganize the coordinates to shape (Nframes, Npersons, Nkpts, Ndims)
    all_pose_coords = np.stack((all_frames_X_homog, all_frames_Y_homog), axis=-1)

    # Select person IDs on click on video/image
    selected_persons = select_persons_on_vid(video_file_path, frame_range, all_pose_coords)

    return selected_persons


def select_persons_on_vid(video_file_path, frame_range, all_pose_coords):
    '''
    Interactive UI to select persons from a video by clicking on their bounding boxes.
    
    INPUTS:
    - video_file_path: path to video file
    - frame_range: tuple (start_frame, end_frame)
    - all_pose_coords: keypoints coordinates. shape (Nframes, Npersons, Nkpts, Ndims)
        
    OUTPUT:
    - selected_persons : list with indices of selected persons
    '''

    BACKGROUND_COLOR = 'white'
    SLIDER_COLOR = '#4682B4'
    SLIDER_EDGE_COLOR = (0.5, 0.5, 0.5, 0.5)
    UNSELECTED_COLOR = (1, 1, 1, 0.1)
    LINE_UNSELECTED_COLOR = 'white'
    LINE_SELECTED_COLOR = 'darkorange'


    def get_frame(frame_idx):
        """Get frame with caching"""
        actual_frame_idx = start_frame + frame_idx
        
        # Check cache first
        if actual_frame_idx in frame_cache:
            # Move to end of cache order (recently used)
            cache_order.remove(actual_frame_idx)
            cache_order.append(actual_frame_idx)
            return frame_cache[actual_frame_idx]
        
        # Load from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        success, frame = cap.read()
        if not success:
            raise ValueError(f"Could not read frame {actual_frame_idx}")
        
        # Add to cache
        frame_cache[actual_frame_idx] = frame.copy()
        cache_order.append(actual_frame_idx)
        
        # Remove old frames if cache too large
        while len(frame_cache) > cache_size:
            oldest_frame = cache_order.pop(0)
            if oldest_frame in frame_cache:
                del frame_cache[oldest_frame]
        
        return frame
    

    def update_frame(val):
        # Update image
        frame_idx = int(frame_slider.val)
        frame = get_frame(frame_idx)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update bboxes and annotations
        for items in [rects, annotations]:
            for item in items:
                item.remove()
            items.clear()

        for person_idx, bbox in enumerate(all_bboxes[frame_idx]):
            if ~np.isnan(bbox).any():
                x_min, y_min, x_max, y_max = bbox.astype(int)
                rect = plt.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=1, edgecolor='white', facecolor=UNSELECTED_COLOR,
                    linestyle='-', path_effects=[patheffects.withSimplePatchShadow()], zorder=2
                ) 
                ax_video.add_patch(rect)
                rects.append(rect)

                annotation = ax_video.text(
                    x_min, y_min - 10, f'{person_idx}', color=LINE_UNSELECTED_COLOR, fontsize=7, fontweight='normal',
                    bbox=dict(facecolor=UNSELECTED_COLOR, edgecolor=LINE_UNSELECTED_COLOR, boxstyle='square,pad=0.3'), path_effects=[patheffects.withSimplePatchShadow()], zorder=3
                )
                annotations.append(annotation)
            else:
                rect = plt.Rectangle((np.nan, np.nan), np.nan, np.nan)
                ax_video.add_patch(rect)
                rects.append(rect)

        # Update plot
        img_plot.set_data(frame_rgb)
        fig.canvas.draw_idle()


    def on_click(event):
        if event.inaxes != ax_video:
            return
        
        frame_idx = int(frame_slider.val)
        x, y = event.xdata, event.ydata
        
        # Check if click is inside any bounding box
        for person_idx, bbox in enumerate(all_bboxes[frame_idx]):
            if ~np.isnan(bbox).any():
                x_min, y_min, x_max, y_max = bbox.astype(int)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    # Toggle selection
                    if person_idx in selected_persons:
                        rects[person_idx].set_linewidth(1)
                        rects[person_idx].set_edgecolor(LINE_UNSELECTED_COLOR)
                        selected_persons.remove(person_idx)
                    else:
                        rects[person_idx].set_linewidth(2)
                        rects[person_idx].set_edgecolor(LINE_SELECTED_COLOR)
                        selected_persons.append(person_idx)
                    
                    # Update display
                    status_text.set_text(f"Selected: {selected_persons}")
                    # draw_bounding_boxes(frame_idx)
                    fig.canvas.draw_idle()
                    break
    

    def on_hover(event):
        if event.inaxes != ax_video:
            return
        
        frame_idx = int(frame_slider.val)
        x, y = event.xdata, event.ydata

        # Change color on hover
        for person_idx, bbox in enumerate(all_bboxes[frame_idx]):
            if person_idx >= len(rects):  # Skip if rect doesn't exist
                continue
            if ~np.isnan(bbox).any():
                x_min, y_min, x_max, y_max = bbox.astype(int)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    rects[person_idx].set_linewidth(2)
                    rects[person_idx].set_edgecolor(LINE_SELECTED_COLOR)
                    rects[person_idx].set_facecolor((1, 1, 0, 0.2))
                else:
                    rects[person_idx].set_facecolor(UNSELECTED_COLOR)
                    if person_idx in selected_persons:
                        rects[person_idx].set_linewidth(2)
                        rects[person_idx].set_edgecolor(LINE_SELECTED_COLOR)
                    else:
                        rects[person_idx].set_linewidth(1)
                        rects[person_idx].set_edgecolor(LINE_UNSELECTED_COLOR)
                fig.canvas.draw_idle()


    def on_ok(event):
        plt.close(fig)


    # Open video
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_file_path}")
    start_frame, end_frame = frame_range
    

    # Frame cache for efficiency - only keep recently accessed frames
    frame_cache = {}
    cache_size = 20  # Keep last 20 frames in memory
    cache_order = []

    # Calculate bounding boxes for each person in each frame
    selected_persons = []
    n_frames, n_persons = all_pose_coords.shape[0], all_pose_coords.shape[1]
    all_bboxes = []
    for frame_idx in range(n_frames):
        frame_bboxes = []
        for person_idx in range(n_persons):
            # Get keypoints for current person
            keypoints = all_pose_coords[frame_idx, person_idx]
            valid_keypoints = keypoints[~np.isnan(keypoints).all(axis=1)]
            if len(valid_keypoints) > 0:
                # Calculate bounding box
                x_min, y_min = np.min(valid_keypoints, axis=0)
                x_max, y_max = np.max(valid_keypoints, axis=0)
                frame_bboxes.append((x_min, y_min, x_max, y_max))
            else:
                frame_bboxes.append((np.nan, np.nan, np.nan, np.nan))  # No valid bounding box for this person
        all_bboxes.append(frame_bboxes)
    all_bboxes = np.array(all_bboxes)  # Shape: (Nframes, Npersons, 4)
    
    # Create figure, axes, and slider
    first_frame = get_frame(0)
    frame_height, frame_width = first_frame.shape[:2]
    is_vertical = frame_height > frame_width
    if is_vertical:
        fig_height = frame_height / 250  # For vertical videos
    else:
        fig_height = max(frame_height / 300, 6)  # For horizontal videos
    fig = plt.figure(figsize=(8, fig_height), num=f'Select the persons to analyze in the desired order')
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    video_axes_height = 0.7 if is_vertical else 0.6
    ax_video = plt.axes([0.1, 0.2, 0.8, video_axes_height])
    ax_video.axis('off')
    ax_video.set_facecolor(BACKGROUND_COLOR)    

    # First image
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    rects, annotations = [], []
    for person_idx, bbox in enumerate(all_bboxes[0]):
        if ~np.isnan(bbox).any():
            x_min, y_min, x_max, y_max = bbox.astype(int)
            rect = plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1, edgecolor=LINE_UNSELECTED_COLOR, facecolor=UNSELECTED_COLOR,
                linestyle='-', path_effects=[patheffects.withSimplePatchShadow()], zorder=2
            ) 
            ax_video.add_patch(rect)
            annotation = ax_video.text(
                x_min, y_min - 10, f'{person_idx}', color=LINE_UNSELECTED_COLOR, fontsize=7, fontweight='normal',
                bbox=dict(facecolor=UNSELECTED_COLOR, edgecolor=LINE_UNSELECTED_COLOR, boxstyle='square,pad=0.3', path_effects=[patheffects.withSimplePatchShadow()]), zorder=3
            )
            rects.append(rect)
            annotations.append(annotation)
    img_plot = ax_video.imshow(frame_rgb)

    # Slider
    ax_slider = plt.axes([ax_video.get_position().x0, ax_video.get_position().y0-0.05, ax_video.get_position().width, 0.04])
    ax_slider.set_facecolor(BACKGROUND_COLOR)
    frame_slider = Slider(
        ax=ax_slider,
        label='',
        valmin=0,
        valmax=len(all_pose_coords)-1,
        valinit=0,
        valstep=1,
        valfmt=None 
    )
    frame_slider.poly.set_edgecolor(SLIDER_EDGE_COLOR)
    frame_slider.poly.set_facecolor(SLIDER_COLOR)
    frame_slider.poly.set_linewidth(1)
    frame_slider.valtext.set_visible(False)


    # Status text and OK button
    ax_status = plt.axes([ax_video.get_position().x0, ax_video.get_position().y0-0.1, 2*ax_video.get_position().width/3, 0.04])
    ax_status.axis('off')
    status_text = ax_status.text(0.0, 0.5, f"Selected: None", color='black', fontsize=10)

    ax_button = plt.axes([ax_video.get_position().x0 + 3*ax_video.get_position().width/4, ax_video.get_position().y0-0.1, ax_video.get_position().width/4, 0.04])
    ok_button = Button(ax_button, 'OK', color=BACKGROUND_COLOR)
    

    # Connect events
    frame_slider.on_changed(update_frame)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    ok_button.on_clicked(on_ok)

    plt.show()

    return selected_persons


def compute_floor_line(trc_data, score_data, keypoint_names = ['LBigToe', 'RBigToe'], toe_speed_below = 7, score_threshold=0.3):
    '''
    Compute the floor line equation, angle, and direction
    from the feet keypoints when they have zero speed.

    N.B.: Y coordinates point downwards

    INPUTS:
    - trc_data: pd.DataFrame. The trc data
    - keypoint_names: list of str. The names of the keypoints to use
    - toe_speed_below: float. The speed threshold (px/frame) below which the keypoints are considered as not moving

    OUTPUT:
    - angle: float. The angle of the floor line in radians
    - xy_origin: list. The origin of the floor line
    - gait_direction: float. Left if < 0, 'right' otherwise
    '''

    # Retrieve zero-speed coordinates for the foot
    low_speeds_X, low_speeds_Y = [], []
    gait_direction_val = []
    for kpt in keypoint_names:
        # Remove frames without data
        trc_data_kpt = trc_data[kpt].iloc[:,:2]
        score_data_kpt = score_data[kpt]
        start, end = indices_of_first_last_non_nan_chunks(score_data_kpt, chunk_choice_method='all')
        trc_data_kpt_trim = trc_data_kpt.iloc[start:end].reset_index(drop=True)
        score_data_kpt_trim = score_data_kpt.iloc[start:end].reset_index(drop=True)

        # Compute euclidean speed
        speeds = np.linalg.norm(trc_data_kpt_trim.diff(), axis=1)

        # Remove speeds with low confidence
        speeds = np.where(score_data_kpt_trim>score_threshold, speeds, np.nan)

        # Get coordinates with low speeds, high
        low_speeds_coords = trc_data_kpt_trim[speeds<toe_speed_below]
        low_speeds_coords = low_speeds_coords[low_speeds_coords!=0]

        low_speeds_X_kpt = low_speeds_coords.iloc[:,0].tolist()
        low_speeds_X += low_speeds_X_kpt
        low_speeds_Y += low_speeds_coords.iloc[:,1].tolist()

        # gait direction (between [-1,1])
        X_trend_val = np.polyfit(range(len(low_speeds_X_kpt)), low_speeds_X_kpt, 1)[0]
        gait_direction_kpt = X_trend_val * len(low_speeds_X_kpt) / (np.max(low_speeds_X_kpt) - np.min(low_speeds_X_kpt))
        gait_direction_val.append(gait_direction_kpt)

    # Fit a line to the zero-speed coordinates
    floor_line = np.polyfit(low_speeds_X, low_speeds_Y, 1) # (slope, intercept)
    angle = -np.arctan(floor_line[0]) # angle of the floor line in radians
    xy_origin = [0, floor_line[1]] # origin of the floor line

    # Gait direction
    gait_direction = np.mean(gait_direction_val)
    
    return angle, xy_origin, gait_direction


def get_distance_from_camera(perspective_value=10, perspective_unit='distance_m', calib_file =None, height_px=1, height_m=1, cam_width=1, cam_height=1):
    '''
    Compute the distance between the camera and the person based on the chosen perspective unit.

    INPUTS:
    - perspective_value: Value associated with the chosen perspective unit.
    - perspective_unit: Unit used to compute the distance. Can be 'distance_m', 'f_px', 'fov_rad', 'fov_deg', or 'from_calib'.
    - calib_file: Path to the toml calibration file.
    - height_px: Height of the person in pixels.
    - height_m: Height of the first person in meters.
    - cam_width: Width of the camera frame in pixels.
    - cam_height: Height of the camera frame in pixels.
    
    OUTPUTS:
    - distance_m: Distance between the camera and the person in meters.
    '''

    if perspective_unit == 'from_calib':
        if not calib_file:
            perspective_unit = 'distance_m'
            distance_m = 10.0
            logging.warning(f'No calibration file provided. Using a default distance of {distance_m} m between the camera and the person to convert px to meters.')
        else:
            calib_params_dict = retrieve_calib_params(calib_file)
            f_px = calib_params_dict['K'][0][0][0]
            distance_m = f_px / height_px * height_m
    elif perspective_unit == 'distance_m':
        distance_m = perspective_value
    elif perspective_unit == 'f_px':
        f_px = perspective_value
        distance_m = f_px / height_px * height_m
    elif perspective_unit == 'fov_rad':
        fov_rad = perspective_value
        f_px = max(cam_width, cam_height)/2 / np.tan(fov_rad / 2)
        distance_m = f_px / height_px * height_m
    elif perspective_unit == 'fov_deg':
        fov_rad = np.radians(perspective_value)
        f_px = max(cam_width, cam_height)/2 / np.tan(fov_rad / 2)
        distance_m = f_px / height_px * height_m

    return distance_m


def get_floor_params(floor_angle='auto', xy_origin=['auto'], 
                     calib_file=None, height_px=1, height_m=1, 
                     fps=30, trc_data=pd.DataFrame(), score_data=pd.DataFrame(), toe_speed_below=1, score_threshold=0.5, 
                     cam_width=1, cam_height=1):
    '''
    Compute the floor angle and the xy_origin based on calibration file, kinematics, or user input.

    INPUTS:
    - floor_angle: Method to compute the floor angle. Can be 'auto', 'from_calib', 'from_kinematics', or a numeric value in degrees.
    - xy_origin: Method to compute the xy_origin. Can be ['auto'], ['from_calib'], ['from_kinematics'], or a list of two numeric values in pixels [cx, cy].
    - calib_file: Path to a toml calibration file.
    - height_px: Height of the person in pixels.
    - height_m: Height of the first person in meters.
    - fps: Framerate of the video in frames per second. Used if estimating floor line from kinematics.
    - trc_data: DataFrame containing the pose data in pixels for one person. Used if estimating floor line from kinematics.
    - score_data: DataFrame containing the keypoint scores for one person. Used if estimating floor line from kinematics.
    - toe_speed_below: Speed below which the foot is considered to be stationary, in m/s. Used if estimating floor line from kinematics.
    - score_threshold: Minimum average keypoint score to consider a frame for floor line estimation.
    - cam_width: Width of the camera frame in pixels. Used if failed to estimate floor line from kinematics.
    - cam_height: Height of the camera frame in pixels. Used if failed to estimate floor line from kinematics.

    OUTPUTS:
    - floor_angle_estim: Estimated floor angle in radians.
    - xy_origin_estim: Estimated xy_origin as a list of two numeric values in pixels [cx, cy].
    - gait_direction: Estimated gait direction. 'left' if < 0, 'right' otherwise.
    '''
    
    # Estimate floor angle from the calibration file
    if floor_angle == 'from_calib' or xy_origin == ['from_calib']:
        if not calib_file:
            if floor_angle == 'from_calib':
                floor_angle = 'auto'
            if  xy_origin == ['from_calib']:
                xy_origin = ['auto']
            logging.warning(f'No calibration file provided. Estimating floor angle and xy_origin from the pose of the first selected person.')
        else:
            calib_params_dict = retrieve_calib_params(calib_file)

            R90z = np.array([[0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]])
            R270x = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0]])

            R_cam = cv2.Rodrigues(calib_params_dict['R'][0])[0]
            T_cam = np.array(calib_params_dict['T'][0])
            R_world, T_world = world_to_camera_persp(R_cam, T_cam)
            Rfloory = R90z.T @ R_world @ R270x.T
            T_world = R90z.T @ T_world
            floor_angle_calib = np.arctan2(Rfloory[0,2], Rfloory[0,0])
            
            cu = calib_params_dict['K'][0][0][2]
            cv = calib_params_dict['K'][0][1][2]
            cx = 0.0
            cy = cv + T_world[2]* height_px / height_m
            xy_origin_calib = [cx, cy]

    # Estimate xy_origin from the line formed by the toes when they are on the ground (where speed = 0)
    px_per_m = height_px/height_m
    toe_speed_below_px_frame = toe_speed_below * px_per_m / fps # speed below which the foot is considered to be stationary
    try:
        if all(key in trc_data for key in ['LBigToe', 'RBigToe']):
            floor_angle_kin, xy_origin_kin, gait_direction = compute_floor_line(trc_data, score_data, keypoint_names=['LBigToe', 'RBigToe'], toe_speed_below=toe_speed_below_px_frame, score_threshold=score_threshold)
        else:
            floor_angle_kin, xy_origin_estim, gait_direction = compute_floor_line(trc_data, score_data, keypoint_names=['LAnkle', 'RAnkle'], toe_speed_below=toe_speed_below_px_frame, score_threshold=score_threshold)
            xy_origin_kin[1] = xy_origin_kin[1] + 0.13*px_per_m # approx. height of the ankle above the floor
            logging.warning(f'The RBigToe and LBigToe are missing from your pose estimation model. Using ankles - 13 cm to compute the floor line.')
    except:
        floor_angle_kin = 0
        xy_origin_kin = cam_width/2, cam_height/2
        gait_direction = 1
        logging.warning(f'Could not estimate the floor angle, xy_origin, and visible from person {0}. Make sure that the full body is visible. Using floor angle = 0°, xy_origin = [{cam_width/2}, {cam_height/2}] px, and visible_side = right.')

    # Determine final floor angle estimation
    if floor_angle == 'from_calib':
        floor_angle_estim = floor_angle_calib
    elif floor_angle in ['auto', 'from_kinematics']:
        floor_angle_estim = floor_angle_kin
    else:
        try:
            floor_angle_estim =  np.radians(float(floor_angle))
        except:
            raise ValueError(f'Invalid floor_angle: {floor_angle}. Must be "auto", "from_calib", "from_kinematics", or a numeric value in degrees.')

    # Determine final xy_origin estimation
    if xy_origin == ['from_calib']:
        xy_origin_estim = xy_origin_calib
    elif xy_origin in [['auto'], ['from_kinematics']]:
        xy_origin_estim = xy_origin_kin
    else:
        try:
            xy_origin_estim = [float(v) for v in xy_origin]
        except:
            raise ValueError(f'Invalid xy_origin: {xy_origin}. Must be "auto", "from_calib", "from_kinematics", or a list of two numeric values in pixels.')
    
    return floor_angle_estim, xy_origin_estim, gait_direction


def convert_px_to_meters(Q_coords_kpt, first_person_height, height_px, distance_m, cam_width, cam_height, cx, cy, floor_angle, visible_side='none'):
    '''
    Convert pixel coordinates to meters.
    Corrects for floor angle, floor level, and depth perspective errors.

    INPUTS:
    - Q_coords_kpt: pd.DataFrame. The xyz coordinates of a keypoint in pixels, with z filled with zeros
    - first_person_height: float. The height of the person in meters
    - height_px: float. The height of the person in pixels
    - cam_width: float. The width of the camera frame in pixels
    - cam_height: float. The height of the camera frame in pixels
    - distance_m: float. The distance between the camera and the person in meters
    - cx, cy: float. The origin of the image in pixels
    - floor_angle: float. The angle of the floor in radians
    - visible_side: str. The side of the person that is visible ('right', 'left', 'front', 'back', 'none')

    OUTPUT:
    - Q_coords_kpt_m: pd.DataFrame. The XYZ coordinates of a keypoint in meters
    '''

    # u,v coordinates
    u = Q_coords_kpt.iloc[:,0]
    v = Q_coords_kpt.iloc[:,1]
    cu = cam_width / 2
    cv = cam_height / 2

    # Normative Z coordinates
    marker_name = Q_coords_kpt.columns[0]
    if 'marker_Z_positions' in globals() and visible_side!='none' and marker_name in marker_Z_positions[visible_side].keys():
        Z = u.copy()
        Z[:] = marker_Z_positions[visible_side][marker_name]
    else:
        Z = np.zeros_like(u)

    ## Compute X and Y coordinates in meters
    # X =   first_person_height / height_px * (u-cu)
    # Y = - first_person_height / height_px * (v-cv)
    ## With floor angle and level correction:
    # X =   first_person_height / height_px * ((u-cx)*np.cos(floor_angle) + (v-cy)*np.sin(floor_angle))
    # Y = - first_person_height / height_px * ((v-cy)*np.cos(floor_angle) + (u-cx)*np.sin(floor_angle))
    ## With floor angle and level correction, and depth perspective correction:
    scaling_factor = first_person_height / height_px
    X = scaling_factor * ( \
            ((u-cx) - Z/distance_m * (u-cu)) * np.cos(floor_angle) + \
            ((v-cy) - Z/distance_m * (v-cv)) * np.sin(floor_angle) )
    Y = - scaling_factor * ( \
            ((v-cy) - Z/distance_m * (v-cv)) * np.cos(floor_angle) - \
            ((u-cx) - Z/distance_m * (u-cu)) * np.sin(floor_angle) )

    # Assemble results
    Q_coords_kpt_m = pd.DataFrame(np.array([X, Y, Z]).T, columns=Q_coords_kpt.columns)

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
        
    ⚠ Warning ⚠ 
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
    video_dir = Path(config_dict.get('base').get('video_dir'))
    
    nb_persons_to_detect = config_dict.get('base').get('nb_persons_to_detect')
    if nb_persons_to_detect != 'all': 
        try:
            nb_persons_to_detect = int(nb_persons_to_detect)
            if nb_persons_to_detect < 1:
                logging.warning('nb_persons_to_detect must be "all" or > 1. Detecting all persons instead.')
                nb_persons_to_detect = 'all'
        except:
            logging.warning('nb_persons_to_detect must be "all" or an integer. Detecting all persons instead.')
            nb_persons_to_detect = 'all'

    person_ordering_method = config_dict.get('base').get('person_ordering_method')

    first_person_height = config_dict.get('base').get('first_person_height')
    visible_side = config_dict.get('base').get('visible_side')
    if isinstance(visible_side, str): visible_side = [visible_side]

    # Pose from file
    load_trc_px = config_dict.get('base').get('load_trc_px')
    if load_trc_px == '': load_trc_px = None
    else: load_trc_px = Path(load_trc_px).resolve()
    compare = config_dict.get('base').get('compare')

    # Webcam settings
    webcam_id =  config_dict.get('base').get('webcam_id')
    input_size = config_dict.get('base').get('input_size')

    # Output settings    
    show_realtime_results = config_dict.get('base').get('show_realtime_results')
    save_vid = config_dict.get('base').get('save_vid')
    save_img = config_dict.get('base').get('save_img')
    save_pose = config_dict.get('base').get('save_pose')
    calculate_angles = config_dict.get('base').get('calculate_angles')
    save_angles = config_dict.get('base').get('save_angles')

    # Pose_advanced settings
    slowmo_factor = config_dict.get('pose').get('slowmo_factor')
    pose_model = config_dict.get('pose').get('pose_model')
    mode = config_dict.get('pose').get('mode')
    det_frequency = config_dict.get('pose').get('det_frequency')
    backend = config_dict.get('pose').get('backend')
    device = config_dict.get('pose').get('device')
    tracking_mode = config_dict.get('pose').get('tracking_mode')
    if tracking_mode == 'deepsort':
        from deep_sort_realtime.deepsort_tracker import DeepSort
        deepsort_params = config_dict.get('pose').get('deepsort_params')
        try:
            deepsort_params = ast.literal_eval(deepsort_params)
        except: # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
            deepsort_params = deepsort_params.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/',':/').replace('":"\\',':\\')
            deepsort_params = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', deepsort_params) # changes "[640", "640]" to [640,640]
            deepsort_params = json.loads(deepsort_params)
        deepsort_tracker = DeepSort(**deepsort_params)
        deepsort_tracker.tracker.tracks.clear()

    keypoint_likelihood_threshold = config_dict.get('pose').get('keypoint_likelihood_threshold')
    average_likelihood_threshold = config_dict.get('pose').get('average_likelihood_threshold')
    keypoint_number_threshold = config_dict.get('pose').get('keypoint_number_threshold')
    max_distance = config_dict.get('pose').get('max_distance', None)

    # Pixel to meters conversion
    to_meters = config_dict.get('px_to_meters_conversion').get('to_meters')
    make_c3d = config_dict.get('px_to_meters_conversion').get('make_c3d')
    save_calib = config_dict.get('px_to_meters_conversion').get('save_calib')
    # Correct perspective effects
    perspective_value = config_dict.get('px_to_meters_conversion', {}).get('perspective_value', 10.0)
    perspective_unit = config_dict.get('px_to_meters_conversion', {}).get('perspective_unit', 'distance_m')
    # Calibration from person height
    floor_angle = config_dict.get('px_to_meters_conversion').get('floor_angle', 'auto') # 'auto' or float
    xy_origin = config_dict.get('px_to_meters_conversion').get('xy_origin', ['auto']) # ['auto'] or [x, y]    
    # Calibration from file
    calib_file = config_dict.get('px_to_meters_conversion').get('calib_file')
    if calib_file == '': 
        calib_file = None
    else: 
        calib_file = Path(calib_file).resolve()
        if not calib_file.is_file():
            raise FileNotFoundError(f'Error: Could not find calibration file {calib_file}. Check that the file exists.')

    # Angles advanced settings
    display_angle_values_on = config_dict.get('angles').get('display_angle_values_on')
    fontSize = config_dict.get('angles').get('fontSize')
    thickness = 1 if fontSize < 0.8 else 2
    joint_angle_names = config_dict.get('angles').get('joint_angles')
    segment_angle_names = config_dict.get('angles').get('segment_angles')
    angle_names = joint_angle_names + segment_angle_names
    angle_names = [angle_name.lower() for angle_name in angle_names]
    flip_left_right = config_dict.get('angles').get('flip_left_right')
    correct_segment_angles_with_floor_angle = config_dict.get('angles').get('correct_segment_angles_with_floor_angle')

    # Post-processing settings
    interpolate = config_dict.get('post-processing').get('interpolate')    
    interp_gap_smaller_than = config_dict.get('post-processing').get('interp_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('post-processing').get('fill_large_gaps_with')
    sections_to_keep = config_dict.get('post-processing').get('sections_to_keep')
    min_chunk_size = config_dict.get('post-processing').get('min_chunk_size')
    do_filter = config_dict.get('post-processing').get('filter')
    handle_LR_swap = config_dict.get('post-processing').get('handle_LR_swap', False)
    reject_outliers = config_dict.get('post-processing').get('reject_outliers', False)
    show_plots = config_dict.get('post-processing').get('show_graphs')
    save_plots = config_dict.get('post-processing').get('save_graphs')
    filter_type = config_dict.get('post-processing').get('filter_type')
    butterworth_filter_order = config_dict.get('post-processing').get('butterworth', {}).get('order')
    butterworth_filter_cutoff = config_dict.get('post-processing').get('butterworth', {}).get('cut_off_frequency')
    gcv_filter_cutoff = config_dict.get('post-processing').get('gcv_spline', {}).get('gcv_cut_off_frequency')
    gcv_smoothing_factor = config_dict.get('post-processing').get('gcv_spline', {}).get('gcv_smoothing_factor')
    kalman_filter_trust_ratio = config_dict.get('post-processing').get('kalman', {}).get('trust_ratio')
    kalman_filter_smooth = config_dict.get('post-processing').get('kalman', {}).get('smooth')
    gaussian_filter_kernel = config_dict.get('post-processing').get('gaussian', {}).get('sigma_kernel')
    loess_filter_kernel = config_dict.get('post-processing').get('loess', {}).get('nb_values_used')
    median_filter_kernel = config_dict.get('post-processing').get('median', {}).get('kernel_size')
    butterworthspeed_filter_order = config_dict.get('post-processing').get('butterworth_on_speed', {}).get('order')
    butterworthspeed_filter_cutoff = config_dict.get('post-processing').get('butterworth_on_speed', {}).get('cut_off_frequency')

    # Create output directories
    if video_file == "webcam":
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file_stem = f'webcam_{current_date}'
        output_dir_name = f'{video_file_stem}_Sports2D'
        video_file_path = result_dir / output_dir_name / f'webcam_{current_date}_raw.mp4'
    else:
        video_file_stem = video_file.stem
        output_dir_name = f'{video_file_stem}_Sports2D'    
        video_file_path = video_dir / video_file
    output_dir = result_dir / output_dir_name
    plots_output_dir = output_dir / f'{output_dir_name}_graphs'
    img_output_dir = output_dir / f'{output_dir_name}_img'
    vid_output_path = output_dir / f'{output_dir_name}.mp4'
    pose_output_path = output_dir / f'{output_dir_name}_px.trc'
    pose_output_path_m = output_dir / f'{output_dir_name}_m.trc'
    angles_output_path = output_dir / f'{output_dir_name}_angles.mot'
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_img:
        img_output_dir.mkdir(parents=True, exist_ok=True)
    if save_plots:
        plots_output_dir.mkdir(parents=True, exist_ok=True)

    # Inverse kinematics settings
    do_ik = config_dict.get('kinematics').get('do_ik')
    use_augmentation = config_dict.get('kinematics').get('use_augmentation')
    participant_masses = config_dict.get('kinematics').get('participant_mass')
    participant_masses = participant_masses if isinstance(participant_masses, list) else [participant_masses]
    feet_on_floor = config_dict.get('kinematics').get('feet_on_floor')
    fastest_frames_to_remove_percent = config_dict.get('kinematics').get('fastest_frames_to_remove_percent')
    large_hip_knee_angles = config_dict.get('kinematics').get('large_hip_knee_angles')
    trimmed_extrema_percent = config_dict.get('kinematics').get('trimmed_extrema_percent')
    close_to_zero_speed_px = config_dict.get('kinematics').get('close_to_zero_speed_px')
    close_to_zero_speed_m = config_dict.get('kinematics').get('close_to_zero_speed_m')
    # Create a Pose2Sim dictionary and fill in missing keys
    recursivedict = lambda: defaultdict(recursivedict)
    Pose2Sim_config_dict = recursivedict()
    if do_ik or use_augmentation:
        try:
            if use_augmentation:
                from Pose2Sim.markerAugmentation import augment_markers_all
            if do_ik:
                from Pose2Sim.kinematics import kinematics_all
        except ImportError:
            logging.error("OpenSim package is not installed. Please install it to use inverse kinematics or marker augmentation features (see 'Full install' section of the documentation).")
            raise ImportError("OpenSim package is not installed. Please install it to use inverse kinematics or marker augmentation features (see 'Full install' section of the documentation).")
        
        # Fill Pose2Sim dictionary (height and mass will be filled later)
        Pose2Sim_config_dict['project']['project_dir'] = str(output_dir)
        Pose2Sim_config_dict['markerAugmentation']['make_c3d'] = make_c3d
        Pose2Sim_config_dict['kinematics'] = config_dict.get('kinematics')
        # Temporarily recreate Pose2Sim file hierarchy
        pose3d_dir = Path(output_dir) / 'pose-3d'
        pose3d_dir.mkdir(parents=True, exist_ok=True)
        kinematics_dir = Path(output_dir) / 'kinematics'
        kinematics_dir.mkdir(parents=True, exist_ok=True)
    
    if do_filter:
        Pose2Sim_config_dict['personAssociation']['handle_LR_swap'] = handle_LR_swap
        Pose2Sim_config_dict['filtering']['reject_outliers'] = reject_outliers
        Pose2Sim_config_dict['filtering']['filter'] = do_filter
        Pose2Sim_config_dict['filtering']['type'] = filter_type
        Pose2Sim_config_dict['filtering']['gcv_spline']['cut_off_frequency'] = gcv_filter_cutoff
        Pose2Sim_config_dict['filtering']['gcv_spline']['smoothing_factor'] = gcv_smoothing_factor
        Pose2Sim_config_dict['filtering']['butterworth']['cut_off_frequency'] = butterworth_filter_cutoff
        Pose2Sim_config_dict['filtering']['butterworth']['order'] = butterworth_filter_order
        Pose2Sim_config_dict['filtering']['kalman']['trust_ratio'] = kalman_filter_trust_ratio
        Pose2Sim_config_dict['filtering']['kalman']['smooth'] = kalman_filter_smooth
        Pose2Sim_config_dict['filtering']['gaussian']['sigma_kernel'] = gaussian_filter_kernel
        Pose2Sim_config_dict['filtering']['loess']['nb_values_used'] = loess_filter_kernel
        Pose2Sim_config_dict['filtering']['median']['kernel_size'] = median_filter_kernel
        Pose2Sim_config_dict['filtering']['butterworth_on_speed']['order'] = butterworthspeed_filter_order
        Pose2Sim_config_dict['filtering']['butterworth_on_speed']['cut_off_frequency'] = butterworthspeed_filter_cutoff


    # Set up video capture
    if video_file == "webcam":
        cap, out_vid, cam_width, cam_height, fps = setup_webcam(webcam_id, vid_output_path, input_size)
        frame_rate = fps
        frame_range = [0,sys.maxsize]
        frame_iterator = range(*frame_range)
        logging.warning('Webcam input: the framerate may vary. If results are filtered, Sports2D will use the average framerate as input.')
    else:
        cap, out_vid, cam_width, cam_height, fps = setup_video(video_file_path, vid_output_path, save_vid)
        fps *= slowmo_factor
        start_time = get_start_time_ffmpeg(video_file_path)
        frame_range = [int((time_range[0]-start_time) * frame_rate), int((time_range[1]-start_time) * frame_rate)] if time_range else [0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_iterator = tqdm(range(*frame_range)) # use a progress bar
    if show_realtime_results:
        try: 
            screen_width, screen_height = get_screen_size()
            display_width, display_height = calculate_display_size(cam_width, cam_height, screen_width, screen_height, margin=50)
            cv2.namedWindow(f'{video_file} Sports2D', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'{video_file} Sports2D', display_width, display_height)
        except: # if Pose2Sim < v0.10.29
            cv2.namedWindow(f'{video_file} Sports2D', cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

    # Select the appropriate model based on the model_type
    logging.info('\nEstimating pose...')
    pose_model_name = pose_model
    pose_model, ModelClass, mode = setup_model_class_mode(pose_model, mode, config_dict)
    
    # Select device and backend
    backend, device = setup_backend_device(backend=backend, device=device)

    # Skip pose estimation or set it up:
    if load_trc_px:
        if not '_px' in str(load_trc_px): 
            logging.error(f'\n{load_trc_px} file needs to be in px, not in meters.')
        logging.info(f'\nUsing a pose file instead of running pose estimation and tracking: {load_trc_px}.')
        # Load pose file in px
        Q_coords, _, time_col, keypoints_names, _ = read_trc(load_trc_px)
        t0 = time_col[0]
        tf = time_col.iloc[-1]
        keypoints_ids = [i for i in range(len(keypoints_names))]
        keypoints_all, scores_all = load_pose_file(Q_coords)

        for pre, _, node in RenderTree(pose_model):
            if node.name in keypoints_names:
                node.id = keypoints_names.index(node.name)
        if time_range:
            frame_range = [abs(time_col - time_range[0]).idxmin(), abs(time_col - time_range[1]).idxmin()+1]
        else:
            frame_range = [0, len(Q_coords)]
        frame_iterator = tqdm(range(*frame_range))
    
    else:
        # Retrieve keypoint names from model
        keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if node.id!=None]
        keypoints_names = [node.name for _, _, node in RenderTree(pose_model) if node.id!=None]
        t0 = 0
        tf = (cap.get(cv2.CAP_PROP_FRAME_COUNT)-1) / fps if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else float('inf')

        # Set up pose tracker
        try:
            pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
        except: # for multi-threading
            try:
                import time
                time.sleep(3)
                pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
            except:
                logging.error('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')
                raise ValueError('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')
        logging.info(f'Persons detection is run every {det_frequency} frames (pose estimation is run at every frame). Tracking is done with {tracking_mode}.')
        
        if tracking_mode == 'deepsort': 
            logging.info(f'Deepsort parameters: {deepsort_params}.')
        if tracking_mode not in ['deepsort', 'sports2d']:
            logging.warning(f"Tracking mode {tracking_mode} is not implemented. 'sports2d' is recommended.")
        logging.info(f'{"All persons are" if nb_persons_to_detect=="all" else f"{nb_persons_to_detect} persons are" if nb_persons_to_detect>1 else "1 person is"} analyzed. Person ordering method is {person_ordering_method}.')
        logging.info(f"{keypoint_likelihood_threshold=}, {average_likelihood_threshold=}, {keypoint_number_threshold=}")

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
                logging.warning(f"Skipping {ang_name} angle computation because at least one of the following keypoints is not provided by the pose estimation model: {ang_params[0]}.")


    #%% ==================================================
    # Process video or webcam feed
    # ====================================================
    logging.info(f"\nProcessing video stream...")
    # logging.info(f"{'Video, ' if save_vid else ''}{'Images, ' if save_img else ''}{'Pose, ' if save_pose else ''}{'Angles ' if save_angles else ''}{'and ' if save_angles or save_img or save_pose or save_vid else ''}Logs will be saved in {result_dir}.")
    all_frames_X, all_frames_X_flipped, all_frames_Y, all_frames_scores, all_frames_angles = [], [], [], [], []
    frame_processing_times = []
    frame_count = 0
    first_frame = max(int(t0 * fps), frame_range[0])
    last_frame = min(int(tf * fps), frame_range[1]-1)
    if first_frame >= last_frame:
        logging.error('Error: No frames to process. Check that your time_range is coherent with the video duration.')
        raise ValueError('Error: No frames to process. Check that your time_range is coherent with the video duration.')
    
    while cap.isOpened():
        # Skip to the starting frame
        if frame_count < first_frame:
            cap.read()
            frame_count += 1
            continue

        for frame_nb in frame_iterator:
            start_time = datetime.now()
            success, frame = cap.read()
            frame_count += 1

            # If frame not grabbed
            if not success:
                logging.warning(f"Failed to grab frame {frame_count-1}.")
                if save_pose:
                    all_frames_X.append([])
                    all_frames_Y.append([])
                    all_frames_scores.append([])
                if save_angles:
                    all_frames_angles.append([])
                continue

            # Retrieve pose or Estimate pose and track people
            if load_trc_px: 
                if frame_nb >= len(keypoints_all):
                    break
                keypoints = keypoints_all[frame_nb]
                scores = scores_all[frame_nb]
            else: 
                # Save video on the fly if the input is a webcam
                if video_file == "webcam":
                    out_vid.write(frame)

                # Detect poses
                keypoints, scores = pose_tracker(frame)

                # Non maximum suppression (at pose level, not detection, and only using likely keypoints)
                frame_shape = frame.shape
                mask_scores = np.mean(scores, axis=1) > 0.2

                likely_keypoints = np.where(mask_scores[:, np.newaxis, np.newaxis], keypoints, np.nan)
                likely_scores = np.where(mask_scores[:, np.newaxis], scores, np.nan)
                likely_bboxes = bbox_xyxy_compute(frame_shape, likely_keypoints, padding=0)
                score_likely_bboxes = np.nanmean(likely_scores, axis=1)

                valid_indices = np.where(~np.isnan(score_likely_bboxes))[0]
                if len(valid_indices) > 0:
                    valid_bboxes = likely_bboxes[valid_indices]
                    valid_scores = score_likely_bboxes[valid_indices]
                    keep_valid = nms(valid_bboxes, valid_scores, nms_thr=0.45)
                    keep = valid_indices[keep_valid]
                else:
                    keep = []
                keypoints, scores = likely_keypoints[keep], likely_scores[keep]

                # # Debugging: display detected keypoints on the frame
                # colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128)]
                # bboxes = likely_bboxes[keep]
                # for person_idx in range(len(keypoints)):
                #     for kpt_idx, kpt in enumerate(keypoints[person_idx]):
                #         if not np.isnan(kpt).any():
                #             cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, colors[person_idx%len(colors)], -1)
                #     if not np.isnan(bboxes[person_idx]).any():
                #         cv2.rectangle(frame, (int(bboxes[person_idx][0]), int(bboxes[person_idx][1])), (int(bboxes[person_idx][2]), int(bboxes[person_idx][3])), colors[person_idx%len(colors)], 1)
                #     cv2.imshow(f'{video_file} Sports2D', frame)

                # Track poses across frames
                if tracking_mode == 'deepsort':
                    keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_count)
                if tracking_mode == 'sports2d': 
                    if 'prev_keypoints' not in locals(): prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores, max_dist=max_distance)
                else:
                    pass
                
                # # Debugging: display detected keypoints on the frame
                # colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128)]
                # for person_idx in range(len(keypoints)):
                #     for kpt_idx, kpt in enumerate(keypoints[person_idx]):
                #         if not np.isnan(kpt).any():
                #             cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, colors[person_idx%len(colors)], -1)
                #         # if not np.isnan(bboxes[person_idx]).any():
                #         #     cv2.rectangle(frame, (int(bboxes[person_idx][0]), int(bboxes[person_idx][1])), (int(bboxes[person_idx][2]), int(bboxes[person_idx][3])), colors[person_idx%len(colors)], 1)
                #     cv2.imshow(f'{video_file} Sports2D', frame)
                # # if (cv2.waitKey(1) & 0xFF) == ord('q') or (cv2.waitKey(1) & 0xFF) == 27:
                # #     break
                # # input()
            
            # Process coordinates and compute angles
            valid_X, valid_Y, valid_scores = [], [], []
            valid_X_flipped, valid_angles_flipped, valid_angles = [], [], []
            for person_idx in range(len(keypoints)):
                if load_trc_px:
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

                    
                    
                    ## RECREATE KEYPOINTS, SCORES









                # Check whether the person is looking to the left or right
                if flip_left_right:
                    person_X_flipped = flip_left_right_direction(person_X, L_R_direction_idx, keypoints_names, keypoints_ids)
                else:
                    person_X_flipped = person_X.copy()
                
                # Add Neck and Hip if not provided
                new_keypoints_names, new_keypoints_ids = keypoints_names.copy(), keypoints_ids.copy()
                for kpt in ['Hip', 'Neck']:
                    if kpt not in new_keypoints_names:
                        person_X_flipped, person_Y, person_scores = add_neck_hip_coords(kpt, person_X_flipped, person_Y, person_scores, new_keypoints_ids, new_keypoints_names)
                        person_X, _, _ = add_neck_hip_coords(kpt, person_X, person_Y, person_scores, new_keypoints_ids, new_keypoints_names)
                        new_keypoints_names.append(kpt)
                        new_keypoints_ids.append(len(person_X_flipped)-1)

                # Compute angles
                if calculate_angles:
                    person_angles = []
                    for ang_name in angle_names:
                        ang_params = angle_dict.get(ang_name)
                        kpts = ang_params[0]
                        if not any(item not in new_keypoints_names for item in kpts):
                            ang = compute_angle(ang_name, person_X_flipped, person_Y, angle_dict, new_keypoints_ids, new_keypoints_names)
                        else:
                            ang = np.nan
                        person_angles.append(ang)
                    
                    # flip angles on the left side if flip_left_right false
                    if len(visible_side) <= person_idx:
                        visible_side += ['auto'] # set to 'auto' if list too short
                    if visible_side[person_idx] == 'left' and not flip_left_right:
                        person_angles_flipped = list(-np.array(person_angles))
                    else:
                        person_angles_flipped = person_angles.copy()

                    valid_angles.append(person_angles)
                    valid_angles_flipped.append(person_angles_flipped)
                    valid_X_flipped.append(person_X_flipped)
                valid_X.append(person_X)
                valid_Y.append(person_Y)
                valid_scores.append(person_scores)

            # Draw keypoints and skeleton
            if show_realtime_results:
                img = frame.copy()
                cv2.putText(img, f"Press 'q' to stop", (cam_width-int(600*fontSize), cam_height-20), cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (255,255,255), thickness+1, cv2.LINE_AA)
                cv2.putText(img, f"Press 'q' to stop", (cam_width-int(600*fontSize), cam_height-20), cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (0,0,255), thickness, cv2.LINE_AA)
                img = draw_bounding_box(img, valid_X, valid_Y, colors=colors, fontSize=fontSize, thickness=thickness)
                img = draw_keypts(img, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
                img = draw_skel(img, valid_X, valid_Y, pose_model)
                if calculate_angles:
                    img = draw_angles(img, valid_X, valid_Y, valid_angles_flipped, valid_X_flipped, new_keypoints_ids, new_keypoints_names, angle_names, display_angle_values_on=display_angle_values_on, colors=colors, fontSize=fontSize, thickness=thickness)
                cv2.imshow(f'{video_file} Sports2D', img)
                if (cv2.waitKey(1) & 0xFF) == ord('q') or (cv2.waitKey(1) & 0xFF) == 27:
                    break

                # # Debugging
                # img_output_path = img_output_dir / f'{video_file_stem}_frame{frame_nb:06d}.png'
                # cv2.imwrite(str(img_output_path), img)

            all_frames_X.append(np.array(valid_X))
            all_frames_X_flipped.append(np.array(valid_X_flipped))
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
        if save_vid or video_file == "webcam":
            out_vid.release()
            if video_file == "webcam":
                vid_output_path.absolute().rename(video_file_path)

        if show_realtime_results:
            cv2.destroyAllWindows()


    #%% ==================================================
    # Post-processing: Select persons, Interpolate, filter, and save pose and angles
    # ====================================================
    all_frames_X_homog = make_homogeneous(all_frames_X)
    all_frames_X_homog = all_frames_X_homog[...,new_keypoints_ids]
    if calculate_angles or save_angles:
        all_frames_X_flipped_homog = make_homogeneous(all_frames_X_flipped)
        all_frames_X_flipped_homog = all_frames_X_flipped_homog[...,new_keypoints_ids]
        all_frames_angles_homog = make_homogeneous(all_frames_angles)
    else:
        all_frames_X_flipped_homog = all_frames_X_flipped
        all_frames_angles_homog = all_frames_angles
    all_frames_Y_homog = make_homogeneous(all_frames_Y)
    all_frames_Y_homog = all_frames_Y_homog[...,new_keypoints_ids]
    all_frames_Z_homog = pd.DataFrame(np.zeros_like(all_frames_X_homog)[:,0,:], columns=new_keypoints_names)
    all_frames_scores_homog = make_homogeneous(all_frames_scores)
    all_frames_scores_homog = all_frames_scores_homog[...,new_keypoints_ids]

    frame_range = [0,frame_count] if video_file == 'webcam' else frame_range
    all_frames_time = pd.Series(np.linspace(frame_range[0]/fps, frame_range[1]/fps, frame_count-frame_range[0]), name='time')
    if load_trc_px:
        selected_persons = [0]
    else:
        # Select persons 
        nb_detected_persons = all_frames_scores_homog.shape[1]
        if nb_persons_to_detect == 'all':
            nb_persons_to_detect = all_frames_scores_homog.shape[1]
        if nb_detected_persons < nb_persons_to_detect:
            logging.warning(f'Less than the {nb_persons_to_detect} required persons were detected. Analyzing all {nb_detected_persons} persons.')
            nb_persons_to_detect = nb_detected_persons

        if person_ordering_method == 'on_click':
            selected_persons = get_personIDs_on_click(video_file_path, frame_range, all_frames_X_homog, all_frames_Y_homog)
            if len(selected_persons) == 0:
                logging.warning('No persons selected. Analyzing all detected persons.')
                selected_persons = list(range(nb_detected_persons))
            if len(selected_persons) != nb_persons_to_detect:
                logging.warning(f'You selected more (or less) than the required {nb_persons_to_detect} persons. "nb_persons_to_detect" will be set to {len(selected_persons)}.')
                nb_persons_to_detect = len(selected_persons)
        elif person_ordering_method == 'highest_likelihood':
            selected_persons = get_personIDs_with_highest_scores(all_frames_scores_homog, nb_persons_to_detect)
        elif person_ordering_method == 'first_detected':
            selected_persons = get_personIDs_in_detection_order(nb_persons_to_detect)
        elif person_ordering_method == 'last_detected':
            selected_persons = get_personIDs_in_detection_order(nb_persons_to_detect, reverse=True)
        elif person_ordering_method == 'largest_size':
            selected_persons = get_personIDs_with_largest_size(all_frames_X_homog, all_frames_Y_homog, nb_persons_to_detect=nb_persons_to_detect, vertical=False)
        elif person_ordering_method == 'smallest_size':
            selected_persons = get_personIDs_with_largest_size(all_frames_X_homog, all_frames_Y_homog, nb_persons_to_detect=nb_persons_to_detect, vertical=False, reverse=True)
        elif person_ordering_method == 'greatest_displacement':
            selected_persons = get_personIDs_with_greatest_displacement(all_frames_X_homog, all_frames_Y_homog, nb_persons_to_detect=nb_persons_to_detect, horizontal=True)
        elif person_ordering_method == 'least_displacement':
            selected_persons = get_personIDs_with_greatest_displacement(all_frames_X_homog, all_frames_Y_homog, nb_persons_to_detect=nb_persons_to_detect, horizontal=True, reverse=True)
        else:
            raise ValueError(f"Invalid person_ordering_method: {person_ordering_method}. Must be 'on_click', 'highest_likelihood', 'greatest_displacement', 'first_detected', or 'last_detected'.")
        logging.info(f'Reordered persons: IDs of persons {selected_persons} become {list(range(len(selected_persons)))}.')
    

    #%% ==================================================
    # Post-processing pose
    # ====================================================
    all_frames_X_processed, all_frames_X_flipped_processed, all_frames_Y_processed, all_frames_scores_processed, all_frames_angles_processed = all_frames_X_homog.copy(), all_frames_X_flipped_homog.copy(), all_frames_Y_homog.copy(), all_frames_scores_homog.copy(), all_frames_angles_homog.copy()
    if save_pose:
        logging.info('\nPost-processing pose:')
        # Process pose for each person
        trc_data, trc_data_unfiltered, score_data = [], [], []
        first_run_starts_everyone, last_run_ends_everyone = [], []
        for i, idx_person in enumerate(selected_persons):
            pose_path_person = pose_output_path.parent / (pose_output_path.stem + f'_person{i:02d}.trc')
            all_frames_X_person = pd.DataFrame(all_frames_X_processed[:,idx_person,:], columns=new_keypoints_names)
            all_frames_Y_person = pd.DataFrame(all_frames_Y_processed[:,idx_person,:], columns=new_keypoints_names)
            score_data.append(pd.DataFrame(all_frames_scores_processed[:,idx_person,:], columns=new_keypoints_names))
            if calculate_angles or save_angles:
                all_frames_X_flipped_person = pd.DataFrame(all_frames_X_flipped_processed[:,idx_person,:], columns=new_keypoints_names)

            # Interpolate
            if not interpolate:
                logging.info(f'- Person {i}: No interpolation.')
                all_frames_X_person_interp = all_frames_X_person
                all_frames_Y_person_interp = all_frames_Y_person
            else:
                logging.info(f'- Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames. Large gaps filled with {fill_large_gaps_with}.')
                all_frames_X_person_interp = all_frames_X_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])
                all_frames_Y_person_interp = all_frames_Y_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])

            # Find the first and last valid chunks of data
            first_run_starts, last_run_ends = [], []
            for col in all_frames_X_person.columns:
                first_run_start, last_run_end = indices_of_first_last_non_nan_chunks(all_frames_X_person_interp[col], min_chunk_size=min_chunk_size, chunk_choice_method=sections_to_keep)
                first_run_starts += [first_run_start]
                last_run_ends += [last_run_end]
            first_run_start_min, last_run_end_max = min(first_run_starts), max(last_run_ends)
            first_run_starts_everyone += [first_run_starts]
            last_run_ends_everyone += [last_run_ends]

            # Do not process person if no section of min_chunk_size valid frames in a row
            if (first_run_start_min, last_run_end_max) == (0,0):
                all_frames_X_processed[:,idx_person,:], all_frames_X_flipped_processed[:,idx_person,:], all_frames_Y_processed[:,idx_person,:] = np.nan, np.nan, np.nan
                columns=np.array([[c]*3 for c in all_frames_X_person.columns]).flatten()
                trc_data_i = pd.DataFrame(0, index=all_frames_X_person.index, columns=['time']+list(columns))
                trc_data_i['time'] = all_frames_time
                trc_data.append(trc_data_i)
                trc_data_unfiltered_i = trc_data_i.copy()
                trc_data_unfiltered.append(trc_data_unfiltered_i)
                logging.info(f'  Person {i}: Less than {min_chunk_size} valid frames in a row. Deleting person.')
                continue

            # Fill remaining gaps
            if fill_large_gaps_with.lower() == 'last_value':
                for col_id, col in enumerate(all_frames_X_person_interp.columns):
                    first_run_start, last_run_end = first_run_starts[col_id], last_run_ends[col_id]
                    for coord_df in [all_frames_X_person_interp, all_frames_Y_person_interp, all_frames_Z_homog]:
                        coord_df.loc[:first_run_start, col] = np.nan
                        coord_df.loc[last_run_end:, col] = np.nan
                        coord_df.loc[first_run_start:last_run_end, col] = coord_df.loc[first_run_start:last_run_end, col].ffill().bfill()
            elif fill_large_gaps_with.lower() == 'zeros':
                all_frames_X_person_interp.replace(np.nan, 0, inplace=True)
                all_frames_Y_person_interp.replace(np.nan, 0, inplace=True)

            # if handle_LR_swap:
            #     logging.info(f'Handling left-right swaps.')
            #     all_frames_X_person_interp = all_frames_X_person_interp.apply(LR_unswap, axis=0)
            #     all_frames_Y_person_interp = all_frames_Y_person_interp.apply(LR_unswap, axis=0)

            if reject_outliers:
                logging.info('Rejecting outliers with a Hampel filter.')
                all_frames_X_person_interp = all_frames_X_person_interp.apply(hampel_filter, axis=0, args = [round(7*frame_rate/30), 2])
                all_frames_Y_person_interp = all_frames_Y_person_interp.apply(hampel_filter, axis=0, args = [round(7*frame_rate/30), 2])

            if not do_filter:
                logging.info(f'No filtering.')
                all_frames_X_person_filt = all_frames_X_person_interp
                all_frames_Y_person_filt = all_frames_Y_person_interp
            else:
                if filter_type == ('butterworth' or 'butterworth_on_speed'):
                    cutoff = butterworth_filter_cutoff
                    if video_file == 'webcam':
                        if cutoff / (fps / 2) >= 1:
                            cutoff_old = cutoff
                            cutoff = fps/(2+0.001)
                            args = f'\n{cutoff_old:.1f} Hz cut-off framerate too large for a real-time framerate of {fps:.1f} Hz. Using a cut-off framerate of {cutoff:.1f} Hz instead.'
                            butterworth_filter_cutoff = cutoff
                    filt_type = 'Butterworth' if filter_type == 'butterworth' else 'Butterworth on speed'
                    args = f'{filt_type} filter, {butterworth_filter_order}th order, {butterworth_filter_cutoff} Hz.'
                    frame_rate = fps
                elif filter_type == 'gcv_spline':
                    args = f'GVC Spline filter, which automatically evaluates the best trade-off between smoothness and fidelity to data.'
                elif filter_type == 'kalman':
                    args = f'Kalman filter, trusting measurement {kalman_filter_trust_ratio} times more than the process matrix.'
                elif filter_type == 'gaussian':
                    args = f'Gaussian filter, Sigma kernel {gaussian_filter_kernel}.'
                elif filter_type == 'loess':
                    args = f'LOESS filter, window size of {loess_filter_kernel} frames.'
                elif filter_type == 'median':
                    args = f'Median filter, kernel of {median_filter_kernel}.'
                else:
                    logging.error(f"Invalid filter_type: {filter_type}. Must be 'butterworth', 'gcv_spline', 'kalman', 'gaussian', 'loess', or 'median'.")
                    raise ValueError(f"Invalid filter_type: {filter_type}. Must be 'butterworth', 'gcv_spline', 'kalman', 'gaussian', 'loess', or 'median'.")

                logging.info(f'Filtering with {args}')
                all_frames_X_person_filt = all_frames_X_person_interp.apply(filter1d, axis=0, args = [Pose2Sim_config_dict, filter_type, frame_rate])
                all_frames_Y_person_filt = all_frames_Y_person_interp.apply(filter1d, axis=0, args = [Pose2Sim_config_dict, filter_type, frame_rate])

            # Build TRC file
            trc_data_i = trc_data_from_XYZtime(all_frames_X_person_filt, all_frames_Y_person_filt, all_frames_Z_homog, all_frames_time)
            trc_data.append(trc_data_i)
            if not load_trc_px:
                make_trc_with_trc_data(trc_data_i, str(pose_path_person), fps=fps)
                logging.info(f'Pose in pixels saved to {pose_path_person.resolve()}.')

            # Plotting coordinates before and after interpolation and filtering
            columns_to_concat = []
            for kpt in range(len(all_frames_X_person.columns)):
                columns_to_concat.extend([all_frames_X_person.iloc[:,kpt], all_frames_Y_person.iloc[:,kpt], all_frames_Z_homog.iloc[:,kpt]])
            trc_data_unfiltered_i = pd.concat([all_frames_time] + columns_to_concat, axis=1)
            trc_data_unfiltered.append(trc_data_unfiltered_i)
            if not to_meters and (show_plots or save_plots):
                pw = pose_plots(trc_data_unfiltered_i, trc_data_i, i, show=show_plots)
                if save_plots:
                    for n, f in enumerate(pw.figure_handles):
                        dpi = pw.canvases[i].figure.dpi
                        f.set_size_inches(1280/dpi, 720/dpi)
                        title = pw.tabs.tabText(n)
                        plot_path = plots_output_dir / (pose_output_path.stem + f'_person{i:02d}_px_{title.replace(" ","_").replace("/","_")}.png')
                        f.savefig(plot_path, dpi=dpi, bbox_inches='tight')
                    logging.info(f'Pose plots (px) saved in {plots_output_dir}.')
                    
            all_frames_X_processed[:,idx_person,:], all_frames_Y_processed[:,idx_person,:] = all_frames_X_person_filt, all_frames_Y_person_filt
            if calculate_angles or save_angles:
                all_frames_X_flipped_processed[:,idx_person,:] = all_frames_X_flipped_person
                

        #%% Convert px to meters
        trc_data_m = []
        if to_meters and save_pose:
            logging.info('\nConverting pose to meters:')
            
            # Compute height in px of the first person
            height_px = compute_height(trc_data[0].iloc[:,1:], new_keypoints_names,
                                        fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, close_to_zero_speed=close_to_zero_speed_px, large_hip_knee_angles=large_hip_knee_angles, trimmed_extrema_percent=trimmed_extrema_percent)

            # Compute distance from camera to compensate for perspective effects
            distance_m = get_distance_from_camera(perspective_value=perspective_value, perspective_unit=perspective_unit, 
                                                  calib_file=calib_file, height_px=height_px, height_m=first_person_height, 
                                                  cam_width=cam_width, cam_height=cam_height)

            # Compute floor angle and xy_origin to compensate for camera horizon and person position
            floor_angle_estim, xy_origin_estim, gait_direction = get_floor_params(floor_angle=floor_angle, xy_origin=xy_origin, 
                                                    calib_file=calib_file, height_px=height_px, height_m=first_person_height, 
                                                    fps=fps, trc_data=trc_data[0], score_data=score_data[0], toe_speed_below=1, score_threshold=average_likelihood_threshold, 
                                                    cam_width=cam_width, cam_height=cam_height)
            cx, cy = xy_origin_estim
            direction_person0 = 'right' if gait_direction > 0.3 \
                                else 'left' if gait_direction < -0.3 \
                                else 'front'

            logging.info(f'Converting from pixels to meters using a person height of {first_person_height:.2f} in meters (manual input), and of {height_px:.2f} in pixels (calculated).')

            perspective_messages = {
                "distance_m": f"(obtained from a manual input).",
                "f_px": f"(calculated from a focal length of {perspective_value:.2f} m).",
                "fov_deg": f"(calculated from a field of view of {perspective_value:.2f} deg).",
                "fov_rad": f"(calculated from a field of view of {perspective_value:.2f} rad).",
                "from_calib": f"(calculated from a calibration file: {calib_file}).",
            }
            message = perspective_messages.get(perspective_unit, "")
            logging.info(f'Perspective effects corrected using a camera-to-person distance of {distance_m:.2f} m {message}')

            floor_angle_messages = {
                "manual": "manual input.",
                "auto": "gait kinematics.",
                "from_kinematics": "gait kinematics.",
                "from_calib": "a calibration file.",
            }
            if isinstance(floor_angle, (int, float)):
                key = "manual"
            else:
                key = floor_angle
            message = floor_angle_messages.get(key, "")
            logging.info(f'Camera horizon: {np.degrees(floor_angle_estim):.2f}°, corrected using {message}')

            def get_correction_message(xy_origin):
                if all(isinstance(o, (int, float)) for o in xy_origin) and len(xy_origin) == 2:
                    return "manual input."
                elif xy_origin == ["auto"] or xy_origin == ["from_kinematics"]:
                    return "gait kinematics."
                elif xy_origin == ["from_calib"]:
                    return "a calibration file."
                else:
                    return "."
            message = get_correction_message(xy_origin)
            logging.info(f'Floor level: {cy:.2f} px (from the top of the image), gait starting at {cx:.2f} px in the {direction_person0} direction for the first person. Corrected using {message}\n')

            # Prepare calibration data
            R90z = np.array([[0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]])
            R270x = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0]])
            
            calib_file_path = output_dir / f'{video_file_stem}_Sports2D_calib.toml'
            
            # name, size, distortions
            N = [video_file_stem]
            S = [[cam_width, cam_height]]
            D = [[0.0, 0.0, 0.0, 0.0]]

            # Intrinsics
            f = height_px / first_person_height * distance_m
            cu = cam_width/2
            cv = cam_height/2
            K = np.array([[[f, 0.0, cu], [0.0, f, cv], [0.0, 0.0, 1.0]]])

            # Extrinsics
            Rfloory = np.array([[np.cos(floor_angle_estim), 0.0, np.sin(floor_angle_estim)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(floor_angle_estim), 0.0, np.cos(floor_angle_estim)]])
            R_world = R90z @ Rfloory @ R270x
            T_world = R90z @ np.array([-(cx-cu)/f*distance_m, -distance_m, (cy-cv)/f*distance_m])
            
            R_cam, T_cam = world_to_camera_persp(R_world, T_world)
            Tvec_cam = T_cam.reshape(1,3).tolist()
            Rvec_cam = cv2.Rodrigues(R_cam)[0].reshape(1,3).tolist()

            # Save calibration file
            if save_calib and not calib_file:
                toml_write(calib_file_path, N, S, D, K, Rvec_cam, Tvec_cam)
                logging.info(f'Calibration saved to {calib_file_path}.')


            # Coordinates in m
            new_visible_side = []
            for i in range(len(trc_data)):
                if not np.array(trc_data[i].iloc[:,1:] ==0).all():
                    # Automatically determine visible side
                    visible_side_i = visible_side[i] if len(visible_side)>i else 'auto' # set to 'auto' if list too short
                    # Set to 'front' if slope of X values between [-5,5]
                    if visible_side_i == 'auto':
                        try:
                            if all(key in trc_data[i] for key in ['LBigToe', 'RBigToe']):
                                _, _, gait_direction = compute_floor_line(trc_data[i], score_data[i], keypoint_names=['LBigToe', 'RBigToe'], score_threshold=keypoint_likelihood_threshold) # toe_speed_below=1 bu default 
                            else:
                                _, _, gait_direction = compute_floor_line(trc_data[i], score_data[i], keypoint_names=['LAnkle', 'RAnkle'], score_threshold=keypoint_likelihood_threshold)
                                logging.warning(f'The RBigToe and LBigToe are missing from your model. Gait direction will be determined from the ankle points.')
                            visible_side_i = 'right' if gait_direction > 0.3 \
                                                else 'left' if gait_direction < -0.3 \
                                                else 'front'
                            logging.info(f'- Person {i}: Seen from the {visible_side_i}.')
                        except:
                            visible_side_i = 'none'
                            logging.warning(f'- Person {i}: Could not automatically find gait direction. Please set visible_side to "front", "back", "left", or "right" for this person. Setting to "none".')
                    # skip if none
                    elif visible_side_i == 'none':
                        logging.info(f'- Person {i}: Keeping output in 2D because "visible_side" is set to "none" for person {i}.')
                    else:
                        logging.info(f'- Person {i}: Seen from the {visible_side_i}.')
                    

                    # Convert to meters
                    px_to_m_i = [convert_px_to_meters(trc_data[i][kpt_name], first_person_height, height_px, distance_m, cam_width, cam_height, cx, cy, -floor_angle_estim, visible_side=visible_side_i) for kpt_name in new_keypoints_names]
                    trc_data_m_i = pd.concat([all_frames_time.rename('time')]+px_to_m_i, axis=1)
                    for c_id, c in enumerate(3*np.arange(len(trc_data_m_i.columns[3::3]))+1): # only X coordinates
                        first_run_start, last_run_end = first_run_starts_everyone[i][c_id], last_run_ends_everyone[i][c_id]
                        trc_data_m_i.iloc[:first_run_start,c+2] = np.nan
                        trc_data_m_i.iloc[last_run_end:,c+2] = np.nan
                        trc_data_m_i.iloc[first_run_start:last_run_end,c+2] = trc_data_m_i.iloc[first_run_start:last_run_end,c+2].ffill().bfill()
                    first_trim, last_trim = trc_data_m_i.isnull().any(axis=1).idxmin(), trc_data_m_i[::-1].isnull().any(axis=1).idxmin()
                    trc_data_m_i = trc_data_m_i.iloc[first_trim:last_trim+1,:]
                    px_to_m_unfiltered_i = [convert_px_to_meters(trc_data_unfiltered[i][kpt_name], first_person_height, height_px, distance_m, cam_width, cam_height, cx, cy, -floor_angle_estim, visible_side=visible_side_i) for kpt_name in new_keypoints_names]
                    trc_data_unfiltered_m_i = pd.concat([all_frames_time.rename('time')]+px_to_m_unfiltered_i, axis=1)

                    if to_meters and (show_plots or save_plots):
                        pw = pose_plots(trc_data_unfiltered_m_i, trc_data_m_i, i, show=show_plots)
                        if save_plots:
                            for n, f in enumerate(pw.figure_handles):
                                dpi = pw.canvases[i].figure.dpi
                                f.set_size_inches(1280/dpi, 720/dpi)
                                title = pw.tabs.tabText(n)
                                plot_path = plots_output_dir / (pose_output_path_m.stem + f'_person{i:02d}_m_{title.replace(" ","_").replace("/","_")}.png')
                                f.savefig(plot_path, dpi=dpi, bbox_inches='tight')
                            logging.info(f'Pose plots (m) saved in {plots_output_dir}.')

                    # Write to trc file
                    trc_data_m.append(trc_data_m_i)
                    pose_path_person_m_i = (pose_output_path.parent / (pose_output_path_m.stem + f'_person{i:02d}.trc'))
                    make_trc_with_trc_data(trc_data_m_i, pose_path_person_m_i, fps=fps)
                    if make_c3d:
                        c3d_path = convert_to_c3d(str(pose_path_person_m_i))
                    logging.info(f'Pose in meters saved to {pose_path_person_m_i.resolve()}. {"Also saved in c3d format." if make_c3d else ""}')
                else:
                    visible_side_i = 'none'
                new_visible_side += [visible_side_i]
        else:
            new_visible_side = visible_side.copy()


    #%% ==================================================
    # Post-processing angles
    # ====================================================
    if save_angles and calculate_angles:
        logging.info('\nPost-processing angles (without inverse kinematics):')

        # unwrap angles
        # all_frames_angles_homog = np.unwrap(all_frames_angles_homog, axis=0, period=180) # This give all nan values -> need to mask nans
        for i in range(all_frames_angles_homog.shape[1]):  # for each person
            for j in range(all_frames_angles_homog.shape[2]):  # for each angle
                valid_mask = ~np.isnan(all_frames_angles_homog[:, i, j])
                ang = np.unwrap(all_frames_angles_homog[valid_mask, i, j], period=180)
                ang = ang-360 if ang.mean()> 180 else ang
                ang = ang+360 if ang.mean()<-180 else ang
                all_frames_angles_homog[valid_mask, i, j] = ang
                
        # Process angles for each person
        for i, idx_person in enumerate(selected_persons):
            angles_path_person = angles_output_path.parent / (angles_output_path.stem + f'_person{i:02d}.mot')
            all_frames_angles_person = pd.DataFrame(all_frames_angles_homog[:,idx_person,:], columns=angle_names)

            # Flip angles for left side when flip_left_right false
            if new_visible_side[i] == 'left' and not flip_left_right: 
                all_frames_angles_homog[:, idx_person, :] = -all_frames_angles_homog[:, idx_person, :]

            if not interpolate:
                logging.info(f'- Person {i}: No interpolation.')
                all_frames_angles_person_interp = all_frames_angles_person
            else:
                logging.info(f'- Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames. Large gaps filled with {fill_large_gaps_with}.')
                all_frames_angles_person_interp = all_frames_angles_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])

            # Find the first and last valid chunks of data
            first_run_starts, last_run_ends = [], []
            for col in all_frames_angles_person.columns:
                first_run_start, last_run_end = indices_of_first_last_non_nan_chunks(all_frames_angles_person_interp[col], min_chunk_size=min_chunk_size, chunk_choice_method=sections_to_keep)
                first_run_starts += [first_run_start]
                last_run_ends += [last_run_end]
            first_run_start_min, last_run_end_max = min(first_run_starts), max(last_run_ends)

            # Do not process person if no section of min_chunk_size valid frames in a row
            if (first_run_start_min, last_run_end_max) == (0,0):
                all_frames_angles_processed[:,idx_person,:]= np.nan
                logging.info(f'  Person {i}: Less than {min_chunk_size} valid frames in a row. Deleting person.')
                continue

            # Fill remaining gaps
            if fill_large_gaps_with == 'last_value':
                for col_id, col in enumerate(all_frames_angles_person_interp.columns):
                    first_run_start, last_run_end = first_run_starts[col_id], last_run_ends[col_id]
                    all_frames_angles_person_interp.loc[:first_run_start, col] = np.nan
                    all_frames_angles_person_interp.loc[last_run_end:, col] = np.nan
                    all_frames_angles_person_interp.loc[first_run_start:last_run_end, col] = all_frames_angles_person_interp.loc[first_run_start:last_run_end, col].ffill().bfill()
            elif fill_large_gaps_with == 'zeros':
                all_frames_angles_person_interp.replace(np.nan, 0, inplace=True)
                
            # Filter
            if reject_outliers:
                logging.info(f'Rejecting outliers with a Hampel filter.')
                all_frames_angles_person_interp = all_frames_angles_person_interp.apply(hampel_filter, axis=0)

            if not do_filter:
                logging.info(f'No filtering.')
                all_frames_angles_person_filt = all_frames_angles_person_interp
            else:
                if filter_type == ('butterworth' or 'butterworth_on_speed'):
                    cutoff = butterworth_filter_cutoff
                    if video_file == 'webcam':
                        if cutoff / (fps / 2) >= 1:
                            cutoff_old = cutoff
                            cutoff = fps/(2+0.001)
                            args = f'\n{cutoff_old:.1f} Hz cut-off framerate too large for a real-time framerate of {fps:.1f} Hz. Using a cut-off framerate of {cutoff:.1f} Hz instead.'
                            butterworth_filter_cutoff = cutoff
                    filt_type = 'Butterworth' if filter_type == 'butterworth' else 'Butterworth on speed'
                    args = f'{filt_type} filter, {butterworth_filter_order}th order, {butterworth_filter_cutoff} Hz.'
                    frame_rate = fps
                elif filter_type == 'gcv_spline':
                    args = f'GVC Spline filter, which automatically evaluates the best trade-off between smoothness and fidelity to data.'
                elif filter_type == 'kalman':
                    args = f'Kalman filter, trusting measurement {kalman_filter_trust_ratio} times more than the process matrix.'
                elif filter_type == 'gaussian':
                    args = f'Gaussian filter, Sigma kernel {gaussian_filter_kernel}.'
                elif filter_type == 'loess':
                    args = f'LOESS filter, window size of {loess_filter_kernel} frames.'
                elif filter_type == 'median':
                    args = f'Median filter, kernel of {median_filter_kernel}.'
                else:
                    logging.error(f"Invalid filter_type: {filter_type}. Must be 'butterworth', 'gcv_spline', 'kalman', 'gaussian', 'loess', or 'median'.")
                    raise ValueError(f"Invalid filter_type: {filter_type}. Must be 'butterworth', 'gcv_spline', 'kalman', 'gaussian', 'loess', or 'median'.")

                logging.info(f'Filtering with {args}')
                all_frames_angles_person_filt = all_frames_angles_person_interp.apply(filter1d, axis=0, args = [Pose2Sim_config_dict, filter_type, frame_rate])

            # Add floor_angle_estim to segment angles
            if correct_segment_angles_with_floor_angle and to_meters:
                logging.info(f'Correcting segment angles by removing the {round(np.degrees(floor_angle_estim),2)}° floor angle.')
                for ang_name in all_frames_angles_person_filt.columns:
                    if 'horizontal' in angle_dict[ang_name][1]:
                        all_frames_angles_person_filt[ang_name] -= np.degrees(floor_angle_estim)

            # Remove columns with all nan values
            all_frames_angles_processed[:,idx_person,:] = all_frames_angles_person_filt
            all_frames_angles_person_filt.dropna(axis=1, how='all', inplace=True)
            all_frames_angles_person = all_frames_angles_person[all_frames_angles_person_filt.columns]

            # Build mot file
            angle_data = make_mot_with_angles(all_frames_angles_person_filt, all_frames_time, str(angles_path_person))
            logging.info(f'Angles saved to {angles_path_person.resolve()}.')

            # Plotting angles before and after interpolation and filtering
            all_frames_angles_person.insert(0, 'time', all_frames_time)
            if show_plots or save_plots:
                pw = angle_plots(all_frames_angles_person, angle_data, i, show=show_plots) # i = current person
                if save_plots:
                    for n, f in enumerate(pw.figure_handles):
                        dpi = pw.canvases[i].figure.dpi
                        f.set_size_inches(1280/dpi, 720/dpi)
                        title = pw.tabs.tabText(n)
                        plot_path = plots_output_dir / (pose_output_path_m.stem + f'_person{i:02d}_ang_{title.replace(" ","_").replace("/","_")}.png')
                        f.savefig(plot_path, dpi=dpi, bbox_inches='tight')
                    logging.info(f'Pose plots (m) saved in {plots_output_dir}.')


    #%% ==================================================
    # Save images/video with processed pose and angles
    # ====================================================
    if save_vid or save_img:
        logging.info('\nSaving images of processed pose and angles:')
        if save_vid:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(str(vid_output_path.absolute()), fourcc, fps, (cam_width, cam_height))
        
        # Reorder persons
        all_frames_X_processed, all_frames_Y_processed = all_frames_X_processed[:,selected_persons,:], all_frames_Y_processed[:,selected_persons,:]
        all_frames_scores_processed = all_frames_scores_processed[:,selected_persons,:]
        if save_angles or calculate_angles:
            all_frames_X_flipped_processed = all_frames_X_flipped_processed[:,selected_persons,:]
            all_frames_angles_processed = all_frames_angles_processed[:,selected_persons,:]

        # Reorder keypoints ids
        pose_model_with_new_ids = copy.deepcopy(pose_model)
        new_id = 0
        for node in PreOrderIter(pose_model_with_new_ids):
            if node.id!=None:
                node.id = new_id
                new_id+=1
        max_id = max(node.id for node in PreOrderIter(pose_model_with_new_ids) if node.id is not None)
        for node in PreOrderIter(pose_model_with_new_ids):
            if node.id==None:
                node.id = max_id+1
                max_id+=1
        new_keypoints_ids = list(range(len(new_keypoints_ids)))

        # Draw pose and angles
        first_frame, last_frame = frame_range
        if 'first_trim' not in locals():
            first_trim, last_trim = first_frame, last_frame
        cap = cv2.VideoCapture(video_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame+first_trim)
        for i in range(first_trim, last_trim):
            success, frame = cap.read()
            if not success:
                raise ValueError(f"Could not read frame {i}")
            img = frame.copy()
            img = draw_bounding_box(img, all_frames_X_processed[i], all_frames_Y_processed[i], colors=colors, fontSize=fontSize, thickness=thickness)
            img = draw_keypts(img, all_frames_X_processed[i], all_frames_Y_processed[i], all_frames_scores_processed[i], cmap_str='RdYlGn')
            img = draw_skel(img, all_frames_X_processed[i], all_frames_Y_processed[i], pose_model_with_new_ids)
            if calculate_angles:
                img = draw_angles(img, all_frames_X_processed[i], all_frames_Y_processed[i], all_frames_angles_processed[i], all_frames_X_flipped_processed[i], new_keypoints_ids, new_keypoints_names, angle_names, display_angle_values_on=display_angle_values_on, colors=colors, fontSize=fontSize, thickness=thickness)

            # Save video or images
            if save_vid:
                out_vid.write(img)
            if save_img:
                cv2.imwrite(str((img_output_dir / f'{output_dir_name}_{(i+frame_range[0]):06d}.png')), img)
        cap.release()

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


    #%% ==================================================
    # OpenSim inverse kinematics (and optional marker augmentation)
    # ====================================================
    if do_ik or use_augmentation:
        import opensim as osim
        logging.info('\nPost-processing angles (with inverse kinematics):')
        if not to_meters:
            logging.warning('Skipping marker augmentation and inverse kinematics as to_meters was set to False.')
        else:
            # move all trc files containing _m_ string to pose3d_dir
            if not load_trc_px: 
                trc_list = output_dir.glob('*_m_*.trc')
            else:
                trc_list = [pose_path_person_m_i]
            for trc_file in trc_list:
                if (pose3d_dir/trc_file.name).exists():
                    os.remove(pose3d_dir/trc_file.name)
                shutil.move(trc_file, pose3d_dir)

            heights_m, masses = [], []
            for i in range(len(trc_data_m)):
                trc_data_m_i = trc_data_m[i]
                if do_ik and not use_augmentation:
                    logging.info(f'- Person {i}: Running scaling and inverse kinematics without marker augmentation. Set use_augmentation to True if you need it.') 
                elif not do_ik and use_augmentation:
                    logging.info(f'- Person {i}: Running marker augmentation without inverse kinematics. Set do_ik to True if you need it.')
                else:
                    logging.info(f'- Person {i}: Running marker augmentation and inverse kinematics.')

                # Delete person if less than 4 valid frames
                pose_path_person = pose_output_path.parent / (pose_output_path.stem + f'_person{i:02d}.trc')
                all_frames_X_person = pd.DataFrame(all_frames_X_homog[:,i,:], columns=new_keypoints_names)
                if new_visible_side[i] == 'none':
                    logging.info(f'Skipping marker augmentation and inverse kinematics because visible_side is "none".')
                else:
                    # Provide missing data to Pose2Sim_config_dict
                    height_m_i = compute_height(trc_data_m_i.iloc[:,1:], keypoints_names,
                        fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, close_to_zero_speed=close_to_zero_speed_m, large_hip_knee_angles=large_hip_knee_angles, trimmed_extrema_percent=trimmed_extrema_percent)
                    mass_i = participant_masses[i] if len(participant_masses)>i else DEFAULT_MASS
                    if len(participant_masses)<=i:
                        logging.warning(f'No mass provided. Using {DEFAULT_MASS} kg as default.')
                    heights_m.append(height_m_i)
                    masses.append(mass_i)
                
            Pose2Sim_config_dict['project']['participant_height'] = heights_m
            Pose2Sim_config_dict['project']['participant_mass'] = masses
            Pose2Sim_config_dict['project']['frame_range'] = 'all'
            Pose2Sim_config_dict['markerAugmentation']['feet_on_floor'] = feet_on_floor
            Pose2Sim_config_dict['pose']['pose_model'] = pose_model_name.upper()
            Pose2Sim_config_dict = to_dict(Pose2Sim_config_dict)

            # Marker augmentation
            if use_augmentation:
                logging.info('Running marker augmentation...')
                augment_markers_all(Pose2Sim_config_dict)
                logging.info(f'Augmented trc results saved to {pose3d_dir.resolve()}.\n')

            if do_ik:
                if not save_angles or not calculate_angles:
                    logging.warning(f'Skipping inverse kinematics because save_angles or calculate_angles is set to False.')
                else:
                    logging.info('Running inverse kinematics...')
                    kinematics_all(Pose2Sim_config_dict)
                    for mot_file in kinematics_dir.glob('*.mot'):
                        if (mot_file.parent/(mot_file.stem+'_ik.mot')).exists():
                            os.remove(mot_file.parent/(mot_file.stem+'_ik.mot'))
                        os.rename(mot_file, mot_file.parent/(mot_file.stem+'_ik.mot'))
                    logging.info(f'.osim model and .mot motion file results saved to {kinematics_dir.resolve().parent}.\n')
                            
            # Move all files in pose-3d and kinematics to the output_dir
            osim.Logger.removeFileSink()
            for directory in [pose3d_dir, kinematics_dir]:
                for file in directory.glob('*'):
                    if (output_dir/file.name).exists():
                        os.remove(output_dir/file.name)
                    shutil.move(file, output_dir)
            pose3d_dir.rmdir()
            kinematics_dir.rmdir()
