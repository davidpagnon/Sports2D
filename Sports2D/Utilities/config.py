#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Configuration and Initialization             ##
    ##################################################

    - Functions for setting up logging configurations.
    - Initialization routines for pose tracking systems.
    - Webcam setup for video capturing.
    - Video setup for processing and recording.
    - Directory setup for saving captured data.
    - Video capture configuration for various devices.
'''

import os
import cv2
import logging
from datetime import datetime

from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def setup_logging(dir):
    '''
    Create logging file and stream handlers
    '''

    logging.basicConfig(format='%(message)s', level=logging.INFO,
        handlers = [logging.handlers.TimedRotatingFileHandler(os.path.join(dir, 'logs.txt'), when='D', interval=7), logging.StreamHandler()])


def setup_pose_tracker(det_frequency, mode, pose_model = "HALPE_26"):
    '''
    Set up the RTMLib pose tracker with the appropriate model and backend.
    If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino

    INPUTS:
    - det_frequency: int. The frequency of pose detection (every N frames)
    - mode: str. The mode of the pose tracker ('lightweight', 'balanced', 'performance')
    - tracking: bool. Whether to track persons across frames with RTMlib tracker

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

    if det_frequency>1:
        logging.info(f'Inference run only every {det_frequency} frames. Inbetween, pose estimation tracks previously detected points.')
    elif det_frequency==1:
        logging.info(f'Inference run on every single frame.')
    else:
        raise ValueError(f"Invalid det_frequency: {det_frequency}. Must be an integer greater or equal to 1.")

    # Select the appropriate model based on the model_type
    if pose_model.upper() == 'HALPE_26':
        ModelClass = BodyWithFeet
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation.")
    elif pose_model.upper() == 'COCO_133':
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
    elif pose_model.upper() == 'COCO_17':
        ModelClass = Body # 26 keypoints(halpe26)
        logging.info(f"Using COCO_17 model (body) for pose estimation.")
    else:
        raise ValueError(f"Invalid model_type: {pose_model}. Must be 'HALPE_26', 'COCO_133', or 'COCO_17'. Use another network (MMPose, DeepLabCut, OpenPose, AlphaPose, BlazePose...) and convert the output files if you need another model. See documentation.")
    logging.info(f'Mode: {mode}.\n')

    # Initialize the pose tracker with Halpe26 model
    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=False,
        to_openpose=False)
        
    return pose_tracker


def setup_webcam(webcam_id, save_video, vid_output_path, input_size):
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    if cam_width != input_size[0] or cam_height != input_size[1]:
        logging.warning(f"Warning: Your webcam does not support {input_size[0]}x{input_size[1]} resolution. Resolution set to the closest supported one: {cam_width}x{cam_height}.")
    
    out_vid = None
    if save_video:
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


def setup_video(video_file_path, save_video, vid_output_path):
    '''
    Set up video capture with OpenCV.

    INPUTS:
    - video_file_path: Path. The path to the video file
    - save_video: bool. Whether to save the video output
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
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            raise
    except:
        raise NameError(f"{video_file_path} is not a video. Images must be put in one subdirectory per camera.")
    
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    out_vid = None

    if save_video:
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


def setup_capture_directories(file_path, output_dir):
    """
    Sets up directories for output and prepares for video capture.

    Parameters:
        file_path (str): Path to the file or 'webcam' for webcam usage.
        output_dir (str): Base directory to store the output directories and files.

    Returns:
        dict: A dictionary containing paths for image output, JSON output, and output video.
    """
    # Create output directories based on the file path or webcam
    if file_path == "webcam":
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f'webcam_{current_date}'
    else:
        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        output_dir_name = f'{file_stem}_Sports2D'

    # Define the full path for the output directory
    output_dir_full = os.path.abspath(os.path.join(output_dir, output_dir_name))
    
    # Create output directories if they do not exist
    if not os.path.isdir(output_dir_full):
        os.makedirs(output_dir_full)
    
    # Prepare directories for images and JSON outputs
    img_output_dir = os.path.join(output_dir_full, f'{output_dir_name}_img')
    json_output_dir = os.path.join(output_dir_full, f'{output_dir_name}_json')
    if not os.path.isdir(img_output_dir):
        os.makedirs(img_output_dir)
    if not os.path.isdir(json_output_dir):
        os.makedirs(json_output_dir)
    
    # Define the path for the output video file
    output_video_path = os.path.join(output_dir_full, f'{output_dir_name}_pose.mp4')

    return output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path


def setup_video_capture(video_file_path, webcam_id=None, save_video=False, output_video_path=None, input_size=None, input_frame_range=[]):
    """
    Sets up video capture from a webcam or a video file. Optionally saves the output.
    """
    import cv2, sys, logging
    from tqdm import tqdm

    if video_file_path == "webcam":
        cap, out_vid, cam_width, cam_height, fps = setup_webcam(webcam_id, save_video, output_video_path, input_size)
        frame_range = [0, sys.maxsize]
        frame_iterator = range(*frame_range)
        logging.warning('Webcam input: the framerate may vary. If results are filtered, Sports2D will use the average framerate as input.')
    else:
        cap, out_vid, cam_width, cam_height, fps = setup_video(video_file_path, save_video, output_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if input_frame_range == []:
            frame_range = [0, total_frames]
        else:
            frame_range = input_frame_range
        frame_iterator = tqdm(range(*frame_range), desc=f'Processing {os.path.basename(video_file_path)}') # use a progress bar
        start_frame = input_frame_range[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    return cap, frame_iterator, out_vid, cam_width, cam_height, fps