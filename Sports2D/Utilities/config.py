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
import sys
import logging
import threading
import time

from datetime import datetime
from tqdm import tqdm
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


## CLASSES
class WebcamStream:
    def __init__(self, src=0, input_size=(640, 480)):
        self.src = src
        self.input_size = input_size
        self.cap = None
        self.cam_width = None
        self.cam_height = None
        self.fps = None
        self.frame = None
        self.open_camera()
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, args=(), daemon=True).start()

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            logging.warning(f"Could not open webcam #{self.src}. Retrying...")
            self.cap = None
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_size[1])
            self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            logging.info(f"Opened webcam #{self.src} with resolution {self.cam_width}x{self.cam_height} at {self.fps} FPS.")

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                # Try to reopen the webcam
                self.open_camera()
                time.sleep(0.5)
                with self.lock:
                    self.frame = None  # Set frame to None when the webcam is not opened
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Reading problem, release and retry
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
                with self.lock:
                    self.frame = None  # Set frame to None on read failure
                continue

            with self.lock:
                self.frame = frame
                
    def read(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        return frame

    def stop(self):
        self.stopped = True
        if self.cap and self.cap.isOpened():
            self.cap.release()

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
    if pose_model.upper()  in ('HALPE_26', 'BODY_WITH_FEET'):
        ModelClass = BodyWithFeet
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation.")
    elif pose_model.upper()  in ('COCO_133', 'WHOLE_BODY'):
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
    elif pose_model.upper()  in ('COCO_17', 'BODY'):
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

    cap = WebcamStream(webcam_id, input_size)

    cam_width = cap.cam_width
    cam_height = cap.cam_height
    fps = cap.fps

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


def setup_capture_directories(file_path, output_dir, save_images):
    """
    Sets up directories for output and prepares for video capture.

    Parameters:
        file_path (str): Path to the file or 'webcam' for webcam usage.
        output_dir (str): Base directory to store the output directories and files.

    Returns:
        dict: A dictionary containing paths for image output, JSON output, and output video.
    """
    # Create output directories based on the file path or webcam
    if str(file_path).startswith("webcam"):
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f'{file_path}_{current_date}'
    else:
        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        output_dir_name = f'{file_stem}'

    # Define the full path for the output directory
    output_dir_full = os.path.abspath(os.path.join(output_dir, "pose"))
    
    # Create output directories if they do not exist
    if not os.path.isdir(output_dir_full):
        os.makedirs(output_dir_full)
    
    # Prepare directories for images and JSON outputs
    img_output_dir = os.path.join(output_dir_full, f'{output_dir_name}_img')
    json_output_dir = os.path.join(output_dir_full, f'{output_dir_name}_json')
    if not os.path.isdir(img_output_dir) and save_images:
        os.makedirs(img_output_dir)
    if not os.path.isdir(json_output_dir):
        os.makedirs(json_output_dir)
    
    # Define the path for the output video file
    output_video_path = os.path.join(output_dir_full, f'{output_dir_name}_pose.mp4')

    return output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path


def setup_video_capture(video_file_path, save_video=False, output_video_path=None, input_size=None, input_frame_range=[], position=0):
    """
    Sets up video capture from a webcam or a video file. Optionally saves the output.
    """

    if str(video_file_path).startswith("webcam"):
        try:
            webcam_id = int(str(video_file_path).replace('webcam', ''))
        except ValueError:
            raise ValueError(f"Invalid webcam ID in {video_file_path}. Expected format 'webcamX' where X is an integer.")
        cap, out_vid, cam_width, cam_height, fps = setup_webcam(webcam_id, save_video, output_video_path, input_size)
        frame_range = [0, sys.maxsize]
        frame_iterator = range(*frame_range)
        logging.warning('Webcam input: the framerate may vary. If results are filtered, the average framerate will be used as input.')
    else:
        cap, out_vid, cam_width, cam_height, fps = setup_video(video_file_path, save_video, output_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if input_frame_range == []:
            frame_range = [0, total_frames]
        else:
            frame_range = input_frame_range
        frame_iterator = tqdm(range(*frame_range), desc=f'Processing {os.path.basename(video_file_path)}', position=position)
        start_frame = input_frame_range[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    return cap, frame_iterator, out_vid, cam_width, cam_height, fps


def process_video_frames(config_dict, video_paths):
    frame_rates = []
    total_frames_list = []

    # Extraction des propriétés vidéo
    for video_path in video_paths:
        if not video_path.exists():
            raise FileNotFoundError(f'Error: Could not find video file {video_path}.')
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise IOError(f'Error: Could not open {video_path}. Check that the file is a valid video.')
        
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        if not frame_rate or frame_rate <= 0:
            frame_rate = 30
            logging.warning(f'Could not retrieve frame rate from {video_path}. Defaulting to 30fps.')
        frame_rates.append(frame_rate)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_list.append(total_frames)
        video.release()

    # Traitement des plages spécifiées
    def process_ranges(range_name):
        ranges = config_dict['project'].get(range_name)
        num_videos = len(video_paths)
        if ranges is None or ranges == []:
            return [None] * num_videos
        elif isinstance(ranges[0], (int, float)) and len(ranges) == 2:
            return [ranges] * num_videos
        elif all(isinstance(r, list) and len(r) == 2 for r in ranges):
            if len(ranges) != num_videos:
                raise ValueError(f'Length of {range_name} does not match number of videos.')
            return ranges
        else:
            raise ValueError(f'{range_name} must be empty, [start, end], or a list of [start, end] for each video.')

    time_ranges = process_ranges('time_range')
    frame_ranges = process_ranges('frame_range')

    # Calcul final des plages de frames
    for i, (time_range, frame_range, frame_rate, total_frames) in enumerate(zip(time_ranges, frame_ranges, frame_rates, total_frames_list)):
        if time_range and frame_range:
            raise ValueError(f"Error: Both time_range and frame_range are specified for video {video_paths[i]}. Only one should be provided.")
        if time_range:
            frame_ranges[i] = [int(time_range[0] * frame_rate), int(time_range[1] * frame_rate)]
        elif frame_range:
            frame_ranges[i] = [int(frame_range[0]), int(frame_range[1])]
        else:
            # Assign full frame range
            frame_ranges[i] = [0, total_frames]

    return frame_ranges
