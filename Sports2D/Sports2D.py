#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##############################################################
    ## SPORTS2D                                                 ##
    ##############################################################
    
    This repository provides a workflow to compute 2D markerless
    joint and segment angles from videos. 
    These angles can be plotted and processed with any 
    spreadsheet software or programming language.
    
    This is a headless version, but apps will be released 
    for Windows, Linux, MacOS, as well as Android and iOS.
    Mobile versions will only support exploratory joint detection 
    from BlazePose, hence less accurately and less tunable.
    
    If you need to detect several persons and want more accurate results, 
    you can install and use OpenPose: 
    https://github.com/CMU-Perceptual-Computing-Lab/openpose
    
    -----
    Sports2D installation:
    -----
    Optional: 
    - Install Miniconda
    - Open a Anaconda Prompt and type: 
    `conda create -n Sports2D python>=3.7`
    `conda activate Sports2D`
    pip install 
    
    - Open a python prompt and type `pip install sports2d`
    - `pip show sports2d`
    - Adjust your settings (in particular video path) in `Config_demo.toml`
    - ```from Sports2D import Sports2D
    Sports2D.detect_pose('Config_demo.toml')
    Sports2D.compute_angles('Config_demo.toml')```
    
    -----
    /!\ Warning /!\
    -----
    - The angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will only lead to acceptable results if the persons move in the 2D plane (sagittal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
    
    -----
    Pose detection:
    -----
    Detect joint centers from a video with OpenPose or BlazePose.
    Save a 2D csv position file per person, and optionally json files, image files, and video files.
    
    If OpenPose is used, multiple persons can be consistently detected across frames.
    Interpolates sequences of missing data if they are less than N frames long.
    Optionally filters results with Butterworth, gaussian, median, or loess filter.
    Optionally displays figures.

    If BlazePose is used, only one person can be detected.
    No interpolation nor filtering options available. Not plotting available.
    
    -----
    Angle computation:
    -----
    Compute joint and segment angles from csv position files.
    Automatically adjust angles when person switches to face the other way.
    Save a 2D csv angle file per person.
    Optionally filters results with Butterworth, gaussian, median, or loess filter.
    Optionally displays figures.
    Optionally saves images and video with overlaid angles.

    Joint angle conventions:
    - Ankle dorsiflexion: Between heel and big toe, and ankle and knee
    - Knee flexion: Between hip, knee, and ankle 
    - Hip flexion: Between knee, hip, and shoulder
    - Shoulder flexion: Between hip, shoulder, and elbow
    - Elbow flexion: Between wrist, elbow, and shoulder

    Segment angle conventions:
    Angles are measured anticlockwise between the horizontal and the segment.
    - Foot: Between heel and big toe
    - Shank: Between ankle and knee
    - Thigh: Between hip and knee
    - Arm: Between shoulder and elbow
    - Forearm: Between elbow and wrist
    - Trunk: Between hip midpoint and shoulder midpoint
    
    -----
    To-do list:
    -----
    - GUI applications for all platforms (with Kivy: https://kivy.org/)
    - Pose refinement: click and move badly estimated 2D points (cf DeepLabCut: https://www.youtube.com/watch?v=bEuBKB7eqmk)
    - Include OpenPose in Sports2D (dockerize it cf https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/demo/Dockerfile)
    - Constrain points to OpenSim skeletal model for better angle estimation (cf Pose2Sim but in 2D https://github.com/perfanalytics/pose2sim)
    
'''


## INIT
import toml
import os
import time
import cv2
from pathlib import Path
import logging, logging.handlers


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.1"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def read_config_file(config):
    '''
    Read configation file.
    '''

    config_dict = toml.load(config)
    return config_dict


def base_params(config_dict):
    '''
    Retrieve sequence name and frames to be analyzed.
    '''

    video_dir = Path(config_dict.get('project').get('video_dir')).resolve()
    if video_dir == '': video_dir = os.getcwd()
    video_file = Path(config_dict.get('project').get('video_file'))
    result_dir = Path(config_dict.get('project').get('result_dir')).resolve()
    if result_dir == '': result_dir = os.getcwd()
    
    video = cv2.VideoCapture(str(video_dir / video_file))
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    try:
        1/frame_rate
    except ZeroDivisionError:
        print('Frame rate could not be retrieved: check that your video exists at the correct path')
        raise
    video.release()

    return video_dir, video_file, result_dir, frame_rate


def detect_pose(config='Config_demo.toml'):
    '''
    Compute 2D pose from video.
    Save 2D csv file, and optionally json files, image files, and video file.
    Optionally interpolates missing data, filters them, and displays figures.
    '''

    from Sports2D.detect_pose import detect_pose_fun
    
    config_dict = read_config_file(config)
    _, video_file, result_dir, _ = base_params(config_dict)
        
    with open(result_dir / 'logs.txt', 'a+') as log_f: pass
    logging.basicConfig(format='%(message)s', level=logging.INFO, force=True, 
        handlers = [logging.handlers.TimedRotatingFileHandler(result_dir / 'logs.txt', when='D', interval=7), logging.StreamHandler()]) 
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info(f"Detect pose for video {video_file}")
    logging.info("---------------------------------------------------------------------")
    start = time.time()
    
    detect_pose_fun(config_dict)
    
    end = time.time()
    logging.info(f'Pose detection took {end-start:.2f} s.')
    
    
def compute_angles(config='Config_demo.toml'):
    '''
    Compute joint and segment angles from 2D points coordinates in csv file.
    Save csv file.
    Optionally interpolates missing data, filters them, and displays figures.
    '''

    from Sports2D.compute_angles import compute_angles_fun

    config_dict = read_config_file(config)
    _, video_file, result_dir, _ = base_params(config_dict)
        
    with open(result_dir / 'logs.txt', 'a+') as log_f: pass
    logging.basicConfig(format='%(message)s', level=logging.INFO, force=True, 
        handlers = [logging.handlers.TimedRotatingFileHandler(result_dir / 'logs.txt', when='D', interval=7), logging.StreamHandler()])
        
    joint_angles = config_dict.get('compute_angles').get('joint_angles')
    segment_angles = config_dict.get('compute_angles').get('segment_angles')

    logging.info("\n\n---------------------------------------------------------------------")
    logging.info(f"Compute angles for video {video_file} ")
    logging.info(f"for {joint_angles}")
    logging.info(f"and {segment_angles}.")
    logging.info("---------------------------------------------------------------------")
    start = time.time()
    
    compute_angles_fun(config_dict)
    
    end = time.time()
    logging.info(f'Joint and segment computation took {end-start:.2f} s.')
    
    

