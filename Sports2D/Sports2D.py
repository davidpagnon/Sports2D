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

import argparse
import toml
from datetime import datetime
from pathlib import Path
import logging, logging.handlers
import cv2
import numpy as np

from Sports2D import Sports2D


## CONSTANTS
DEFAULT_CONFIG =   {'project': {'video_input': 'Demo.mp4',
                                'time_range': [],
                                'video_dir': '',
                                'webcam_id': 0,
                                'input_size': [1280, 720]
                                },
                    'process': {'multiperson': True,
                                'show_realtime_results': True,
                                'save_vid': True,
                                'save_img': True,
                                'save_pose': True,
                                'save_angles': True,
                                'result_dir': ''
                                },
                    'pose':     { 'pose_model': 'body_with_feet',
                                'mode': 'balanced',
                                'det_frequency': 1,
                                'tracking_mode': 'sports2d',
                                'keypoint_likelihood_threshold': 0.3,
                                'average_likelihood_threshold': 0.5,
                                'keypoint_number_threshold': 0.3
                                },
                    'angles':   {'display_angle_values_on': ['body', 'list'],
                                'joint_angles': [   'Right ankle',
                                                    'Left ankle',
                                                    'Right knee',
                                                    'Left knee',
                                                    'Right hip',
                                                    'Left hip',
                                                    'Right shoulder',
                                                    'Left shoulder',
                                                    'Right elbow',
                                                    'Left elbow'],
                                'segment_angles': [ 'Right foot',
                                                    'Left foot',
                                                    'Right shank',
                                                    'Left shank',
                                                    'Right thigh',
                                                    'Left thigh',
                                                    'Pelvis',
                                                    'Trunk',
                                                    'Shoulders',
                                                    'Head',
                                                    'Right arm',
                                                    'Left arm',
                                                    'Right forearm',
                                                    'Left forearm'],
                                'flip_left_right': True
                                },
                    'post-processing': {'interpolate': True,
                                        'interp_gap_smaller_than': 10,
                                        'fill_large_gaps_with': 'last_value',
                                        'filter': True,
                                        'show_plots': True,
                                        'filter_type': 'butterworth',
                                        'butterworth': {'order': 4, 'cut_off_frequency': 3},
                                        'gaussian': {'sigma_kernel': 1},
                                        'loess': {'nb_values_used': 5},
                                        'median': {'kernel_size': 3}
                                        }
                    }


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

    # videod_dir and result_dir
    video_dir = Path(config_dict.get('project').get('video_dir')).resolve()
    if video_dir == '': video_dir = Path.cwd()
    result_dir = Path(config_dict.get('process').get('result_dir')).resolve()
    if result_dir == '': result_dir = Path.cwd()

    # video_files, frame_rates, time_ranges
    video_input = config_dict.get('project').get('video_input')
    if video_input == "webcam":
        video_files = ['webcam']  # No video files for webcam
        frame_rates = [None]  # No frame rate for webcam
        time_ranges = [None]
    else:
        # video_files
        video_files = config_dict.get('project').get('video_input')
        if isinstance(video_files, str):
            video_files = [Path(video_files)]
        else: 
            video_files = [Path(v) for v in video_files]

        # frame_rates
        frame_rates = []
        for video_file in video_files:
            video = cv2.VideoCapture(str(video_dir / video_file))
            frame_rate = video.get(cv2.CAP_PROP_FPS)
            try:
                1/frame_rate
                frame_rates.append(frame_rate)
            except ZeroDivisionError:
                print('Frame rate could not be retrieved: check that your video exists at the correct path')
                raise
            video.release()

        # time_ranges
        time_ranges = np.array(config_dict.get('project').get('time_range'))
        if time_ranges.shape == (0,):
            time_ranges = [None] * len(video_files)
        elif time_ranges.shape == (2,):
            time_ranges = [time_ranges.tolist()] * len(video_files)
        elif time_ranges.shape == (len(video_files), 2):
            time_ranges = time_ranges.tolist()
        else:
            raise ValueError('Time range must be [] for analysing all frames of all videos, or [start_time, end_time] for analysing all videos from start_time to end_time, or [[start_time1, end_time1], [start_time2, end_time2], ...] for analysing each video for a different time_range.')
    
    return video_dir, video_files, frame_rates, time_ranges, result_dir


def get_leaf_keys(config, prefix=''):
    '''
    Flatten configuration to map leaf keys to their full path
    '''

    leaf_keys = {}
    for key, value in config.items():
        if isinstance(value, dict):
            leaf_keys.update(get_leaf_keys(value, prefix=prefix + key + '.'))
        else:
            leaf_keys[prefix + key] = value
    return leaf_keys


def update_nested_dict(config, key_path, value):
    '''
    Update a nested dictionary based on a key path string like 'process.multiperson'.
    '''

    keys = key_path.split('.')
    d = config
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def set_nested_value(config, flat_key, value):
    '''
    Update the nested dictionary based on flattened keys
    '''

    keys = flat_key.split('.')
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def process(config='Config_demo.toml'):
    '''
    Read video or webcam input
    Compute 2D pose with RTMPose
    Compute joint and segment angles
    Optionally interpolate missing data, filter them, and display figures
    Save image and video results, save pose as trc files, save angles as csv files
    '''

    from Sports2D.process import process_fun
    
    if type(config) == dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    video_dir, video_files, frame_rates, time_ranges, result_dir = base_params(config_dict)
        
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / 'logs.txt', 'a+') as log_f: pass
    logging.basicConfig(format='%(message)s', level=logging.INFO, force=True, 
        handlers = [logging.handlers.TimedRotatingFileHandler(result_dir / 'logs.txt', when='D', interval=7), logging.StreamHandler()]) 
    
    for video_file, time_range, frame_rate in zip(video_files, time_ranges, frame_rates):
        currentDateAndTime = datetime.now()
        time_range_str = f' from {time_range[0]} to {time_range[1]} seconds' if time_range else ''

        logging.info("\n\n---------------------------------------------------------------------")
        logging.info(f"Processing {video_file}{time_range_str}")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"{f'Video input directory: {video_dir}' if video_file != 'webcam' else ''}")
        logging.info("---------------------------------------------------------------------")

        process_fun(config_dict, video_file, time_range, frame_rate, result_dir)

        elapsed_time = (datetime.now() - currentDateAndTime).total_seconds()        
        logging.info(f'\nProcessing {video_file} took {elapsed_time:.2f} s.')

    logging.shutdown()


def main():
    '''
    Run Sports2D from command line with entry points
    Run sports2d --help for more information

    Usage:
    - Run on Demo video with default parameters: 
        sports2d
    - Run on custom video with default parameters:
        sports2d --video_input path_to_video.mp4
    - Run on multiple videos with default parameters:
        sports2d --video_input path_to_video1.mp4 path_to_video2.mp4
    - Run on webcam with default parameters: 
        sports2d --video_input webcam
    - Run with custom parameters (all non specified are set to default): 
        sports2d --show_plots False --video_input webcam
    - Run with a toml configuration file: 
        sports2d --config path_to_config.toml
    '''

    # Dynamically add arguments for each leaf key in the DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description='Process 2D sports videos')
    parser.add_argument('--config', type=str, required=False, help='Path to the toml configuration file')
    leaf_keys = get_leaf_keys(DEFAULT_CONFIG)
    for key in leaf_keys:
        leaf_name = key.split('.')[-1]
        if leaf_name == 'video_input':
            parser.add_argument(f'--{leaf_name}', type=str, nargs='+', help=f'Override for {leaf_name}')
        else:
            parser.add_argument(f'--{leaf_name}', type=str, help=f'Override for {leaf_name}')
    args = parser.parse_args()

    # If config.toml file is provided, load it, else, use default config
    if args.config:
        new_config = toml.load(args.config)
    else:
        new_config = DEFAULT_CONFIG.copy()
        new_config.get('project').update({'video_dir': Path(__file__).resolve().parent / 'Demo'})

    # Override dictionary with command-line arguments if provided
    leaf_keys = get_leaf_keys(new_config)
    for leaf_key, default_value in leaf_keys.items():
        leaf_name = leaf_key.split('.')[-1]
        cli_value = getattr(args, leaf_name)
        if cli_value is not None:
            set_nested_value(new_config, leaf_key, cli_value)

    # Run process with the new configuration dictionary
    Sports2D.process(new_config)

if __name__ == "__main__":
    main()
