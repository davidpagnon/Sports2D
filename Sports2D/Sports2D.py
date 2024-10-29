#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##############################################################
    ## SPORTS2D                                                 ##
    ##############################################################
    
    Use sports2d to compute your athlete's pose, joint, and segment angles

    -----
    Help
    -----
    See https://github.com/davidpagnon/Sports2D
    Or run: sports2d --help
    Or check: https://github.com/davidpagnon/Sports2D/blob/main/Sports2D/Demo/Config_demo.toml

    -----
    Usage
    -----
    - Run on Demo video with default parameters: 
        sports2d
    - Run on custom video with default parameters:
        sports2d --video_input path_to_video.mp4
    - Run on multiple videos with default parameters:
        sports2d --video_input path_to_video1.mp4 path_to_video2.mp4
    - Run on webcam with default parameters: 
        sports2d --video_input webcam
    - Run with custom parameters (all non specified are set to default): 
        sports2d --show_plots False --time_range 0 2.1 --result_dir path_to_result_dir
        sports2d --multi_person false --mode lightweight --det_frequency 50
    - Run with a toml configuration file: 
        sports2d --config path_to_config.toml
   
    -----
    Installation
    -----
    Optional: 
    - Install Miniconda
    - Open a Anaconda Prompt and type: 
    `conda create -n Sports2D python>=3.7`
    `conda activate Sports2D`
    pip install .
    
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
    How it works
    -----
    Detects 2D joint centers from a video or a webcam with RTMLib.
    Computes selected joint and segment angles. 
    Optionally saves processed image files and video file.
    Optionally saves processed poses as a TRC file, and angles as a MOT file (OpenSim compatible).

    Further details. Sports2D:
    - loads skeleton information
    - reads stream from a video or a webcam
    - sets up the RTMLib pose tracker from RTMlib with specified parameters
    - detects poses within the selected time or frame range
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

    -----
    Angle conventions
    -----
    Joint angles:
    - Ankle dorsiflexion: Between heel and big toe, and ankle and knee + 90°
    - Knee flexion: Between hip, knee, and ankle 
    - Hip flexion: Between knee, hip, and shoulder
    - Shoulder flexion: Between hip, shoulder, and elbow
    - Elbow flexion: Between wrist, elbow, and shoulder

    Segment angles:
    Angles are measured anticlockwise between the horizontal and the segment.
    - Foot: Between heel and big toe
    - Shank: Between ankle and knee
    - Thigh: Between hip and knee
    - Pelvis: Between right and left hip
    - Trunk: Between hip midpoint and neck
    - Shoulders: Between right and left shoulder
    - Arm: Between shoulder and elbow
    - Forearm: Between elbow and wrist

    -----
    To-do list
    -----
    - GUI applications for all platforms
    - Constrain points to OpenSim skeletal model for better angle estimation (cf Pose2Sim but in 2D https://github.com/perfanalytics/pose2sim)
    - Pose refinement: click and move badly estimated 2D points (cf DeepLabCut: https://www.youtube.com/watch?v=bEuBKB7eqmk)
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
from Sports2D.Utilities.common import *


## CONSTANTS
DEFAULT_CONFIG =   {'project': {'video_input': ['demo.mp4'],
                                'time_range': [],
                                'frame_range': [],
                                'video_dir': '',
                                'webcam_id': 0,
                                'input_size': [1280, 720],
                                'save_video': 'to_video'
                                },
                    'process': {'multi_person': True,
                                'multiperson': True,
                                'show_realtime_results': True,
                                'save_pose': True,
                                'save_angles': True,
                                'result_dir': ''
                                },
                    'pose':     {'pose_model': 'body_with_feet',
                                'mode': 'balanced',
                                'det_frequency': 1,
                                'tracking_mode': 'sports2d',
                                'keypoint_likelihood_threshold': 0.3,
                                'average_likelihood_threshold': 0.5,
                                'keypoint_number_threshold': 0.3
                                },
                    'angles':   {'display_angle_values_on': ['body', 'list'],
                                'fontSize': 0.3,
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
                                        'show_graphs': True,
                                        'filter_type': 'butterworth',
                                        'butterworth': {'order': 4, 'cut_off_frequency': 3},
                                        'gaussian': {'sigma_kernel': 1},
                                        'loess': {'nb_values_used': 5},
                                        'median': {'kernel_size': 3}
                                        }
                    }

CONFIG_HELP =   {'config': ["c", "Path to a toml configuration file"],
                'video_input': ["i", "webcam, or video_path.mp4, or video1_path.avi video2_path.mp4 ... Beware that images won't be saved if paths contain non ASCII characters"],
                'webcam_id': ["w", "webcam ID. 0 if not specified"],
                'time_range': ["t", "start_time, end_time. In seconds. Whole video if not specified"],
                'frame_range': ["F", "start_frame, end_frame. Whole video if not specified"],
                'video_dir': ["d", "Current directory if not specified"],
                'result_dir': ["r", "Current directory if not specified"],
                'show_realtime_results': ["R", "show results in real-time. true if not specified"],
                'display_angle_values_on': ["a", '"body", "list", "body" "list", or "None". body list if not specified'],
                'show_graphs': ["G", "Show plots of raw and processed results. true if not specified"],
                'joint_angles': ["j", '"Right ankle" "Left ankle" "Right knee" "Left knee" "Right hip" "Left hip" "Right shoulder" "Left shoulder" "Right elbow" "Left elbow" if not specified'],
                'segment_angles': ["s", '"Right foot" "Left foot" "Right shank" "Left shank" "Right thigh" "Left thigh" "Pelvis" "Trunk" "Shoulders" "Head" "Right arm" "Left arm" "Right forearm" "Left forearm" if not specified'],
                'save_video': ["V", "Specify output format: 'to_video', 'to_images', 'none', or a combination as a list ['to_video', 'to_images']. Default is 'to_video' if not specified."],
                'save_pose': ["P", "save pose as trc files. true if not specified"],
                'save_angles': ["A", "save angles as mot files. true if not specified"],
                'pose_model': ["p", "Only body_with_feet is available for now. body_with_feet if not specified"],
                'mode': ["m", "light, balanced, or performance. balanced if not specified"],
                'det_frequency': ["f", "Run person detection only every N frames, and inbetween track previously detected bounding boxes. keypoint detection is still run on all frames.\n\
                                 Equal to or greater than 1, can be as high as you want in simple uncrowded cases. Much faster, but might be less accurate. 1 if not specified: detection runs on all frames"],
                'multi_person': ["M", "Multi-person involves tracking: will be faster if set to false. true if not specified"],
                'multiperson': ["", "Warning: 'multiperson' is deprecated. Please switch to 'multi_person'. Multi-person involves tracking: will be faster if set to false. true if not specified"],
                'tracking_mode': ["", "sports2d or rtmlib. sports2d is generally much more accurate and comparable in speed. sports2d if not specified"],
                'input_size': ["", "width, height. 1280, 720 if not specified. Lower resolution will be faster but less precise"],
                'keypoint_likelihood_threshold': ["", "Detected keypoints are not retained if likelihood is below this threshold. 0.3 if not specified"],
                'average_likelihood_threshold': ["", "Detected persons are not retained if average keypoint likelihood is below this threshold. 0.5 if not specified"],
                'keypoint_number_threshold': ["", "Detected persons are not retained if number of detected keypoints is below this threshold. 0.3 if not specified, i.e., i.e., 30 percent"],
                'fontSize': ["", "Font size for angle values. 0.3 if not specified"],
                'flip_left_right': ["", "true or false. true to get consistent angles with people facing both left and right sides. Set it to false if you want timeseries to be continuous even when the participent switches their stance. true if not specified"],
                'interpolate': ["", "Interpolate missing data. true if not specified"],
                'interp_gap_smaller_than': ["", "Interpolate sequences of missing data if they are less than N frames long. 10 if not specified"],
                'fill_large_gaps_with': ["", "last_value, nan, or zeros. last_value if not specified"],
                'filter': ["", "Filter results. true if not specified"],
                'filter_type': ["", "butterworth, gaussian, median, or loess. butterworth if not specified"],
                'order': ["", "Order of the Butterworth filter. 4 if not specified"],
                'cut_off_frequency': ["", "Cut-off frequency of the Butterworth filter. 3 if not specified"],
                'sigma_kernel': ["", "Sigma of the gaussian filter. 1 if not specified"],
                'nb_values_used': ["", "Number of values used for the loess filter. 5 if not specified"],
                'kernel_size': ["", "Kernel size of the median filter. 3 if not specified"]
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
def main():
    '''
    Use sports2d to compute your athlete's pose, joint, and segment angles

    Help:
    See https://github.com/davidpagnon/Sports2D
    Or run: sports2d --help
    Or check: https://github.com/davidpagnon/Sports2D/blob/main/Sports2D/Demo/Config_demo.toml

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
        sports2d --show_plots False --time_range 0 2.1 --result_dir path_to_result_dir
        sports2d --multi_person false --mode lightweight --det_frequency 50
    - Run with a toml configuration file: 
        sports2d --config path_to_config.toml
    '''

    # Dynamically add arguments for each leaf key in the DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description="Use sports2d to compute your athlete's pose, joint, and segment angles. See https://github.com/davidpagnon/Sports2D")
    parser.add_argument('-C', '--config', type=str, required=False, help='Path to a toml configuration file')

    leaf_keys = get_leaf_keys(DEFAULT_CONFIG)
    leaf_keys = {k.split('.')[-1]:v for k,v in leaf_keys.items()}
    for leaf_name in list(CONFIG_HELP.keys())[1:]:
        short_key = CONFIG_HELP[leaf_name][0]
        arg_str = [f'-{short_key}', f'--{leaf_name}'] if short_key else [f'--{leaf_name}']
        if type(leaf_keys[leaf_name]) == bool:
            parser.add_argument(*arg_str, type=str2bool, help=CONFIG_HELP[leaf_name][1])
        elif type(leaf_keys[leaf_name]) == list:
            if len(leaf_keys[leaf_name])==0: 
                list_type = float # time_range for example
            else:
                list_type = type(leaf_keys[leaf_name][0])
            parser.add_argument(*arg_str, type=list_type, nargs='*', help=CONFIG_HELP[leaf_name][1])
        else:
            parser.add_argument(*arg_str, type=type(leaf_keys[leaf_name]), help=CONFIG_HELP[leaf_name][1])
    args = parser.parse_args()

    # If config.toml file is provided, load it, else, use default config
    if args.config:
        new_config = toml.load(args.config)
    else:
        new_config = DEFAULT_CONFIG.copy()
        if not args.video_input: 
            new_config.get('project').update({'video_dir': Path(__file__).resolve().parent / 'Demo'})

    # Override dictionary with command-line arguments if provided
    leaf_keys = get_leaf_keys(new_config)
    for leaf_key, _ in leaf_keys.items():
        leaf_name = leaf_key.split('.')[-1]
        cli_value = getattr(args, leaf_name)
        if cli_value is not None:
            set_nested_value(new_config, leaf_key, cli_value)

    # Run process with the new configuration dictionary
    Sports2D.process(new_config)


def process(config='Config_demo.toml'):
    '''
    Read video or webcam input
    Compute 2D pose with RTMPose
    Compute joint and segment angles
    Optionally interpolate missing data, filter them, and display figures
    Save image and video results, save pose as trc files, save angles as csv files
    '''

    from Sports2D.process import process_fun, post_processing
    from Sports2D.Utilities.common import setup_pose_tracker
    
    if type(config) == dict:
        config_dict = config
    else:
        config_dict = toml.load(config)

    mode = config_dict.get('pose').get('mode')
    det_frequency = config_dict.get('pose').get('det_frequency')

    video_paths, frame_ranges, output_dir = base_params(config_dict)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'logs.txt', 'a+'): pass

    setup_logging(output_dir)

    for video_path, frame_range in zip(video_paths, frame_ranges):
        currentDateAndTime = datetime.now()

        range_str = ''
        if video_path != "webcam":
            range_str = f' from frame {frame_range[0]} to frame {frame_range[1]}'

        logging.info("\n\n---------------------------------------------------------------------")
        logging.info(f"Processing {video_path} {range_str}")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info("---------------------------------------------------------------------")
        
        pose_tracker = setup_pose_tracker(det_frequency, mode)

        logging.info(f'Pose tracking set up for BodyWithFeet model in {mode} mode.')
        logging.info(f'Persons are detected every {det_frequency} frames and tracked inbetween.')

        frame_idx, fps, output_dir_name, all_frames_X, all_frames_Y, all_frames_angles = process_fun(config_dict, video_path, pose_tracker, frame_range, output_dir)

        post_processing(config_dict, video_path, frame_idx, fps, frame_range, output_dir, output_dir_name, all_frames_X, all_frames_Y, all_frames_angles)

        elapsed_time = (datetime.now() - currentDateAndTime).total_seconds()        
        logging.info(f'\nProcessing {video_path} took {elapsed_time:.2f} s.')

    logging.shutdown()


def base_params(config_dict):
    '''
    Retrieve sequence name and frames to be analyzed.
    '''


    # Resolve video_dir and output_dir
    video_dir = Path(config_dict['project'].get('video_dir', '')).resolve()
    if not video_dir.exists():
        video_dir = Path.cwd()
    output_dir = Path(config_dict['process'].get('result_dir', '')).resolve()
    if not output_dir.exists():
        output_dir = Path.cwd()

    # Get video_input
    video_input = config_dict['project'].get('video_input')

    if video_input == "webcam" or video_input == ["webcam"]:
        video_paths = ['webcam']
        frame_rates = [None]
        total_frames_list = [None]
        frame_ranges = [None]
    else:
        # Ensure video_input is a list
        if isinstance(video_input, str):
            video_input = [video_input]
        video_paths = [video_dir / Path(v) for v in video_input]

        # Verify video files exist and get frame rates and total frames
        frame_rates = []
        total_frames_list = []
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

        # Helper function to process ranges
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

        # Process time_ranges and frame_ranges
        time_ranges = process_ranges('time_range')
        frame_ranges = process_ranges('frame_range')

        # Combine time_ranges and frame_ranges into final frame_ranges
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

    return video_paths, frame_ranges, output_dir

if __name__ == "__main__":
    main()
