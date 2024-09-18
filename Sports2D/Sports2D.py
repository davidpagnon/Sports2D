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
        sports2d --multiperson false --mode lightweight --det_frequency 50
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
import sys
import toml
from datetime import datetime
from pathlib import Path
import logging, logging.handlers
import cv2
import numpy as np

from Sports2D import Sports2D


## CONSTANTS
DEFAULT_CONFIG =   {'project': {'video_input': ['demo.mp4'],
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
                'video_dir': ["d", "Current directory if not specified"],
                'result_dir': ["r", "Current directory if not specified"],
                'show_realtime_results': ["R", "show results in real-time. true if not specified"],
                'display_angle_values_on': ["a", '"body", "list", "body" "list", or "None". body list if not specified'],
                'show_graphs': ["G", "Show plots of raw and processed results. true if not specified"],
                'joint_angles': ["j", '"Right ankle" "Left ankle" "Right knee" "Left knee" "Right hip" "Left hip" "Right shoulder" "Left shoulder" "Right elbow" "Left elbow" if not specified'],
                'segment_angles': ["s", '"Right foot" "Left foot" "Right shank" "Left shank" "Right thigh" "Left thigh" "Pelvis" "Trunk" "Shoulders" "Head" "Right arm" "Left arm" "Right forearm" "Left forearm" if not specified'],
                'save_vid': ["V", "save processed video. true if not specified"],
                'save_img': ["I", "save processed images. true if not specified"],
                'save_pose': ["P", "save pose as trc files. true if not specified"],
                'save_angles': ["A", "save angles as mot files. true if not specified"],
                'pose_model': ["p", "Only body_with_feet is available for now. body_with_feet if not specified"],
                'mode': ["m", "light, balanced, or performance. balanced if not specified"],
                'det_frequency': ["f", "Run person detection only every N frames, and inbetween track previously detected bounding boxes. keypoint detection is still run on all frames.\n\
                                 Equal to or greater than 1, can be as high as you want in simple uncrowded cases. Much faster, but might be less accurate. 1 if not specified: detection runs on all frames"],
                'multiperson': ["M", "Multiperson involves tracking: will be faster if set to false. true if not specified"],
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
    if video_input == "webcam" or video_input == ["webcam"]:
        video_files = ['webcam']  # No video files for webcam
        frame_rates = [None]  # No frame rate for webcam
        time_ranges = [None]
    else:
        # video_files
        if isinstance(video_input, str):
            video_files = [Path(video_input)]
        else: 
            video_files = [Path(v) for v in video_input]

        # frame_rates
        frame_rates = []
        for video_file in video_files:
            video = cv2.VideoCapture(str(video_dir / video_file)) if video_dir else cv2.VideoCapture(str(video_file))
            if not video.isOpened():
                raise FileNotFoundError(f'Error: Could not open {video_dir/video_file}. Check that the file exists.')
            frame_rate = video.get(cv2.CAP_PROP_FPS)
            if frame_rate == 0:
                frame_rate = 30
                logging.warning(f'Error: Could not retrieve frame rate from {video_dir/video_file}. Defaulting to 30fps.')
            frame_rates.append(frame_rate)
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


def str2bool(v):
    '''
    Convert a string to a boolean value.
    '''
    
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

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
        logging.info("---------------------------------------------------------------------")

        process_fun(config_dict, video_file, time_range, frame_rate, result_dir)

        elapsed_time = (datetime.now() - currentDateAndTime).total_seconds()        
        logging.info(f'\nProcessing {video_file} took {elapsed_time:.2f} s.')

    logging.shutdown()


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
        sports2d --multiperson false --mode lightweight --det_frequency 50
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
    for leaf_key, default_value in leaf_keys.items():
        leaf_name = leaf_key.split('.')[-1]
        cli_value = getattr(args, leaf_name)
        if cli_value is not None:
            set_nested_value(new_config, leaf_key, cli_value)

    # Run process with the new configuration dictionary
    Sports2D.process(new_config)


if __name__ == "__main__":
    main()
