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
import subprocess
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
import imageio_ffmpeg as ffmpeg


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
angle_dict = { # lowercase!
    # joint angles
    'right ankle': [['RKnee', 'RAnkle', 'RBigToe', 'RHeel'], 'dorsiflexion', 90, 1],
    'left ankle': [['LKnee', 'LAnkle', 'LBigToe', 'LHeel'], 'dorsiflexion', 90, 1],
    'right knee': [['RAnkle', 'RKnee', 'RHip'], 'flexion', -180, 1],
    'left knee': [['LAnkle', 'LKnee', 'LHip'], 'flexion', -180, 1],
    'right hip': [['RKnee', 'RHip', 'Hip', 'Neck'], 'flexion', 0, -1],
    'left hip': [['LKnee', 'LHip', 'Hip', 'Neck'], 'flexion', 0, -1],
    # 'lumbar': [['Neck', 'Hip', 'RHip', 'LHip'], 'flexion', -180, -1],
    # 'neck': [['Head', 'Neck', 'RShoulder', 'LShoulder'], 'flexion', -180, -1],
    'right shoulder': [['RElbow', 'RShoulder', 'Hip', 'Neck'], 'flexion', 0, -1],
    'left shoulder': [['LElbow', 'LShoulder', 'Hip', 'Neck'], 'flexion', 0, -1],
    'right elbow': [['RWrist', 'RElbow', 'RShoulder'], 'flexion', 180, -1],
    'left elbow': [['LWrist', 'LElbow', 'LShoulder'], 'flexion', 180, -1],
    'right wrist': [['RElbow', 'RWrist', 'RIndex'], 'flexion', -180, 1],
    'left wrist': [['LElbow', 'LWrist', 'LIndex'], 'flexion', -180, 1],

    # segment angles
    'right foot': [['RBigToe', 'RHeel'], 'horizontal', 0, -1],
    'left foot': [['LBigToe', 'LHeel'], 'horizontal', 0, -1],
    'right shank': [['RAnkle', 'RKnee'], 'horizontal', 0, -1],
    'left shank': [['LAnkle', 'LKnee'], 'horizontal', 0, -1],
    'right thigh': [['RKnee', 'RHip'], 'horizontal', 0, -1],
    'left thigh': [['LKnee', 'LHip'], 'horizontal', 0, -1],
    'pelvis': [['LHip', 'RHip'], 'horizontal', 0, -1],
    'trunk': [['Neck', 'Hip'], 'horizontal', 0, -1],
    'shoulders': [['LShoulder', 'RShoulder'], 'horizontal', 0, -1],
    'head': [['Head', 'Neck'], 'horizontal', 0, -1],
    'right arm': [['RElbow', 'RShoulder'], 'horizontal', 0, -1],
    'left arm': [['LElbow', 'LShoulder'], 'horizontal', 0, -1],
    'right forearm': [['RWrist', 'RElbow'], 'horizontal', 0, -1],
    'left forearm': [['LWrist', 'LElbow'], 'horizontal', 0, -1],
    'right hand': [['RIndex', 'RWrist'], 'horizontal', 0, -1],
    'left hand': [['LIndex', 'LWrist'], 'horizontal', 0, -1]
    }

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

colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255),
            (125, 0, 0), (0, 125, 0), (0, 0, 125), (125, 125, 0), (125, 0, 125), (0, 125, 125), 
            (255, 125, 125), (125, 255, 125), (125, 125, 255), (255, 255, 125), (255, 125, 255), (125, 255, 255), (125, 125, 125),
            (255, 0, 125), (255, 125, 0), (0, 125, 255), (0, 255, 125), (125, 0, 255), (125, 255, 0), (0, 255, 0)]
thickness = 1

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
            return [len(list_of_arrays)] + [max(arr.shape[i] for arr in list_of_arrays if arr.size > 0) for i in range(list_of_arrays[0].ndim)]

    def pad_with_nans(list_of_arrays, target_shape):
        '''
        Recursively pad list_of_arrays with nans to match the target shape.
        '''
        if isinstance(list_of_arrays, np.ndarray):        
            # Pad the current array to the target shape        
            pad_width = []        
            for dim_index in range(0, len(target_shape)):
                if dim_index == len(list_of_arrays.shape) or dim_index > len(list_of_arrays.shape):
                    list_of_arrays = np.expand_dims(list_of_arrays, 0)
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
    Get the start time of a video using FFmpeg.
    '''

    try:
        ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        logging.warning(f"No ffmpeg exe could be found. Starting time set to 0.0. Error: {e}")
        return 0.0
    
    cmd = [ffmpeg_path, "-i", video_path]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    for line in result.stderr.splitlines():
        if "start:" in line:
            parts = line.split("start:")
            if len(parts) > 1:
                start_time = parts[1].split(",")[0].strip()
                return float(start_time)
    return 0.0  # Default to 0 if not found


def resample_video(vid_output_path, fps, desired_framerate):
    '''
    Resample video to the desired fps using ffmpeg.
    '''
   
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    new_vid_path = vid_output_path.parent / Path(vid_output_path.stem+'_2'+vid_output_path.suffix)
    subprocess.run([ffmpeg_path, '-i', vid_output_path, '-filter:v', f'setpts={fps/desired_framerate}*PTS', '-r', str(desired_framerate), new_vid_path])
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
