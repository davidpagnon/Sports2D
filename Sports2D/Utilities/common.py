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
import re
import sys
import toml
import subprocess
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from scipy import interpolate
import imageio_ffmpeg as ffmpeg
import cv2

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4.0"
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
    'left wrist': [['LElbow', 'LIndex', 'LWrist'], 'flexion', -180, 1],

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

colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255),
            (125, 0, 0), (0, 125, 0), (0, 0, 125), (125, 125, 0), (125, 0, 125), (0, 125, 125), 
            (255, 125, 125), (125, 255, 125), (125, 125, 255), (255, 255, 125), (255, 125, 255), (125, 255, 255), (125, 125, 125),
            (255, 0, 125), (255, 125, 0), (0, 125, 255), (0, 255, 125), (125, 0, 255), (125, 255, 0), (0, 255, 0)]
thickness = 1

## CLASSES
class plotWindow():
    '''
    Display several figures in tabs
    Taken from https://github.com/superjax/plotWindow/blob/master/plotWindow.py

    USAGE:
    pw = plotWindow()
    f = plt.figure()
    plt.plot(x1, y1)
    pw.addPlot("1", f)
    f = plt.figure()
    plt.plot(x2, y2)
    pw.addPlot("2", f)
    '''
    def __init__(self, parent=None):
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.setWindowTitle("Multitabs figure")
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(1280, 720)
        self.MainWindow.show()

    def addPlot(self, title, figure):
        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        figure.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.91, wspace=0.2, hspace=0.2)
        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        self.app.exec_()
        
## FUNCTIONS
def read_trc(trc_path):
    '''
    Read a TRC file and extract its contents.

    INPUTS:
    - trc_path (str): The path to the TRC file.

    OUTPUTS:
    - tuple: A tuple containing the Q coordinates, frames column, time column, marker names, and header.
    '''

    try:
        with open(trc_path, 'r') as trc_file:
            header = [next(trc_file) for _ in range(5)]
        markers = header[3].split('\t')[2::3]
        markers = [m.strip() for m in markers if m.strip()] # remove last \n character
       
        trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding='utf-8')
        frames_col, time_col = trc_df.iloc[:, 0], trc_df.iloc[:, 1]
        Q_coords = trc_df.drop(trc_df.columns[[0, 1]], axis=1)
        Q_coords = Q_coords.loc[:, ~Q_coords.columns.str.startswith('Unnamed')] # remove unnamed columns
        Q_coords.columns = np.array([[m,m,m] for m in markers]).ravel().tolist()

        return Q_coords, frames_col, time_col, markers, header
    
    except Exception as e:
        raise ValueError(f"Error reading TRC file at {trc_path}: {e}")


def interpolate_zeros_nans(col, *args):
    '''
    Interpolate missing points (of value zero),
    unless more than N contiguous values are missing.

    INPUTS:
    - col: pandas column of coordinates
    - args[0] = N: max number of contiguous bad values, above which they won't be interpolated
    - args[1] = kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default: 'cubic'

    OUTPUT:
    - col_interp: interpolated pandas column
    '''

    if len(args)==2:
        N, kind = args
    if len(args)==1:
        N = np.inf
        kind = args[0]
    if not args:
        N = np.inf
    
    # Interpolate nans
    mask = ~(np.isnan(col) | col.eq(0)) # true where nans or zeros
    idx_good = np.where(mask)[0]
    if len(idx_good) <= 4:
        return col
    
    if 'kind' not in locals(): # 'linear', 'slinear', 'quadratic', 'cubic'
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind="linear", bounds_error=False)
    else:
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind, fill_value='extrapolate', bounds_error=False)
    col_interp = np.where(mask, col, f_interp(col.index)) #replace at false index with interpolated values
    
    # Reintroduce nans if length of sequence > N
    idx_notgood = np.where(~mask)[0]
    gaps = np.where(np.diff(idx_notgood) > 1)[0] + 1 # where the indices of true are not contiguous
    sequences = np.split(idx_notgood, gaps)
    if sequences[0].size>0:
        for seq in sequences:
            if len(seq) > N: # values to exclude from interpolation are set to false when they are too long 
                col_interp[seq] = np.nan
    
    return col_interp


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


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


def points_to_angles(points_list):
    '''
    If len(points_list)==2, computes clockwise angle of ab vector w.r.t. horizontal (e.g. RBigToe, RHeel) 
    If len(points_list)==3, computes clockwise angle from a to c around b (e.g. Neck, Hip, Knee) 
    If len(points_list)==4, computes clockwise angle between vectors ab and cd (e.g. Neck Hip, RKnee RHip)
    
    Points can be 2D or 3D.
    If parameters are float, returns a float between 0.0 and 360.0
    If parameters are arrays, returns an array of floats between 0.0 and 360.0

    INPUTS:
    - points_list: list of arrays of points

    OUTPUTS:
    - ang_deg: float or array of floats. The angle(s) in degrees.
    '''

    if len(points_list) < 2: # if not enough points, return None
        return np.nan
    
    points_array = np.array(points_list)
    dimensions = points_array.shape[-1]

    if len(points_list) == 2:
        vector_u = points_array[0] - points_array[1]
        if len(points_array.shape)==2:
            vector_v = np.array([1, 0, 0]) # Here vector X, could be any horizontal vector
        else:
            vector_v = np.array([[1, 0, 0],] * points_array.shape[1]) 

    elif len(points_list) == 3:
        vector_u = points_array[0] - points_array[1]
        vector_v = points_array[2] - points_array[1]

    elif len(points_list) == 4:
        vector_u = points_array[1] - points_array[0]
        vector_v = points_array[3] - points_array[2]
        
    else:
        return np.nan

    if dimensions == 2: 
        vector_u = vector_u[:2]
        vector_v = vector_v[:2]
        ang = np.arctan2(vector_u[1], vector_u[0]) - np.arctan2(vector_v[1], vector_v[0])
    else:
        cross_product = np.cross(vector_u, vector_v)
        dot_product = np.einsum('ij,ij->i', vector_u, vector_v) # np.dot(vector_u, vector_v) # does not work with time series
        ang = np.arctan2(np.linalg.norm(cross_product, axis=1), dot_product)

    ang_deg = np.degrees(ang)
    # ang_deg = np.array(np.degrees(np.unwrap(ang*2)/2))
    
    return ang_deg


def fixed_angles(points_list, ang_name):
    '''
    Add offset and multiplying factor to angles

    INPUTS:
    - points_list: list of arrays of points
    - ang_name: str. The name of the angle to consider.

    OUTPUTS:
    - ang: float. The angle in degrees.
    '''

    ang_params = angle_dict[ang_name]
    ang = points_to_angles(points_list)
    ang += ang_params[2]
    ang *= ang_params[3]
    if ang_name in ['pelvis', 'shoulders']:
        ang = np.where(ang>90, ang-180, ang)
        ang = np.where(ang<-90, ang+180, ang)
    else:
        ang = np.where(ang>180, ang-360, ang)
        ang = np.where(ang<-180, ang+360, ang)

    return ang


def mean_angles(trc_data, ang_to_consider = ['right knee', 'left knee', 'right hip', 'left hip']):
    '''
    Compute the mean angle time series from 3D points for a given list of angles.

    INPUTS:
    - trc_data (DataFrame): The triangulated coordinates of the markers.
    - ang_to_consider (list): The list of angles to consider (requires angle_dict).

    OUTPUTS:
    - ang_mean: The mean angle time series.
    '''

    ang_to_consider = ['right knee', 'left knee', 'right hip', 'left hip']

    angs = []
    for ang_name in ang_to_consider:
        ang_params = angle_dict[ang_name]
        ang_mk = ang_params[0]
        if 'Neck' not in trc_data.columns:
            df_MidShoulder = pd.DataFrame((trc_data['RShoulder'].values + trc_data['LShoulder'].values) /2)
            df_MidShoulder.columns = ['Neck']*3
            trc_data = pd.concat((trc_data.reset_index(drop=True), df_MidShoulder), axis=1)

        pts_for_angles = []
        for pt in ang_mk:
            # pts_for_angles.append(trc_data.iloc[:,markers.index(pt)*3:markers.index(pt)*3+3])
            pts_for_angles.append(trc_data[pt])

        ang = fixed_angles(pts_for_angles, ang_name)
        ang = np.abs(ang)
        angs.append(ang)

    ang_mean = np.mean(angs, axis=0)

    return ang_mean


def add_neck_hip_coords(kpt_name, p_X, p_Y, p_scores, kpt_ids, kpt_names):
    '''
    Add neck (midshoulder) and hip (midhip) coordinates if neck and hip are not available
    
    INPUTS:
    - kpt_name: name of the keypoint to add (neck, hip)
    - p_X: list of x coordinates after flipping if needed
    - p_Y: list of y coordinates
    - p_scores: list of confidence scores
    - kpt_ids: list of keypoint ids (see skeletons.py)
    - kpt_names: list of keypoint names (see skeletons.py)
    
    OUTPUTS:
    - p_X: list of x coordinates with added missing coordinate
    - p_Y: list of y coordinates with added missing coordinate
    - p_scores: list of confidence scores with added missing score
    '''

    names, ids = kpt_names.copy(), kpt_ids.copy()
    names.append(kpt_name)
    ids.append(len(p_X))
    if kpt_name == 'Neck':
        mid_X = (np.abs(p_X[ids[names.index('LShoulder')]]) + np.abs(p_X[ids[names.index('RShoulder')]])) /2
        mid_Y = (p_Y[ids[names.index('LShoulder')]] + p_Y[ids[names.index('RShoulder')]])/2
        mid_score = (p_scores[ids[names.index('LShoulder')]] + p_scores[ids[names.index('RShoulder')]])/2
    elif kpt_name == 'Hip':
        mid_X = (np.abs(p_X[ids[names.index('LHip')]]) + np.abs(p_X[ids[names.index('RHip')]]) ) /2
        mid_Y = (p_Y[ids[names.index('LHip')]] + p_Y[ids[names.index('RHip')]])/2
        mid_score = (p_scores[ids[names.index('LHip')]] + p_scores[ids[names.index('RHip')]])/2
    else:
        raise ValueError("kpt_name must be 'Neck' or 'Hip'")
    p_X = np.append(p_X, mid_X)
    p_Y = np.append(p_Y, mid_Y)
    p_scores = np.append(p_scores, mid_score)

    return p_X, p_Y, p_scores


def best_coords_for_measurements(trc_data, keypoints_names, fastest_frames_to_remove_percent=0.2, close_to_zero_speed=0.2, large_hip_knee_angles=45):
    '''
    Compute the best coordinates for measurements, after removing:
    - 20% fastest frames (may be outliers)
    - frames when speed is close to zero (person is out of frame): 0.2 m/frame, or 50 px/frame
    - frames when hip and knee angle below 45° (imprecise coordinates when person is crouching)
    
    INPUTS:
    - trc_data: pd.DataFrame. The XYZ coordinates of each marker
    - keypoints_names: list. The list of marker names
    - fastest_frames_to_remove_percent: float
    - close_to_zero_speed: float (sum for all keypoints: about 50 px/frame or 0.2 m/frame)
    - large_hip_knee_angles: int
    - trimmed_extrema_percent

    OUTPUT:
    - trc_data_low_speeds_low_angles: pd.DataFrame. The best coordinates for measurements
    '''

    # Add MidShoulder column
    df_MidShoulder = pd.DataFrame((trc_data['RShoulder'].values + trc_data['LShoulder'].values) /2)
    df_MidShoulder.columns = ['MidShoulder']*3
    trc_data = pd.concat((trc_data.reset_index(drop=True), df_MidShoulder), axis=1)

    # Add Hip column if not present
    n_markers_init = len(keypoints_names)
    if 'Hip' not in keypoints_names:
        df_Hip = pd.DataFrame((trc_data['RHip'].values + trc_data['LHip'].values) /2)
        df_Hip.columns = ['Hip']*3
        trc_data = pd.concat((trc_data.reset_index(drop=True), df_Hip), axis=1)
    n_markers = len(keypoints_names)

    # Using 80% slowest frames
    sum_speeds = pd.Series(np.nansum([np.linalg.norm(trc_data.iloc[:,kpt:kpt+3].diff(), axis=1) for kpt in range(n_markers)], axis=0))
    sum_speeds = sum_speeds[sum_speeds>close_to_zero_speed] # Removing when speeds close to zero (out of frame)
    if len(sum_speeds)==0:
        raise ValueError('All frames have speed close to zero. Make sure the person is moving and correctly detected, or change close_to_zero_speed to a lower value.')
    min_speed_indices = sum_speeds.abs().nsmallest(int(len(sum_speeds) * (1-fastest_frames_to_remove_percent))).index
    trc_data_low_speeds = trc_data.iloc[min_speed_indices].reset_index(drop=True)
    
    # Only keep frames with hip and knee flexion angles below 45% 
    # (if more than 50 of them, else take 50 smallest values)
    try:
        ang_mean = mean_angles(trc_data_low_speeds, ang_to_consider = ['right knee', 'left knee', 'right hip', 'left hip'])
        trc_data_low_speeds_low_angles = trc_data_low_speeds[ang_mean < large_hip_knee_angles]
        if len(trc_data_low_speeds_low_angles) < 50:
            trc_data_low_speeds_low_angles = trc_data_low_speeds.iloc[pd.Series(ang_mean).nsmallest(50).index]
    except:
        logging.warning(f"At least one among the RAnkle, RKnee, RHip, RShoulder, LAnkle, LKnee, LHip, LShoulder markers is missing for computing the knee and hip angles. Not restricting these agles to be below {large_hip_knee_angles}°.")

    if n_markers_init < n_markers:
        trc_data_low_speeds_low_angles = trc_data_low_speeds_low_angles.iloc[:,:-3]

    return trc_data_low_speeds_low_angles


def compute_height(trc_data, keypoints_names, fastest_frames_to_remove_percent=0.1, close_to_zero_speed=50, large_hip_knee_angles=45, trimmed_extrema_percent=0.5):
    '''
    Compute the height of the person from the trc data.

    INPUTS:
    - trc_data: pd.DataFrame. The XYZ coordinates of each marker
    - keypoints_names: list. The list of marker names
    - fastest_frames_to_remove_percent: float. Frames with high speed are considered as outliers
    - close_to_zero_speed: float. Sum for all keypoints: about 50 px/frame or 0.2 m/frame
    - large_hip_knee_angles5: float. Hip and knee angles below this value are considered as imprecise
    - trimmed_extrema_percent: float. Proportion of the most extreme segment values to remove before calculating their mean)
    
    OUTPUT:
    - height: float. The estimated height of the person
    '''
    
    # Retrieve most reliable coordinates, adding MidShoulder and Hip columns if not present
    trc_data_low_speeds_low_angles = best_coords_for_measurements(trc_data, keypoints_names, 
                                                                  fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, close_to_zero_speed=close_to_zero_speed, large_hip_knee_angles=large_hip_knee_angles)

    # Automatically compute the height of the person
    feet_pairs = [['RHeel', 'RAnkle'], ['LHeel', 'LAnkle']]
    try:
        rfoot, lfoot = [euclidean_distance(trc_data_low_speeds_low_angles[pair[0]],trc_data_low_speeds_low_angles[pair[1]]) for pair in feet_pairs]
    except:
        rfoot, lfoot = 10, 10
        logging.warning('The Heel marker is missing from your model. Considering Foot to Heel size as 10 cm.')

    ankle_to_shoulder_pairs =  [['RAnkle', 'RKnee'], ['RKnee', 'RHip'], ['RHip', 'RShoulder'],
                                ['LAnkle', 'LKnee'], ['LKnee', 'LHip'], ['LHip', 'LShoulder']]
    try:
        rshank, rfemur, rback, lshank, lfemur, lback = [euclidean_distance(trc_data_low_speeds_low_angles[pair[0]],trc_data_low_speeds_low_angles[pair[1]]) for pair in ankle_to_shoulder_pairs]
    except:
        logging.error('At least one of the following markers is missing for computing the height of the person:\
                            RAnkle, RKnee, RHip, RShoulder, LAnkle, LKnee, LHip, LShoulder.\n\
                            Make sure that the person is entirely visible, or use a calibration file instead, or set "to_meters=false".')
        raise ValueError('At least one of the following markers is missing for computing the height of the person:\
                         RAnkle, RKnee, RHip, RShoulder, LAnkle, LKnee, LHip, LShoulder.\
                         Make sure that the person is entirely visible, or use a calibration file instead, or set "to_meters=false".')

    try:
        head_pair = [['MidShoulder', 'Head']]
        head = [euclidean_distance(trc_data_low_speeds_low_angles[pair[0]],trc_data_low_speeds_low_angles[pair[1]]) for pair in head_pair][0]
    except:
        head_pair = [['MidShoulder', 'Nose']]
        head = [euclidean_distance(trc_data_low_speeds_low_angles[pair[0]],trc_data_low_speeds_low_angles[pair[1]]) for pair in head_pair][0]\
                *1.33
        logging.warning('The Head marker is missing from your model. Considering Neck to Head size as 1.33 times Neck to MidShoulder size.')
    
    heights = (rfoot + lfoot)/2 + (rshank + lshank)/2 + (rfemur + lfemur)/2 + (rback + lback)/2 + head
    
    # Remove the 20% most extreme values
    height = trimmed_mean(heights, trimmed_extrema_percent=trimmed_extrema_percent)

    return height


def euclidean_distance(q1, q2):
    '''
    Euclidean distance between 2 points (N-dim).
    
    INPUTS:
    - q1: list of N_dimensional coordinates of point
         or list of N points of N_dimensional coordinates
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    '''
    
    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    if np.isnan(dist).all():
        dist =  np.empty_like(dist)
        dist[...] = np.inf
    
    if len(dist.shape)==1:
        euc_dist = np.sqrt(np.nansum( [d**2 for d in dist]))
    else:
        euc_dist = np.sqrt(np.nansum( [d**2 for d in dist], axis=1))
    
    return euc_dist


def trimmed_mean(arr, trimmed_extrema_percent=0.5):
    '''
    Trimmed mean calculation for an array.

    INPUTS:
    - arr (np.array): The input array.
    - trimmed_extrema_percent (float): The percentage of values to be trimmed from both ends.

    OUTPUTS:
    - float: The trimmed mean of the array.
    '''

    # Sort the array
    sorted_arr = np.sort(arr)
    
    # Determine the indices for the 25th and 75th percentiles (if trimmed_percent = 0.5)
    lower_idx = int(len(sorted_arr) * (trimmed_extrema_percent/2))
    upper_idx = int(len(sorted_arr) * (1 - trimmed_extrema_percent/2))
    
    # Slice the array to exclude the 25% lowest and highest values
    trimmed_arr = sorted_arr[lower_idx:upper_idx]
    
    # Return the mean of the remaining values
    return np.mean(trimmed_arr)


def retrieve_calib_params(calib_file):
    '''
    Compute projection matrices from toml calibration file.
    
    INPUT:
    - calib_file: calibration .toml file.
    
    OUTPUT:
    - S: (h,w) vectors as list of 2x1 arrays
    - K: intrinsic matrices as list of 3x3 arrays
    - dist: distortion vectors as list of 4x1 arrays
    - inv_K: inverse intrinsic matrices as list of 3x3 arrays
    - optim_K: intrinsic matrices for undistorting points as list of 3x3 arrays
    - R: rotation rodrigue vectors as list of 3x1 arrays
    - T: translation vectors as list of 3x1 arrays
    '''
    
    calib = toml.load(calib_file)

    cal_keys = [c for c in calib.keys() 
                if c not in ['metadata', 'capture_volume', 'charuco', 'checkerboard'] 
                and isinstance(calib[c],dict)]
    S, K, dist, optim_K, inv_K, R, R_mat, T = [], [], [], [], [], [], [], []
    for c, cam in enumerate(cal_keys):
        S.append(np.array(calib[cam]['size']))
        K.append(np.array(calib[cam]['matrix']))
        dist.append(np.array(calib[cam]['distortions']))
        optim_K.append(cv2.getOptimalNewCameraMatrix(K[c], dist[c], [int(s) for s in S[c]], 1, [int(s) for s in S[c]])[0])
        inv_K.append(np.linalg.inv(K[c]))
        R.append(np.array(calib[cam]['rotation']))
        R_mat.append(cv2.Rodrigues(R[c])[0])
        T.append(np.array(calib[cam]['translation']))
    calib_params_dict = {'S': S, 'K': K, 'dist': dist, 'inv_K': inv_K, 'optim_K': optim_K, 'R': R, 'R_mat': R_mat, 'T': T}
            
    return calib_params_dict


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