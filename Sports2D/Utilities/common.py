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
import itertools as it
import logging
from collections import defaultdict
from anytree import PreOrderIter

import numpy as np
import pandas as pd
from scipy import interpolate
import imageio_ffmpeg as ffmpeg
import cv2
import c3d

import matplotlib.pyplot as plt
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
                    'front':
                        {"RHip": 0.0301, "RKnee": 0.0179, "RAnkle": 0.0230, "RBigToe": 0.2179, "RHeel": -0.0119, "RSmallToe": 0.1804, 
                        "RShoulder": -0.01275, "RElbow": 0.0119, "RWrist": 0.0002, "RThumb": 0.0106, "RIndex": -0.0004, "RPinky": -0.0009, "REye": 0.0702, 
                        "LHip": -0.0301, "LKnee": -0.0179, "LAnkle": 0.0230, "LBigToe": 0.2179, "LHeel": -0.0119, "LSmallToe": 0.1804, 
                        "LShoulder": 0.01275, "LElbow": -0.0119, "LWrist": -0.0002, "LThumb": -0.0106, "LIndex": 0.0004, "LPinky": 0.0009, "LEye": -0.0702, 
                        "Hip": 0.0301, "Neck": -0.0008, "Head": 0.0655, "Nose": 0.1076},
                    'back':
                        {"RHip": -0.0301, "RKnee": -0.0179, "RAnkle": -0.0230, "RBigToe": -0.2179, "RHeel": 0.0119, "RSmallToe": -0.1804, 
                        "RShoulder": 0.01275, "RElbow": -0.0119, "RWrist": -0.0002, "RThumb": -0.0106, "RIndex": 0.0004, "RPinky": 0.0009, "REye": -0.0702, 
                        "LHip": 0.0301, "LKnee": 0.0179, "LAnkle": -0.0230, "LBigToe": -0.2179, "LHeel": 0.0119, "LSmallToe": -0.1804, 
                        "LShoulder": -0.01275, "LElbow": 0.0119, "LWrist": 0.0002, "LThumb": 0.0106, "LIndex": -0.0004, "LPinky": -0.0009, "LEye": 0.0702, 
                        "Hip": 0.0301, "Neck": -0.0008, "Head": -0.0655, "Nose": 0.1076},
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
def to_dict(d):
    '''
    Convert a defaultdict to a dict.
    '''
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    return d


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


def extract_trc_data(trc_path):
    '''
    Extract marker names and coordinates from a trc file.

    INPUTS:
    - trc_path: Path to the trc file

    OUTPUTS:
    - marker_names: List of marker names
    - marker_coords: Array of marker coordinates (n_frames, t+3*n_markers)
    '''

    # marker names
    with open(trc_path, 'r') as file:
        lines = file.readlines()
        marker_names_line = lines[3]
        marker_names = marker_names_line.strip().split('\t')[2::3]

    # time and marker coordinates
    trc_data_np = np.genfromtxt(trc_path, skip_header=5, delimiter = '\t')[:,1:] 

    return marker_names, trc_data_np


def create_c3d_file(c3d_path, marker_names, trc_data_np):
    '''
    Create a c3d file from the data extracted from a trc file.

    INPUTS:
    - c3d_path: Path to the c3d file
    - marker_names: List of marker names
    - trc_data_np: Array of marker coordinates (n_frames, t+3*n_markers)

    OUTPUTS:
    - c3d file
    '''

    # retrieve frame rate
    times = trc_data_np[:,0]
    frame_rate = round((len(times)-1) / (times[-1] - times[0]))

    # write c3d file
    writer = c3d.Writer(point_rate=frame_rate, analog_rate=0, point_scale=1.0, point_units='mm', gen_scale=-1.0)
    writer.set_point_labels(marker_names)
    writer.set_screen_axis(X='+Z', Y='+Y')
    
    for frame in trc_data_np:
        residuals = np.full((len(marker_names), 1), 0.0)
        cameras = np.zeros((len(marker_names), 1))
        coords = frame[1:].reshape(-1,3)*1000
        points = np.hstack((coords, residuals, cameras))
        writer.add_frames([(points, np.array([]))])

    writer.set_start_frame(0)
    writer._set_last_frame(len(trc_data_np)-1)

    with open(c3d_path, 'wb') as handle:
        writer.write(handle)


def convert_to_c3d(trc_path):
    '''
    Make Visual3D compatible c3d files from a trc path

    INPUT:
    - trc_path: string, trc file to convert

    OUTPUT:
    - c3d file
    '''

    trc_path = str(trc_path)
    c3d_path = trc_path.replace('.trc', '.c3d')
    marker_names, trc_data_np = extract_trc_data(trc_path)
    create_c3d_file(c3d_path, marker_names, trc_data_np)

    return c3d_path


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


def best_coords_for_measurements(Q_coords, keypoints_names, beginning_frames_to_remove_percent=0.2, end_frames_to_remove_percent=0.2, fastest_frames_to_remove_percent=0.2, close_to_zero_speed=0.2, large_hip_knee_angles=45):
    '''
    Compute the best coordinates for measurements, after removing:
    - 20% fastest frames (may be outliers)
    - frames when speed is close to zero (person is out of frame): 0.2 m/frame, or 50 px/frame
    - frames when hip and knee angle below 45° (imprecise coordinates when person is crouching)
    
    INPUTS:
    - Q_coords: pd.DataFrame. The XYZ coordinates of each marker
    - keypoints_names: list. The list of marker names
    - beginning_frames_to_remove_percent: float
    - end_frames_to_remove_percent: float
    - fastest_frames_to_remove_percent: float
    - close_to_zero_speed: float (sum for all keypoints: about 50 px/frame or 0.2 m/frame)
    - large_hip_knee_angles: int
    - trimmed_extrema_percent

    OUTPUT:
    - Q_coords_low_speeds_low_angles: pd.DataFrame. The best coordinates for measurements
    '''

    # Add MidShoulder column
    df_MidShoulder = pd.DataFrame((Q_coords['RShoulder'].values + Q_coords['LShoulder'].values) /2)
    df_MidShoulder.columns = ['MidShoulder']*3
    Q_coords = pd.concat((Q_coords.reset_index(drop=True), df_MidShoulder), axis=1)

    # Add Hip column if not present
    n_markers_init = len(keypoints_names)
    if 'Hip' not in keypoints_names:
        df_Hip = pd.DataFrame((Q_coords['RHip'].values + Q_coords['LHip'].values) /2)
        df_Hip.columns = ['Hip']*3
        Q_coords = pd.concat((Q_coords.reset_index(drop=True), df_Hip), axis=1)
    n_markers = len(keypoints_names)

    # Removing first and last frames
    # Q_coords = Q_coords.iloc[int(len(Q_coords) * beginning_frames_to_remove_percent):int(len(Q_coords) * (1-end_frames_to_remove_percent))]

    # Using 80% slowest frames
    sum_speeds = pd.Series(np.nansum([np.linalg.norm(Q_coords.iloc[:,kpt:kpt+3].diff(), axis=1) for kpt in range(n_markers)], axis=0))
    sum_speeds = sum_speeds[sum_speeds>close_to_zero_speed] # Removing when speeds close to zero (out of frame)
    if len(sum_speeds)==0:
        logging.warning('All frames have speed close to zero. Make sure the person is moving and correctly detected, or change close_to_zero_speed to a lower value. Not restricting the speeds to be above any threshold.')
        Q_coords_low_speeds = Q_coords
    else:
        min_speed_indices = sum_speeds.abs().nsmallest(int(len(sum_speeds) * (1-fastest_frames_to_remove_percent))).index
        Q_coords_low_speeds = Q_coords.iloc[min_speed_indices].reset_index(drop=True)
    
    # Only keep frames with hip and knee flexion angles below 45% 
    # (if more than 50 of them, else take 50 smallest values)
    try:
        ang_mean = mean_angles(Q_coords_low_speeds, ang_to_consider = ['right knee', 'left knee', 'right hip', 'left hip'])
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds[ang_mean < large_hip_knee_angles]
        if len(Q_coords_low_speeds_low_angles) < 50:
            Q_coords_low_speeds_low_angles = Q_coords_low_speeds.iloc[pd.Series(ang_mean).nsmallest(50).index]
    except:
        logging.warning(f"At least one among the RAnkle, RKnee, RHip, RShoulder, LAnkle, LKnee, LHip, LShoulder markers is missing for computing the knee and hip angles. Not restricting these angles to be below {large_hip_knee_angles}°.")

    if n_markers_init < n_markers:
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds_low_angles.iloc[:,:-3]

    return Q_coords_low_speeds_low_angles


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
        rfoot, lfoot = 0.10, 0.10
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

    
def pad_shape(arr, target_len, fill_value=np.nan):
    '''
    Pads an array to the target length with specified fill values
    
    INPUTS:
    - arr: Input array to be padded.
    - target_len: The target length of the first dimension after padding.
    - fill_value: The value to use for padding (default: np.nan).
    
    OUTPUTS:
    - Padded array with shape (target_len, ...) matching the input dimensions.
    '''

    if len(arr) < target_len:
        pad_shape = (target_len - len(arr),) + arr.shape[1:]
        padding = np.full(pad_shape, fill_value)
        return np.concatenate((arr, padding))
    
    return arr


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
    

def sort_people_sports2d(keyptpre, keypt, scores=None):
    '''
    Associate persons across frames (Sports2D method)
    Persons' indices are sometimes swapped when changing frame
    A person is associated to another in the next frame when they are at a small distance
    
    N.B.: Requires min_with_single_indices and euclidian_distance function (see common.py)

    INPUTS:
    - keyptpre: (K, L, M) array of 2D coordinates for K persons in the previous frame, L keypoints, M 2D coordinates
    - keypt: idem keyptpre, for current frame
    - score: (K, L) array of confidence scores for K persons, L keypoints (optional) 
    
    OUTPUTS:
    - sorted_prev_keypoints: array with reordered persons with values of previous frame if current is empty
    - sorted_keypoints: array with reordered persons --> if scores is not None
    - sorted_scores: array with reordered scores     --> if scores is not None
    - associated_tuples: list of tuples with correspondences between persons across frames --> if scores is None (for Pose2Sim.triangulation())
    '''
    
    # Generate possible person correspondences across frames
    max_len = max(len(keyptpre), len(keypt))
    keyptpre = pad_shape(keyptpre, max_len, fill_value=np.nan)
    keypt = pad_shape(keypt, max_len, fill_value=np.nan)
    if scores is not None:
        scores = pad_shape(scores, max_len, fill_value=np.nan)
    
    # Compute distance between persons from one frame to another
    personsIDs_comb = sorted(list(it.product(range(len(keyptpre)), range(len(keypt)))))
    frame_by_frame_dist = [euclidean_distance(keyptpre[comb[0]],keypt[comb[1]]) for comb in personsIDs_comb]
    frame_by_frame_dist = np.mean(frame_by_frame_dist, axis=1)
    
    # Sort correspondences by distance
    _, _, associated_tuples = min_with_single_indices(frame_by_frame_dist, personsIDs_comb)
    
    # Associate points to same index across frames, nan if no correspondence
    sorted_keypoints = []
    for i in range(len(keyptpre)):
        id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
        if len(id_in_old) > 0:      sorted_keypoints += [keypt[id_in_old[0]]]
        else:                       sorted_keypoints += [keypt[i]]
    sorted_keypoints = np.array(sorted_keypoints)

    if scores is not None:
        sorted_scores = []
        for i in range(len(keyptpre)):
            id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
            if len(id_in_old) > 0:  sorted_scores += [scores[id_in_old[0]]]
            else:                   sorted_scores += [scores[i]]
        sorted_scores = np.array(sorted_scores)

    # Keep track of previous values even when missing for more than one frame
    sorted_prev_keypoints = np.where(np.isnan(sorted_keypoints) & ~np.isnan(keyptpre), keyptpre, sorted_keypoints)
    
    if scores is not None:
        return sorted_prev_keypoints, sorted_keypoints, sorted_scores
    else: # For Pose2Sim.triangulation()
        return sorted_keypoints, associated_tuples


def sort_people_rtmlib(pose_tracker, keypoints, scores):
    '''
    Associate persons across frames (RTMLib method)

    INPUTS:
    - pose_tracker: PoseTracker. The initialized RTMLib pose tracker object
    - keypoints: array of shape K, L, M with K the number of detected persons,
    L the number of detected keypoints, M their 2D coordinates
    - scores: array of shape K, L with K the number of detected persons,
    L the confidence of detected keypoints

    OUTPUT:
    - sorted_keypoints: array with reordered persons
    - sorted_scores: array with reordered scores
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


def sort_people_deepsort(keypoints, scores, deepsort_tracker, frame,frame_count):
    '''
    Associate persons across frames (DeepSort method)

    INPUTS:
    - keypoints: array of shape K, L, M with K the number of detected persons,
    L the number of detected keypoints, M their 2D coordinates
    - scores: array of shape K, L with K the number of detected persons,
    L the confidence of detected keypoints
    - deepsort_tracker: The initialized DeepSort tracker object
    - frame: np.array. The current image opened with cv2.imread

    OUTPUT:
    - sorted_keypoints: array with reordered persons
    - sorted_scores: array with reordered scores
    '''

    try:
        # Compute bboxes from keypoints and create detections (bboxes, scores, class_ids)
        bboxes_ltwh = bbox_ltwh_compute(keypoints, padding=20)
        bbox_scores = np.mean(scores, axis=1)
        class_ids = np.array(['person']*len(bboxes_ltwh))
        detections = list(zip(bboxes_ltwh, bbox_scores, class_ids))

        # Estimates the tracks and retrieve indexes of the original detections
        det_ids = [i for i in range(len(detections))]
        tracks = deepsort_tracker.update_tracks(detections, frame=frame, others=det_ids)
        track_ids_frame, orig_det_ids = [], []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_ids_frame.append(int(track.track_id)-1)       # ID of people
            orig_det_ids.append(track.get_det_supplementary())  # ID of detections

        # Correspondence between person IDs and original detection IDs
        desired_size = max(track_ids_frame) + 1
        sorted_keypoints = np.full((desired_size, keypoints.shape[1], 2), np.nan)
        sorted_scores = np.full((desired_size, scores.shape[1]), np.nan)
        for i,v in enumerate(track_ids_frame):
            if orig_det_ids[i] is not None:    
                sorted_keypoints[v] = keypoints[orig_det_ids[i]]
                sorted_scores[v] = scores[orig_det_ids[i]]

    except Exception as e:
        sorted_keypoints, sorted_scores = keypoints, scores
        if frame_count > deepsort_tracker.tracker.n_init:
            logging.warning(f"Tracking error: {e}. Sorting persons with DeepSort method failed for this frame.")

    return sorted_keypoints, sorted_scores


def bbox_ltwh_compute(keypoints, padding=0):
    '''
    Compute bounding boxes in (x_min, y_min, width, height) format
    Optionally add padding to the bounding boxes 
    as a percentage of the bounding box size (+padding% horizontally, +padding/2% vertically)

    INPUTS:
    - keypoints: array of shape K, L, M with K the number of detected persons,
                    L the number of detected keypoints, M their 2D coordinates
    - padding: int. The padding to add to the bounding boxes, in perceptage
    '''

    x_coords = keypoints[:, :, 0]
    y_coords = keypoints[:, :, 1]

    x_min, x_max = np.min(x_coords, axis=1), np.max(x_coords, axis=1)
    y_min, y_max = np.min(y_coords, axis=1), np.max(y_coords, axis=1)
    width = x_max - x_min
    height = y_max - y_min

    if padding > 0:
        x_min = x_min - width*padding/100
        y_min = y_min - height/2*padding/100
        width = width + 2*width*padding/100
        height = height + height*padding/100

    bbox_ltwh = np.stack((x_min, y_min, width, height), axis=1)

    return bbox_ltwh


def draw_bounding_box(img, X, Y, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], fontSize=0.3, thickness=1):
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
            cv2.putText(img, str(i), (x_min-30, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, 2, cv2.LINE_AA) 
    
    return img


def draw_skel(img, X, Y, model):
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
    id_pairs, name_pairs = [], []
    for data_i in PreOrderIter(model.root, filter_=lambda node: node.is_leaf):
        node_branch_ids = [node_i.id for node_i in data_i.path]
        node_branch_names = [node_i.name for node_i in data_i.path]
        id_pairs += [[node_branch_ids[i],node_branch_ids[i+1]] for i in range(len(node_branch_ids)-1)]
        name_pairs += [[node_branch_names[i],node_branch_names[i+1]] for i in range(len(node_branch_names)-1)]
    node_pairs = {tuple(name_pair): id_pair for (name_pair,id_pair) in zip(name_pairs,id_pairs)}

    
    # Draw lines
    for (x,y) in zip(X,Y):
        if not np.isnan(x).all():
            for names, ids in node_pairs.items():
                if not None in ids and not (np.isnan(x[ids[0]]) or np.isnan(y[ids[0]]) or np.isnan(x[ids[1]]) or np.isnan(y[ids[1]])):
                    if any(n.startswith('R') for n in names) and not any(n.startswith('L') for n in names):
                        c = (255,128,0)
                    elif any(n.startswith('L') for n in names) and not any(n.startswith('R') for n in names):
                        c = (0,255,0)
                    else:
                        c = (51, 153, 255)
                    cv2.line(img, (int(x[ids[0]]), int(y[ids[0]])), (int(x[ids[1]]), int(y[ids[1]])), c, thickness)

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
