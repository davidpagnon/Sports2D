#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Data Processing and Calculations             ##
    ##################################################

    - A function for interpolating zeros and NaNs in datasets.
    - A function to resample video frames to a target frequency.
    - Converts 2D points into angle measurements.
    - Computes the Euclidean distance between two points.
    - A function to find the minimum values with indices in a dataset.
    - Processes coordinates and angles for biomechanical analysis.
    - Function to flip direction from left to right in datasets.
    - Calculates angles based on coordinate data.
    - A function for generating TRC files with XYZ coordinates for motion analysis.
    - A function for creating MOT files with angular data for biomechanics applications.
'''


## INIT
import subprocess

import numpy as np
import pandas as pd
import imageio_ffmpeg as ffmpeg

from pathlib import Path
from scipy import interpolate


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


def resample_video(vid_output_path, fps, desired_framerate):
    '''
    Resample video to the desired fps using ffmpeg.
    '''
   
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    new_vid_path = vid_output_path.parent / Path(vid_output_path.stem+'_2'+vid_output_path.suffix)
    subprocess.run([ffmpeg_path, '-i', vid_output_path, '-filter:v', f'setpts={fps/desired_framerate}*PTS', '-r', str(desired_framerate), new_vid_path])
    vid_output_path.unlink()
    new_vid_path.rename(vid_output_path)


def points2D_to_angles(points_list):
    '''
    If len(points_list)==2, computes clockwise angle of ab vector w.r.t. horizontal (e.g. RBigToe, RHeel) 
    If len(points_list)==3, computes clockwise angle from a to c around b (e.g. Neck, Hip, Knee) 
    If len(points_list)==4, computes clockwise angle between vectors ab and cd (e.g. Neck Hip, RKnee RHip)
    
    If parameters are float, returns a float between 0.0 and 360.0
    If parameters are arrays, returns an array of floats between 0.0 and 360.0
    '''

    if len(points_list) < 2: # if not enough points, return None
        return np.nan
    
    ax, ay = points_list[0]
    bx, by = points_list[1]

    if len(points_list)==2:
        ux, uy = ax-bx, ay-by
        vx, vy = 1,0
    if len(points_list)==3:
        cx, cy = points_list[2]
        ux, uy = ax-bx, ay-by
        vx, vy = cx-bx, cy-by

    if len(points_list)==4:
        cx, cy = points_list[2]
        dx, dy = points_list[3]
        ux, uy = bx-ax, by-ay
        vx, vy = dx-cx, dy-cy

    ang = np.arctan2(uy, ux) - np.arctan2(vy, vx)
    ang_deg = np.degrees(ang)
    # ang_deg = np.array(np.degrees(np.unwrap(ang*2)/2))
    
    return ang_deg


def euclidean_distance(q1, q2):
    '''
    Euclidean distance between 2 points (N-dim).

    INPUTS:
    - q1: list of N_dimensional coordinates of point
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    '''

    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1

    euc_dist = np.sqrt(np.sum( [d**2 for d in dist]))

    return euc_dist


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


def process_coordinates_and_angles(keypoints, scores, keypoint_likelihood_threshold, keypoint_number_threshold, average_likelihood_threshold, flip_left_right, L_R_direction_idx, keypoints_names, keypoints_ids, angle_names, angle_dict):
    valid_X, valid_Y, valid_scores = [], [], []
    valid_X_flipped, valid_angles = [], []
    for kpts, scrs in zip(keypoints, scores):
        # Filtrage des keypoints avec faible confiance
        mask = scrs >= keypoint_likelihood_threshold
        person_X = np.where(mask, kpts[:, 0], np.nan)
        person_Y = np.where(mask, kpts[:, 1], np.nan)
        person_scores = np.where(mask, scrs, np.nan)

        # Vérification du nombre de keypoints valides
        valid_ratio = np.sum(mask) / len(mask)
        average_score = np.nanmean(person_scores)
        if valid_ratio < keypoint_number_threshold or average_score < average_likelihood_threshold:
            person_X[:] = np.nan
            person_Y[:] = np.nan
            person_scores[:] = np.nan

        valid_X.append(person_X)
        valid_Y.append(person_Y)
        valid_scores.append(person_scores)

        # Gestion du flip gauche-droite
        if flip_left_right:
            person_X_flipped = flip_left_right_direction(
                person_X, L_R_direction_idx, keypoints_names, keypoints_ids
            )
        else:
            person_X_flipped = person_X.copy()
        valid_X_flipped.append(person_X_flipped)

        # Calcul des angles
        person_angles = [
            compute_angle(
                ang_name, person_X_flipped, person_Y, angle_dict, keypoints_ids, keypoints_names
            )
            for ang_name in angle_names
        ]
        valid_angles.append(person_angles)
    return valid_X, valid_Y, valid_X_flipped, valid_angles


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
    Requires points2D_to_angles function (see common.py)

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
        if ang_name in ['pelvis', 'trunk', 'shoulders']:
            angle_coords = [[np.abs(person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]]), person_Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0] if kpt in keypoints_names]
        else:
            angle_coords = [[person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]], person_Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0] if kpt in keypoints_names]
        ang = points2D_to_angles(angle_coords)
        ang += ang_params[2]
        ang *= ang_params[3]
        if ang_name in ['pelvis', 'shoulders']:
            ang = ang-180 if ang>90 else ang
            ang = ang+180 if ang<-90 else ang
        else:
            ang = ang-360 if ang>180 else ang
            ang = ang+360 if ang<-180 else ang
    else:
        ang = np.nan

    return ang


def make_trc_with_XYZ(X, Y, Z, time, trc_path):
    '''
    Write a trc file from 3D coordinates and time, compatible with OpenSim.

    INPUTS:
    - X: pd.DataFrame. The x coordinates of the keypoints
    - Y: pd.DataFrame. The y coordinates of the keypoints
    - Z: pd.DataFrame. The z coordinates of the keypoints
    - time: pd.Series. The time series for the coordinates
    - trc_path: str. The path where to save the trc file

    OUTPUT:
    - trc_data: pd.DataFrame. The data that has been written to the TRC file
    '''

    #Header
    frame_rate = (len(X)-1)/(time.iloc[-1] - time.iloc[0])
    DataRate = CameraRate = OrigDataRate = frame_rate
    NumFrames = len(X)
    NumMarkers = len(X.columns)
    keypoint_names = X.columns
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_path, 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, 0, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoint_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoint_names))])]
    
    # Data
    trc_data = pd.concat([pd.concat([X.iloc[:,kpt], Y.iloc[:,kpt], Z.iloc[:,kpt]], axis=1) for kpt in range(len(X.columns))], axis=1)
    trc_data.insert(0, 't', time)

    # Write file
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        trc_data.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_data


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