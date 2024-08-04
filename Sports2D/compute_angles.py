#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##############################################################
    ## Compute joint and segment angles from csv position files ##
    ##############################################################

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
    
    /!\ Warning /!\
    - The angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
    
    INPUTS:
    - one or several position csv files
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one csv file for joint and segment angles per detected person
    - a logs.txt file 

'''    


## INIT
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import traceback
from Sports2D.Sports2D import base_params
from Sports2D.Utilities import filter, common
from Sports2D.Utilities.skeletons import halpe26_rtm



## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.3.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# CONSTANTS
# dict: name: points, extra, offset, invert. 
# Most angles are multiplied by -1 because the OpenCV y axis points down.

# text dics
joint_text_positions = {}

# define joint angle parameters
def get_joint_angle_params(joint):
    """
    Returns parameters for joint angle calculations.
    
    Args:
    joint (str): Name of the joint

    Returns:
    list: Parameters for angle calculation [keypoints, type, offset, invert]
    """
    joint_angle_dict = {
        'Right ankle': [['right_heel', 'right_big_toe', 'right_ankle', 'right_knee'], 'dorsiflexion', -90, -1],
        'Left ankle': [['left_heel', 'left_big_toe', 'left_ankle', 'left_knee'], 'dorsiflexion', -90, -1],
        'Right knee': [['right_hip', 'right_knee', 'right_ankle'], 'flexion', -180, -1],
        'Left knee': [['left_hip', 'left_knee', 'left_ankle'], 'flexion', -180, -1],
        'Right hip': [['right_knee', 'right_hip', 'right_shoulder'], 'flexion', -180, -1],
        'Left hip': [['left_knee', 'left_hip', 'left_shoulder'], 'flexion', -180, -1],
        'Right shoulder': [['right_hip', 'right_shoulder', 'right_elbow'], 'flexion', 0, 1],
        'Left shoulder': [['left_hip', 'left_shoulder', 'left_elbow'], 'flexion', 0, 1],
        'Right elbow': [['right_wrist', 'right_elbow', 'right_shoulder'], 'flexion', -180, -1],
        'Left elbow': [['left_wrist', 'left_elbow', 'left_shoulder'], 'flexion', -180, -1],
    }
    return joint_angle_dict.get(joint)

# Define segment angle parameters
def get_segment_angle_params(segment):
    """
    Returns parameters for segment angle calculations.
    
    Args:
    segment (str): Name of the segment

    Returns:
    list: Parameters for angle calculation [keypoints, type, offset, invert]
    """
    segment_angle_dict = {
        'Right foot': [['right_heel', 'right_big_toe'], 'horizontal', 0, -1],
        'Left foot': [['left_heel', 'left_big_toe'], 'horizontal', 0, -1],
        'Right shank': [['right_knee', 'right_ankle'], 'horizontal', 0, -1],
        'Left shank': [['left_knee', 'left_ankle'], 'horizontal', 0, -1],
        'Right thigh': [['right_hip', 'right_knee'], 'horizontal', 0, -1],
        'Left thigh': [['left_hip', 'left_knee'], 'horizontal', 0, -1],
        'Trunk': [['hip', 'neck'], 'horizontal', 0, -1],
        'Right arm': [['right_shoulder', 'right_elbow'], 'horizontal', 0, -1],
        'Left arm': [['left_shoulder', 'left_elbow'], 'horizontal', 0, -1],
        'Right forearm': [['right_elbow', 'right_wrist'], 'horizontal', 0, -1],
        'Left forearm': [['left_elbow', 'left_wrist'], 'horizontal', 0, -1],
    }
    return segment_angle_dict.get(segment)

# FUNCTIONS
def display_figures_fun_ang(df_list):
    '''
    Displays filtered and unfiltered data for comparison
    /!\ Crashes on the third window...

    INPUTS:
    - df_list: list of dataframes of angles

    OUTPUT:
    - matplotlib window with tabbed figures for each angle
    '''
    
    angle_names = df_list[0].iloc[:,1:].columns.get_level_values(2)
    time = df_list[0].iloc[:,0]
    
    pw = common.plotWindow()
    for id, angle in enumerate(angle_names): # angles
        f = plt.figure()
        
        plt.plot()
        [plt.plot(time, df.iloc[:,id+1], label=['unfiltered' if i==0 else 'filtered' if i==1 else ''][0]) for i,df in enumerate(df_list)]
        plt.xlabel('Time (seconds)')
        plt.ylabel(angle)
        plt.legend()

        pw.addPlot(angle, f)
    
    if pw.tabs.count() > 0:  # Only show if there are plots
        pw.show()
    else:
        print("No data to display.")
    
    
def points2D_to_angles(points_list):
    '''
    If len(points_list)==2, computes clockwise angle of ab vector w.r.t. horizontal (e.g. RBigToe-RHeel) 
    If len(points_list)==3, computes clockwise angle from a to c around b (e.g. Neck, Hip, Knee) 
    If len(points_list)==4, computes clockwise angle between vectors ab and cd (e.g. CHip-Neck, RHip-RKnee)
    
    If parameters are float, returns a float between 0.0 and 360.0
    If parameters are arrays, returns an array of floats between 0.0 and 360.0
    '''
    if len(points_list) < 2: # if not enough points, return None
        return None
    
    ax, ay = points_list[0]
    bx, by = points_list[1]

    if len(points_list)==2:
        ux, uy = bx-ax, by-ay
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
    ang_deg = np.array(np.degrees(np.unwrap(ang*2)/2))

    return ang_deg

def flip_left_right_direction(df_points, data_type):      # ! should check working on video and webcam (viedo ok, webcam not checked)
    '''
    Inverts X coordinates to get consistent angles when person changes direction and goes to the left.
    The person is deemed to go to the left when their toes are to the left of their heels.
    
    INPUT:
    - df_points: dataframe of pose detection
    
    OUTPUT:
    - df_points: dataframe of pose detection with flipped X coordinates
    '''

    # print(f"Flipping left-right direction for {data_type} data")
    if data_type not in ['video', 'webcam']:
        raise ValueError("data_type must be either 'video' or 'webcam'")
    
    # print('Flipping left-right direction...')
    # print("DataFrame columns:", df_points.columns)
    # print("Index levels:", df_points.columns.names)
    
    # Set the level names explicitly if they are None
    if df_points.columns.names == [None, None, None, None]:
        df_points.columns.names = ['scorer', 'individuals', 'bodyparts', 'coords']
    
    try:
        if data_type == 'video':
            # print(f"Flipping X coordinates for {data_type} data")
            
            # Select all 'x' coordinates
            x_coords = df_points.xs('x', axis=1, level='coords')
            # print(f"X coordinates first 5 values:\n{x_coords.head()}")
            # print(f"X coordinates columns: {x_coords.columns}")
            
            # Access MultiIndex columns correctly
            scorer = x_coords.columns.get_level_values('scorer')[0]
            individual = x_coords.columns.get_level_values('individuals')[0]
            
            try:
                right_toe = x_coords.loc[:, (scorer, individual, 'right_big_toe')]
                right_heel = x_coords.loc[:, (scorer, individual, 'right_heel')]
                left_toe = x_coords.loc[:, (scorer, individual, 'left_big_toe')]
                left_heel = x_coords.loc[:, (scorer, individual, 'left_heel')]
            except KeyError as e:
                print(f"Column not found: {e}")
                raise

        else:  # webcam
            required_columns = ['right_big_toe_x', 'right_heel_x', 'left_big_toe_x', 'left_heel_x']
            for col in required_columns:
                if col not in df_points.columns:
                    raise ValueError(f"Required column {col} is missing from the dataframe")
            right_toe = df_points['right_big_toe_x']
            right_heel = df_points['right_heel_x']
            left_toe = df_points['left_big_toe_x']
            left_heel = df_points['left_heel_x']
        
        # print(f"Right toe first 5 values: {right_toe.head()}")
        # print(f"Right heel first 5 values: {right_heel.head()}")
        # print(f"Left toe first 5 values: {left_toe.head()}")
        # print(f"Left heel first 5 values: {left_heel.head()}")
        
        # Calculate orientation while handling NaN values
        right_orientation = right_toe - right_heel
        left_orientation = left_toe - left_heel
        orientation = right_orientation.add(left_orientation, fill_value=0)

        # print(f"Orientation first 5 values: {orientation.head()}")
        
        # Create flip mask, treating NaN as positive (no flip)
        flip_mask = np.where(orientation.isnull(), 1, np.where(orientation >= 0, 1, -1))

        if data_type == 'video':
            # Update only the 'x' coordinates
            x_coords_flipped = x_coords * flip_mask[:, np.newaxis]
            
            # Update the original dataframe with flipped x coordinates
            df_points.loc[:, (slice(None), slice(None), slice(None), 'x')] = x_coords_flipped.values
            # print(f"Flipped X coordinates first 5 values:\n{x_coords_flipped.head()}")

        else:  # webcam
            x_columns = [col for col in df_points.columns if col.endswith('_x')]
            df_points[x_columns] = df_points[x_columns] * flip_mask.reshape(-1, 1)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    return df_points

def joint_angles_series_from_points(df_points, angle_params, kpt_thr):
    '''
    Obtain joint angle series from point series.
    
    INPUT: 
    - df_points: dataframe of pose detection, from csv
    - angle_params: dictionary specifying which points to use, and what offset and multiplying factor to use
    
    OUTPUT:
    - ang_series: array of time series of the considered angle
    '''
    
    # Retrieve points
    keypt_series = []
    for k in angle_params[0]:
        if f"{k}_x" in df_points.columns and f"{k}_y" in df_points.columns and f"{k}_score" in df_points.columns:
            score = df_points[f"{k}_score"].values[0]
            if score >= kpt_thr:
                keypt = df_points[[f"{k}_x", f"{k}_y"]]
                keypt_series.append(keypt)
            else:
                return None
        else:
            print(f"Error: Keypoint {k} or its score not found in dataframe")
            return None

    if len(keypt_series) != len(angle_params[0]):
        print(f"Error: Not all required keypoints were found or passed the threshold")
        return None

    # Compute angles
    points_list = [k.values.T for k in keypt_series]
    ang_series = points2D_to_angles(points_list)
    if ang_series is None:
        print(f"points2D_to_angles returned None for {angle_params[1]}")
        return None
    
    ang_series += angle_params[2]
    ang_series *= angle_params[3]
    
    if ang_series.mean() > 180: ang_series -= 360
    if ang_series.mean() < -180: ang_series += 360

    return ang_series


def segment_angles_series_from_points(df_points, angle_params, segment, kpt_thr):
    '''
    Obtain segment angle series w/r horizontal from point series.
    For trunk segment: mean of the angles between RHip-RShoulder and LHip-LShoulder
    
    INPUT: 
    - df_points: dataframe of pose detection, from csv
    - angle_params: dictionary specifying which points to use, and what offset and multiplying factor to use
    - segment: which segment angle is considered
    
    OUTPUT:
    - ang_series: array of time series of the considered angle
    '''
    
    # Retrieve points
    keypt_series = []
    for k in angle_params[0]:
        if df_points[f"{k}_score"].values[0] >= kpt_thr:
            keypt_series.append(df_points[[f"{k}_x", f"{k}_y"]])
        else:
            return None

    points_list = [k.values.T for k in keypt_series]
    ang_series = points2D_to_angles(points_list)
    if ang_series is None:
        return None

    ang_series += angle_params[2]
    ang_series *= angle_params[3]
    # ang_series = np.where(ang_series>180,ang_series-360,ang_series)

    # For trunk: mean between angles RHip-RShoulder and LHip-LShoulder
    if segment == 'Trunk':
        ang_seriesR = ang_series
        angle_params[0] = [a.replace('R','L') for a in angle_params[0]]
        keypt_series = []
        for k in angle_params[0]:
            if df_points[f"{k}_score"].values[0] >= kpt_thr:
                keypt_series.append(df_points[[f"{k}_x", f"{k}_y"]])
            else:
                return None

        points_list = [k.values.T for k in keypt_series]
        ang_series = points2D_to_angles(points_list)
        ang_series += angle_params[2]
        ang_series *= angle_params[3]
        ang_series = np.mean((ang_seriesR, ang_series), axis=0)
        # ang_series = np.where(ang_series>180,ang_series-360,ang_series)
        
    if ang_series.mean() > 180: ang_series -= 360
    if ang_series.mean() < -180: ang_series += 360

    return ang_series

def joint_angles_series_from_csv(df_points, angle_params, kpt_thr):
    '''
    Obtain joint angle series from point series.
    
    INPUT: 
    - df_points: dataframe of pose detection, from csv
    - angle_params: dictionary specifying which points to use, and what offset and multiplying factor to use
    - kpt_thr: threshold for keypoint confidence
    
    OUTPUT:
    - ang_series: array of time series of the considered angle
    '''
    print(f"Processing angle for: {angle_params[0]}")
    
    # Retrieve points
    keypt_series = []
    for k in angle_params[0]:
        series = df_points.iloc[:,df_points.columns.get_level_values(2)==k].iloc[:,:2]
        keypt_series.append(series)

    # Compute angles
    points_list = [k.values.T for k in keypt_series]
    
    # Create a mask for valid data (non-NaN)
    valid_mask = np.all([~np.isnan(points).any(axis=0) for points in points_list], axis=0)
    
    # Only compute angles for valid data
    ang_series = np.full(valid_mask.shape, np.nan)
    ang_series[valid_mask] = points2D_to_angles([p[:, valid_mask] for p in points_list])
    
    # Apply offset and scaling only to valid angles
    valid_indices = np.where(valid_mask)[0]
    ang_series[valid_indices] += angle_params[2]
    ang_series[valid_indices] *= angle_params[3]

    # Adjust each angle individually to be within -180 to 180 range
    def adjust_angle(angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    ang_series[valid_indices] = np.array([adjust_angle(angle) for angle in ang_series[valid_indices]])

    return ang_series

def segment_angles_series_from_csv(df_points, angle_params, segment, kpt_thr):
    '''
    Obtain segment angle series w/r horizontal from point series.
    For trunk segment: mean of the angles between RHip-RShoulder and LHip-LShoulder
    
    INPUT: 
    - df_points: dataframe of pose detection, from csv
    - angle_params: dictionary specifying which points to use, and what offset and multiplying factor to use
    - segment: which segment angle is considered
    
    OUTPUT:
    - ang_series: array of time series of the considered angle
    '''
    
    # Retrieve points
    keypt_series = []
    for k in angle_params[0]:
        keypt_series += [df_points.iloc[:,df_points.columns.get_level_values(2)==k].iloc[:,:2]]
    points_list = [k.values.T for k in keypt_series]
    
    # Create a mask for valid data (non-NaN)
    valid_mask = np.all([~np.isnan(points).any(axis=0) for points in points_list], axis=0)
    
    # Only compute angles for valid data
    ang_series = np.full(valid_mask.shape, np.nan)
    ang_series[valid_mask] = points2D_to_angles([p[:, valid_mask] for p in points_list])
    
    # Apply offset and scaling only to valid angles
    valid_indices = np.where(valid_mask)[0]
    ang_series[valid_indices] += angle_params[2]
    ang_series[valid_indices] *= angle_params[3]
    
    # For trunk: mean between angles RHip-RShoulder and LHip-LShoulder
    if segment == 'Trunk':
        ang_seriesR = ang_series.copy()
        angle_params[0] = [a.replace('R','L') for a in angle_params[0]]
        keypt_series = []
        for k in angle_params[0]:
            keypt_series += [df_points.iloc[:,df_points.columns.get_level_values(2)==k].iloc[:,:2]]
        points_list = [k.values.T for k in keypt_series]
        
        # Create a new mask for the left side
        valid_mask_L = np.all([~np.isnan(points).any(axis=0) for points in points_list], axis=0)
        
        # Compute angles for the left side
        ang_seriesL = np.full(valid_mask_L.shape, np.nan)
        ang_seriesL[valid_mask_L] = points2D_to_angles([p[:, valid_mask_L] for p in points_list])
        
        # Apply offset and scaling
        valid_indices_L = np.where(valid_mask_L)[0]
        ang_seriesL[valid_indices_L] += angle_params[2]
        ang_seriesL[valid_indices_L] *= angle_params[3]
        
        # Combine right and left angles
        ang_series = np.nanmean(np.array([ang_seriesR, ang_seriesL]), axis=0)

    # Adjust each angle individually to be within -180 to 180 range
    def adjust_angle(angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    ang_series[~np.isnan(ang_series)] = np.array([adjust_angle(angle) for angle in ang_series[~np.isnan(ang_series)]])
        
    return ang_series

def draw_joint_angle(frame, joint, angle, keypoints, scores, kpt_thr):
    joint_params = get_joint_angle_params(joint)
    if joint_params:
        keypoint_names = joint_params[0]
        pts = [keypoints[halpe26_rtm['keypoint_info'][next(k for k, v in halpe26_rtm['keypoint_info'].items() if v['name'] == name)]['id']] for name in keypoint_names]
        scores_pts = [scores[halpe26_rtm['keypoint_info'][next(k for k, v in halpe26_rtm['keypoint_info'].items() if v['name'] == name)]['id']] for name in keypoint_names]
        
        if all(score >= kpt_thr for score in scores_pts):
            neck = keypoints[halpe26_rtm['keypoint_info'][next(k for k, v in halpe26_rtm['keypoint_info'].items() if v['name'] == 'neck')]['id']]
            mid_hip = keypoints[halpe26_rtm['keypoint_info'][next(k for k, v in halpe26_rtm['keypoint_info'].items() if v['name'] == 'hip')]['id']]
            if 'ankle' in joint.lower():
                draw_angle_arc(frame, joint, pts, angle, neck, mid_hip)
            else:
                pt1, pt2, pt3 = pts[:3]  # Use only the first three points for non-ankle joints
                draw_angle_arc(frame, joint, [pt1, pt2, pt3], angle, neck, mid_hip)

def draw_dotted_line(frame, start, direction, length, color=(0, 255, 0), gap=7, dot_length=3): # default is green color
    for i in range(0, length, gap):
        line_start = start + direction * i
        line_end = line_start + direction * dot_length
        cv2.line(frame, tuple(line_start.astype(int)), tuple(line_end.astype(int)), color, 2)

# everything well done.
def draw_angle_arc(frame, joint, pts, angle, neck, mid_hip):

    start_angle = 0
    end_angle = 0
    ref_point = None
    radius = 0
    # vector_length = 0  # It might be useful to adjust size of the font based on the vector length


    try:
        if 'ankle' in joint.lower():
            heel, toe, ankle, knee = map(np.array, pts)

            # Calculate shank vector (from ankle to knee)
            shank_vec = knee - ankle
            
            # Calculate foot vector (from heel to toe)
            foot_vec = toe - heel

            # Calculate perpendicular vector to shank (shin)
            perpendicular_vec = np.array([-shank_vec[1], shank_vec[0]])
            perpendicular_vec /= np.linalg.norm(perpendicular_vec)
            
            # Ensure the perpendicular vector points towards the toe
            if np.dot(perpendicular_vec, foot_vec) < 0:
                perpendicular_vec = -perpendicular_vec
            
            radius = int(0.2 * np.linalg.norm(knee - ankle))
            ref_point = ankle
            
            # Set start angle perpendicular to the shin
            start_angle = np.degrees(np.arctan2(perpendicular_vec[1], perpendicular_vec[0]))
            
            # Set end angle parallel to heel-toe line, but starting from ankle
            heel_toe_vec = toe - heel
            heel_toe_unit_vec = heel_toe_vec / np.linalg.norm(heel_toe_vec)
            end_angle = np.degrees(np.arctan2(heel_toe_unit_vec[1], heel_toe_unit_vec[0]))
            
            # Draw dotted line perpendicular to shin
            segment_length = np.linalg.norm(knee - ankle)
            dotted_line_length = int(segment_length * 0.3)
            draw_dotted_line(frame, ref_point, perpendicular_vec, dotted_line_length)
        
        elif 'shoulder' in joint.lower() or 'hip' in joint.lower():
            trunk_vec = np.array(mid_hip) - np.array(neck)
            trunk_unit_vec = trunk_vec / np.linalg.norm(trunk_vec)

            if 'shoulder' in joint.lower():
                hip, shoulder, elbow = map(np.array, pts)
                ref_point = shoulder
                other_vec = elbow - shoulder
            else:  # 'hip' in joint.lower()
                knee, hip, shoulder = map(np.array, pts)
                ref_point = hip
                other_vec = knee - hip

            radius = int(0.2 * np.linalg.norm(other_vec))

            start_angle = np.degrees(np.arctan2(trunk_unit_vec[1], trunk_unit_vec[0]))
            end_angle = np.degrees(np.arctan2(other_vec[1], other_vec[0]))

            # Draw dotted line parallel to neck-to-mid-hip vector, starting from ref_point
            segment_length = np.linalg.norm(trunk_vec)
            dotted_line_length = int(segment_length * 0.3)
            draw_dotted_line(frame, ref_point, trunk_unit_vec, dotted_line_length)

        else:
            pt1, pt2, pt3 = map(np.array, pts)
            if 'knee' in joint.lower(): # hip, knee, ankle
                ref_vec = pt2 - pt1  # hip to knee
                other_vec = pt3 - pt2  # knee to ankle
            elif 'elbow' in joint.lower(): # wrist, elbow, shoulder
                ref_vec = pt2 - pt3  # shoulder to elbow
                other_vec = pt1 - pt2  # elbow to wrist
            else:
                raise ValueError(f"Unsupported joint type: {joint}")
            
            direction = ref_vec / np.linalg.norm(ref_vec)
            radius = int(0.2 * (np.linalg.norm(pt1 - pt2) + np.linalg.norm(pt3 - pt2)) / 2)
            ref_point = pt2

            start_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            end_angle = np.degrees(np.arctan2(other_vec[1], other_vec[0]))

            # Draw dotted line
            segment_length = np.linalg.norm(pt1 - pt2)
            dotted_line_length = int(segment_length * 0.3)
            draw_dotted_line(frame, ref_point, direction, dotted_line_length)

        # Ensure the arc is not greater than 180 degrees
        if abs(end_angle - start_angle) > 180:
            if end_angle > start_angle:
                start_angle += 360
            else:
                end_angle += 360

        # Draw arc
        cv2.ellipse(frame, tuple(ref_point.astype(int)), (radius, radius), 0, start_angle, end_angle, (0, 255, 0), 2)
            
            
        # 텍스트 위치 조정
        text_angle = np.radians((start_angle + end_angle) / 2)
        text_radius = radius + 15  # arc 바로 아래에 위치하도록 조정
        text_pos = (
            int(ref_point[0] + text_radius * np.cos(text_angle)),
            int(ref_point[1] + text_radius * np.sin(text_angle))
        )

        # arc의 아래쪽에 텍스트가 오도록 조정
        if text_pos[1] < ref_point[1]:  # 텍스트가 arc의 위쪽에 있다면
            text_pos = (text_pos[0], int(ref_point[1] + radius + 10))  # arc의 아래로 이동
        else:
            text_pos = (text_pos[0], text_pos[1] + 10)  # 약간의 여백 추가
        
        # store text position
        joint_text_positions[joint] = text_pos

        # 텍스트 준비 (기존 코드 유지)
        text = f"{angle:.1f}"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.4
        out_thickness = 2
        thickness = 1

        # 텍스트 그리기 (기존 코드 유지, 색상 변경 없음)
        outline_color = (255, 255, 255)  # White outline for joint text
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cv2.putText(frame, text, 
                        (text_pos[0] + dx, text_pos[1] + dy), 
                        font, font_scale, outline_color, out_thickness, cv2.LINE_AA)
        
        cv2.putText(frame, text, text_pos, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    except Exception as e:
            print(f"Error in draw_angle_arc for joint {joint}: {str(e)}")

def draw_segment_angle(frame, segment, angle, keypoints, scores, kpt_thr):
    thickness = 2
    color = (255, 255, 255)  # White

    segment_params = get_segment_angle_params(segment)
    if segment_params:
        keypoint_names = segment_params[0]
        kpt_indices = [halpe26_rtm['keypoint_info'][next(k for k, v in halpe26_rtm['keypoint_info'].items() if v['name'] == name)]['id'] for name in keypoint_names]
        
        right_foot_keypoints = [keypoints[halpe26_rtm['keypoint_info'][next(k for k, v in halpe26_rtm['keypoint_info'].items() if v['name'] == name)]['id']] for name in ['right_ankle', 'right_big_toe', 'right_heel']]
        left_foot_keypoints = [keypoints[halpe26_rtm['keypoint_info'][next(k for k, v in halpe26_rtm['keypoint_info'].items() if v['name'] == name)]['id']] for name in ['left_ankle', 'left_big_toe', 'left_heel']]

        if segment in ["Right foot", "Left foot"]:
            heel, toe = [keypoints[i] for i in kpt_indices]
            score_heel, score_toe = [scores[i] for i in kpt_indices]
            if min(score_heel, score_toe) >= kpt_thr:
                ankle = keypoints[halpe26_rtm['keypoint_info'][next(k for k, v in halpe26_rtm['keypoint_info'].items() if v['name'] == f"{segment.lower().split()[0]}_ankle")]['id']]
                draw_angle_line(frame, heel, toe, ankle, angle, thickness, color, right_foot_keypoints, left_foot_keypoints, segment)
        else:
            pt1, pt2 = [keypoints[i] for i in kpt_indices]
            score1, score2 = [scores[i] for i in kpt_indices]
            if min(score1, score2) >= kpt_thr:
                draw_angle_line(frame, pt1, pt2, None, angle, thickness, color, right_foot_keypoints, left_foot_keypoints, segment)

    return frame

def draw_angle_line(frame, pt1, pt2, ankle, angle, thickness, color, right_foot_keypoints, left_foot_keypoints, segment):
    """
        segment_angle_dict = {
        'Right foot': [['right_heel', 'right_big_toe'], 'horizontal', 0, -1],
        'Left foot': [['left_heel', 'left_big_toe'], 'horizontal', 0, -1],
        'Right shank': [['right_knee', 'right_ankle'], 'horizontal', 0, -1],
        'Left shank': [['left_knee', 'left_ankle'], 'horizontal', 0, -1],
        'Right thigh': [['right_hip', 'right_knee'], 'horizontal', 0, -1],
        'Left thigh': [['left_hip', 'left_knee'], 'horizontal', 0, -1],
        'Trunk': [['right_hip', 'right_shoulder'], 'horizontal', 0, -1],
        'Right arm': [['right_shoulder', 'right_elbow'], 'horizontal', 0, -1],
        'Left arm': [['left_shoulder', 'left_elbow'], 'horizontal', 0, -1],
        'Right forearm': [['right_elbow', 'right_wrist'], 'horizontal', 0, -1],
        'Left forearm': [['left_elbow', 'left_wrist'], 'horizontal', 0, -1],
    }
    """
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))
    
    # Calculate the direction vector of the segment
    segment_vector = np.array(pt2) - np.array(pt1)
    segment_length = np.linalg.norm(segment_vector)
    length = int(segment_length * 0.12)  # 12% of segment length
    
    # Normalize the segment vector
    if segment_length > 0:
        segment_unit_vector = segment_vector / segment_length
    else:
        segment_unit_vector = np.array([1, 0])  # Default to horizontal if segment length is 0
    
    # Calculate the direction for each foot
    right_foot_direction = np.array(right_foot_keypoints[1]) - np.array(right_foot_keypoints[2])  # right toe to heel
    left_foot_direction = np.array(left_foot_keypoints[1]) - np.array(left_foot_keypoints[2])  # left toe to heel
    
    # Determine the reference direction based on the segment
    if segment.startswith("Right"):
        direction = np.array([1, 0]) if right_foot_direction[0] >= 0 else np.array([-1, 0])
    elif segment.startswith("Left"):
        direction = np.array([1, 0]) if left_foot_direction[0] >= 0 else np.array([-1, 0])
    else:  # For trunk
        average_foot_direction = (right_foot_direction[0] + left_foot_direction[0]) / 2
        direction = np.array([1, 0]) if average_foot_direction >= 0 else np.array([-1, 0])
    
    if segment == "Right foot" or segment == "Left foot":
        ankle = tuple(map(int, ankle))
        heel = np.array(pt1)
        toe = np.array(pt2)
        foot_length = int(segment_length * 0.20)  # 20% of foot length
        
        # Draw the line parallel to heel-toe starting from ankle
        heel_to_toe_vector = toe - heel
        heel_to_toe_unit_vector = heel_to_toe_vector / np.linalg.norm(heel_to_toe_vector)
        foot_end = tuple(map(int, ankle + foot_length * heel_to_toe_unit_vector))
        cv2.line(frame, ankle, foot_end, color, thickness)
        
        # Draw horizontal reference line
        reference_end = tuple(map(int, np.array(ankle) + foot_length * direction))
        cv2.line(frame, ankle, reference_end, color, thickness)
        
    else:
        # For non-foot segments, draw lines as before
        segment_end = tuple(map(int, np.array(pt1) + length * segment_unit_vector))
        reference_end = tuple(map(int, np.array(pt1) + length * direction))
        
        cv2.line(frame, pt1, segment_end, color, thickness)
        cv2.line(frame, pt1, reference_end, color, thickness)

    return frame

    
def overlay_angles(frame, df_angles_list_frame, keypoints, scores, kpt_thr, person_ids):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (255, 255, 255)  # 흰색
    outline_color = (0, 0, 0)  # 검정색
    
    if person_ids is None:
        person_ids = range(len(df_angles_list_frame))

    cmap = plt.cm.hsv  # draw_bounding_box 함수와 동일한 색상 맵 사용
    
    for i, (angles_frame_person, person_keypoints, person_scores, person_id) in enumerate(zip(df_angles_list_frame, keypoints, scores, person_ids)):
        
        person_color = (np.array(cmap((i+1)/len(df_angles_list_frame)))*255).astype(int).tolist()

        # Check if angles_frame_person is a dict, if so, convert to pandas Series
        if isinstance(angles_frame_person, dict):
            angles_frame_person = pd.Series(angles_frame_person)

        # NaN이 아닌 segment angle 값이 있는지 확인
        segment_angles = angles_frame_person.dropna().drop('Time', errors='ignore')
        segment_angles = segment_angles[[col for col in segment_angles.index if not any(joint in str(col).lower() for joint in ["ankle", "knee", "hip", "shoulder", "elbow"])]]
        
        if not segment_angles.empty:
            segment_count = 0
            
            # 사람 구분을 위한 라벨 추가 (이제 person_id 사용)
            person_label = f"person{person_id}"
            person_label_position = (10 + 250*i, 15)
            person_outline_color = (255, 255, 255)  # 흰색
            
            # 사람 라벨 그리기
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(frame, person_label, 
                            (person_label_position[0] + dx, person_label_position[1] + dy), 
                            font, font_scale, outline_color, 2, cv2.LINE_AA)
            cv2.putText(frame, person_label, person_label_position, font, font_scale, person_color, 1, cv2.LINE_AA)
            
            for ang_nb, (angle_name, angle_value) in enumerate(angles_frame_person.items()):
                try:
                    if angle_name == 'Time':  # Skip the 'Time' column
                        continue

                    if isinstance(angle_value, pd.Series):
                        angle_value = angle_value.iloc[0]
                    
                    angle_value = float(angle_value)
                    
                    angle_label = str(angle_name)
                    if isinstance(angle_name, tuple):
                        angle_label = angle_name[2]  # Use the third element of the tuple for the label
                    
                    is_joint = any(joint in angle_label.lower() for joint in ["ankle", "knee", "hip", "shoulder", "elbow"])
                    
                    if is_joint:
                        # Draw joint angle, no text (as before)
                        draw_joint_angle(frame, angle_label, angle_value, person_keypoints, person_scores, kpt_thr)
                    else:
                        # For segment angles, display text and draw angle only if not NaN
                        if not pd.isna(angle_value):
                            y_offset = 35 + 15 * segment_count  # 시작 위치를 약간 아래로 조정
                            segment_count += 1
                            
                            label_position = (10+250*i, y_offset)
                            value_position = (150+250*i, y_offset)
                            progress_bar_start = (170+250*i, y_offset - 14)
                            progress_bar_end = (220+250*i, y_offset + 1)
                            
                            # Draw angle label with outline
                            label_text = f"{angle_label}:"
                            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                                cv2.putText(frame, label_text, 
                                            (label_position[0] + dx, label_position[1] + dy), 
                                            font, font_scale, outline_color, 2, cv2.LINE_AA)
                            cv2.putText(frame, label_text, label_position, font, font_scale, text_color, 1, cv2.LINE_AA)
                            
                            # Draw angle value with outline
                            value_text = f"{angle_value:.1f}"
                            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                                cv2.putText(frame, value_text, 
                                            (value_position[0] + dx, value_position[1] + dy), 
                                            font, font_scale, outline_color, 2, cv2.LINE_AA)
                            cv2.putText(frame, value_text, value_position, font, font_scale, text_color, 1, cv2.LINE_AA)
                            
                            # Draw progress bar
                            x_ang = int(angle_value*35/180)
                            if x_ang != 0:
                                sub_frame = frame[progress_bar_start[1]:progress_bar_end[1], 
                                                  progress_bar_start[0]:progress_bar_start[0]+x_ang]
                                if sub_frame.size > 0:
                                    white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                                    res = cv2.addWeighted(sub_frame, 0.6, white_rect, 0.4, 1.0)
                                    frame[progress_bar_start[1]:progress_bar_end[1], 
                                          progress_bar_start[0]:progress_bar_start[0]+x_ang] = res

                            # Draw segment angle
                            draw_segment_angle(frame, angle_label, angle_value, person_keypoints, person_scores, kpt_thr)
                    
                except Exception as e:
                    # print(f"Warning: Unable to process angle {ang_nb} for person {i}: {str(e)}")
                    continue
    
    return frame
    
def draw_bounding_box(X, Y, img, person_ids=None):
    '''
    Draw bounding boxes and person ID
    around list of lists of X and Y coordinates

    INPUTS:
    - X: list of list of x coordinates
    - Y: list of list of y coordinates
    - img: opencv image
    - person_ids: list of person IDs (optional)

    OUTPUT:
    - img: image with rectangles and person IDs
    '''

    cmap = plt.cm.hsv

    if person_ids is None:
        person_ids = range(len(X))

    # Draw rectangles
    for i, (x, y, person_id) in enumerate(zip(X, Y, person_ids)):
        if np.isnan(x).all():
            continue
        color = (np.array(cmap((i+1)/len(X)))*255).tolist()
        x_min, y_min = np.nanmin(x).astype(int)-25, np.nanmin(y).astype(int)-25
        x_max, y_max = np.nanmax(x).astype(int)+25, np.nanmax(y).astype(int)+25
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Write person ID
        cv2.putText(img, str(person_id),
            (x_min, y_min), 
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            color,
            2, cv2.LINE_AA)

    return img

def draw_keypts_skel(X, Y, scores, img, pose_model, kpt_thr):
    '''
    Draws keypoints and optionally skeleton for each person
    INPUTS:
    - X: list of list of x coordinates
    - Y: list of list of y coordinates
    - img: opencv image
    - pose_model: pose model name (string)
    
    OUTPUT:
    - img: image with keypoints and skeleton
    '''
    
    cmap = plt.cm.hsv
    
    # Draw keypoints
    for person_idx, (x, y, score) in enumerate(zip(X, Y, scores)):
        color = tuple(map(int, np.array(cmap((person_idx + 1) / len(X))) * 255))
        [cv2.circle(img, (int(x[i]), int(y[i])), 5, color, -1)
         for i in range(len(x))
         if not (np.isnan(x[i]) or np.isnan(y[i])) and score[i] >= kpt_thr]
    
    # Draw skeleton
    if pose_model == 'RTMPose':
        keypoint_id_to_index = {kp['id']: i for i, kp in halpe26_rtm['keypoint_info'].items()}

        # Add shoulder-hip connections to skeleton_info for shoulder's arc
        halpe26_rtm['skeleton_info'].update({
            'right_shoulder_hip': {'link': ('right_shoulder', 'right_hip')},
            'left_shoulder_hip': {'link': ('left_shoulder', 'left_hip')}
        })

        for link_info in halpe26_rtm['skeleton_info'].values():
            start_name, end_name = link_info['link']
            start_id = next(kp['id'] for kp in halpe26_rtm['keypoint_info'].values() if kp['name'] == start_name)
            end_id = next(kp['id'] for kp in halpe26_rtm['keypoint_info'].values() if kp['name'] == end_name)
            start_index = keypoint_id_to_index[start_id]
            end_index = keypoint_id_to_index[end_id]
            
            for person_idx, (x, y, score) in enumerate(zip(X, Y, scores)):
                if (not (np.isnan(x[start_index]) or np.isnan(y[start_index]) or 
                         np.isnan(x[end_index]) or np.isnan(y[end_index])) and
                    score[start_index] >= kpt_thr and
                    score[end_index] >= kpt_thr):
                    color = tuple(map(int, np.array(cmap((person_idx + 1) / len(X))) * 255))
                    cv2.line(img,
                        (int(x[start_index]), int(y[start_index])), 
                        (int(x[end_index]), int(y[end_index])),
                        color, 1)
    
    return img

def save_imgvid_reID(video_path, video_result_path, df_angles_list, pose_model, save_vid, save_img, kpt_thr):

    csv_dir = video_result_path.parent / 'video_results'
    print(f'Saving results to {csv_dir}')
    csv_paths = list(csv_dir.glob(f'{video_result_path.stem}_person*_angles.csv'))
    print(f"CSV files found: {csv_paths}")
    
    if not csv_paths:
        print("Error: No CSV files found in the specified directory.")
        return
        
    # Load both angles and points CSV files
    angles_coords = []
    points_coords = []
    person_ids = []
    for c in csv_paths:
        angles_file = c
        points_file = c.parent / (c.stem.replace('angles', 'points') + '.csv')
        
        with open(angles_file) as af, open(points_file) as pf:
            angles_df = pd.read_csv(af, header=[0,1,2,3], index_col=[0,1])
            points_df = pd.read_csv(pf, header=[0,1,2,3], index_col=[0,1])
            angles_coords.append(angles_df)
            points_coords.append(points_df)

            # Extract person_id from the filename
            person_id = int(c.stem.split('_person')[1].split('_')[0])
            person_ids.append(person_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if save_vid:
        video_pose_path = csv_dir / (video_result_path.stem + '_' + pose_model + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_pose_path), fourcc, fps, (int(W), int(H)))

    if save_img:
        img_pose_path = csv_dir / (video_result_path.stem + '_' + pose_model + '_img')
        img_pose_path.mkdir(parents=True, exist_ok=True)
        
    f = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_keypoints = []
        frame_scores = []
        for coord in points_coords:
            if f < len(coord):
                X = np.array(coord.iloc[f, 0::3])
                Y = np.array(coord.iloc[f, 1::3])
                S = np.array(coord.iloc[f, 2::3])
                
                person_keypoints = np.column_stack((X, Y))
                frame_keypoints.append(person_keypoints)
                frame_scores.append(S)
            else:
                if frame_keypoints and frame_scores:
                    frame_keypoints.append(frame_keypoints[-1])
                    frame_scores.append(frame_scores[-1])
                else:
                    # print(f"Warning: No previous keypoints or scores available for frame {f}")
                    break

        frame_keypoints = np.array(frame_keypoints)
        frame_scores = np.array(frame_scores)

        if frame_keypoints.size > 0 and frame_scores.size > 0:
            frame = draw_bounding_box(frame_keypoints[:,:,0], frame_keypoints[:,:,1], frame, person_ids)
            frame = draw_keypts_skel(frame_keypoints[:,:,0], frame_keypoints[:,:,1], frame_scores, frame, pose_model, kpt_thr)

            df_angles_list_frame = []
            for df in angles_coords:
                if f < len(df):
                    df_angles_list_frame.append(df.iloc[f,:])
                else:
                    df_angles_list_frame.append(df.iloc[-1,:])
            
            # person_ids를 overlay_angles 함수에 전달
            frame = overlay_angles(frame, df_angles_list_frame, frame_keypoints, frame_scores, kpt_thr=0.2, person_ids=person_ids)


        if save_vid:
            writer.write(frame)
        if save_img:
            cv2.imwrite(str(img_pose_path / f"{video_result_path.stem}_{pose_model}.{f:05d}.png"), frame)
        f += 1

    cap.release()
    if save_vid:
        writer.release()
    
def compute_angles_fun(config_dict, video_file):
    '''
    Compute joint and segment angles from csv position files.
    Automatically adjust angles when person switches to face the other way.
    Save a 2D csv angle file per person.
    Optionally filters results with Butterworth, gaussian, median, or loess filter.
    Optionally displays figures.

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
    
    /!\ Warning /!\
    - The angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim

    INPUTS:
    - one or several position csv files
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one csv file for joint and segment angles per detected person
    - a logs.txt file 
    '''
    
    # Retrieve parameters
    video_dir, video_files, result_dir, frame_rate, data_type = base_params(config_dict)
    joint_angles = config_dict.get('compute_angles').get('joint_angles')
    segment_angles = config_dict.get('compute_angles').get('segment_angles')
    angle_nb = len(joint_angles) + len(segment_angles)
    
    show_plots = config_dict.get('compute_angles_advanced').get('show_plots')
    flip_left_right = config_dict.get('compute_angles_advanced').get('flip_left_right')
    # print(f"flip_left_right: {flip_left_right}")
    do_filter = config_dict.get('compute_angles_advanced').get('filter')
    filter_type = config_dict.get('compute_angles_advanced').get('filter_type')
    butterworth_filter_order = config_dict.get('compute_angles_advanced').get('butterworth').get('order')
    butterworth_filter_cutoff = config_dict.get('compute_angles_advanced').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel = config_dict.get('compute_angles_advanced').get('gaussian').get('sigma_kernel')
    loess_filter_kernel = config_dict.get('compute_angles_advanced').get('loess').get('nb_values_used')
    median_filter_kernel = config_dict.get('compute_angles_advanced').get('median').get('kernel_size')
    filter_options = (do_filter, filter_type, butterworth_filter_order, butterworth_filter_cutoff, frame_rate, gaussian_filter_kernel, loess_filter_kernel, median_filter_kernel)

    # save
    show_angles_img = config_dict.get('compute_angles_advanced').get('show_angles_on_img')
    show_angles_vid = config_dict.get('compute_angles_advanced').get('show_angles_on_vid')

    # data_type = config_dict.get('pose').get('data_type')
    kpt_thr = config_dict.get('pose_advanced').get('keypoints_threshold')
    
    # Find csv position files in video_dir, search pose_model and video_file.stem
    csv_dir = result_dir / 'video_results' # csv files are in the pose directory
    video_file_name = video_file.stem
    csv_paths = list(csv_dir.glob(f'{video_file_name}_person*_points*.csv'))
    # print(f"found {len(csv_paths)} csv files")

    # Compute angles
    df_angles_list = []
    for i, c in enumerate(csv_paths):
        print(f"Processing {c}")
        df_angles = []
        try:
            logging.info(f'Starting processing for Person {i}')
            # Prepare angle csv header
            scorer = ['DavidPagnon']*(angle_nb+1)
            individuals = [f'person{i}']*(angle_nb+1)
            angs = ['Time'] + joint_angles + segment_angles
            coords = ['seconds']
            for j in joint_angles:
                angle_params = get_joint_angle_params(j) # get parameters for each joint angle
                if angle_params:
                    coords.append(angle_params[1])
                else:
                    coords.append('unknown')
            for s in segment_angles:
                angle_params = get_segment_angle_params(s) # get parameters for each segment angle
                if angle_params:
                    coords.append(angle_params[1])
                else:
                    coords.append('unknown')
            tuples = list(zip(scorer, individuals, angs, coords)) # multiindex
            index_angs_csv = pd.MultiIndex.from_tuples(tuples, names=['scorer', 'individuals', 'angs', 'coords'])  

            # Compute angles for each person, for each angle, with each required keypoint position
            with open(c) as c_f:
                        df_points = pd.read_csv(c_f, header=[0,1,2,3])      
                        if df_points.empty: # if no valid data after removing rows with too many NaN values
                            logging.warning(f'Person {i}: No valid data after removing rows with too many NaN values')
                            continue
                        time = [np.array(df_points.iloc[:,1])]
                        
                        # Flip along x when feet oriented to the left
                        if flip_left_right:
                            # print(f"Flipping left-right for Person {i}")
                            df_points = flip_left_right_direction(df_points, data_type)
                        
            # Joint angles
            joint_angle_series = []
            for j in joint_angles: 
                try:
                    angle_params = get_joint_angle_params(j)
                    if angle_params:
                        print(f"Calculating joint angle {j} for Person {i}")
                        j_ang_series = joint_angles_series_from_csv(df_points, angle_params, kpt_thr)
                        joint_angle_series.append(j_ang_series if j_ang_series is not None else np.nan)
                    else:
                        print(f"Error: No parameters found for joint angle {j}")
                        joint_angle_series.append(np.nan)
                except Exception as e:
                    logging.warning(f'Error calculating joint angle {j} for Person {i}: {str(e)}')
                    joint_angle_series.append(np.nan)

            # Segment angles
            segment_angle_series = []
            for s in segment_angles:
                try:
                    angle_params = get_segment_angle_params(s)
                    if angle_params:
                        s_ang_series = segment_angles_series_from_csv(df_points, angle_params, s, kpt_thr)
                        segment_angle_series.append(s_ang_series if s_ang_series is not None else np.nan)
                    else:
                        segment_angle_series.append(np.nan)
                except Exception as e:
                    logging.warning(f'Error calculating segment angle {s} for Person {i}: {str(e)}')
                    segment_angle_series.append(np.nan)

            angle_series = time + joint_angle_series + segment_angle_series

            # Filter out None values and ensure all elements are iterable
            angle_series = [series if isinstance(series, (list, np.ndarray)) else np.full_like(time[0], np.nan) for series in angle_series if series is not None]

            # Ensure lengths match
            max_length = max(len(series) for series in angle_series)
            angle_series = [np.pad(series, (0, max_length - len(series)), 'constant', constant_values=np.nan) for series in angle_series]

            if len(angle_series) != len(index_angs_csv):
                print(f"Warning: Length of angle_series ({len(angle_series)}) does not match length of index_angs_csv ({len(index_angs_csv)}).")
                # Pad with NaN if necessary
                while len(angle_series) < len(index_angs_csv):
                    angle_series.append(np.full(max_length, np.nan))
                # Truncate if too long
                angle_series = angle_series[:len(index_angs_csv)]
            df_angles = [pd.DataFrame(np.array(angle_series).T, columns=index_angs_csv)]
            
            # Filter
            if filter_options[0]:
                    filter_type = filter_options[1]
                    if filter_type == 'butterworth':
                        args = f'Butterworth filter, {filter_options[2]}th order, {filter_options[3]} Hz.'
                    if filter_type == 'gaussian':
                        args = f'Gaussian filter, Sigma kernel {filter_options[5]}'
                    if filter_type == 'loess':
                        args = f'LOESS filter, window size of {filter_options[6]} frames.'
                    if filter_type == 'median':
                        args = f'Median filter, kernel of {filter_options[7]}.'
                    logging.info(f'Person {i}: Filtering with {args}.')
                    df_angles[0].replace(0, np.nan, inplace=True)
                    df_angles += [df_angles[0].copy()]
                    df_angles[1] = df_angles[1].apply(filter.filter1d, axis=0, args=filter_options)

            # replace NaN with 0
            # df_angles[-1].replace(np.nan, 0, inplace=True)

            # Creation of the csv files
            # print(f"c : {c}")
            csv_angle_path = c.parent / (c.stem.replace('points', 'angles') + '.csv')
            # print(f"Saving angles CSV file to {csv_angle_path}")
            df_angles[-1].to_csv(csv_angle_path, sep=',', index=True, lineterminator='\n')
            
            if os.path.exists(csv_angle_path):
                logging.info(f'Successfully saved angles CSV for Person {i}')
            else:
                logging.error(f'Failed to save angles CSV for Person {i}')
            
            # Display figures
            if show_plots:
                if not df_angles[0].empty:
                    display_figures_fun_ang(df_angles)
                else:
                    logging.info(f'Person {i}: No angle data to display.')
                plt.close('all')  # always close figures to avoid memory leak

            df_angles_list += [df_angles[-1]]
            
        except Exception as e:
            continue  # if error, continue to next person

    # Add angles to vid and img
    if show_angles_img or show_angles_vid:
        video_base = Path(video_dir / video_file)
        video_pose = result_dir / (video_base.stem + '.mp4')
        save_imgvid_reID(video_base, video_pose, df_angles_list, 'RTMPose', save_vid=show_angles_vid, save_img=show_angles_img, kpt_thr=kpt_thr)