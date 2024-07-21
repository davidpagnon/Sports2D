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
        'Trunk': [['right_shoulder', 'right_hip'], 'horizontal', 0, 1],
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

def flip_left_right_direction(df_points, data_type):
    '''
    Inverts X coordinates to get consistent angles when person changes direction and goes to the left.
    The person is deemed to go to the left when their toes are to the left of their heels.
    
    INPUT:
    - df_points: dataframe of pose detection
    - data_type: 'video' or 'webcam'
    
    OUTPUT:
    - df_points: dataframe of pose detection with flipped X coordinates
    '''
    
    if data_type not in ['video', 'webcam']:
        raise ValueError("data_type must be either 'video' or 'webcam'")
    
    if data_type == 'video':
        right_toe = df_points.iloc[:,df_points.columns.get_level_values(2)=='right_big_toe'].iloc[:,0]
        right_heel = df_points.iloc[:,df_points.columns.get_level_values(2)=='right_heel'].iloc[:,0]
        left_toe = df_points.iloc[:,df_points.columns.get_level_values(2)=='left_big_toe'].iloc[:,0]
        left_heel = df_points.iloc[:,df_points.columns.get_level_values(2)=='left_heel'].iloc[:,0]
    else:  # webcam
        required_columns = ['right_big_toe_x', 'right_heel_x', 'left_big_toe_x', 'left_heel_x']
        for col in required_columns:
            if col not in df_points.columns:
                raise ValueError(f"Required column {col} is missing from the dataframe")
        right_toe = df_points['right_big_toe_x']
        right_heel = df_points['right_heel_x']
        left_toe = df_points['left_big_toe_x']
        left_heel = df_points['left_heel_x']
    
    right_orientation = right_toe - right_heel
    left_orientation = left_toe - left_heel
    orientation = right_orientation + left_orientation
    
    if data_type == 'video':
        df_points.iloc[:,2::3] = df_points.iloc[:,2::3] * np.where(orientation>=0, 1, -1).reshape(-1,1)
    else:  # webcam
        x_columns = [col for col in df_points.columns if col.endswith('_x')]
        df_points[x_columns] = df_points[x_columns] * np.where(orientation >= 0, 1, -1).reshape(-1, 1)

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
    
    OUTPUT:
    - ang_series: array of time series of the considered angle
    '''
    
    keypt_series = []
    
    for k in angle_params[0]:
        try:
            score_series = df_points.xs((k, 'score'), level=[2, 3], axis=1)
        except KeyError:
            # print(f"Key 'score' not found for keypoint '{k}'.")
            continue
        
        if score_series.max().max() >= kpt_thr:
            x_series = df_points.xs((k, 'x'), level=[2, 3], axis=1)
            y_series = df_points.xs((k, 'y'), level=[2, 3], axis=1)
            keypt_series.append(pd.DataFrame({'x': x_series.values.flatten(), 'y': y_series.values.flatten()}))
        else:
            return None

    if not keypt_series:
        return None

    # Compute angles
    points_list = [k.values.T for k in keypt_series]
    ang_series = points2D_to_angles(points_list)
    if ang_series is None:
        return None
    
    ang_series += angle_params[2]
    ang_series *= angle_params[3]
    
    if ang_series.mean() > 180: ang_series -= 360
    if ang_series.mean() < -180: ang_series += 360

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
    
    keypt_series = []
    for k in angle_params[0]:
        try:
            score_series = df_points.xs((k, 'score'), level=[2, 3], axis=1)
        except KeyError:
            continue
        
        if score_series.max().max() >= kpt_thr:
            x_series = df_points.xs((k, 'x'), level=[2, 3], axis=1)
            y_series = df_points.xs((k, 'y'), level=[2, 3], axis=1)
            keypt_series.append(pd.DataFrame({'x': x_series.values.flatten(), 'y': y_series.values.flatten()}))
        else:
            return None

    if not keypt_series:
        return None

    points_list = [k.values.T for k in keypt_series]
    ang_series = points2D_to_angles(points_list)
    if ang_series is None:
        return None

    ang_series += angle_params[2]
    ang_series *= angle_params[3]

    # For trunk: mean between angles RHip-RShoulder and LHip-LShoulder
    if segment == 'Trunk':
        ang_seriesR = ang_series
        angle_params[0] = [a.replace('R', 'L') for a in angle_params[0]]
        keypt_series = []
        for k in angle_params[0]:
            try:
                score_series = df_points.xs((k, 'score'), level=[2, 3], axis=1)
            except KeyError:
                continue
            
            if score_series.max().max() >= kpt_thr:
                x_series = df_points.xs((k, 'x'), level=[2, 3], axis=1)
                y_series = df_points.xs((k, 'y'), level=[2, 3], axis=1)
                keypt_series.append(pd.DataFrame({'x': x_series.values.flatten(), 'y': y_series.values.flatten()}))
            else:
                return None

        if not keypt_series:
            return None

        points_list = [k.values.T for k in keypt_series]
        ang_series = points2D_to_angles(points_list)
        if ang_series is None:
            return None
        
        ang_series += angle_params[2]
        ang_series *= angle_params[3]
        ang_series = np.mean((ang_seriesR, ang_series), axis=0)
        
    if ang_series.mean() > 180: ang_series -= 360
    if ang_series.mean() < -180: ang_series += 360

    return ang_series

def adjust_text_scale(frame, base_scale=0.25, base_thickness=1):
    """
    Adjusts the text scale and thickness based on the frame size.

    This function calculates appropriate text scale and thickness for overlay text
    on a given frame. It scales the text properties relative to a base frame width
    of 640 pixels, ensuring that text appears proportional on different frame sizes.

    Args:
        frame (numpy.ndarray): The input frame (image) on which text will be drawn.
                               Expected to be a 3-dimensional array (height, width, channels).
        base_scale (float, optional): The base scale factor for text. Defaults to 0.25.
        base_thickness (int, optional): The base thickness for text. Defaults to 1.

    Returns:
        tuple: A tuple containing two elements:
               - scale (float): The adjusted scale factor for text.
               - thickness (int): The adjusted thickness for text.
    """
    height, width, _ = frame.shape
    scale = base_scale * (width / 640)
    thickness = int(base_thickness * (width / 640))
    return scale, thickness

def draw_joint_angle(frame, joint, angle, keypoints, scores, kpt_thr):
    """
    Draws the joint angle on the given frame.

    This function identifies the keypoints for a specific joint and draws the angle
    between them if their scores are above the threshold.

    Args:
        frame (numpy.ndarray): The image frame to draw on.
        joint (str): The name of the joint (e.g., "Right knee", "Left elbow").
        angle (float): The angle to be displayed.
        keypoints (list): List of keypoint coordinates for the entire pose.
        scores (list): List of confidence scores for each keypoint.
        kpt_thr (float): Threshold for keypoint confidence scores.

    Returns:
        None: The function modifies the input frame in-place.

    Note:
        The function uses a predefined mapping of joints to keypoint indices.
        It only draws the angle if all relevant keypoint scores are above the threshold.
    """
    joint_to_keypoints = {
        "Right ankle": [14, 16, 21],  # knee, ankle, big_toe
        "Left ankle": [13, 15, 20],   # knee, ankle, big_toe
        "Right knee": [12, 14, 16],   # hip, knee, ankle
        "Left knee": [11, 13, 15],    # hip, knee, ankle
        "Right hip": [12, 12, 14],    # hip center, hip, knee
        "Left hip": [11, 11, 13],     # hip center, hip, knee
        "Right shoulder": [18, 6, 8], # neck, shoulder, elbow
        "Left shoulder": [18, 5, 7],  # neck, shoulder, elbow
        "Right elbow": [6, 8, 10],    # shoulder, elbow, wrist
        "Left elbow": [5, 7, 9],      # shoulder, elbow, wrist
    }

    if joint in joint_to_keypoints:
        pts = [keypoints[i] for i in joint_to_keypoints[joint]]
        scores_pts = [scores[i] for i in joint_to_keypoints[joint]]
        if all(score >= kpt_thr for score in scores_pts):
            pt1, pt2, pt3 = pts
            draw_angle_arc(frame, joint, pt1, pt2, pt3, angle)

def draw_angle_arc(frame, joint, pt1, pt2, pt3, angle):
    """
    Draws an arc representing the angle between three points on the frame.

    This function calculates and draws an arc representing the angle formed by
    three points, typically representing a joint angle. It also adds text showing
    the angle value.

    Args:
        frame (numpy.ndarray): The image frame to draw on.
        joint (str): The name of the joint (unused in this function, but kept for potential future use).
        pt1 (tuple): Coordinates of the first point.
        pt2 (tuple): Coordinates of the second point (vertex of the angle).
        pt3 (tuple): Coordinates of the third point.
        angle (float): The angle to be displayed.

    Returns:
        None: The function modifies the input frame in-place.

    Note:
        The function draws an arc, the angle value, and lines connecting the points.
        The radius of the arc is calculated as 20% of the average length of the two vectors.
    """
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))
    pt3 = tuple(map(int, pt3))
    
    # calculate vectors
    v1 = np.array(pt1) - np.array(pt2)
    v2 = np.array(pt3) - np.array(pt2)
    
    # calculate start and end angles
    start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
    end_angle = np.degrees(np.arctan2(v2[1], v2[0]))
    
    # angles adjustment
    if abs(end_angle - start_angle) > 180:
        if end_angle > start_angle:
            start_angle += 360
        else:
            end_angle += 360
    
    # start_angle is always smaller than end_angle
    if start_angle > end_angle:
        start_angle, end_angle = end_angle, start_angle
    
    # radius is 20% of the average of the two vectors
    radius = int(0.2 * (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2)
    
    # draw arc
    cv2.ellipse(frame, pt2, (radius, radius), 0, start_angle, end_angle, (0, 255, 0), 2)
    
    # position of the text
    text_angle = np.radians((start_angle + end_angle) / 2)
    text_pos = (
        int(pt2[0] + (radius + 20) * np.cos(text_angle)),
        int(pt2[1] + (radius + 20) * np.sin(text_angle))
    )
    
    # draw text
    cv2.putText(frame, f"{angle:.1f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # draw lines
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    cv2.line(frame, pt2, pt3, (0, 255, 0), 2)

def draw_segment_angle(frame, segment, angle, keypoints, scores, kpt_thr):
    """
    Draws the segment angle on the given frame.

    This function identifies the keypoints for a specific body segment and draws the angle
    between the segment and the horizontal axis if the keypoint scores are above the threshold.

    Args:
        frame (numpy.ndarray): The image frame to draw on.
        segment (str): The name of the body segment (e.g., "Right thigh", "Left arm").
        angle (float): The angle to be displayed.
        keypoints (list): List of keypoint coordinates for the entire pose.
        scores (list): List of confidence scores for each keypoint.
        kpt_thr (float): Threshold for keypoint confidence scores.

    Returns:
        None: The function modifies the input frame in-place.

    Note:
        The function uses a predefined mapping of segments to keypoint indices.
        It only draws the angle if both relevant keypoint scores are above the threshold.
    """
    thickness = 2
    length = 20
    color = (255, 0, 0)  # blue 

    segment_to_keypoints = {
        "Right foot": [16, 23],
        "Left foot": [15, 22],
        "Right shank": [14, 16],
        "Left shank": [13, 15],
        "Right thigh": [12, 14],
        "Left thigh": [11, 13],
        "Trunk": [19, 18],
        "Right arm": [6, 8],
        "Left arm": [5, 7],
        "Right forearm": [8, 10],
        "Left forearm": [7, 9]
    }

    if segment in segment_to_keypoints:
        pt1, pt2 = [keypoints[i] for i in segment_to_keypoints[segment]]
        score1, score2 = [scores[i] for i in segment_to_keypoints[segment]]
        if score1 >= kpt_thr and score2 >= kpt_thr:
            draw_angle_line(frame, pt1, pt2, angle, thickness, length, color)

def draw_angle_line(frame, pt1, pt2, angle, thickness, length, color):
    """
    Draws a line representing the segment angle on the frame.

    This function draws a horizontal reference line and an angled line representing
    the segment, along with the angle between them.

    Args:
        frame (numpy.ndarray): The image frame to draw on.
        pt1 (tuple): Coordinates of the first point of the segment.
        pt2 (tuple): Coordinates of the second point of the segment (unused in this function).
        angle (float): The angle to be displayed.
        thickness (int): Thickness of the lines to be drawn.
        length (int): Length of the lines to be drawn.
        color (tuple): RGB color of the lines.

    Returns:
        None: The function modifies the input frame in-place.

    Note:
        The function draws a horizontal reference line and an angled line from the first point (pt1).
        The second point (pt2) is not used in the current implementation but kept for potential future use.
    """
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))

    end_point = (int(pt1[0] + length), pt1[1])
    cv2.line(frame, pt1, end_point, color, thickness)
    
    segment_end = (int(pt1[0] + length * math.cos(math.radians(angle))), int(pt1[1] + length * math.sin(math.radians(angle))))
    cv2.line(frame, pt1, segment_end, color, thickness)

def overlay_angles(frame, df_angles_list_frame, keypoints, scores, kpt_thr):
    """
    Overlays angle information on the frame for each detected person, including text and progress bars.

    This function draws angle labels, values, and a progress bar for each angle
    on the frame. It also visualizes joint and segment angles using appropriate
    drawing functions.

    Args:
        frame (numpy.ndarray): The image frame to draw on.
        df_angles_list_frame (list): List of DataFrames containing angle information for each person.
        keypoints (list): List of keypoint coordinates for each person.
        scores (list): List of confidence scores for each person's keypoints.
        kpt_thr (float): Threshold for keypoint confidence scores.

    Returns:
        numpy.ndarray: The frame with overlaid angle information.

    Note:
        - The function uses a color map to assign different colors to different persons.
        - Angle labels and values are displayed in separate columns.
        - A progress bar is drawn to visually represent the angle value.
        - The function distinguishes between joint angles and segment angles for visualization.
    """
    cmap = plt.cm.hsv
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, (angles_frame_person, person_keypoints, person_scores) in enumerate(zip(df_angles_list_frame, keypoints, scores)):
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
                
                # Draw angle label
                cv2.putText(frame, 
                    angle_label + ':',
                    (10+250*i, 15+15*ang_nb), 
                    font, 0.5, 
                    (0,0,0), 
                    2, 
                    cv2.LINE_4)
                frame = cv2.putText(frame, 
                    angle_label + ':',
                    (10+250*i, 15+15*ang_nb), 
                    font, 0.5, 
                    (np.array(cmap((i+1)/len(df_angles_list_frame)))*255).tolist(), 
                    1, 
                    cv2.LINE_4)
                
                # Draw angle value
                cv2.putText(frame, 
                    f"{angle_value:.1f}",
                    (150+250*i, 15+15*ang_nb), 
                    font, 0.5, 
                    (0,0,0), 
                    2, 
                    cv2.LINE_4)
                frame = cv2.putText(frame, 
                    f"{angle_value:.1f}",
                    (150+250*i, 15+15*ang_nb), 
                    font, 0.5, 
                    (np.array(cmap((i+1)/len(df_angles_list_frame)))*255).tolist(), 
                    1, 
                    cv2.LINE_4)
                
                # Draw progress bar
                x_ang = int(angle_value*50/180)
                if x_ang != 0:
                    sub_frame = frame[1+15*ang_nb : 16+15*ang_nb, 170+250*i + min(0,x_ang) : 170+250*i + max(0,x_ang)]
                    if sub_frame.size > 0:
                        white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                        res = cv2.addWeighted(sub_frame, 0.6, white_rect, 0.4, 1.0)
                        frame[1+15*ang_nb : 16+15*ang_nb, 170+250*i + min(0,x_ang) : 170+250*i + max(0,x_ang)] = res

                # Draw joint or segment angle
                if any(joint in angle_label.lower() for joint in ["ankle", "knee", "hip", "shoulder", "elbow"]):
                    draw_joint_angle(frame, angle_label, angle_value, person_keypoints, person_scores, kpt_thr)
                else:
                    draw_segment_angle(frame, angle_label, angle_value, person_keypoints, person_scores, kpt_thr)
                
            except Exception as e:
                print(f"Warning: Unable to process angle {ang_nb} for person {i}: {str(e)}")
                continue
    
    return frame

def overlay_angles_video(frame, df_angles_list_frame, keypoints, scores, kpt_thr):
    """
    Overlays angle information on a video frame for each detected person.

    This function draws angle labels, values, and a progress bar for each angle
    on the frame. It also visualizes joint and segment angles using appropriate
    drawing functions.

    Args:
        frame (numpy.ndarray): The video frame to draw on.
        df_angles_list_frame (list): List of DataFrames containing angle information for each person.
        keypoints (list): List of keypoint coordinates for each person.
        scores (list): List of confidence scores for each person's keypoints.
        kpt_thr (float): Threshold for keypoint confidence scores.

    Returns:
        numpy.ndarray: The frame with overlaid angle information.

    Note:
        - The function uses a color map to assign different colors to different persons.
        - Angle labels and values are displayed in separate columns.
        - A progress bar is drawn to visually represent the angle value.
        - The function distinguishes between joint angles and segment angles for visualization.
    """
    cmap = plt.cm.hsv
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (angles_frame_person, person_keypoints, person_scores) in enumerate(zip(df_angles_list_frame, keypoints, scores)):
        for ang_nb, (angle_name, angle_value) in enumerate(angles_frame_person.items()):
            if angle_name == 'Time':  # Skip the 'Time' column
                continue

            angle_label = angle_name[2]  # make sure it's a joint or segment name
            angle_value = float(angle_value)

            # Ensure angle_name is a string
            if isinstance(angle_name, tuple):
                angle_name = angle_name[0]  # Use the first element of the tuple

            # Angle label
            cv2.putText(frame, 
                angles_frame_person.index[ang_nb][2] + ':',
                (10+250*i, 15+15*ang_nb), 
                font, 0.5, 
                (0,0,0), 
                2, 
                cv2.LINE_4)
            frame = cv2.putText(frame, 
                angles_frame_person.index[ang_nb][2] + ':',
                (10+250*i, 15+15*ang_nb), 
                font, 0.5, 
                (np.array(cmap((i+1)/len(df_angles_list_frame)))*255).tolist(), 
                1, 
                cv2.LINE_4)
            
            # Angle value
            cv2.putText(frame, 
                str(round(angles_frame_person.iloc[ang_nb],1)),
                (150+250*i, 15+15*ang_nb), 
                font, 0.5, 
                (0,0,0), 
                2, 
                cv2.LINE_4)
            frame = cv2.putText(frame, 
                str(round(angles_frame_person.iloc[ang_nb],1)),
                (150+250*i, 15+15*ang_nb), 
                font, 0.5, 
                (np.array(cmap((i+1)/len(df_angles_list_frame)))*255).tolist(), 
                1, 
                cv2.LINE_4)
            
            # Progress bar
            x_ang = int(angles_frame_person.iloc[ang_nb]*50/180)
            if x_ang > 0:
                sub_frame = frame[ 1+15*ang_nb : 16+15*ang_nb , 170+250*i : 170+250*i+x_ang ]
                if sub_frame.size>0:
                    white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                    res = cv2.addWeighted(sub_frame, 0.6, white_rect, 0.4, 1.0)
                    frame[ 1+15*ang_nb : 16+15*ang_nb , 170+250*i : 170+250*i+x_ang ] = res
            elif x_ang < 0:
                sub_frame = frame[ 1+15*ang_nb : 16+15*ang_nb , 170+250*i+x_ang : 170+250*i ]
                if sub_frame.size>0:
                    white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                    res = cv2.addWeighted(sub_frame, 0.6, white_rect, 0.4, 1.0)
                    frame[ 1+15*ang_nb : 16+15*ang_nb , 170+250*i+x_ang : 170+250*i ] = res

            # Draw joint or segment angle
            if "ankle" in angle_label.lower() or "knee" in angle_label.lower() or "hip" in angle_label.lower() or "shoulder" in angle_label.lower() or "elbow" in angle_label.lower():
                # print(f"Drawing joint angle for {angle_label}")
                draw_joint_angle(frame, angle_label, angle_value, person_keypoints, person_scores, kpt_thr)
            else:
                # print(f"Drawing segment angle for {angle_label}")
                draw_segment_angle(frame, angle_label, angle_value, person_keypoints, person_scores, kpt_thr)
        
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

def draw_keypts_skel(X, Y, img, pose_model):
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
    for (x,y) in zip(X,Y):
        [cv2.circle(img, (int(x[i]), int(y[i])), 5, (255,255,255), -1)
         for i in range(len(x))
         if not (np.isnan(x[i]) or np.isnan(y[i]))]
    
    # Draw skeleton
    if pose_model == 'halpe26_rtm':
        keypoint_id_to_index = {kp['id']: i for i, kp in halpe26_rtm['keypoint_info'].items()}
        for link_info in halpe26_rtm['skeleton_info'].values():
            start_name, end_name = link_info['link']
            start_id = next(kp['id'] for kp in halpe26_rtm['keypoint_info'].values() if kp['name'] == start_name)
            end_id = next(kp['id'] for kp in halpe26_rtm['keypoint_info'].values() if kp['name'] == end_name)
            start_index = keypoint_id_to_index[start_id]
            end_index = keypoint_id_to_index[end_id]
            
            for (x,y) in zip(X,Y):
                if not (np.isnan(x[start_index]) or np.isnan(y[start_index]) or 
                        np.isnan(x[end_index]) or np.isnan(y[end_index])):
                    cv2.line(img,
                        (int(x[start_index]), int(y[start_index])), 
                        (int(x[end_index]), int(y[end_index])),
                        tuple(link_info['color']), 2)
    
    return img

def save_imgvid_reID(video_path, video_result_path, df_angles_list, pose_model, save_vid, save_img):

    csv_dir = video_result_path.parent / 'pose'
    csv_paths = list(csv_dir.glob(f'{video_result_path.stem}_person*_angles.csv'))
    
    if not csv_paths:
        print("Error: No CSV files found in the specified directory.")
        return
        
    # Load both angles and points CSV files
    angles_coords = []
    points_coords = []
    for c in csv_paths:
        angles_file = c
        points_file = c.parent / (c.stem.replace('angles', 'points') + '.csv')
        
        with open(angles_file) as af, open(points_file) as pf:
            angles_df = pd.read_csv(af, header=[0,1,2,3], index_col=[0,1])
            points_df = pd.read_csv(pf, header=[0,1,2,3], index_col=[0,1])
            angles_coords.append(angles_df)
            points_coords.append(points_df)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if save_vid:
        video_pose_path = video_result_path.parent / (video_result_path.stem + '_' + pose_model + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_pose_path), fourcc, fps, (int(W), int(H)))

    if save_img:
        img_pose_path = video_result_path.parent / (video_result_path.stem + '_' + pose_model + '_img')
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
                    print(f"Warning: No previous keypoints or scores available for frame {f}")
                    break

        frame_keypoints = np.array(frame_keypoints)
        frame_scores = np.array(frame_scores)

        if frame_keypoints.size > 0 and frame_scores.size > 0:
            frame = draw_bounding_box(frame_keypoints[:,:,0], frame_keypoints[:,:,1], frame)
            frame = draw_keypts_skel(frame_keypoints[:,:,0], frame_keypoints[:,:,1], frame, pose_model)

            df_angles_list_frame = []
            for df in angles_coords:
                if f < len(df):
                    df_angles_list_frame.append(df.iloc[f,:])
                else:
                    df_angles_list_frame.append(df.iloc[-1,:])
            frame = overlay_angles_video(frame, df_angles_list_frame, frame_keypoints, frame_scores, kpt_thr=0.2)

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
    video_dir, video_files, result_dir, frame_rate = base_params(config_dict)
    joint_angles = config_dict.get('compute_angles').get('joint_angles')
    segment_angles = config_dict.get('compute_angles').get('segment_angles')
    angle_nb = len(joint_angles) + len(segment_angles)
    
    show_plots = config_dict.get('compute_angles_advanced').get('show_plots')
    flip_left_right = config_dict.get('compute_angles_advanced').get('flip_left_right')
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

    data_type = config_dict.get('pose').get('data_type')
    kpt_thr = config_dict.get('pose').get('keypoints_threshold')
    
    # Find csv position files in video_dir, search pose_model and video_file.stem
    csv_dir = result_dir / 'pose' # csv files are in the pose directory
    video_file_name = video_file.stem
    csv_paths = list(csv_dir.glob(f'{video_file_name}_person*_points*.csv'))

    # Compute angles
    df_angles_list = []
    for i, c in enumerate(csv_paths):
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
                            df_points = flip_left_right_direction(df_points, data_type)
                        
            # Joint angles
            joint_angle_series = []
            for j in joint_angles: 
                try:
                    angle_params = get_joint_angle_params(j)
                    if angle_params:
                        j_ang_series = joint_angles_series_from_csv(df_points, angle_params, kpt_thr)
                        joint_angle_series.append(j_ang_series if j_ang_series is not None else np.nan)
                    else:
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
            
            # already filtered in coords.csv, we need filtering again?
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
            df_angles[-1].replace(np.nan, 0, inplace=True)

            # Creation of the csv files
            csv_angle_path = c.parent / (c.stem.replace('points', 'angles') + '.csv')
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
        save_imgvid_reID(video_base, video_pose, df_angles_list, 'halpe26_rtm', save_vid=show_angles_vid, save_img=show_angles_img)