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
from Sports2D.Utilities.skeletons import *



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
def get_joint_angle_params(joint):
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
def display_figures_fun(df_list):
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
    # print(f"points2D_to_angles received points_list: {points_list}")
    # print(f"Number of points: {len(points_list)}")

    if len(points_list) < 2:
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

def flip_left_right_direction(df_points):
    '''
    Inverts X coordinates to get consistent angles when person changes direction and goes to the left.
    The person is deemed to go to the left when their toes are to the left of their heels.
    
    INPUT:
    - df_points: dataframe of pose detection
    
    OUTPUT:
    - df_points: dataframe of pose detection with flipped X coordinates
    '''
    
    # Check if the required columns exist
    required_columns = ['right_big_toe', 'right_heel', 'left_big_toe', 'left_heel']
    for col in required_columns:
        if col not in df_points.columns.get_level_values(2):
            print(f"Debug: Column '{col}' not found in df_points")
    
    try:
        righ_orientation = df_points.iloc[:,df_points.columns.get_level_values(2)=='right_big_toe'].iloc[:,0] - df_points.iloc[:,df_points.columns.get_level_values(2)=='right_heel'].iloc[:,0]
    except Exception as e:
        righ_orientation = pd.Series([0] * len(df_points))
    
    try:
        left_orientation = df_points.iloc[:,df_points.columns.get_level_values(2)=='left_big_toe'].iloc[:,0] - df_points.iloc[:,df_points.columns.get_level_values(2)=='left_heel'].iloc[:,0]
    except Exception as e:
        left_orientation = pd.Series([0] * len(df_points))
    
    orientation = righ_orientation + left_orientation

    try:
        df_points.iloc[:,2::3] = df_points.iloc[:,2::3] * np.where(orientation>=0, 1, -1).reshape(-1,1)
    except Exception as e:
        print(f"Debug: Error in flipping coordinates: {str(e)}")

    return df_points

def flip_left_right_direction_webcam(df_points):
    required_columns = ['right_big_toe_x', 'right_heel_x', 'left_big_toe_x', 'left_heel_x']
    for col in required_columns:
        if col not in df_points.columns:
            raise ValueError(f"Required column {col} is missing from the dataframe")
    
    # Calculate orientation
    right_orientation = df_points['right_big_toe_x'] - df_points['right_heel_x']
    left_orientation = df_points['left_big_toe_x'] - df_points['left_heel_x']
    orientation = right_orientation + left_orientation
    
    # Flip X coordinates based on orientation
    x_columns = [col for col in df_points.columns if col.endswith('_x')]
    df_points[x_columns] = df_points[x_columns] * np.where(orientation >= 0, 1, -1).reshape(-1, 1)
    
    if (orientation < 0).any():
        print("Some coordinates flipped")
    else:
        print("No coordinates flipped")
    
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
    # print(f"Processing joint: {angle_params[1]}")
    # print(f"Keypoints to use: {angle_params[0]}")
    
    keypt_series = []
    # print(f"df_points columns: {df_points.columns}")
    
    for k in angle_params[0]:
        if f"{k}_x" in df_points.columns and f"{k}_y" in df_points.columns and f"{k}_score" in df_points.columns:
            score = df_points[f"{k}_score"].values[0]
            if score >= kpt_thr:
                keypt = df_points[[f"{k}_x", f"{k}_y"]]
                # print(f"Keypoint {k} added: {keypt.values[0]}, score: {score}")
                keypt_series.append(keypt)
            else:
                # print(f"Keypoint {k} skipped due to low score: {score}")
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

    # print(f"Final angle for {angle_params[1]}: {ang_series}")
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
    
    ang_series = np.abs(ang_series)
    if ang_series is None or len(ang_series) == 0:
        return None
    return ang_series

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

        points_list = [k.values.T for k in keypt_series]
        ang_series = points2D_to_angles(points_list)
        if ang_series is None:
            return None
        
        # print(f"offset : {angle_params[2]}")
        ang_series += angle_params[2]
        # print(f"direction : {angle_params[3]}")
        ang_series *= angle_params[3]
        ang_series = np.mean((ang_seriesR, ang_series), axis=0)
        
    if ang_series.mean() > 180: ang_series -= 360
    if ang_series.mean() < -180: ang_series += 360
        
    ang_series = np.abs(ang_series)
    if ang_series is None or len(ang_series) == 0:
        return None
    return ang_series

def adjust_text_scale(frame, base_scale=0.25, base_thickness=1):
    height, width, _ = frame.shape
    scale = base_scale * (width / 640)
    thickness = int(base_thickness * (width / 640))
    return scale, thickness

def draw_joint_angle(frame, joint, angle, keypoints, scores, kpt_thr):
    joint_to_keypoints = {
        "Right ankle": [14, 16, 21],  # knee, ankle, big_toe
        "Left ankle": [13, 15, 20],   # knee, ankle, big_toe
        "Right knee": [12, 14, 16],   # hip, knee, ankle
        "Left knee": [11, 13, 15],    # hip, knee, ankle
        "Right hip": [19, 12, 14],    # hip center, hip, knee
        "Left hip": [19, 11, 13],     # hip center, hip, knee
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
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))
    pt3 = tuple(map(int, pt3))
    
    # calculate vectors
    v1 = np.array(pt1) - np.array(pt2)
    v2 = np.array(pt3) - np.array(pt2)
    
    # 시작 각도와 끝 각도 계산
    start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
    end_angle = np.degrees(np.arctan2(v2[1], v2[0]))
    
    # 항상 작은 각을 그리도록 조정
    if abs(end_angle - start_angle) > 180:
        if end_angle > start_angle:
            start_angle += 360
        else:
            end_angle += 360
    
    # 시작 각도가 항상 작은 값이 되도록 조정
    if start_angle > end_angle:
        start_angle, end_angle = end_angle, start_angle
    
    # 반지름 계산 (벡터 길이의 평균의 20%)
    radius = int(0.2 * (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2)
    
    # 각도 그리기
    cv2.ellipse(frame, pt2, (radius, radius), 0, start_angle, end_angle, (0, 255, 0), 2)
    
    # 텍스트 위치 계산
    text_angle = np.radians((start_angle + end_angle) / 2)
    text_pos = (
        int(pt2[0] + (radius + 20) * np.cos(text_angle)),
        int(pt2[1] + (radius + 20) * np.sin(text_angle))
    )
    
    # 각도 텍스트 추가
    cv2.putText(frame, f"{angle:.1f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 선 그리기
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    cv2.line(frame, pt2, pt3, (0, 255, 0), 2)

def draw_segment_angle(frame, segment, angle, keypoints, scores, kpt_thr):
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
        "Left forearm": [7, 9],
    }

    if segment in segment_to_keypoints:
        pt1, pt2 = [keypoints[i] for i in segment_to_keypoints[segment]]
        score1, score2 = [scores[i] for i in segment_to_keypoints[segment]]
        if score1 >= kpt_thr and score2 >= kpt_thr:
            draw_angle_line(frame, pt1, pt2, angle, thickness, length, color)

def draw_angle_line(frame, pt1, pt2, angle, thickness, length, color):
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))

    end_point = (int(pt1[0] + length), pt1[1])
    cv2.line(frame, pt1, end_point, color, thickness)
    
    segment_end = (int(pt1[0] + length * math.cos(math.radians(angle))), int(pt1[1] + length * math.sin(math.radians(angle))))
    cv2.line(frame, pt1, segment_end, color, thickness)

def overlay_angles(frame, df_angles_list_frame, keypoints, scores, kpt_thr):
    # print("Debug: overlay_angles function called")
    cmap = plt.cm.hsv
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    height, width, _ = frame.shape
    scale, thickness = adjust_text_scale(frame)
    
    base_y = int(0.03 * height)  # 3% from the top
    y_step = int(0.05 * height)  # Step size for each line
    
    for i, (angles_frame_person, person_keypoints, person_scores) in enumerate(zip(df_angles_list_frame, keypoints, scores)):
        # print(f"Debug: Processing person {i}")
        for ang_nb, (angle_name, angle_value) in enumerate(angles_frame_person.items()):
            try:
                if isinstance(angle_value, pd.Series):
                    angle_value = angle_value.iloc[0]
                
                angle_value = float(angle_value)
                
                # print(f"Debug: Processing angle {ang_nb}: {angle_name} = {angle_value}")
                
                angle_label = str(angle_name)
                text = f"{angle_label}: {angle_value:.1f}"
                
                text = text.encode('utf-8').decode('utf-8')
                
                # Text color based on joint type
                if "ankle" in angle_name.lower() or "knee" in angle_name.lower() or "hip" in angle_name.lower() or "shoulder" in angle_name.lower() or "elbow" in angle_name.lower():
                    text_color = (0, 255, 0)  # green
                else:
                    text_color = (255, 0, 0)  # blue    
                
                # Draw text
                text_x = 10 + (width // 3) * i
                text_y = base_y + y_step * ang_nb
                
                cv2.putText(frame, text, (text_x, text_y), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
                frame = cv2.putText(frame, text, (text_x, text_y), font, scale, text_color, thickness, cv2.LINE_AA)
                
                # Draw joint or segment angle
                if "ankle" in angle_name.lower() or "knee" in angle_name.lower() or "hip" in angle_name.lower() or "shoulder" in angle_name.lower() or "elbow" in angle_name.lower():
                    draw_joint_angle(frame, angle_name, angle_value, person_keypoints, person_scores, kpt_thr)
                else:
                    draw_segment_angle(frame, angle_name, angle_value, person_keypoints, person_scores, kpt_thr)
                
            except Exception as e:
                print(f"Warning: Unable to process angle {ang_nb} for person {i}: {str(e)}")
                continue
    
    return frame

def overlay_angles_video(frame, df_angles_list_frame, keypoints, scores, kpt_thr):
    '''
    Overlays a text box for each detected person with joint and segment angles
    
    INPUT:
    - frame: a frame opened with OpenCV
    - df_angles_list_frame: list of one frame for all angles
    '''
    # logging.info(f"Number of angle dataframes: {len(df_angles_list_frame)}")
    # logging.info(f"Shape of keypoints: {keypoints.shape}")
    # logging.info(f"Shape of scores: {scores.shape}") 
    cmap = plt.cm.hsv
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (angles_frame_person, person_keypoints, person_scores) in enumerate(zip(df_angles_list_frame, keypoints, scores)):
        for ang_nb, (angle_name, angle_value) in enumerate(angles_frame_person.items()):
            if angle_name == 'Time':  # Skip the 'Time' column
                continue
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
        
    return frame


def draw_bounding_box(X, Y, img):
    '''
    Draw bounding boxes and person ID
    around list of lists of X and Y coordinates
    
    INPUTS:
    - X: list of list of x coordinates
    - Y: list of list of y coordinates
    - img: opencv image
    
    OUTPUT:
    - img: image with rectangles and person IDs
    '''
    
    cmap = plt.cm.hsv
    
    # Draw rectangles
    [cv2.rectangle(img, 
        (np.nanmin(x).astype(int)-25, np.nanmin(y).astype(int)-25), 
        (np.nanmax(x).astype(int)+25, np.nanmax(y).astype(int)+25), 
        (np.array(cmap((i+1)/len(X)))*255).tolist(), 
        2) 
        for i,(x,y) in enumerate(zip(X,Y)) if not np.isnan(x).all()]
 
    # Write person ID
    [cv2.putText(img, str(i),
        (np.nanmin(x).astype(int), np.nanmin(y).astype(int)), 
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (np.array(cmap((i+1)/len(X)))*255).tolist(),
        2, cv2.LINE_AA) 
        for i,(x,y) in enumerate(zip(X,Y)) if not np.isnan(x).all()]
    
    return img

def draw_keypts_skel(X, Y, img, *pose_model):
    '''
    Draws keypoints and optionally skeleton for each person

    INPUTS:
    - X: list of list of x coordinates
    - Y: list of list of y coordinates
    - img: opencv image
    
    OUTPUT:
    - img: image with keypoints and skeleton
    '''
    
    model = eval(pose_model[0])
    cmap = plt.cm.hsv
    
    # Draw keypoints (same color for same keypoint)
    for (x,y) in zip(X,Y):
        [cv2.circle(img, (int(x[i]), int(y[i])), 5,
            (255,255,255),
            -1)
            for i in range(len(x))
            if not (np.isnan(x[i]) or np.isnan(y[i]))]
    
    # Draw skeleton
    if pose_model != None:
        eval(pose_model[0])
        # Get (unique) pairs between which to draw a line
        node_pairs = []
        for data_i in PreOrderIter(model.root, filter_=lambda node: node.is_leaf):
            node_branches = [node_i.id for node_i in data_i.path[1:]]
            node_pairs += [[node_branches[i],node_branches[i+1]] for i in range(len(node_branches)-1)]
        node_pairs = [list(x) for x in set(tuple(x) for x in node_pairs)]
        # Draw lines
        for (x,y) in zip(X,Y):
            [cv2.line(img,
            (int(x[n[0]]), int(y[n[0]])), (int(x[n[1]]), int(y[n[1]])),
            (np.array(cmap((i+1)/len(node_pairs)))*255).tolist(), 
            2)
            for i, n in enumerate(node_pairs)
            if not (np.isnan(x[n[0]]) or np.isnan(y[n[0]]) or np.isnan(x[n[1]]) or np.isnan(y[n[1]]))]
    
    return img

def save_imgvid_reID(video_path, video_result_path, df_angles_list, pose_model, save_vid, save_img):
    '''
    Displays json 2d detections overlayed on original raw images.
    High confidence keypoints are green, low confidence ones are red.
     
    Note: See 'json_display_without_img.py' if you only want to display the
    json coordinates on an animated graph or if don't have the original raw
    images.
    
    Usage: 
    json_display_with_img -j "<json_folder>" -i "<raw_img_folder>"
    json_display_with_img -j "<json_folder>" -i "<raw_img_folder>" -o "<output_img_folder>" -d True -s True
    import json_display_with_img; json_display_with_img.json_display_with_img_func(json_folder=r'<json_folder>', raw_img_folder=r'<raw_img_folder>')
    '''

    logging.info(f"Starting save_imgvid_reID with video_path: {video_path}, video_result_path: {video_result_path}")
    logging.info(f"save_vid: {save_vid}, save_img: {save_img}")
    logging.info(f"Number of dataframes in df_angles_list: {len(df_angles_list)}")
    if not df_angles_list:
        logging.error("df_angles_list is empty")
        return
    # Find csv position files, prepare video and image saving paths
    pose_model = pose_model[0]
    csv_dir = video_result_path.parent / 'pose'
    # print(f"csv_dir: {csv_dir}")
    csv_paths = list(csv_dir.glob(f'{video_result_path.stem}_person*_angles.csv'))
    # print(f"csv_paths: {csv_paths}")
    
    if not csv_paths:
        logging.error("No CSV files found in the specified directory.")
        return
        
    # Open csv files
    coords = []
    try:
        for c in csv_paths:
            with open(c) as c_f:
                coord_df = pd.read_csv(c_f, header=[0,1,2,3])
                logging.info(f"Loaded CSV: {c} with shape: {coord_df.shape}")
                coords.append(coord_df)
    except Exception as e:
        logging.warning(f"No csv files found or error reading CSV: {str(e)}")
        return

    # Open video frame by frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save_vid:
        video_pose_path = video_result_path.parent / (video_result_path.stem + '_' + pose_model + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_pose_path), fourcc, fps, (int(W), int(H)))
        if not writer.isOpened():
            logging.error(f"Error creating video writer for: {video_result_path}")
            return
    if save_img:
        img_pose_path = video_result_path.parent / (video_result_path.stem + '_' + pose_model + '_img')
        img_pose_path.mkdir(parents=True, exist_ok=True)  
        
    f = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            logging.info(f"Finished reading video after {f} frames")
            break
        try:
            logging.info(f"Processing frame {f}")
            
            # Extract X, Y coordinates and scores
            frame_keypoints = []
            frame_scores = []
            for coord in coords:
                if f < len(coord):
                    X = np.array(coord.iloc[f, 2::3])
                    Y = np.array(coord.iloc[f, 3::3])
                    S = np.array(coord.iloc[f, 4::3])
                    
                    # Replace 0 with NaN for consistency
                    X = np.where(X == 0., np.nan, X)
                    Y = np.where(Y == 0., np.nan, Y)
                    
                    # Combine X and Y into keypoints
                    person_keypoints = np.column_stack((X, Y))
                    frame_keypoints.append(person_keypoints)
                    frame_scores.append(S)
                else:
                    logging.warning(f"Frame {f} exceeds coord data length {len(coord)}")
                    if frame_keypoints and frame_scores:
                        frame_keypoints.append(frame_keypoints[-1])
                        frame_scores.append(frame_scores[-1])
                    else:
                        logging.error(f"No previous keypoints or scores available for frame {f}")
                        break

            frame_keypoints = np.array(frame_keypoints)
            frame_scores = np.array(frame_scores)

            logging.info(f"Frame {f}: keypoints shape {frame_keypoints.shape}, scores shape {frame_scores.shape}")

            # 키포인트와 스코어가 비어있지 않은 경우에만 처리
            if frame_keypoints.size > 0 and frame_scores.size > 0:
                # Draw bounding box
                frame = draw_bounding_box(frame_keypoints[:,:,0], frame_keypoints[:,:,1], frame)
                logging.info(f"Frame {f}: Bounding box drawn")

                # Draw keypoints and skeleton using draw_skeleton
                try:
                    frame = draw_keypts_skel(frame_keypoints[:,:,0], frame_keypoints[:,:,1], frame, pose_model)
                    logging.info(f"Frame {f}: Skeleton drawn")
                except Exception as e:
                    logging.error(f"Error in draw_skeleton for frame {f}: {str(e)}")

                # Overlay angles
                df_angles_list_frame = []
                for df in df_angles_list:
                    if f < len(df):
                        df_angles_list_frame.append(df.iloc[f,:])
                    else:
                        df_angles_list_frame.append(df.iloc[-1,:])
                frame = overlay_angles_video(frame, df_angles_list_frame, frame_keypoints, frame_scores, kpt_thr=0.2)
                logging.info(f"Frame {f}: Angles overlaid")

            else:
                logging.warning(f"Frame {f}: No valid keypoints or scores")

            # Save video and images
            if save_vid:
                writer.write(frame)
                logging.info(f"Frame {f}: Saved to video")
            if save_img:
                cv2.imwrite(str(img_pose_path / f"{video_result_path.stem}_{pose_model}.{f:05d}.png"), frame)
                logging.info(f"Frame {f}: Saved as image")

        except Exception as e:
            logging.error(f"Error processing frame {f}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            continue

        f += 1

    cap.release()
    if save_vid:
        writer.release()
    logging.info("Finished processing video and saving results")
    
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
    
    show_angles_img = config_dict.get('compute_angles_advanced').get('show_angles_on_img')
    show_angles_vid = config_dict.get('compute_angles_advanced').get('show_angles_on_vid')

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

                        # replace 0 with NaN
                        df_points = df_points.replace(0, np.nan)
                        
                        # Remove rows with too many NaN values
                        df_points = df_points.dropna(axis=0, thresh=len(df_points)*0.8)
                        
                        if df_points.empty: # if no valid data after removing rows with too many NaN values
                            logging.warning(f'Person {i}: No valid data after removing rows with too many NaN values')
                            continue

                        time = [np.array(df_points.iloc[:,1])]
                        
                        # Flip along x when feet oriented to the left
                        if flip_left_right:
                            df_points = flip_left_right_direction(df_points)
                        
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
                    display_figures_fun(df_angles)
                else:
                    logging.info(f'Person {i}: No angle data to display.')
                plt.close('all')  # always close figures to avoid memory leak

            df_angles_list += [df_angles[-1]]
            
        except Exception as e:
            continue  # if error, continue to next person

    # Add angles to vid and img
    if show_angles_img or show_angles_vid:
        # print(f"Debug: show_angles_img: {show_angles_img}, show_angles_vid: {show_angles_vid}")
        video_base = Path(video_dir / video_file)
        video_pose = result_dir / (video_base.stem + '.mp4')
        
        logging.info(f'Saving video with angles and skeleton in {str(video_pose)}.')
        logging.info(f"video_base exists: {video_base.exists()}")
        logging.info(f"Number of dataframes in df_angles_list: {len(df_angles_list)}")
        for i, df in enumerate(df_angles_list):
            logging.info(f"Shape of df_angles_list[{i}]: {df.shape}")
        
        save_imgvid_reID(video_base, video_pose, df_angles_list, 'HALPE_26', save_vid=show_angles_vid, save_img=show_angles_img)

    logging.info("Angle computation and visualization completed.")