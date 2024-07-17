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
joint_angle_dict = { 
    'Right ankle': [['RHeel', 'RBigToe', 'RAnkle', 'RKnee'], 'dorsiflexion', -90, -1],
    'Left ankle': [['LHeel', 'LBigToe', 'LAnkle', 'LKnee'], 'dorsiflexion', -90, -1],
    'Right knee': [['RHip', 'RKnee', 'RAnkle'], 'flexion', -180, -1],
    'Left knee': [['LHip', 'LKnee', 'LAnkle'], 'flexion', -180, -1],
    'Right hip': [['RKnee', 'RHip', 'RShoulder'], 'flexion', -180, -1],
    'Left hip': [['LKnee', 'LHip', 'LShoulder'], 'flexion', -180, -1],
    'Right shoulder': [['RHip', 'RShoulder', 'RElbow'], 'flexion', 0, 1],
    'Left shoulder': [['LHip', 'LShoulder', 'LElbow'], 'flexion', 0, 1],
    'Right elbow': [['RWrist', 'RElbow', 'RShoulder'], 'flexion', -180, -1],
    'Left elbow': [['LWrist', 'LElbow', 'LShoulder'], 'flexion', -180, -1],
    'Right wrist': [['RIndex', 'RWrist', 'RElbow'], 'flexion', -180, -1],
    'Left wrist': [['LIndex', 'LWrist', 'LElbow'], 'flexion', -180, -1]
    }

segment_angle_dict = {     
    'Right foot': [['RHeel', 'RBigToe'], 'horizontal', 0, -1],
    'Left foot': [['LHeel', 'LBigToe'], 'horizontal', 0, -1],
    'Right shank': [['RKnee', 'RAnkle'], 'horizontal', 0, -1],
    'Left shank': [['LKnee', 'LAnkle'], 'horizontal', 0, -1],
    'Right thigh': [['RHip', 'RKnee'], 'horizontal', 0, -1],
    'Left thigh': [['LHip', 'LKnee'], 'horizontal', 0, -1],
    'Trunk': [['RShoulder', 'RHip'], 'horizontal', 0, 1],
    'Right arm': [['RShoulder', 'RElbow'], 'horizontal', 0, -1],
    'Left arm': [['LShoulder', 'LElbow'], 'horizontal', 0, -1],
    'Right forearm': [['RElbow', 'RWrist'], 'horizontal', 0, -1],
    'Left forearm': [['LElbow', 'LWrist'], 'horizontal', 0, -1],
    'Right hand': [['RWrist', 'RIndex'], 'horizontal', 0, -1],
    'Left hand': [['LWrist', 'LIndex'], 'horizontal', 0, -1]
    }
    

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
    
    pw.show()
    
    
def points2D_to_angles(points_list):
    '''
    If len(points_list)==2, computes clockwise angle of ab vector w.r.t. horizontal (e.g. RBigToe-RHeel) 
    If len(points_list)==3, computes clockwise angle from a to c around b (e.g. Neck, Hip, Knee) 
    If len(points_list)==4, computes clockwise angle between vectors ab and cd (e.g. CHip-Neck, RHip-RKnee)
    
    If parameters are float, returns a float between 0.0 and 360.0
    If parameters are arrays, returns an array of floats between 0.0 and 360.0
    '''
    if len(points_list) < 2:
        return None
    
    ax, ay = points_list[0]
    bx, by = points_list[1]
    
    if len(points_list)==2:
        ux, uy = bx-ax, by-ay
        vx, vy = 1,0

    elif len(points_list) == 3:
        cx, cy = points_list[2]
        ux, uy = ax-bx, ay-by
        vx, vy = cx-bx, cy-by

    elif len(points_list) == 4:
        cx, cy = points_list[2]
        dx, dy = points_list[3]
        ux, uy = bx-ax, by-ay
        vx, vy = dx-cx, dy-cy

    ang = np.arctan2(uy, ux) - np.arctan2(vy, vx)
    ang_deg = np.array(np.degrees(np.unwrap(ang * 2) / 2))

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
    
    righ_orientation = df_points.iloc[:,df_points.columns.get_level_values(2)=='RBigToe'].iloc[:,0] - df_points.iloc[:,df_points.columns.get_level_values(2)=='RHeel'].iloc[:,0]
    left_orientation = df_points.iloc[:,df_points.columns.get_level_values(2)=='LBigToe'].iloc[:,0] - df_points.iloc[:,df_points.columns.get_level_values(2)=='LHeel'].iloc[:,0]
    orientation = righ_orientation + left_orientation
    df_points.iloc[:,2::3] = df_points.iloc[:,2::3] * np.where(orientation>=0, 1, -1).reshape(-1,1)
    
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
        if df_points[f"{k}_score"].values[0] >= kpt_thr:
            keypt_series.append(df_points[[f"{k}_x", f"{k}_y"]])
        else:
            return None

    # Compute angles
    points_list = [k.values.T for k in keypt_series]
    ang_series = points2D_to_angles(points_list)
    if ang_series is None:
        return None
    
    ang_series += angle_params[2]
    ang_series *= angle_params[3]
    # ang_series = np.where(ang_series>180,ang_series-360,ang_series) # handled by np.unwrap in points2D_to_angles()
    # ang_series = np.where((ang_series==0) | (ang_series==90) | (ang_series==180), +0, ang_series)
    if ang_series.mean() > 180: ang_series -= 360
    if ang_series.mean() < -180: ang_series += 360
    
    # abs value of angle
    ang_series = np.abs(ang_series)

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
        
    # abs value of angle
    ang_series = np.abs(ang_series)

    return ang_series

def adjust_text_scale(frame, base_scale=0.25, base_thickness=1):
    height, width, _ = frame.shape
    scale = base_scale * (width / 640)
    thickness = int(base_thickness * (width / 640))
    return scale, thickness

def draw_joint_angle(frame, joint, angle, keypoints, scores, kpt_thr):
    # print(f"Debug: draw_joint_angle called for {joint} with angle {angle}")
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
    # Convert points to integer tuples
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))
    pt3 = tuple(map(int, pt3))
    
    # Calculate vectors
    v1 = np.array(pt1) - np.array(pt2)
    v2 = np.array(pt3) - np.array(pt2)
    
    # Calculate angle
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # Determine the direction of the angle
    cross_product = np.cross(v1, v2)
    
    # Calculate start and end angles
    start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
    end_angle = np.degrees(np.arctan2(v2[1], v2[0]))
    
    # Ensure the smaller angle is always drawn
    if abs(end_angle - start_angle) > 180:
        if end_angle > start_angle:
            end_angle -= 360
        else:
            start_angle -= 360
    
    # Calculate radius (adjust based on the length of the limb)
    limb_length = min(np.linalg.norm(v1), np.linalg.norm(v2))
    radius = int(limb_length * 0.15)
    
    # Determine the direction to draw the arc
    if cross_product < 0:
        start_angle, end_angle = end_angle, start_angle
    
    # Draw the arc
    cv2.ellipse(frame, pt2, (radius, radius), 0, start_angle, end_angle, (0, 255, 0), 2)
    
    # Add text for angle
    text_pos = (pt2[0] + int(radius * 1.2), pt2[1] - int(radius * 0.2))
    cv2.putText(frame, f"{angle:.1f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw lines to connect points
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    cv2.line(frame, pt2, pt3, (0, 255, 0), 2)

def draw_segment_angle(frame, segment, angle, keypoints, scores, kpt_thr):
    thickness = 2
    length = 40
    color = (255, 0, 0)  # blue 

    segment_to_keypoints = {
        "Right foot": [22, 20],
        "Left foot": [19, 17],
        "Right shank": [14, 16],
        "Left shank": [13, 15],
        "Right thigh": [12, 14],
        "Left thigh": [11, 13],
        "Trunk": [6, 12],
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
    pose_algo = config_dict.get('pose').get('pose_algo')
    if pose_algo == 'OPENPOSE':
        pose_model = config_dict.get('pose').get('OPENPOSE').get('openpose_model')
    elif pose_algo == 'BLAZEPOSE':
        pose_model = 'BLAZEPOSE'
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
    
    # Find csv position files in video_dir, search pose_model and video_file.stem
    logging.info(f'Retrieving csv position files in {result_dir}...')
    csv_paths = list(result_dir.glob(f'*{video_file.stem}_*points*.csv'))
    logging.info(f'{len(csv_paths)} persons found.')

    # Compute angles
    df_angles_list = []
    for i, c in enumerate(csv_paths):
        # Prepare angle csv header
        scorer = ['DavidPagnon']*(angle_nb+1)
        individuals = [f'person{i}']*(angle_nb+1)
        angs = ['Time'] + joint_angles + segment_angles
        coords = ['seconds'] + [joint_angle_dict.get(j)[1] for j in joint_angles] + [segment_angle_dict.get(s)[1] for s in segment_angles]
        tuples = list(zip(scorer, individuals, angs, coords))
        index_angs_csv = pd.MultiIndex.from_tuples(tuples, names=['scorer', 'individuals', 'angs', 'coords'])    

        # Compute angles for each person, for each angle, with each required keypoint position
        logging.info(f'Person {i}: Computing 2D joint and segment angles.')
        with open(c) as c_f:
            df_points = pd.read_csv(c_f, header=[0,1,2,3])
            time = [np.array(df_points.iloc[:,1])]
            
            # Flip along x when feet oriented to the left
            if flip_left_right:
                df_points = flip_left_right_direction(df_points)
            
            # Joint angles
            joint_angle_series = []
            for j in joint_angles: 
                angle_params = joint_angle_dict.get(j)
                j_ang_series = joint_angles_series_from_points(df_points, angle_params)
                joint_angle_series += [j_ang_series]
    
            # Segment angles
            segment_angle_series = []
            for s in segment_angles:
                angle_params = segment_angle_dict.get(s)
                s_ang_series = segment_angles_series_from_points(df_points, angle_params, s)
                segment_angle_series += [s_ang_series]
            
            angle_series = time + joint_angle_series + segment_angle_series
            df_angles = []
            df_angles += [pd.DataFrame(angle_series, index=index_angs_csv).T]
            
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
            logging.info(f'Person {i}: Saving csv angle file in {csv_angle_path}.')
            df_angles[-1].to_csv(csv_angle_path, sep=',', index=True, lineterminator='\n')
            
            # Display figures
            if show_plots:
                logging.info(f'Person {i}: Displaying figures.')
                display_figures_fun(df_angles)

            df_angles_list += [df_angles[-1]]

    # Add angles to vid and img
    if show_angles_img or show_angles_vid:
        video_base = Path(video_dir / video_file)
        img_pose = result_dir / (video_base.stem + '_img')
        video_pose = result_dir / (video_base.stem + '.mp4')
        video_pose2 = result_dir / (video_base.stem + '2.mp4')
        
        if show_angles_vid:
            logging.info(f'Saving video in {str(video_pose)}.')
            cap = [cv2.VideoCapture(str(video_pose)) if Path.exists(video_pose) else cv2.VideoCapture(str(video_base))][0]
            fps = cap.get(cv2.CAP_PROP_FPS)
            W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_pose2), fourcc, fps, (int(W), int(H)))
        if show_angles_img:
            logging.info(f'Saving images in {img_pose}.')
            
        # Preferentially from pose image files
        frames_img = sorted(list(img_pose.glob('*')))
        if len(frames_img)>0:
            for frame_nb in range(df_angles_list[0].shape[0]):
                df_angles_list_frame = [df_angles_list[n].iloc[frame_nb,:] for n in range(len(df_angles_list))]
                frame = cv2.imread(str(frames_img[frame_nb]))
                frame = overlay_angles(frame, df_angles_list_frame)
                if show_angles_img:
                    cv2.imwrite(str(frames_img[frame_nb]), frame)
                if show_angles_vid:
                    writer.write(frame)
            if show_angles_vid:
                writer.release()
                
        # Else from pose video (or base video if pose video does not exist)
        elif Path.exists(video_base) or Path.exists(video_pose):
            frame_nb = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    df_angles_list_frame = [df_angles_list[n].iloc[frame_nb,:] for n in range(len(df_angles_list))]
                    frame = overlay_angles(frame, df_angles_list_frame)
                    if show_angles_img:
                        if frame_nb==0: img_pose.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(img_pose / (video_base.stem + '_' + '.' + str(frame_nb).zfill(5)+'.png')), frame)
                    if show_angles_vid:
                        writer.write(frame)
                    frame_nb+=1
                else:
                    break

        if show_angles_vid:
            cap.release()
            writer.release()
            if Path.exists(video_pose): os.remove(video_pose)
            os.rename(video_pose2,video_pose)
            if Path.exists(video_pose2): os.remove(video_pose2)
