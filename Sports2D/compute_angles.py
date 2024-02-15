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
            
    ang = np.array(np.degrees(np.arctan2(uy, ux) - np.arctan2(vy, vx)))
    
    return ang
            

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
            

def joint_angles_series_from_points(df_points, angle_params):
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
        keypt_series += [df_points.iloc[:,df_points.columns.get_level_values(2)==k].iloc[:,:2]]

    # Compute angles
    points_list = [k.values.T for k in keypt_series]
    ang_series = points2D_to_angles(points_list)
    ang_series += angle_params[2]
    ang_series *= angle_params[3]
    ang_series = np.where(ang_series>180,ang_series-360,ang_series)
    ang_series = np.where((ang_series==0) | (ang_series==90) | (ang_series==180), +0, ang_series)

    return ang_series


def segment_angles_series_from_points(df_points, angle_params, segment):
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
    
    # Compute angles
    ang_series = points2D_to_angles(points_list)
    ang_series += angle_params[2]
    ang_series *= angle_params[3]
    ang_series = np.where(ang_series>180,ang_series-360,ang_series)
    
    # For trunk: mean between angles RHip-RShoulder and LHip-LShoulder
    if segment == 'Trunk':
        ang_seriesR = ang_series
        angle_params[0] = [a.replace('R','L') for a in angle_params[0]]
        keypt_series = []
        for k in angle_params[0]:
            keypt_series += [df_points.iloc[:,df_points.columns.get_level_values(2)==k].iloc[:,:2]]
        # Compute angles
        points_list = [k.values.T for k in keypt_series]
        ang_series = points2D_to_angles(points_list)
        ang_series += angle_params[2]
        ang_series *= angle_params[3]
        ang_series = np.mean((ang_seriesR, ang_series), axis=0)
        ang_series = np.where(ang_series>180,ang_series-360,ang_series)
        
    return ang_series
    
    
def overlay_angles(frame, df_angles_list_frame):
    '''
    Overlays a text box for each detected person with joint and segment angles
    
    INPUT:
    - frame: a frame opened with OpenCV
    - df_angles_list_frame: list of one frame for all angles
    '''
    
    cmap = plt.cm.hsv
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, angles_frame_person in enumerate(df_angles_list_frame):
        for ang_nb in range(len(angles_frame_person)):
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
    csv_paths = list(result_dir.glob(f'*{video_file.stem}_{pose_model}_*points*.csv'))
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
        img_pose = result_dir / (video_base.stem + '_' + pose_model + '_img')
        video_pose = result_dir / (video_base.stem + '_' + pose_model + '.mp4')
        video_pose2 = result_dir / (video_base.stem + '_' + pose_model + '2.mp4')
        
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
                        cv2.imwrite(str(img_pose / (video_base.stem + '_' + pose_model + '.' + str(frame_nb).zfill(5)+'.png')), frame)
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
