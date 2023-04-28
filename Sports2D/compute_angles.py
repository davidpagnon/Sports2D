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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Sports2D.Sports2D import base_params
from Sports2D.Utilities import filter, common
from Sports2D.Utilities.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.1"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"



# CONSTANTS
# dict: name: points, extra, offset, invert. 
# Most angles are multiplied by -1 because the OpenCV y axis points down.
joint_angle_dict = { 
    'rankle': [['RHeel', 'RBigToe', 'RAnkle', 'RKnee'], 'dorsiflexion', -90, -1],
    'rknee': [['RHip', 'RKnee', 'RAnkle'], 'flexion', -180, -1],
    'rhip': [['RKnee', 'RHip', 'RShoulder'], 'flexion', -180, -1],
    'rshoulder': [['RHip', 'RShoulder', 'RElbow'], 'flexion', 0, 1],
    'relbow': [['RWrist', 'RElbow', 'RShoulder'], 'flexion', -180, -1],
    'lankle': [['LHeel', 'LBigToe', 'LAnkle', 'LKnee'], 'dorsiflexion', -90, -1],
    'lknee': [['LHip', 'LKnee', 'LAnkle'], 'flexion', -180, -1],
    'lhip': [['LKnee', 'LHip', 'LShoulder'], 'flexion', -180, -1],
    'lshoulder': [['LHip', 'LShoulder', 'LElbow'], 'flexion', 0, 1],
    'lelbow': [['LWrist', 'LElbow', 'LShoulder'], 'flexion', -180, -1]
    }

segment_angle_dict = {     
    'rfoot': [['RHeel', 'RBigToe'], 'horizontal', 0, -1],
    'lfoot': [['LHeel', 'LBigToe'], 'horizontal', 0, -1],
    'rshank': [['RAnkle', 'RKnee'], 'horizontal', 0, -1],
    'lshank': [['LAnkle', 'LKnee'], 'horizontal', 0, -1],
    'rthigh': [['RHip', 'RKnee'], 'horizontal', 0, -1],
    'lthigh': [['LHip', 'LKnee'], 'horizontal', 0, -1],
    'trunk': [['RHip', 'RShoulder'], 'horizontal', 0, 1],
    'rarm': [['RShoulder', 'RElbow'], 'horizontal', 0, -1],
    'larm': [['LShoulder', 'LElbow'], 'horizontal', 0, -1],
    'rforearm': [['RElbow', 'RWrist'], 'horizontal', 0, -1],
    'lforearm': [['LElbow', 'LWrist'], 'horizontal', 0, -1]
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
    
    angle_names = df_list[0].columns.get_level_values(2)
    
    pw = common.plotWindow()
    for id, angle in enumerate(angle_names): # angles
        f = plt.figure()
        
        plt.plot()
        [plt.plot(df_list[0].index, df.iloc[:,id], label=['unfiltered' if i==0 else 'filtered' if i==1 else ''][0]) for i,df in enumerate(df_list)]
        plt.ylabel(angle) # nom angle
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
    ang = np.where(ang>180,ang-360,ang)

    return ang
            

def compute_angles_fun(config_dict):
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
    video_dir, video_file, frame_rate = base_params(config_dict)
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
    
    # Find csv position files in video_dir, search pose_model and video_file.stem
    logging.info(f'Retrieving csv position files in {video_dir}...')
    csv_paths = list(video_dir.glob(f'*{video_file.stem}_{pose_model}_*points*.csv'))
    logging.info(f'{len(csv_paths)} persons found.')

    # Compute angles
    for i, c in enumerate(csv_paths):
        # Prepare angle csv header
        scorer = ['DavidPagnon']*angle_nb
        individuals = [f'person{i}']*angle_nb
        angs = joint_angles + segment_angles
        coords = [joint_angle_dict.get(j)[1] for j in joint_angles] + [segment_angle_dict.get(s)[1] for s in segment_angles]
        tuples = list(zip(scorer, individuals, angs, coords))
        index_angs_csv = pd.MultiIndex.from_tuples(tuples, names=['scorer', 'individuals', 'angs', 'coords'])    

        # Compute angles for each person, for each angle, with each required keypoint position
        logging.info(f'Person {i}: Computing 2D joint and segment angles.')
        with open(c) as c_f:
            df_points = pd.read_csv(c_f, header=[0,1,2,3])
            
            # Flip along x when feet oriented to the left
            righ_orientation = df_points.iloc[:,df_points.columns.get_level_values(2)=='RBigToe'].iloc[:,0] - df_points.iloc[:,df_points.columns.get_level_values(2)=='RHeel'].iloc[:,0]
            left_orientation = df_points.iloc[:,df_points.columns.get_level_values(2)=='LBigToe'].iloc[:,0] - df_points.iloc[:,df_points.columns.get_level_values(2)=='LHeel'].iloc[:,0]
            orientation = righ_orientation + left_orientation
            df_points.iloc[:,1::3] = df_points.iloc[:,1::3] * np.where(orientation>=0, 1, -1).reshape(-1,1)
            
            # joint angles
            joint_angle_series = []
            for j in joint_angles: 
                angle_params = joint_angle_dict.get(j)
                keypt_series = []
                for k in angle_params[0]:
                    keypt_series += [df_points.iloc[:,df_points.columns.get_level_values(2)==k].iloc[:,:2]]
                # Compute angles
                points_list = [k.values.T for k in keypt_series]
                ang_series = points2D_to_angles(points_list)
                ang_series += angle_params[2]
                ang_series *= angle_params[3]
                ang_series = np.where(ang_series>180,ang_series-360,ang_series)
                joint_angle_series += [ang_series]
    
            # segment angles
            segment_angle_series = []
            for s in segment_angles:
                angle_params = segment_angle_dict.get(s)
                keypt_series = []
                for k in angle_params[0]:
                    keypt_series += [df_points.iloc[:,df_points.columns.get_level_values(2)==k].iloc[:,:2]]
                # Compute angles
                points_list = [k.values.T for k in keypt_series]
                ang_series = points2D_to_angles(points_list)
                ang_series += angle_params[2]
                ang_series *= angle_params[3]
                ang_series = np.where(ang_series>180,ang_series-360,ang_series)
                # for trunk: mean between angles RHip-RShoulder and LHip-LShoulder
                if s == 'trunk':
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
                segment_angle_series += [ang_series]
                
            angle_series = joint_angle_series + segment_angle_series
            df_list = []
            df_list += [ pd.DataFrame(angle_series, index=index_angs_csv).T]
            
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
                df_list[0].replace(0, np.nan, inplace=True)
                df_list += [df_list[0].copy()]
                df_list[1] = df_list[1].apply(filter.filter1d, axis=0, args=filter_options)
            df_list[-1].replace(np.nan, 0, inplace=True)
               
            # Creation of the csv files
            csv_angle_path = c.parent / (c.stem.replace('points', 'angles') + '.csv')
            logging.info(f'Person {i}: Saving csv angle file in {csv_angle_path}.')
            df_list[-1].to_csv(csv_angle_path, sep=',', index=True, lineterminator='\n')
    
            # Display figures
            if show_plots:
                logging.info(f'Person {i}: Displaying figures.')
                display_figures_fun(df_list)

    logging.info(f'Done.')