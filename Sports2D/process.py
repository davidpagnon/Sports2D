#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##############################################################
    ## Compute pose and angles from video or webcam input       ##
    ##############################################################
    
    Detects 2D joint centers from a video or a webcam with RTMLib.
    Computes selected joint and segment angles. 
    Optionally saves processed image files and video file.
    Optionally saves processed poses as a TRC file, and angles as a MOT file (OpenSim compatible).

    This scripts:
    - loads skeleton information
    - reads stream from a video or a webcam
    - sets up the RTMLib pose tracker from RTMlib with specified parameters
    - detects poses within the selected time or frame range
    - tracks people so that their IDs are consistent across frames
    - retrieves the keypoints with high enough confidence, and only keeps the persons with enough high-confidence keypoints
    - computes joint and segment angles, and flips those on the left/right side them if the respective foot is pointing to the left
    - draws bounding boxes around each person with their IDs
    - draws joint and segment angles on the body, and writes the values either near the joint/segment, or on the upper-left of the image with a progress bar
    - draws the skeleton and the keypoints, with a green to red color scale to account for their confidence
    - optionally show processed images, saves them, or saves them as a video
    - interpolates missing pose and angle sequences if gaps are not too large
    - filters them with the selected filter and parameters
    - optionally plots pose and angle data before and after processing for comparison
    - optionally saves poses for each person as a trc file, and angles as a mot file
        
    /!\ Warning /!\
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal or frontal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
        
    INPUTS:
    - a video or a webcam
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one trc file of joint coordinates per detected person
    - one mot file of joint angles per detected person
    - image files, video
    - a logs.txt file 
'''    


## INIT
import os
import logging
from datetime import datetime
import itertools as it
from anytree import RenderTree, PreOrderIter

import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

from Sports2D.Utilities import filter
from Sports2D.Utilities.common import *
from Sports2D.Utilities.skeletons import *


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


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def draw_dotted_line(img, start, direction, length, color=(0, 255, 0), gap=7, dot_length=3, thickness=thickness):
    '''
    Draw a dotted line with on a cv2 image

    INPUTS:
    - img: opencv image
    - start: np.array. The starting point of the line
    - direction: np.array. The direction of the line
    - length: int. The length of the line
    - color: tuple. The color of the line
    - gap: int. The distance between each dot
    - dot_length: int. The length of each dot
    - thickness: int. The thickness of the line

    OUTPUT:
    - img: image with the dotted line
    '''

    for i in range(0, length, gap):
        line_start = start + direction * i
        line_end = line_start + direction * dot_length
        cv2.line(img, tuple(line_start.astype(int)), tuple(line_end.astype(int)), color, thickness)


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
            cv2.putText(img, str(i), (x_min-30, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, fontSize+1, color, 2, cv2.LINE_AA) 
    
    return img


def draw_skel(img, X, Y, model, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
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
    node_pairs = []
    for data_i in PreOrderIter(model.root, filter_=lambda node: node.is_leaf):
        node_branches = [node_i.id for node_i in data_i.path]
        node_pairs += [[node_branches[i],node_branches[i+1]] for i in range(len(node_branches)-1)]
    node_pairs = [list(x) for x in set(tuple(x) for x in node_pairs)]
    
    # Draw lines
    color_cycle = it.cycle(colors)
    for (x,y) in zip(X,Y):
        c = next(color_cycle)
        if not np.isnan(x).all():
            [cv2.line(img,
                (int(x[n[0]]), int(y[n[0]])), (int(x[n[1]]), int(y[n[1]])), c, thickness)
                for n in node_pairs
                if not (np.isnan(x[n[0]]) or np.isnan(y[n[0]]) or np.isnan(x[n[1]]) or np.isnan(y[n[1]]))]

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


def draw_angles(img, valid_X, valid_Y, valid_angles, valid_X_flipped, keypoints_ids, keypoints_names, angle_names, display_angle_values_on= ['body', 'list'], colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], fontSize=0.3, thickness=1):
    '''
    Draw angles on the image.
    Angles are displayed as a list on the image and/or on the body.

    INPUTS:
    - img: opencv image
    - valid_X: list of list of x coordinates
    - valid_Y: list of list of y coordinates
    - valid_angles: list of list of angles
    - valid_X_flipped: list of list of x coordinates after flipping if needed
    - keypoints_ids: list of keypoint ids (see skeletons.py)
    - keypoints_names: list of keypoint names (see skeletons.py)
    - angle_names: list of angle names
    - display_angle_values_on: list of str. 'body' and/or 'list'
    - colors: list of colors to cycle through

    OUTPUT:
    - img: image with angles
    '''

    color_cycle = it.cycle(colors)
    for person_id, (X,Y,angles, X_flipped) in enumerate(zip(valid_X, valid_Y, valid_angles, valid_X_flipped)):
        c = next(color_cycle)
        if not np.isnan(X).all():
            # person label
            if 'list' in display_angle_values_on:
                person_label_position = (int(10 + fontSize*150/0.3*person_id), int(fontSize*50))
                cv2.putText(img, f'person {person_id}', person_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (255,255,255), thickness+1, cv2.LINE_AA)
                cv2.putText(img, f'person {person_id}', person_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, c, thickness, cv2.LINE_AA)
            
            # angle lines, names and values
            ang_label_line = 1
            for k, ang in enumerate(angles):
                if not np.isnan(ang):
                    ang_name = angle_names[k]
                    ang_params = angle_dict.get(ang_name)
                    if ang_params is not None:
                        ang_coords = np.array([[X[keypoints_ids[keypoints_names.index(kpt)]], Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0] if kpt in keypoints_names])
                        X_flipped_coords = [X_flipped[keypoints_ids[keypoints_names.index(kpt)]] for kpt in ang_params[0] if kpt in keypoints_names]
                        flip = -1 if any(x_flipped < 0 for x_flipped in X_flipped_coords) else 1
                        flip = 1 if ang_name in ['pelvis', 'trunk', 'shoulders'] else flip
                        right_angle = True if ang_params[2]==90 else False
                        
                        # Draw angle
                        if len(ang_coords) == 2: # segment angle
                            app_point, vec = draw_segment_angle(img, ang_coords, flip)
                        else: # joint angle
                            app_point, vec1, vec2 = draw_joint_angle(img, ang_coords, flip, right_angle)
    
                        # Write angle on body
                        if 'body' in display_angle_values_on:
                            if len(ang_coords) == 2: # segment angle
                                write_angle_on_body(img, ang, app_point, vec, np.array([1,0]), dist=20, color=(255,255,255), fontSize=fontSize, thickness=thickness)
                            else: # joint angle
                                write_angle_on_body(img, ang, app_point, vec1, vec2, dist=40, color=(0,255,0), fontSize=fontSize, thickness=thickness)

                        # Write angle as a list on image with progress bar
                        if 'list' in display_angle_values_on:
                            if len(ang_coords) == 2: # segment angle
                                ang_label_line = write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color = (255,255,255), fontSize=fontSize, thickness=thickness)
                            else:
                                ang_label_line = write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color = (0,255,0), fontSize=fontSize, thickness=thickness)

    return img


def draw_segment_angle(img, ang_coords, flip):
    '''
    Draw a segment angle on the image.

    INPUTS:
    - img: opencv image
    - ang_coords: np.array. The 2D coordinates of the keypoints
    - flip: int. Whether the angle should be flipped

    OUTPUT:
    - app_point: np.array. The point where the angle is displayed
    - unit_segment_direction: np.array. The unit vector of the segment direction
    - img: image with the angle
    '''
    
    if not np.any(np.isnan(ang_coords)):
        app_point = np.int32(np.mean(ang_coords, axis=0))

        # segment line
        segment_direction = np.int32(ang_coords[0]) - np.int32(ang_coords[1])
        if (segment_direction==0).all():
            return app_point, np.array([0,0])
        unit_segment_direction = segment_direction/np.linalg.norm(segment_direction)
        cv2.line(img, app_point, np.int32(app_point+unit_segment_direction*20), (255,255,255), thickness)

        # horizontal line
        cv2.line(img, app_point, (np.int32(app_point[0])+flip*20, np.int32(app_point[1])), (255,255,255), thickness)

        return app_point, unit_segment_direction


def draw_joint_angle(img, ang_coords, flip, right_angle):
    '''
    Draw a joint angle on the image.

    INPUTS:
    - img: opencv image
    - ang_coords: np.array. The 2D coordinates of the keypoints
    - flip: int. Whether the angle should be flipped
    - right_angle: bool. Whether the angle should be offset by 90 degrees

    OUTPUT:
    - app_point: np.array. The point where the angle is displayed
    - unit_segment_direction: np.array. The unit vector of the segment direction
    - unit_parentsegment_direction: np.array. The unit vector of the parent segment direction
    - img: image with the angle
    '''
    
    if not np.any(np.isnan(ang_coords)):
        app_point = np.int32(ang_coords[1])
        
        segment_direction = np.int32(ang_coords[0] - ang_coords[1])
        parentsegment_direction = np.int32(ang_coords[-2] - ang_coords[-1])
        if (segment_direction==0).all() or (parentsegment_direction==0).all():
            return app_point, np.array([0,0]), np.array([0,0])
        
        if right_angle:
            segment_direction = np.array([-flip*segment_direction[1], flip*segment_direction[0]])
            segment_direction, parentsegment_direction = parentsegment_direction, segment_direction

        # segment line
        unit_segment_direction = segment_direction/np.linalg.norm(segment_direction)
        cv2.line(img, app_point, np.int32(app_point+unit_segment_direction*40), (0,255,0), thickness)
        
        # parent segment dotted line
        unit_parentsegment_direction = parentsegment_direction/np.linalg.norm(parentsegment_direction)
        draw_dotted_line(img, app_point, unit_parentsegment_direction, 40, color=(0, 255, 0), gap=7, dot_length=3, thickness=thickness)

        # arc
        start_angle = np.degrees(np.arctan2(unit_segment_direction[1], unit_segment_direction[0]))
        end_angle = np.degrees(np.arctan2(unit_parentsegment_direction[1], unit_parentsegment_direction[0]))
        if abs(end_angle - start_angle) > 180:
            if end_angle > start_angle: start_angle += 360
            else: end_angle += 360
        cv2.ellipse(img, app_point, (20, 20), 0, start_angle, end_angle, (0, 255, 0), thickness)

        return app_point, unit_segment_direction, unit_parentsegment_direction


def write_angle_on_body(img, ang, app_point, vec1, vec2, dist=40, color=(255,255,255), fontSize=0.3, thickness=1):
    '''
    Write the angle on the body.

    INPUTS:
    - img: opencv image
    - ang: float. The angle value to display
    - app_point: np.array. The point where the angle is displayed
    - vec1: np.array. The unit vector of the first segment
    - vec2: np.array. The unit vector of the second segment
    - dist: int. The distance from the origin where to write the angle
    - color: tuple. The color of the angle

    OUTPUT:
    - img: image with the angle
    '''

    vec_sum = vec1 + vec2
    if (vec_sum == 0.).all():
        return
    unit_vec_sum = vec_sum/np.linalg.norm(vec_sum)
    text_position = np.int32(app_point + unit_vec_sum*dist)
    cv2.putText(img, f'{ang:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, f'{ang:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)


def write_angle_as_list(img, ang, ang_name, person_label_position, ang_label_line, color=(255,255,255), fontSize=0.3, thickness=1):
    '''
    Write the angle as a list on the image with a progress bar.

    INPUTS:
    - img: opencv image
    - ang: float. The value of the angle to display
    - ang_name: str. The name of the angle
    - person_label_position: tuple. The position of the person label
    - ang_label_line: int. The line where to write the angle
    - color: tuple. The color of the angle

    OUTPUT:
    - ang_label_line: int. The updated line where to write the next angle
    - img: image with the angle
    '''
    
    if not np.any(np.isnan(ang)):
        # angle names and values
        ang_label_position = (person_label_position[0], person_label_position[1]+int((ang_label_line)*40*fontSize))
        ang_value_position = (ang_label_position[0]+int(250*fontSize), ang_label_position[1])
        cv2.putText(img, f'{ang_name}:', ang_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, f'{ang_name}:', ang_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)
        cv2.putText(img, f'{ang:.1f}', ang_value_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, f'{ang:.1f}', ang_value_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)
        
        # progress bar
        ang_percent = int(ang*50/180)
        y_crop, y_crop_end = ang_value_position[1] - int(35*fontSize), ang_value_position[1]
        x_crop, x_crop_end = ang_label_position[0]+int(300*fontSize), ang_label_position[0]+int(300*fontSize)+int(ang_percent*fontSize/0.3)
        if ang_percent < 0:
            x_crop, x_crop_end = x_crop_end, x_crop
        img_crop = img[y_crop:y_crop_end, x_crop:x_crop_end]
        if img_crop.size>0:
            white_rect = np.ones(img_crop.shape, dtype=np.uint8)*255
            alpha_rect = cv2.addWeighted(img_crop, 0.6, white_rect, 0.4, 1.0)
            img[y_crop:y_crop_end, x_crop:x_crop_end] = alpha_rect

        ang_label_line += 1
    
    return ang_label_line


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


def pose_plots(trc_data_unfiltered, trc_data, person_id):
    '''
    Displays trc filtered and unfiltered data for comparison
    /!\ Often crashes on the third window...

    INPUTS:
    - trc_data_unfiltered: pd.DataFrame. The unfiltered trc data
    - trc_data: pd.DataFrame. The filtered trc data

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''
    
    mpl.use('qt5agg')
    mpl.rc('figure', max_open_warning=0)

    keypoints_names = trc_data.columns[1::3]
    
    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person'+ str(person_id) + ' coordinates') # Main title

    for id, keypoint in enumerate(keypoints_names):
        f = plt.figure()
        f.canvas.manager.window.setWindowTitle(keypoint + ' Plot')

        axX = plt.subplot(211)
        plt.plot(trc_data_unfiltered.iloc[:,0], trc_data_unfiltered.iloc[:,id*3+1], label='unfiltered')
        plt.plot(trc_data.iloc[:,0], trc_data.iloc[:,id*3+1], label='filtered')
        plt.setp(axX.get_xticklabels(), visible=False)
        axX.set_ylabel(keypoint+' X')
        plt.legend()

        axY = plt.subplot(212)
        plt.plot(trc_data_unfiltered.iloc[:,0], trc_data_unfiltered.iloc[:,id*3+2], label='unfiltered')
        plt.plot(trc_data.iloc[:,0], trc_data.iloc[:,id*3+2], label='filtered')
        axY.set_xlabel('Time (seconds)')
        axY.set_ylabel(keypoint+' Y')

        pw.addPlot(keypoint, f)
    
    pw.show()


def angle_plots(angle_data_unfiltered, angle_data, person_id):
    '''
    Displays angle filtered and unfiltered data for comparison
    /!\ Often crashes on the third window...

    INPUTS:
    - angle_data_unfiltered: pd.DataFrame. The unfiltered angle data
    - angle_data: pd.DataFrame. The filtered angle data

    OUTPUT:
    - matplotlib window with tabbed figures for each angle
    '''

    mpl.use('qt5agg')
    mpl.rc('figure', max_open_warning=0)

    angles_names = angle_data.columns[1:]

    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person'+ str(person_id) + ' angles') # Main title

    for id, angle in enumerate(angles_names):
        f = plt.figure()
        
        ax = plt.subplot(111)
        plt.plot(angle_data_unfiltered.iloc[:,0], angle_data_unfiltered.iloc[:,id+1], label='unfiltered')
        plt.plot(angle_data.iloc[:,0], angle_data.iloc[:,id+1], label='filtered')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(angle+' (°)')
        plt.legend()

        pw.addPlot(angle, f)

    pw.show()


def process_fun(config_dict, video_file_path, pose_tracker, input_frame_range, output_dir):
    '''
    Detect 2D joint centers from a video or a webcam with RTMLib.
    Compute selected joint and segment angles. 
    Optionally save processed image files and video file.
    Optionally save processed poses as a TRC file, and angles as a MOT file (OpenSim compatible).

    This scripts:
    - loads skeleton information
    - reads stream from a video or a webcam
    - sets up the RTMLib pose tracker from RTMlib with specified parameters
    - detects poses within the selected frame range
    - tracks people so that their IDs are consistent across frames
    - retrieves the keypoints with high enough confidence, and only keeps the persons with enough high-confidence keypoints
    - computes joint and segment angles, and flips those on the left/right side them if the respective foot is pointing to the left
    - draws bounding boxes around each person with their IDs
    - draws joint and segment angles on the body, and writes the values either near the joint/segment, or on the upper-left of the image with a progress bar
    - draws the skeleton and the keypoints, with a green to red color scale to account for their confidence
    - optionally show processed images, saves them, or saves them as a video
    - interpolates missing pose and angle sequences if gaps are not too large
    - filters them with the selected filter and parameters
    - optionally plots pose and angle data before and after processing for comparison
    - optionally saves poses for each person as a trc file, and angles as a mot file
        
    /!\ Warning /!\d
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal or frontal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
        
    INPUTS:
    - a video or a webcam
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one trc file of joint coordinates per detected person
    - one mot file of joint angles per detected person
    - image files, video
    - a logs.txt file 
    '''

    # Base parameters
    webcam_id =  config_dict.get('project').get('webcam_id')
    input_size = config_dict.get('project').get('input_size')
    
    save_video = True if 'to_video' in config_dict['project']['save_video'] else False
    save_images = True if 'to_images' in config_dict['project']['save_video'] else False

    # Process settings
    multi_person = config_dict.get('process', {}).get('multi_person')

    if multi_person is None:
        multi_person = config_dict.get('process', {}).get('multiperson')
        print("Warning: 'multiperson' is deprecated. Please switch to 'multi_person'.")
    show_realtime_results = config_dict.get('process').get('show_realtime_results')
    save_pose = config_dict.get('process').get('save_pose')
    save_angles = config_dict.get('process').get('save_angles')

    # Pose_advanced settings
    pose_model = config_dict.get('pose').get('pose_model')
    tracking_mode = config_dict.get('pose').get('tracking_mode')

    keypoint_likelihood_threshold = config_dict.get('pose').get('keypoint_likelihood_threshold')
    average_likelihood_threshold = config_dict.get('pose').get('average_likelihood_threshold')
    keypoint_number_threshold = config_dict.get('pose').get('keypoint_number_threshold')

    # Angles advanced settings
    joint_angle_names = config_dict.get('angles').get('joint_angles')
    segment_angle_names = config_dict.get('angles').get('segment_angles')
    angle_names = joint_angle_names + segment_angle_names
    angle_names = [angle_name.lower() for angle_name in angle_names]
    display_angle_values_on = config_dict.get('angles').get('display_angle_values_on')
    fontSize = config_dict.get('angles').get('fontSize')
    thickness = 1 if fontSize < 0.8 else 2
    flip_left_right = config_dict.get('angles').get('flip_left_right')

    # Retrieve keypoint names from model
    model = eval(pose_model)
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]

    Ltoe_idx = keypoints_ids[keypoints_names.index('LBigToe')]
    LHeel_idx = keypoints_ids[keypoints_names.index('LHeel')]
    Rtoe_idx = keypoints_ids[keypoints_names.index('RBigToe')]
    RHeel_idx = keypoints_ids[keypoints_names.index('RHeel')]
    L_R_direction_idx = [Ltoe_idx, LHeel_idx, Rtoe_idx, RHeel_idx]


    logging.info(f'Multi-person is {"" if multi_person else "not "}selected.')
    logging.info(f"Parameters: {f'{tracking_mode=}, ' if multi_person else ''}{keypoint_likelihood_threshold=}, {average_likelihood_threshold=}, {keypoint_number_threshold=}")

    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = setup_capture_directories(video_file_path, output_dir)

    try:
        cap = cv2.VideoCapture(video_file_path)
        cap.read()
        if cap.read()[0] == False:
            raise
    except:
        raise NameError(f"{video_file_path} is not a video. Images must be put in one subdirectory per camera.")

    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = setup_capture_directories(video_file_path, output_dir)

    validate_video_file(video_file_path)

    # Set up video capture
    cap, frame_iterator, out_vid, cam_width, cam_height = setup_video_capture(video_file_path, webcam_id, save_video, output_video_path, input_size, input_frame_range)

    # Call to display real-time results if needed
    if show_realtime_results:
        display_realtime_results(video_file_path)
    # Process video or webcam feed
    # logging.info(f"{'Video, ' if save_video else ''}{'Images, ' if save_images else ''}{'Pose, ' if save_pose else ''}{'Angles ' if save_angles else ''}{'and ' if save_angles or save_images or save_pose or save_video else ''}Logs will be saved in {result_dir}.")
    all_frames_X, all_frames_Y, all_frames_scores, all_frames_angles = [], [], [], []

    if video_file_path == 'webcam' and save_video:
        total_processing_start_time = datetime.now()

    frames_processed = 0

    for frame_idx in frame_iterator:
        # If frame not grabbed
        frame = read_frame(cap, frame_idx)
        if frame is None:
            if save_pose:
                all_frames_X.append([])
                all_frames_Y.append([])
                all_frames_scores.append([])
            if save_angles:
                all_frames_angles.append([])
            continue

        display_quit_message(frame, cam_width, cam_height, fontSize, thickness)

         # Perform pose estimation on the frame
        keypoints, scores = pose_tracker(frame)

        # Tracking people IDs across frames
        keypoints, scores, prev_keypoints = track_people(
            keypoints, scores, multi_person, tracking_mode, prev_keypoints, pose_tracker
        )

        # Process coordinates and compute angles
        valid_X, valid_Y, valid_scores, valid_X_flipped, valid_angles = process_coordinates_and_angles(
            keypoints, scores, keypoint_likelihood_threshold, keypoint_number_threshold,
            average_likelihood_threshold, flip_left_right, L_R_direction_idx,
            keypoints_names, keypoints_ids, angle_names, angle_dict
        )

        if save_pose:
            all_frames_X.append(np.array(valid_X))
            all_frames_Y.append(np.array(valid_Y))
            all_frames_scores.append(np.array(valid_scores))
        if save_angles:
            all_frames_angles.append(np.array(valid_angles))

        # Draw keypoints and skeleton
        if show_realtime_results or save_video or save_images:
            img_show = frame.copy()
            img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors=colors, fontSize=fontSize, thickness=thickness)
            img_show = draw_keypts(img_show, valid_X, valid_Y, scores, cmap_str='RdYlGn')
            img_show = draw_skel(img_show, valid_X, valid_Y, model, colors=colors)
            img_show = draw_angles(img_show, valid_X, valid_Y, valid_angles, valid_X_flipped, keypoints_ids, keypoints_names, angle_names, display_angle_values_on=display_angle_values_on, colors=colors, fontSize=fontSize, thickness=thickness)

        if show_realtime_results:
            cv2.imshow(f"Pose Estimation {os.path.basename(video_file_path)}", img_show)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        if save_video:
            out_vid.write(img_show)
        if save_images:
            os.makedirs(img_output_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(img_output_dir, f'{output_dir_name}_{frame_idx:06d}.jpg'),
                img_show
            )

        frames_processed += 1

    cap.release()

    logging.info(f"Video processing completed.")
    
    if save_video:
        out_vid.release()
        if video_file_path == 'webcam' and frames_processed > 0:
            fps = finalize_video_processing(frames_processed, total_processing_start_time, output_video_path, fps)
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if show_realtime_results:
        cv2.destroyAllWindows()

    # def post_processing(config_dict, video_file, frame_rate):
    save_pose = config_dict.get('process').get('save_pose')
    save_angles = config_dict.get('process').get('save_angles')

    # Post-processing settings
    interpolate = config_dict.get('post-processing').get('interpolate')    
    interp_gap_smaller_than = config_dict.get('post-processing').get('interp_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('post-processing').get('fill_large_gaps_with')

    show_plots = config_dict.get('post-processing').get('show_graphs')
    filter_type = config_dict.get('post-processing').get('filter_type')
    do_filter = config_dict.get('post-processing').get('filter')
    butterworth_filter_order = config_dict.get('post-processing').get('butterworth').get('order')
    butterworth_filter_cutoff = config_dict.get('post-processing').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel = config_dict.get('post-processing').get('gaussian').get('sigma_kernel')
    loess_filter_kernel = config_dict.get('post-processing').get('loess').get('nb_values_used')
    median_filter_kernel = config_dict.get('post-processing').get('median').get('kernel_size')

    filter_options = [do_filter, filter_type,
                           butterworth_filter_order, butterworth_filter_cutoff, fps,
                           gaussian_filter_kernel, loess_filter_kernel, median_filter_kernel]

    # Post-processing: Interpolate, filter, and save pose and angles
    frame_range = [0,frame_idx] if video_file_path == 'webcam' else frame_range
    all_frames_time = pd.Series(np.linspace(frame_range[0]/fps, frame_range[1]/fps, frame_idx), name='time')

    if save_pose:
        logging.info('\nPost-processing pose:')
        # Select only the keypoints that are in the model from skeletons.py, invert Y axis, divide pixel values by 1000
        all_frames_X = make_homogeneous(all_frames_X)
        all_frames_X = all_frames_X[...,keypoints_ids] / 1000
        all_frames_Y = make_homogeneous(all_frames_Y)
        all_frames_Y = -all_frames_Y[...,keypoints_ids] / 1000
        all_frames_Z_person = pd.DataFrame(np.zeros_like(all_frames_X)[:,0,:], columns=keypoints_names)
        
        # Process pose for each person
        for i in range(all_frames_X.shape[1]):
            pose_path_person = os.path.join(output_dir, (output_dir_name + '_px' + f'_person{i:02d}.trc'))
            all_frames_X_person = pd.DataFrame(all_frames_X[:,i,:], columns=keypoints_names)
            all_frames_Y_person = pd.DataFrame(all_frames_Y[:,i,:], columns=keypoints_names)

            # Delete person if less than 4 valid frames
            pose_nan_count = len(np.where(all_frames_X_person.sum(axis=1)==0)[0])
            if frame_idx - pose_nan_count <= 4:
                logging.info(f'- Person {i}: Less than 4 valid frames. Deleting person.')

            else:
                # Interpolate
                if not interpolate:
                    logging.info(f'- Person {i}: No interpolation.')
                    all_frames_X_person_interp = all_frames_X_person
                    all_frames_Y_person_interp = all_frames_Y_person
                else:
                    logging.info(f'- Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames. Large gaps filled with {fill_large_gaps_with}.')
                    all_frames_X_person_interp = all_frames_X_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])
                    all_frames_Y_person_interp = all_frames_Y_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])
                    if fill_large_gaps_with == 'last_value':
                        all_frames_X_person_interp = all_frames_X_person_interp.ffill(axis=0).bfill(axis=0)
                        all_frames_Y_person_interp = all_frames_Y_person_interp.ffill(axis=0).bfill(axis=0)
                    elif fill_large_gaps_with == 'zeros':
                        all_frames_X_person_interp.replace(np.nan, 0, inplace=True)
                        all_frames_Y_person_interp.replace(np.nan, 0, inplace=True)

                # Filter
                if not filter_options[0]:
                    logging.info(f'No filtering.')
                    all_frames_X_person_filt = all_frames_X_person_interp
                    all_frames_Y_person_filt = all_frames_Y_person_interp
                else:
                    filter_type = filter_options[1]
                    if filter_type == 'butterworth':
                        if video_file_path == 'webcam':
                            cutoff = filter_options[3]
                            if cutoff / (fps / 2) >= 1:
                                cutoff_old = cutoff
                                cutoff = fps/(2+0.001)
                                args = f'\n{cutoff_old:.1f} Hz cut-off framerate too large for a real-time framerate of {fps:.1f} Hz. Using a cut-off framerate of {cutoff:.1f} Hz instead.'
                                filter_options[3] = cutoff
                        else: 
                            args = ''
                        args = f'Butterworth filter, {filter_options[2]}th order, {filter_options[3]} Hz.'
                        filter_options[4] = fps
                    if filter_type == 'gaussian':
                        args = f'Gaussian filter, Sigma kernel {filter_options[5]}.'
                    if filter_type == 'loess':
                        args = f'LOESS filter, window size of {filter_options[6]} frames.'
                    if filter_type == 'median':
                        args = f'Median filter, kernel of {filter_options[7]}.'
                    logging.info(f'Filtering with {args}')
                    all_frames_X_person_filt = all_frames_X_person_interp.apply(filter.filter1d, axis=0, args=filter_options)
                    all_frames_Y_person_filt = all_frames_Y_person_interp.apply(filter.filter1d, axis=0, args=filter_options)

                # Build TRC file
                trc_data = make_trc_with_XYZ(all_frames_X_person_filt, all_frames_Y_person_filt, all_frames_Z_person, all_frames_time, str(pose_path_person))
                logging.info(f'Pose saved to {pose_path_person}.')

                # Plotting coordinates before and after interpolation and filtering
                if show_plots:
                    trc_data_unfiltered = pd.concat([pd.concat([all_frames_X_person.iloc[:,kpt], all_frames_Y_person.iloc[:,kpt], all_frames_Z_person.iloc[:,kpt]], axis=1) for kpt in range(len(all_frames_X_person.columns))], axis=1)
                    trc_data_unfiltered.insert(0, 't', all_frames_time)
                    pose_plots(trc_data_unfiltered, trc_data, i) # i = current person


    # Angles post-processing
    if save_angles:
        logging.info('\nPost-processing angles:')
        all_frames_angles = make_homogeneous(all_frames_angles)

        # Process angles for each person
        for i in range(all_frames_angles.shape[1]):
            angles_path_person = os.path.join(output_dir, (output_dir_name + '_angles' + f'_person{i:02d}.mot'))
            all_frames_angles_person = pd.DataFrame(all_frames_angles[:,i,:], columns=angle_names)
            
            # Delete person if less than 4 valid frames
            angle_nan_count = len(np.where(all_frames_angles_person.sum(axis=1)==0)[0])
            if frame_idx - angle_nan_count <= 4:
                logging.info(f'- Person {i}: Less than 4 valid frames. Deleting person.')

            else:
                # Interpolate
                if not interpolate:
                    logging.info(f'- Person {i}: No interpolation.')
                    all_frames_angles_person_interp = all_frames_angles_person
                else:
                    logging.info(f'- Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames. Large gaps filled with {fill_large_gaps_with}.')
                    all_frames_angles_person_interp = all_frames_angles_person.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])
                    if fill_large_gaps_with == 'last_value':
                        all_frames_angles_person_interp = all_frames_angles_person_interp.ffill(axis=0).bfill(axis=0)
                    elif fill_large_gaps_with == 'zeros':
                        all_frames_angles_person_interp.replace(np.nan, 0, inplace=True)
                
                # Filter
                if not filter_options[0]:
                    logging.info(f'No filtering.')
                    all_frames_angles_person_filt = all_frames_angles_person_interp
                else:
                    filter_type = filter_options[1]
                    if filter_type == 'butterworth':
                        if video_file_path == 'webcam':
                            cutoff = filter_options[3]
                            if cutoff / (fps / 2) >= 1:
                                cutoff_old = cutoff
                                cutoff = fps/(2+0.001)
                                args = f'\n{cutoff_old:.1f} Hz cut-off framerate too large for a real-time framerate of {fps:.1f} Hz. Using a cut-off framerate of {cutoff:.1f} Hz instead.'
                                filter_options[3] = cutoff
                        else: 
                            args = ''
                        args = f'Butterworth filter, {filter_options[2]}th order, {filter_options[3]} Hz. ' + args
                        filter_options[4] = fps
                    if filter_type == 'gaussian':
                        args = f'Gaussian filter, Sigma kernel {filter_options[5]}.'
                    if filter_type == 'loess':
                        args = f'LOESS filter, window size of {filter_options[6]} frames.'
                    if filter_type == 'median':
                        args = f'Median filter, kernel of {filter_options[7]}.'
                    logging.info(f'Filtering with {args}')
                    all_frames_angles_person_filt = all_frames_angles_person_interp.apply(filter.filter1d, axis=0, args=filter_options)

                # Build mot file
                angle_data = make_mot_with_angles(all_frames_angles_person_filt, all_frames_time, str(angles_path_person))
                logging.info(f'Angles saved to {angles_path_person}.')

                # Plotting angles before and after interpolation and filtering
                if show_plots:
                    all_frames_angles_person.insert(0, 't', all_frames_time)
                    angle_plots(all_frames_angles_person, angle_data, i) # i = current person
