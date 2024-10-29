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

from Sports2D.Utilities.video_management import draw_bounding_box, draw_keypts, draw_skel, draw_angles
from Sports2D.Utilities.data_processing import make_trc_with_XYZ, make_mot_with_angles
from Sports2D.Utilities.visualisation import pose_plots, angle_plots


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

    # Set up video capture
    cap, frame_iterator, out_vid, cam_width, cam_height, fps = setup_video_capture(video_file_path, webcam_id, save_video, output_video_path, input_size, input_frame_range)

    # Call to display real-time results if needed
    if show_realtime_results:
        display_realtime_results(video_file_path)
    # Process video or webcam feed
    # logging.info(f"{'Video, ' if save_video else ''}{'Images, ' if save_images else ''}{'Pose, ' if save_pose else ''}{'Angles ' if save_angles else ''}{'and ' if save_angles or save_images or save_pose or save_video else ''}Logs will be saved in {result_dir}.")
    all_frames_X, all_frames_Y, all_frames_angles = [], [], []

    if video_file_path == 'webcam' and save_video:
        total_processing_start_time = datetime.now()

    frames_processed = 0
    prev_keypoints = None
    for frame_idx in frame_iterator:
        frame = read_frame(cap, frame_idx)

        # If frame not grabbed
        if frame is None:
            logging.warning(f"Failed to grab frame {frame_idx}.")
            if save_pose:
                all_frames_X.append([])
                all_frames_Y.append([])
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
        valid_X, valid_Y, valid_X_flipped, valid_angles = process_coordinates_and_angles(
            keypoints, scores, keypoint_likelihood_threshold, keypoint_number_threshold,
            average_likelihood_threshold, flip_left_right, L_R_direction_idx,
            keypoints_names, keypoints_ids, angle_names, angle_dict
        )

        if save_pose:
            all_frames_X.append(np.array(valid_X))
            all_frames_Y.append(np.array(valid_Y))
        if save_angles:
            all_frames_angles.append(np.array(valid_angles))

        # Draw keypoints and skeleton
        if show_realtime_results or save_video or save_images:
            img_show = frame.copy()
            img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors, fontSize, thickness)
            img_show = draw_keypts(img_show, valid_X, valid_Y, scores, thickness, cmap_str='RdYlGn')
            img_show = draw_skel(img_show, valid_X, valid_Y, model, colors, thickness)
            img_show = draw_angles(img_show, valid_X, valid_Y, valid_angles, valid_X_flipped, keypoints_ids, keypoints_names, angle_names, display_angle_values_on, colors, fontSize, thickness)

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

    return frame_idx, fps, output_dir_name, all_frames_X, all_frames_Y, all_frames_angles

def post_processing(config_dict, video_file_path, frame_idx, fps, frame_range, output_dir, output_dir_name, all_frames_X, all_frames_Y, all_frames_angles):
    save_pose = config_dict.get('process').get('save_pose')
    save_angles = config_dict.get('process').get('save_angles')

    # Post-processing settings
    interpolate = config_dict.get('post-processing').get('interpolate')    
    interp_gap_smaller_than = config_dict.get('post-processing').get('interp_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('post-processing').get('fill_large_gaps_with')

    # Retrieve keypoint names from model
    pose_model = config_dict.get('pose').get('pose_model')
    model = eval(pose_model)
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]

    joint_angle_names = config_dict.get('angles').get('joint_angles')
    segment_angle_names = config_dict.get('angles').get('segment_angles')
    angle_names = joint_angle_names + segment_angle_names
    angle_names = [angle_name.lower() for angle_name in angle_names]

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
