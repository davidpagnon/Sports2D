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
from anytree import RenderTree

import numpy as np
import cv2

from Sports2D.Utilities.skeletons import *
from Sports2D.Utilities.utilities import read_frame
from Sports2D.Utilities.config import setup_capture_directories, setup_video_capture
from Sports2D.Utilities.video_management import draw_bounding_box, draw_keypts, draw_skel, draw_angles, display_realtime_results, track_people, finalize_video_processing
from Sports2D.Utilities.data_processing import process_coordinates_and_angles
from Sports2D.Utilities.visualisation import display_quit_message


## CONSTANTS
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
    input_size = config_dict.get('project').get('input_size')
    video_input = config_dict['project'].get('video_input')
    
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

    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = setup_capture_directories(video_file_path, output_dir, save_images)

    # Set up video capture
    cap, frame_iterator, out_vid, cam_width, cam_height, fps = setup_video_capture(video_file_path, save_video, output_video_path, input_size, input_frame_range)

    # Call to display real-time results if needed
    if show_realtime_results:
        display_realtime_results(video_file_path)
    # Process video or webcam feed
    # logging.info(f"{'Video, ' if save_video else ''}{'Images, ' if save_images else ''}{'Pose, ' if save_pose else ''}{'Angles ' if save_angles else ''}{'and ' if save_angles or save_images or save_pose or save_video else ''}Logs will be saved in {result_dir}.")
    all_frames_X, all_frames_Y, all_frames_angles = [], [], []

    if video_input == "webcam" and save_video:
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
            keypoints_names, keypoints_ids, angle_names)

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
            cv2.imwrite(
                os.path.join(img_output_dir, f'{output_dir_name}_{frame_idx:06d}.jpg'),
                img_show
            )

        frames_processed += 1

    cap.release()

    logging.info(f"Video processing completed.")
    
    if save_video:
        out_vid.release()
        if video_input == "webcam"  and frames_processed > 0:
            fps = finalize_video_processing(frames_processed, total_processing_start_time, output_video_path, fps)
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if show_realtime_results:
        cv2.destroyAllWindows()

    return frame_idx, fps, output_dir_name, all_frames_X, all_frames_Y, all_frames_angles