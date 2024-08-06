#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##############################################################
    ## Compute angles from 2D pose detection                    ##
    ##############################################################
    
    Detect joint centers from a video with OpenPose or BlazePose.
    Save a 2D csv position file per person, and optionally json files, image files, and video files.
    
    If OpenPose is used, multiple persons can be consistently detected across frames.
    Interpolates sequences of missing data if they are less than N frames long.
    Optionally filters results with Butterworth, gaussian, median, or loess filter.
    Optionally displays figures.

    If BlazePose is used, only one person can be detected.
    No interpolation nor filtering options available. Not plotting available.

    /!\ Warning /!\
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
   
    INPUTS:
    - a video
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one csv file of joint coordinates per detected person
    - optionally json directory, image directory, video
    - a logs.txt file 

'''    


## INIT
import os
import logging
import time as time_module # Avoid conflict
from pathlib import Path
from sys import platform
import json
import itertools as it
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import onnxruntime as ort
from datetime import datetime
import sys
from IPython.display import clear_output

from Sports2D.Sports2D import base_params
from Sports2D.compute_angles import (
    get_joint_angle_params,
    get_segment_angle_params, 
    joint_angles_series_from_points, 
    segment_angles_series_from_points,
    joint_angles_series_from_csv,
    segment_angles_series_from_csv, 
    draw_bounding_box,
    overlay_angles,
    flip_left_right_direction,
    display_figures_fun_ang,
    draw_keypts_skel)
from Sports2D.Utilities import filter, common
from Sports2D.Utilities.skeletons import halpe26_rtm
from rtmlib import PoseTracker, BodyWithFeet, draw_skeleton


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.3.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"



# FUNCTIONS
def display_figures_fun_cords(df_list):
    '''
    Displays filtered and unfiltered data for comparison
    /!\ Crashes on the third window...

    INPUTS:
    - df_list: list of dataframes of 3N columns, only 3i and 3i+1 are displayed

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''
    
    mpl.use('qt5agg')
    mpl.rc('figure', max_open_warning=0)

    keypoints_names = df_list[0].columns.get_level_values(2)[1::3]
    
    pw = common.plotWindow()
    for id, keypoint in enumerate(keypoints_names):
        f = plt.figure()
        
        axX = plt.subplot(211)
        [plt.plot(df.iloc[:,0], df.iloc[:,id*3+1], label=['unfiltered' if i==0 else 'filtered' if i==1 else ''][0]) for i,df in enumerate(df_list)]
        plt.setp(axX.get_xticklabels(), visible=False)
        axX.set_ylabel(keypoint+' X')
        plt.legend()

        axY = plt.subplot(212)
        [plt.plot(df.iloc[:,0], df.iloc[:,id*3+2]) for df in df_list]
        axY.set_xlabel('Time (seconds)')
        axY.set_ylabel(keypoint+' Y')

        pw.addPlot(keypoint, f)
    
    pw.show()
    
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
    - argminL: list of indices of smallest values of L
    - T_minL: list of tuples associated with smallest values of L
    '''

    try:
        minL = [np.min(L)]
    except:
        return [], [], []
    argminL = [np.argmin(L)]
    T_minL = [T[argminL[0]]]
    
    mask_tokeep = np.array([True for t in T])
    i=0
    while mask_tokeep.any()==True:
        mask_tokeep = mask_tokeep & np.array([t[0]!=T_minL[i][0] and t[1]!=T_minL[i][1] for t in T])
        if mask_tokeep.any()==True:
            indicesL_tokeep = np.where(mask_tokeep)[0]
            minL += [np.min(np.array(L)[indicesL_tokeep])]
            argminL += [indicesL_tokeep[np.argmin(np.array(L)[indicesL_tokeep])]]
            T_minL += (T[argminL[i+1]],)
            i+=1
    
    return minL, argminL, T_minL
    
    
def sort_people(keyptpre, keypt, nb_persons_to_detect):
    '''
    Associate persons across frames
    Persons' indices are sometimes swapped when changing frame
    A person is associated to another in the next frame when they are at a small distance
    
    INPUTS:
    - keyptpre: array of shape K, L, M with K the number of detected persons,
    L the number of detected keypoints, M their 2D coordinates + confidence
    for the previous frame
    - keypt: idem keyptpre, for current frame
    
    OUTPUT:
    - keypt: array with reordered persons
    '''
    
    # Generate possible person correspondences across frames
    personsIDs_comb = sorted(list(it.product(range(len(keyptpre)),range(len(keypt)))))
    # Compute distance between persons from one frame to another
    frame_by_frame_dist = []
    for comb in personsIDs_comb:
        frame_by_frame_dist += [np.mean([euclidean_distance(i,j) for (i,j) in zip(keyptpre[comb[0]][:,:2],keypt[comb[1]][:,:2])])]
    # sort correspondences by distance
    _, index_best_comb, _ = min_with_single_indices(frame_by_frame_dist, personsIDs_comb)
    index_best_comb.sort()
    personsIDs_sorted = np.array(personsIDs_comb)[index_best_comb][:,1]
    # rearrange persons
    keypt = np.array(keypt)[personsIDs_sorted]
    
    return keypt

def save_to_openpose(json_file_path, keypoints, scores, kpt_thr):
    """
    Saves the detected keypoints and scores to a JSON file in the OpenPose format.

    Args:
        json_file_path (str): The path to the output JSON file.
        keypoints (list): A list of detected keypoints, where each element is a list of 2D coordinates for each keypoint.
        scores (list): A list of scores for each detected keypoint, where each element is a list of confidence scores for each keypoint.
        kpt_thr (float): The minimum average score threshold for a detection to be included in the output.

    Returns:
        None
    """

    score_threshold = kpt_thr
    nb_detections = len(keypoints)
    # print(f" number of detections: {nb_detections}")
    detections = []

    for i in range(nb_detections):
        # average score of all keypoints
        average_score = sum(score.item() for score in scores[i]) / len(scores[i])
        
        # if average score is above threshold, add to detections
        if average_score >= score_threshold:
            keypoints_with_confidence_i = []
            for kp, score in zip(keypoints[i], scores[i]):
                keypoints_with_confidence_i.extend([kp[0].item(), kp[1].item(), score.item()])
            detections.append({
                "person_id": [-1], # Maybe change to i ?
                "pose_keypoints_2d": keypoints_with_confidence_i,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            })
            # print(f"Person {i} passed the threshold. Average score: {average_score:.4f}")
        # else:
        #     print(f"Person {i} did not pass the threshold. Average score: {average_score:.4f}")
    
    json_output = {"version": 1.3, "people": detections}
    
    json_output_dir = os.path.abspath(os.path.join(json_file_path, '..'))
    if not os.path.isdir(json_output_dir): 
        os.makedirs(json_output_dir)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)


def json_to_csv(json_path, frame_rate, interp_gap_smaller_than, filter_options, show_plots, min_detection_time, tracking):
    '''
    Converts frame-by-frame json coordinate files 
    to one csv files per detected person

    INPUTS:
    - json_path: directory path of json files
    - frame_rate: frame rate of the video
    - interp_gap_smaller_than: integer, maximum number of missing frames for conducting interpolation
    - filter_options: list, options for filtering
    - show_plots: boolean, show plots or not
    - min_detection_time: minimum detection time in seconds (default: 1 second)
    - tracking: boolean, use tracking or not (when input_data is video)

    OUTPUTS:
    - Creation of one csv files per detected person
    '''
        
    # Retrieve keypoint names from model
    model = halpe26_rtm
    keypoints_info = model['keypoint_info']
    keypoints_ids = [kp['id'] for kp in keypoints_info.values()]
    keypoints_names = [kp['name'] for kp in keypoints_info.values()]
    keypoints_nb = len(keypoints_ids)

    # Retrieve coordinates
    logging.info('Sorting people across frames.')
    json_fnames = sorted(list(json_path.glob('*.json')))
    nb_persons_to_detect = max([len(json.load(open(json_fname))['people']) for json_fname in json_fnames])
    Coords = [np.array([]).reshape(0,keypoints_nb*3)] * nb_persons_to_detect
    for json_fname in json_fnames:    # for each frame
        with open(json_fname) as json_f:
            json_file = json.load(json_f)
            keypt = []
            # Retrieve coords for this frame 
            for ppl in range(len(json_file['people'])):  # for each detected person
                keypt_all = np.asarray(json_file['people'][ppl]['pose_keypoints_2d']).reshape(-1,3)[keypoints_ids]
                keypt += [keypt_all]
            keypt = np.array(keypt)
            # Make sure keypt is as large as the number of persons that need to be detected
            if len(keypt) < nb_persons_to_detect:
                empty_keypt_to_add = np.concatenate( [[ np.zeros([keypoints_nb,3]) ]] * (nb_persons_to_detect-len(keypt)) )
                keypt = [np.concatenate([keypt, empty_keypt_to_add]) if list(keypt)!=[] else empty_keypt_to_add][0]
            if 'keyptpre' not in locals():
                keyptpre = keypt
            # Associate persons across frames if tracking is enabled
            if tracking and keyptpre is not None:
                keypt = sort_people(keyptpre, keypt, nb_persons_to_detect)
            # Concatenate to coordinates of previous frames
            for i in range(nb_persons_to_detect): 
                Coords[i] = np.vstack([Coords[i], keypt[i].reshape(-1)])
            keyptpre = keypt
    logging.info(f'{nb_persons_to_detect} persons found.')

    # Inject coordinates in dataframes and save
    for i in range(nb_persons_to_detect): 
        # Calculate detection time
        detection_mask = np.any(Coords[i] != 0, axis=1)
        detection_time = np.sum(detection_mask) / frame_rate

        # Skip if detected for less than min_detection_time
        if detection_time < min_detection_time:
            logging.info(f'Person {i}: Detected for less than {min_detection_time} second(s). Skipping.')
            continue

        # Prepare csv header
        scorer = ['DavidPagnon']*(keypoints_nb*3+1)
        individuals = [f'person{i}']*(keypoints_nb*3+1)
        bodyparts = [[p]*3 for p in keypoints_names]
        bodyparts = ['Time']+[item for sublist in bodyparts for item in sublist]
        coords = ['seconds']+['x', 'y', 'score']*keypoints_nb
        tuples = list(zip(scorer, individuals, bodyparts, coords))
        index_csv = pd.MultiIndex.from_tuples(tuples, names=['scorer', 'individuals', 'bodyparts', 'coords'])

        # Create dataframe
        df_list=[]
        time = np.expand_dims( np.arange(0,len(Coords[i]), 1)/frame_rate, axis=0 )
        time_coords = np.concatenate(( time, Coords[i].T ))
        df_list += [pd.DataFrame(time_coords, index=index_csv).T]

        # Interpolate
        logging.info(f'Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames.')
        df_list[0] = df_list[0].apply(common.interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, 'linear'])
        
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
        
        # empty values
        df_list[-1] = df_list[-1].replace({np.nan: None, 0: None})
        
        # Save csv
        csv_path = json_path.parent / Path(json_path.name[:-5]+f'_person{i}_points.csv')
        logging.info(f'Person {i}: Saving csv position file in {csv_path}.')
        df_list[-1].to_csv(csv_path, sep=',', index=True, lineterminator='\n', na_rep='')
        
        # Display figures
        if show_plots:
            logging.info(f'Person {i}: Displaying figures.')
            display_figures_fun_cords(df_list)

def show_image(img, title):
    if 'google.colab' in sys.modules:
        # Colab
        from google.colab.patches import cv2_imshow
        cv2_imshow(img)
        clear_output(wait=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    else:
        # Local
        cv2.imshow(title, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True
    
# If input is a video
def process_video(video_path, video_result_path,pose_tracker, output_format, save_video, save_images, display_detection, frame_range, kpt_thr):
    '''
    Estimate pose from a video file
    
    INPUTS:
    - video_path: str. Path to the input video file
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process
    - kpt_thr: float. Keypoint score threshold

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''

    try:
        cap = cv2.VideoCapture(video_path)
        cap.read()
        if cap.read()[0] == False:
            raise
    except:
        raise NameError(f"{video_path} is not a video. Images must be put in one subdirectory per camera.")
    
    pose_dir = os.path.abspath(os.path.join(video_result_path, '..', 'video_results'))
    if not os.path.isdir(pose_dir): os.makedirs(pose_dir)
    video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
    json_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_json')
    output_video_path = os.path.join(pose_dir, f'{video_name_wo_ext}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_img')
    
    if save_video: # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        fps = cap.get(cv2.CAP_PROP_FPS) # Get the frame rate from the raw video
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the width and height from the raw video
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file

    frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_range = [[total_frames] if frame_range==[] else frame_range][0]
    with tqdm(total=total_frames, desc=f'Processing {os.path.basename(video_path)}') as pbar:
        while cap.isOpened():
            # print('\nFrame ', frame_idx)
            success, frame = cap.read()
            if not success:
                break
            
            if frame_idx in range(*f_range):
                # Perform pose estimation on the frame
                keypoints, scores = pose_tracker(frame)

                # Save to json
                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores, kpt_thr)

                # Draw skeleton on the frame
                if display_detection or save_video or save_images:
                    img_show = frame.copy()
                    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low
                
                if display_detection:
                    if not show_image(img_show, f"Pose Estimation {os.path.basename(video_path)}"):
                        break

                if save_video:
                    out.write(img_show)

                if save_images:
                    if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                    cv2.imwrite(os.path.join(img_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.png'), img_show)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection and 'google.colab' not in sys.modules:
        cv2.destroyAllWindows()

def process_webcam(cam_id, pose_tracker, joint_angles, segment_angles, 
                       save_vid, save_img, interp_gap_smaller_than, filter_options, show_plots, flip_left_right,
                         kpt_thr, data_type, min_detection_time, filter_options_ang, show_plots_ang):
    """
    Process a live webcam feed to detect poses and calculate angles.

    This function captures video from a webcam, performs pose detection, calculates joint and segment angles,
    and optionally saves the processed video, images, and angle data.

    Parameters:
    - cam_id (int): The ID of the webcam to capture from
    - pose_tracker (object): Pose detection and tracking object
    - tracking (bool): Whether to use object tracking
    - joint_angles (list): List of joint angles to calculate
    - segment_angles (list): List of segment angles to calculate
    - save_video (bool): Whether to save the processed video
    - save_images (bool): Whether to save individual frame images
    - interp_gap_smaller_than (int): Maximum frame gap to interpolate
    - filter_options (tuple): Filtering options for smoothing angle data
    - show_plots (bool): Whether to display result graphs
    - flip_left_right (bool): Whether to apply left-right flip
    - kpt_thr (float): Keypoint detection threshold
    - data_type (str): Type of data processing ('webcam' in this case)

    The function performs the following steps:
    1. Set up webcam capture
    2. Process frames in real-time, detecting poses and calculating angles (with tracking)
    3. Save JSON files with pose data
    4. Optionally save processed video and images
    5. Convert JSON files to CSV
    6. Recalculate angles from filtered CSV data
    7. Apply filtering to angle data
    8. Save final angle data as CSV
    9. Optionally display plots of angle data

    Output:
    - Processed video file (if save_video is True)
    - Frame images (if save_images is True)
    - JSON files with pose data
    - CSV files with filtered pose data
    - CSV files with calculated and filtered angle data
    - Plots of angle data (if show_plots_ang is True)

    Note: The function can be interrupted by pressing 'q' or using a keyboard interrupt (Ctrl+C).
    """

    is_colab = 'google.colab' in sys.modules
    if is_colab:
        from IPython.display import display, Javascript, clear_output
        from google.colab.output import eval_js
        from base64 import b64decode, b64encode
        import matplotlib.pyplot as plt
        import PIL.Image
        import io
    
        # JavaScript to properly create our live video stream using our webcam as input
        def video_stream():
            js = Javascript('''
                var video;
                var div = null;
                var stream;
                var captureCanvas;
                var imgElement;
                var labelElement;
                var resultImgElement;
                
                var pendingResolve = null;
                var shutdown = false;
                
                function removeDom() {
                   stream.getVideoTracks()[0].stop();
                   video.remove();
                   div.remove();
                   video = null;
                   div = null;
                   stream = null;
                   imgElement = null;
                   captureCanvas = null;
                   labelElement = null;
                   resultImgElement = null;
                }
                
                function onAnimationFrame() {
                  if (!shutdown) {
                    window.requestAnimationFrame(onAnimationFrame);
                  }
                  if (pendingResolve) {
                    var result = "";
                    if (!shutdown) {
                      captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
                      result = captureCanvas.toDataURL('image/jpeg', 0.8)
                    }
                    var lp = pendingResolve;
                    pendingResolve = null;
                    lp(result);
                  }
                }
                
                async function createDom() {
                  if (div !== null) {
                    return stream;
                  }
            
                  div = document.createElement('div');
                  div.style.border = '2px solid black';
                  div.style.padding = '3px';
                  div.style.width = '100%';
                  div.style.maxWidth = '600px';
                  document.body.appendChild(div);
                  
                  const modelOut = document.createElement('div');
                  modelOut.innerHTML = "<span>Status:</span>";
                  labelElement = document.createElement('span');
                  labelElement.innerText = 'No data';
                  labelElement.style.fontWeight = 'bold';
                  modelOut.appendChild(labelElement);
                  div.appendChild(modelOut);
                       
                  video = document.createElement('video');
                  video.style.display = 'block';
                  video.width = div.clientWidth - 6;
                  video.setAttribute('playsinline', '');
                  video.onclick = () => { shutdown = true; };
                  stream = await navigator.mediaDevices.getUserMedia(
                      {video: { facingMode: "environment"}});
                  div.appendChild(video);
            
                  resultImgElement = document.createElement('img');
                  resultImgElement.style.position = 'absolute';
                  resultImgElement.style.zIndex = 1;
                  resultImgElement.onclick = () => { shutdown = true; };
                  div.appendChild(resultImgElement);
                  
                  const instruction = document.createElement('div');
                  instruction.innerHTML = 
                      '<span style="color: red; font-weight: bold;">' +
                      'When finished, click here or on the video to stop this demo</span>';
                  div.appendChild(instruction);
                  instruction.onclick = () => { shutdown = true; };
                  
                  video.srcObject = stream;
                  await video.play();
            
                  captureCanvas = document.createElement('canvas');
                  captureCanvas.width = 640; //video.videoWidth;
                  captureCanvas.height = 480; //video.videoHeight;
                  window.requestAnimationFrame(onAnimationFrame);
                  
                  return stream;
                }
                async function stream_frame(label, imgData) {
                  if (shutdown) {
                    removeDom();
                    shutdown = false;
                    return '';
                  }
            
                  var preCreate = Date.now();
                  stream = await createDom();
                  
                  var preShow = Date.now();
                  if (label != "") {
                    labelElement.innerHTML = label;
                  }
                        
                  if (imgData != "") {
                    var videoRect = video.getClientRects()[0];
                    resultImgElement.style.top = videoRect.top + "px";
                    resultImgElement.style.left = videoRect.left + "px";
                    resultImgElement.style.width = videoRect.width + "px";
                    resultImgElement.style.height = videoRect.height + "px";
                    resultImgElement.src = imgData;
                  }
                  
                  var preCapture = Date.now();
                  var result = await new Promise(function(resolve, reject) {
                    pendingResolve = resolve;
                  });
                  shutdown = false;
                  
                  return {'create': preShow - preCreate, 
                          'show': preCapture - preShow, 
                          'capture': Date.now() - preCapture,
                          'img': result};
                }
                ''')
            
            display(js)
          
        def video_frame(label, bbox):
          data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
          return data

        
    # Output directory setup
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.path.join(os.getcwd(), f'webcam_results_{current_time}'))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir = output_dir / 'json'
    json_output_dir.mkdir(parents=True, exist_ok=True)

    if is_colab:
        #initialize video stream
        video_stream()
        
        # Get initial frame to determine resolution
        initial_frame_data = video_frame("Initializing...", "")
        frame = np.frombuffer(b64decode(initial_frame_data['img'].split(',')[1]), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        cam_width, cam_height = frame.shape[1], frame.shape[0]
    
        print(f"Webcam resolution: {cam_width}x{cam_height}")
    else:
        # Local webcam setup
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW) # CAP_DSHOW would be better for Windows
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Set extremely high resolution for automatically using maximum resolution of webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        # MJPG would be better when high resolution is needed
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Get the actual resolution of the webcam
        cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Resolution: {cam_width}x{cam_height}")

        # Check if the webcam is working
        ret, frame = cap.read()
        if ret:
            print(f"Actual frame size: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("Failed to capture frame")

        cv2.namedWindow("Real-time Analysis", cv2.WINDOW_NORMAL)


        # Image output directory setup
        if save_img:
            img_output_dir = output_dir / 'webcam_images'
            img_output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    keypoints_data = {}
    scores_data = {}
    processed_frames = []
    processing_times = []
    last_process_time = time_module.time()

    # Initialize previous frame keypoints for person tracking
    prev_keypoints = None

    try:
        while True:
            if is_colab:
                frame_data = video_frame("Processing...", "")
                if frame_data == '':
                    print("Processing stopped by user.")
                    break
    
                frame = np.frombuffer(b64decode(frame_data['img'].split(',')[1]), dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

            frame_count += 1
            current_time = time_module.time()
            elapsed_time = current_time - last_process_time
            processing_times.append(elapsed_time)

            keypoints, scores = pose_tracker(frame)

            # Apply person tracking
            if prev_keypoints is not None:
                keypoints = sort_people(prev_keypoints, keypoints, len(keypoints))
            prev_keypoints = keypoints

            df_angles_list_frame = []
            valid_X, valid_Y, valid_scores, valid_person_ids = [], [], [], []
            keypoints_flipped = keypoints.copy()

            for person_idx in range(len(keypoints)):
                person_keypoints = keypoints[person_idx]
                person_scores = scores[person_idx]

                # This would be useful for more accurate detection of people
                # If the person has less than 30% keypoints, skip this person.
                # if np.sum(person_scores >= kpt_thr) < len(person_keypoints) * 0.3:
                    # continue  

                df_points = common.convert_keypoints_to_dataframe(person_keypoints, person_scores)

                X = df_points[[col for col in df_points.columns if col.endswith('_x')]]
                Y = df_points[[col for col in df_points.columns if col.endswith('_y')]]

                valid_X.append(X.values[0])
                valid_Y.append(Y.values[0])
                valid_scores.append(person_scores)
                valid_person_ids.append(person_idx)

                # check if the person is on the left or right side of the screen
                if flip_left_right:
                    df_points = flip_left_right_direction(df_points, data_type)

                    # Only flip the x-coordinates of the keypoints
                    x_columns = [col for col in df_points.columns if col.endswith('_x')]
                    flipped_x = df_points[x_columns].values.flatten()
                    keypoints_flipped[person_idx, :, 0] = flipped_x
                
                joint_angle_values = {}
                for joint in joint_angles:
                    angle_params = get_joint_angle_params(joint) # Get the joint dictionaries
                    if angle_params:
                        angle = joint_angles_series_from_points(df_points, angle_params, kpt_thr)
                        if angle is not None:
                            joint_angle_values[joint] = angle[0]
                
                segment_angle_values = {}
                for segment in segment_angles:
                    angle_params = get_segment_angle_params(segment) # Get the segment dictionaries
                    if angle_params:
                        angle = segment_angles_series_from_points(df_points, angle_params, segment, kpt_thr)
                        if angle is not None:
                            segment_angle_values[segment] = angle[0]
                
                df_angles_list_frame.append({**joint_angle_values, **segment_angle_values})
                
                if person_idx not in keypoints_data:
                    keypoints_data[person_idx] = []
                    scores_data[person_idx] = []
                
                keypoints_data[person_idx].append(person_keypoints)
                scores_data[person_idx].append(person_scores)

            img_show = frame.copy()
            if valid_X and valid_Y:
                img_show = draw_bounding_box(valid_X, valid_Y, img_show, valid_person_ids)
                img_show = draw_keypts_skel(valid_X, valid_Y, valid_scores, img_show, 'RTMPose', kpt_thr)

            img_show = overlay_angles(img_show, df_angles_list_frame, keypoints, scores, kpt_thr, valid_person_ids)

            processed_frames.append(img_show)

            cv2.imshow("Real-time Analysis", img_show)


            if is_colab:
                # Incoding image to base64 for display
                _, buffer = cv2.imencode('.jpg', img_show)
                img_base64 = b64encode(buffer).decode('utf-8')
                
                # Display image in Colab
                _ = video_frame("Processing...", f"data:image/jpeg;base64,{img_base64}")
            else:
                window_size = cv2.getWindowImageRect("Real-time Analysis")
                if window_size[2] > 0 and window_size[3] > 0:
                    img_show = cv2.resize(img_show, (window_size[2], window_size[3]))
                cv2.imshow("Real-time Analysis", img_show)
            
            if save_img:
                cv2.imwrite(str(img_output_dir / f'frame_{frame_count:06d}.png'), img_show)
            
            json_file_path = json_output_dir / f'frame_{frame_count:06d}.json'
            if flip_left_right:
                save_to_openpose(json_file_path, keypoints_flipped, scores, kpt_thr)
            else:
                save_to_openpose(json_file_path, keypoints, scores, kpt_thr)
            
            if not is_colab and cv2.waitKey(1) & 0xFF == ord('q'):
                break

            last_process_time = current_time

            if frame_count % 100 == 0:
                print(f"Processed frame {frame_count}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if is_colab:
            eval_js('stream_frame("", "")')
        else:
            cap.release()
        cv2.destroyAllWindows()

    print(f"\nTotal frames processed: {frame_count}")

    # Actual processing FPS
    actual_processing_fps = len(processed_frames) / sum(processing_times)
    print(f"Actual processing FPS: {actual_processing_fps:.2f}")

    if save_vid:
        video_path = str(output_dir / 'webcam_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, actual_processing_fps, (cam_width, cam_height))
        
        for frame in processed_frames:
            writer.write(cv2.resize(frame, (cam_width, cam_height)))
        
        writer.release()

    # Filtering for cords
    if filter_options[0]:
        filter_options = list(filter_options)
        filter_options[4] = actual_processing_fps
        filter_options = tuple(filter_options)
    
    # Save CSV filee, tracking already done before
    json_to_csv(json_output_dir, actual_processing_fps, interp_gap_smaller_than, filter_options, show_plots, min_detection_time, tracking=False)

    # Recalculate angles from filtered CSV files and apply filtering for each person
    for person_idx in keypoints_data.keys():
        csv_path = output_dir / f'_person{person_idx}_points.csv'
        if os.path.exists(csv_path):
            df_points = pd.read_csv(csv_path, header=[0,1,2,3])

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
                    logging.warning(f'Error calculating joint angle {j} for Person {person_idx}: {str(e)}')
                    joint_angle_series.append(np.nan)

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
                    logging.warning(f'Error calculating segment angle {s} for Person {person_idx}: {str(e)}')
                    segment_angle_series.append(np.nan)

            time = [np.array(df_points.iloc[:,1])]
            angle_series = time + joint_angle_series + segment_angle_series

            angle_series = [series if isinstance(series, (list, np.ndarray)) else np.full_like(time[0], np.nan) for series in angle_series if series is not None]

            max_length = max(len(series) for series in angle_series)
            angle_series = [np.pad(series, (0, max_length - len(series)), 'constant', constant_values=np.nan) for series in angle_series]

            scorer = ['DavidPagnon'] * (len(joint_angles) + len(segment_angles) + 1)
            individuals = [f'person{person_idx}'] * len(scorer)
            angs = ['Time'] + joint_angles + segment_angles
            coords = ['seconds'] + [get_joint_angle_params(j)[1] if get_joint_angle_params(j) else 'unknown' for j in joint_angles] + [get_segment_angle_params(s)[1] if get_segment_angle_params(s) else 'unknown' for s in segment_angles]
            index_angs_csv = pd.MultiIndex.from_tuples(list(zip(scorer, individuals, angs, coords)), 
                                                      names=['scorer', 'individuals', 'angs', 'coords'])

            df_angles = [pd.DataFrame(np.array(angle_series).T, columns=index_angs_csv)]

            # Initialize df_list
            df_list = df_angles.copy()

            # Filtering for angles
            if filter_options_ang[0]:
                filter_options_ang = list(filter_options_ang)
                filter_options_ang[4] = actual_processing_fps 
                filter_options_ang = tuple(filter_options_ang)

                filter_type = filter_options_ang[1]
                if filter_type == 'butterworth':
                    args = f'Butterworth filter, {filter_options_ang[2]}th order, {filter_options_ang[3]} Hz.'
                elif filter_type == 'gaussian':
                    args = f'Gaussian filter, Sigma kernel {filter_options_ang[5]}'
                elif filter_type == 'loess':
                    args = f'LOESS filter, window size of {filter_options_ang[6]} frames.'
                elif filter_type == 'median':
                    args = f'Median filter, kernel of {filter_options_ang[7]}.'
                logging.info(f'Person {person_idx}: Filtering with {args}.')
                df_list[0].replace(0, np.nan, inplace=True)
                df_list += [df_list[0].copy()]
                df_list[1] = df_list[1].apply(filter.filter1d, axis=0, args=filter_options_ang)
        
            # empty values
            df_list[-1] = df_list[-1].replace({np.nan: None, 0: None})

            # Replace NaN with 0 in the final DataFrame
            df_list[-1].replace(np.nan, 0, inplace=True)

            csv_angle_path = output_dir / f'_person{person_idx}_angles.csv'
            df_list[-1].to_csv(csv_angle_path, sep=',', index=True, lineterminator='\n')
            
            if os.path.exists(csv_angle_path):
                logging.info(f'Successfully saved angles CSV for Person {person_idx}')
            else:
                logging.error(f'Failed to save angles CSV for Person {person_idx}')

            # Display figures
            if show_plots_ang:
                if not df_list[0].empty:
                    display_figures_fun_ang(df_list)
                else:
                    logging.info(f'Person {person_idx}: No angle data to display.')

    logging.info(f"Output saved to: {output_dir}")
    if save_vid:
        logging.info(f"Video file: {video_path}")
    if save_img:
        logging.info(f"Image files: {img_output_dir}")
    logging.info(f"CSV files: {output_dir}")





def detect_pose_fun(config_dict, video_file):
    '''
    Detect joint centers from a video with OpenPose or BlazePose.
    Save a 2D csv file per person, and optionally json files, image files, and video file.
    
    If OpenPose is used, multiple persons can be consistently detected across frames.
    Interpolates sequences of missing data if they are less than N frames long.
    Optionally filters results with Butterworth, gaussian, median, or loess filter.
    Optionally displays figures.

    If BlazePose is used, only one person can be detected.
    No interpolation nor filtering options available. Not plotting available.

    /!\ Warning /!\
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
        
    INPUTS:
    - a video or a webcam
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one csv file of joint coordinates per detected person
    - optionally json directory, image directory, video
    - a logs.txt file 
    '''
    
    # Retrieve parameters
    root_dir = os.getcwd()
    video_dir, video_files, result_dir, frame_rate, data_type = base_params(config_dict)

    # Pose settings
    time_range = config_dict.get('pose', {}).get('time_range', []) 
    frame_range = [int(time_range[0] * frame_rate), int(time_range[1] * frame_rate)] if len(time_range) == 2 else []
    display_detection = config_dict.get('pose_advanced').get('display_detection')
    output_format = "openpose"
    openpose_skeleton = False
    
    # Pose_advanced settings
    cam_id =  config_dict.get('pose_advanced').get('webcam_id')
    load_pose = not config_dict.get('pose_advanced').get('overwrite_pose')
    save_vid = config_dict.get('pose_advanced').get('save_vid')
    save_img = config_dict.get('pose_advanced').get('save_img')
    interp_gap_smaller_than = config_dict.get('pose_advanced').get('interp_gap_smaller_than')
    min_detection_time = config_dict.get('pose_advanced').get('min_detection_time') # If lower than this, person will be ignored. For webcams, 
                                                                                    #it is possible to detect the wrong person if the person is only partially detected (usually less than a second).
    mode = config_dict.get('pose_advanced').get('mode')
    det_frequency = config_dict.get('pose_advanced').get('det_frequency')
    kpt_thr = config_dict.get('pose_advanced').get('keypoints_threshold') # If only part of a person is on screen, increase this number to ensure that only correctly detected keypoints are used.
    
    ## Filtering options for cords
    show_plots = config_dict.get('pose_advanced').get('show_plots')
    do_filter = config_dict.get('pose_advanced').get('filter')
    filter_type = config_dict.get('pose_advanced').get('filter_type')
    butterworth_filter_order = config_dict.get('pose_advanced').get('butterworth').get('order')
    butterworth_filter_cutoff = config_dict.get('pose_advanced').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel = config_dict.get('pose_advanced').get('gaussian').get('sigma_kernel')
    loess_filter_kernel = config_dict.get('pose_advanced').get('loess').get('nb_values_used')
    median_filter_kernel = config_dict.get('pose_advanced').get('median').get('kernel_size')
    filter_options = (do_filter, filter_type, butterworth_filter_order, butterworth_filter_cutoff, frame_rate,
                      gaussian_filter_kernel, loess_filter_kernel, median_filter_kernel)

    # Compute_angles_advanced settings
    ## Filter options for angles
    show_plots_ang = config_dict.get('compute_angles_advanced').get('show_plots')
    flip_left_right = config_dict.get('compute_angles_advanced').get('flip_left_right')
    do_filter_ang = config_dict.get('compute_angles_advanced').get('filter')
    filter_type_ang = config_dict.get('compute_angles_advanced').get('filter_type')
    butterworth_filter_order_ang = config_dict.get('compute_angles_advanced').get('butterworth').get('order')
    butterworth_filter_cutoff_ang = config_dict.get('compute_angles_advanced').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel_ang = config_dict.get('compute_angles_advanced').get('gaussian').get('sigma_kernel')
    loess_filter_kernel_ang = config_dict.get('compute_angles_advanced').get('loess').get('nb_values_used')
    median_filter_kernel_ang = config_dict.get('compute_angles_advanced').get('median').get('kernel_size')
    filter_options_ang = (do_filter_ang, filter_type_ang, butterworth_filter_order_ang, butterworth_filter_cutoff_ang,
                          frame_rate, gaussian_filter_kernel_ang, loess_filter_kernel_ang, median_filter_kernel_ang)



    # Determine device and backend for RTMPose
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            backend = 'onnxruntime'
        logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
    elif 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
        device = 'mps'
        backend = 'onnxruntime'
        logging.info(f"\nValid MPS installation found: using ONNXRuntime backend with GPU.")
    else:
        device = 'cpu'
        backend = 'openvino'
        logging.info(f"\nNo valid CUDA installation found: using OpenVINO backend with CPU.")

    # Initialize the pose tracker with Halpe26 model(I should add the other models later)
    pose_tracker = PoseTracker(
        BodyWithFeet,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=False, # we will use the Sports2D's tracking
        to_openpose=False)

    if data_type == 'video':
        if video_file is None:
            raise ValueError("Video file path is required when data_type is 'video'")
        
        video_file_stem = video_file.stem
        video_path = video_dir / video_file
        video_result_path = result_dir / video_file
        pose_dir = result_dir / 'video_results'
        json_path = pose_dir / '_'.join((video_file_stem,'json'))

        # Pose detection skipped if load existing json files
        if load_pose and len(list(json_path.glob('*.json')))>0:
            logging.info(f'2D joint positions have already been detected. To run the analysis over again from the beginning, set "overwrite_pose" to true in Advanced pose settings.')
        else:
            logging.info(f'Detecting 2D joint positions with RTMPose Halpe26 model, for {video_file}.')
            
            json_path.mkdir(parents=True, exist_ok=True)

            # Process video with RTMPose
            process_video(str(video_path), video_result_path, pose_tracker, output_format, save_vid, save_img, display_detection, frame_range, kpt_thr)

        # Sort people and save to csv, optionally display plot
        try:
            json_to_csv(json_path, frame_rate, interp_gap_smaller_than, filter_options, show_plots, min_detection_time, tracking=True)
        except:
            logging.warning('No person detected or persons could not be associated across frames.')
            return

    elif data_type == 'webcam':
        # Define necessary parameters for webcam processing
        joint_angles = config_dict.get('compute_angles').get('joint_angles', [])
        segment_angles = config_dict.get('compute_angles').get('segment_angles', [])

        # Warning message about filtering
        print('Webcam input: the framerate may vary. Sports2D will use the average framerate as input to the filter (if results are filtered).')

        # Process webcam feed
        process_webcam(cam_id, pose_tracker, joint_angles, segment_angles, 
                       save_vid, save_img, interp_gap_smaller_than, filter_options, show_plots, flip_left_right,
                         kpt_thr, data_type, min_detection_time, filter_options_ang, show_plots_ang)

    else:
        raise ValueError(f"Invalid input_source: {data_type}. Must be 'video' or 'webcam'.")

    logging.info("Pose detection and analysis completed.")