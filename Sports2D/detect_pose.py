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
from pathlib import Path
from sys import platform
import json
import subprocess
import itertools as it
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from Sports2D.Sports2D import base_params
from Sports2D.Utilities import Blazepose_runsave, filter, common
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


# FUNCTIONS
def display_figures_fun(df_list):
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
    

def run_openpose_windows(video_path, json_path, pose_model):
    '''
    Use a subprocess to run OpenPoseDemo.exe, and saves json coordinate files.
     
    INPUTS:
    - video_path: Path of the video to analyze
    - json_path: Path of the directory where to save json files
    - pose_model: string. "BODY_25B", "BODY_25", or others.
          
    OUTPUTS:
    - json files in json_path
    '''

    subprocess.run(["bin\OpenPoseDemo.exe", "--video", video_path, \
    "--model_pose", pose_model, \
    "--write_json", json_path, \
    "--render_pose", "0", "--display", "0"])


def run_openpose_linux(video_path, json_path, pose_model):
    '''
    Use a subprocess to run openpose.bin, and saves json coordinate files.
     
    INPUTS:
    - video_path: Path of the video to analyze
    - json_path: Path of the directory where to save json files
    - pose_model: string. "BODY_25B", "BODY_25", or others.
          
    OUTPUTS:
    - json files in json_path
    '''
    
    subprocess.run(["./build/examples/openpose/openpose.bin", "--video", video_path, \
    "--model_pose", pose_model, \
    "--write_json", json_path, \
    "--render_pose", "0", "--display", "0"])


def run_openpose_mac(video_path, json_path, pose_model):
    '''
    Use a subprocess to run openpose.bin, and saves json coordinate files.
    WARNING: not tested.
     
    INPUTS:
    - video_path: Path of the video to analyze
    - json_path: Path of the directory where to save json files
    - pose_model: string. "BODY_25B", "BODY_25", or others.
          
    OUTPUTS:
    - json files in json_path
    '''

    subprocess.run(["./build/examples/openpose/openpose.bin", "--video", video_path, \
    "--model_pose", pose_model, \
    "--write_json", json_path, \
    "--render_pose", "0", "--display", "0"])
    
    
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
    - 2nd smallest value when excluding indices [(0,0),(0,1),(0,2),X,(1,0),(1,1),(1,2),X,X,X,X,X]:
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

    minL = [np.min(L)]
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
    personsIDs_comb = list(it.product(range(len(keyptpre)),range(len(keypt))))
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


def json_to_csv(json_path, frame_rate, pose_model, interp_gap_smaller_than, filter_options, show_plots):
    '''
    Converts frame-by-frame json coordinate files 
    to one csv files per detected person

    INPUTS:
    - json_path: directory path of json files
    - pose_model: string, to get tree from skeletons.py
    - interp_gap_smaller_than: integer, maximum number of missing frames for conducting interpolation
    - filter_options: list, options for filtering
    - show_plots: boolean, show plots or not

    OUTPUTS:
    - Creation of one csv files per detected person
    '''
        
    # Retrieve keypoint names from model
    model = eval(pose_model)
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names_rearranged = [y for x,y in sorted(zip(keypoints_ids,keypoints_names))]
    keypoints_nb = len(keypoints_ids)

    # Retrieve coordinates
    logging.info('Sorting people across frames.')
    json_fnames = list(json_path.glob('*.json'))
    nb_persons_to_detect = max([len(json.load(open(json_fname))['people']) for json_fname in json_fnames])
    Coords = [np.array([]).reshape(0,keypoints_nb*3)] * nb_persons_to_detect
    for json_fname in json_fnames:    # for each frame
        with open(json_fname) as json_f:
            json_file = json.load(json_f)
            keypt = []
            # Retrieve coords for this frame 
            for ppl in range(len(json_file['people'])):  # for each detected person
                keypt += [np.asarray(json_file['people'][ppl]['pose_keypoints_2d']).reshape(-1,3)]
            keypt = np.array(keypt)
            # Make sure keypt is as large as the number of persons that need to be detected
            if len(keypt) < nb_persons_to_detect:
                empty_keypt_to_add = np.concatenate( [[ np.zeros([25,3]) ]] * (nb_persons_to_detect-len(keypt)) )
                keypt = [np.concatenate([keypt, empty_keypt_to_add]) if keypt!=[] else empty_keypt_to_add][0]
            if 'keyptpre' not in locals():
                keyptpre = keypt
            # Associate persons across frames
            keypt = sort_people(keyptpre, keypt, nb_persons_to_detect)
            # Concatenate to coordinates of previous frames
            for i in range(nb_persons_to_detect): 
                Coords[i] = np.vstack([Coords[i], keypt[i].reshape(-1)])
            keyptpre = keypt
    logging.info(f'{nb_persons_to_detect} persons found.')
    
    # Inject coordinates in dataframes and save
    for i in range(nb_persons_to_detect): 
        # Prepare csv header
        scorer = ['DavidPagnon']*(keypoints_nb*3+1)
        individuals = [f'person{i}']*(keypoints_nb*3+1)
        bodyparts = [[p]*3 for p in keypoints_names_rearranged]
        bodyparts = ['Time']+[item for sublist in bodyparts for item in sublist]
        coords = ['seconds']+['x', 'y', 'likelihood']*keypoints_nb
        tuples = list(zip(scorer, individuals, bodyparts, coords))
        index_csv = pd.MultiIndex.from_tuples(tuples, names=['scorer', 'individuals', 'bodyparts', 'coords'])

        # Create dataframe
        df_list=[]
        time = np.expand_dims( np.arange(0,len(Coords[i])/frame_rate, 1/frame_rate), axis=0 )
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
        df_list[-1].replace(np.nan, 0, inplace=True)
           
        # Save csv
        csv_path = json_path.parent / Path(json_path.name[:-5]+f'_person{i}_points.csv')
        logging.info(f'Person {i}: Saving csv position file in {csv_path}.')
        df_list[-1].to_csv(csv_path, sep=',', index=True, lineterminator='\n')
        
        # Display figures
        if show_plots:
            logging.info(f'Person {i}: Displaying figures.')
            display_figures_fun(df_list)
            

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


def save_imgvid_reID(video_path, video_result_path, save_vid=1, save_img=1, *pose_model):
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
            
   # Find csv position files, prepare video and image saving paths
    pose_model = pose_model[0]
    csv_paths = list(video_result_path.parent.glob(f'*{video_result_path.stem}*{pose_model}*points*refined*.csv'))
    if csv_paths == []:
        csv_paths = list(video_result_path.parent.glob(f'*{video_result_path.stem}*{pose_model}*points*.csv'))
        
    # Open csv files
    coords = []
    for c in csv_paths:
        with open(c) as c_f:
            coords += [pd.read_csv(c_f, header=[0,1,2,3])]

    # Open video frame by frame
    cap = cv2.VideoCapture(str(video_path))
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
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
        try:
            X = [np.array(coord.iloc[f,2::3]) for coord in coords]
            X = [np.where(x==0., np.nan, x) for x in X]
            Y = [np.array(coord.iloc[f,3::3]) for coord in coords]
            Y = [np.where(y==0., np.nan, y) for y in Y]

            # Draw bounding box
            frame = draw_bounding_box(X, Y, frame)

            # Draw keypoints and skeleton
            frame = draw_keypts_skel(X, Y, frame, pose_model)
            
            # Save video and images
            if save_vid:
                writer.write(frame)
            if save_img:
                cv2.imwrite(str( img_pose_path / (video_result_path.stem+'_'+pose_model+'.'+str(f).zfill(5)+'.png' )), frame)

        except: 
            break
        f += 1
    cap.release()
    writer.release()


def detect_pose_fun(config_dict):
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
    - a video
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one csv file of joint coordinates per detected person
    - optionally json directory, image directory, video
    - a logs.txt file 
    '''
    
    # Retrieve parameters
    root_dir = os.getcwd()
    video_dir, video_file, result_dir, frame_rate = base_params(config_dict)
    pose_algo = config_dict.get('pose').get('pose_algo')
    
    load_pose = config_dict.get('pose_advanced').get('load_pose')
    save_vid = config_dict.get('pose_advanced').get('save_vid')
    save_img = config_dict.get('pose_advanced').get('save_img')
    interp_gap_smaller_than = config_dict.get('pose_advanced').get('interp_gap_smaller_than')
    
    show_plots = config_dict.get('pose_advanced').get('show_plots')
    do_filter = config_dict.get('pose_advanced').get('filter')
    filter_type = config_dict.get('pose_advanced').get('filter_type')
    butterworth_filter_order = config_dict.get('pose_advanced').get('butterworth').get('order')
    butterworth_filter_cutoff = config_dict.get('pose_advanced').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel = config_dict.get('pose_advanced').get('gaussian').get('sigma_kernel')
    loess_filter_kernel = config_dict.get('pose_advanced').get('loess').get('nb_values_used')
    median_filter_kernel = config_dict.get('pose_advanced').get('median').get('kernel_size')
    filter_options = (do_filter, filter_type, butterworth_filter_order, butterworth_filter_cutoff, frame_rate, gaussian_filter_kernel, loess_filter_kernel, median_filter_kernel)
    
    video_file_stem = video_file.stem
    video_path = video_dir / video_file
    video_result_path = result_dir / video_file

    if pose_algo == 'OPENPOSE':
        pose_model = config_dict.get('pose').get('OPENPOSE').get('openpose_model')
        json_path = result_dir / '_'.join((video_file_stem,pose_model,'json'))

        # Pose detection skipped if load existing json files
        if load_pose and len(list(json_path.glob('*')))>0:
            pass
        else:
            logging.info(f'Detecting 2D joint positions with OpenPose model {pose_model}, for {video_file}.')
            json_path.mkdir(parents=True, exist_ok=True)
            openpose_path = config_dict.get('pose').get('OPENPOSE').get('openpose_path')
            os.chdir(openpose_path)
            if platform =="win32":
                run_openpose_windows(video_path, json_path, pose_model)
            elif platform == "darwin":
                run_openpose_mac(video_path, json_path, pose_model)
            elif platform == "linux" or platform=="linux2":
                run_openpose_linux(video_path, json_path, pose_model)
            os.chdir(root_dir)
        
    # Sort people and save to csv, optionally display plot
        json_to_csv(json_path, frame_rate, pose_model, interp_gap_smaller_than, filter_options, show_plots)
        
    # Save images and files after reindentification
        if save_img and save_vid:
            logging.info(f'Saving images and video in {result_dir}.')
        if save_img and not save_vid:
            logging.info(f'Saving images in {result_dir}.')
        if not save_img and save_vid:
            logging.info(f'Saving video in {result_dir}.')
        if save_vid or save_img:
            save_imgvid_reID(video_path, video_result_path, save_vid, save_img, pose_model)
   
     
    elif pose_algo == 'BLAZEPOSE':
        model_complexity = config_dict.get('pose').get('BLAZEPOSE').get('model_complexity')
        Blazepose_runsave.blazepose_detec_func(input_file=video_path, save_images=save_img, to_json=True, save_video=save_vid, to_csv=True, output_folder=result_dir, model_complexity=model_complexity)

    logging.info(f'Done.')
