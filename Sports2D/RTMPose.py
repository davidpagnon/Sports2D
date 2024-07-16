import cv2
import numpy as np
from rtmlib import PoseTracker, BodyWithFeet, draw_skeleton
import pandas as pd
from Sports2D.compute_angles import (
    joint_angles_series_from_points, 
    segment_angles_series_from_points, 
    overlay_angles, 
    display_figures_fun
)
from Sports2D.detect_pose import (
    draw_bounding_box,
    draw_keypts_skel,
    save_imgvid_reID,
    json_to_csv
)
from Sports2D.Utilities import filter
import math
import os
import glob
import logging
from tqdm import tqdm
import json
import re
from datetime import datetime
from pathlib import Path
# Define joint angle parameters
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

# Save keypoints and scores to a JSON file in OpenPose format
def save_to_openpose(json_file_path, keypoints, scores):
    nb_detections = len(keypoints)
    detections = []
    for i in range(nb_detections):
        keypoints_with_confidence_i = []
        for kp, score in zip(keypoints[i], scores[i]):
            keypoints_with_confidence_i.extend([kp[0].item(), kp[1].item(), score.item()])
        detections.append({
            "person_id": [-1],
            "pose_keypoints_2d": keypoints_with_confidence_i,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        })
    
    json_output = {"version": 1.3, "people": detections}
    
    json_output_dir = os.path.abspath(os.path.join(json_file_path, '..'))
    if not os.path.isdir(json_output_dir): 
        os.makedirs(json_output_dir)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)

# Sort list of strings with numbers in natural order
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# Process video input for pose estimation
def process_video(video_path, pose_tracker, tracking, output_format, save_video, save_images, display_detection, frame_range):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.read()[0]:
            raise ValueError
    except:
        raise NameError(f"{video_path} is not a video. Images must be put in one subdirectory per camera.")
    
    pose_dir = os.path.abspath(os.path.join(video_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): 
        os.makedirs(pose_dir)
    video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
    json_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_json')
    output_video_path = os.path.join(pose_dir, f'{video_name_wo_ext}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_img')
    
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
        
    if display_detection:
        cv2.namedWindow(f"Pose Estimation {os.path.basename(video_path)}", cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

    frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_range = frame_range if frame_range else [0, total_frames]
    with tqdm(total=total_frames, desc=f'Processing {os.path.basename(video_path)}') as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            if frame_idx in range(*f_range):
                keypoints, scores = pose_tracker(frame)

                if tracking:
                    max_id = max(pose_tracker.track_ids_last_frame)
                    num_frames, num_points, num_coordinates = keypoints.shape
                    keypoints_filled = np.zeros((max_id + 1, num_points, num_coordinates))
                    scores_filled = np.zeros((max_id + 1, num_points))
                    keypoints_filled[pose_tracker.track_ids_last_frame] = keypoints
                    scores_filled[pose_tracker.track_ids_last_frame] = scores
                    keypoints = keypoints_filled
                    scores = scores_filled

                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores)

                if display_detection or save_video or save_images:
                    img_show = frame.copy()
                    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)
                
                if display_detection:
                    cv2.imshow(f"Pose Estimation {os.path.basename(video_path)}", img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save_video:
                    out.write(img_show)

                if save_images:
                    if not os.path.isdir(img_output_dir): 
                        os.makedirs(img_output_dir)
                    cv2.imwrite(os.path.join(img_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.png'), img_show)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()

# Process image input for pose estimation
def process_images(image_folder_path, vid_img_extension, pose_tracker, tracking, output_format, fps, save_video, save_images, display_detection, frame_range):
    pose_dir = os.path.abspath(os.path.join(image_folder_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): 
        os.makedirs(pose_dir)
    json_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_json')
    output_video_path = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_img')

    image_files = glob.glob(os.path.join(image_folder_path, '*' + vid_img_extension))
    image_files.sort(key=natural_sort_key)

    if save_video:
        logging.warning('Using default framerate of 60 fps.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        W, H = cv2.imread(image_files[0]).shape[:2][::-1]
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    if display_detection:
        cv2.namedWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", cv2.WINDOW_NORMAL)
    
    f_range = frame_range if frame_range else [0, len(image_files)]
    for frame_idx, image_file in enumerate(tqdm(image_files, desc=f'\nProcessing {os.path.basename(img_output_dir)}')):
        if frame_idx in range(*f_range):
            try:
                frame = cv2.imread(image_file)
            except:
                raise NameError(f"{image_file} is not an image. Videos must be put in the video directory, not in subdirectories.")
            
            keypoints, scores = pose_tracker(frame)

            if tracking:
                max_id = max(pose_tracker.track_ids_last_frame)
                num_frames, num_points, num_coordinates = keypoints.shape
                keypoints_filled = np.zeros((max_id + 1, num_points, num_coordinates))
                scores_filled = np.zeros((max_id + 1, num_points))
                keypoints_filled[pose_tracker.track_ids_last_frame] = keypoints
                scores_filled[pose_tracker.track_ids_last_frame] = scores
                keypoints = keypoints_filled
                scores = scores_filled            
            
            if 'openpose' in output_format:
                json_file_path = os.path.join(json_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.json")
                save_to_openpose(json_file_path, keypoints, scores)

            if display_detection or save_video or save_images:
                img_show = frame.copy()
                img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)

            if display_detection:
                cv2.imshow(f"Pose Estimation {os.path.basename(image_folder_path)}", img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video:
                out.write(img_show)

            if save_images:
                if not os.path.isdir(img_output_dir): 
                    os.makedirs(img_output_dir)
                cv2.imwrite(os.path.join(img_output_dir, f'{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.png'), img_show)

    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()

def set_webcam_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def adjust_text_scale(frame, base_scale=0.25, base_thickness=1):
    height, width, _ = frame.shape
    scale = base_scale * (width / 640)
    thickness = int(base_thickness * (width / 640))
    return scale, thickness

def process_webcam(pose_tracker, openpose_skeleton, joint_angles, segment_angles, save_video, save_images, interp_gap_smaller_than, filter_options, show_plots):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution 
    width = 1980
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Set resolution to: {width}x{height}")

    cv2.namedWindow("Real-time Pose Estimation", cv2.WINDOW_NORMAL)
    kpt_thr = 0.3

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0 or frame_rate is None:
        frame_rate = 30
    print(f"Webcam frame rate: {frame_rate}")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.path.join(os.getcwd(), f'realtime_pose_output_{current_time}'))
    output_dir.mkdir(parents=True, exist_ok=True)

    json_output_dir = output_dir / 'json'
    json_output_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if save_video:
        video_output_path = str(output_dir / 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, frame_rate, (width, height))
        if not video_writer.isOpened():
            print("Error: Could not create video writer.")
            save_video = False

    if save_images:
        img_output_dir = output_dir / 'images'
        img_output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0

    # Joint angles and segment angles lists
    joint_angles_data = []
    segment_angles_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1
            scale, thickness = adjust_text_scale(frame)
            keypoints, scores = pose_tracker(frame)
            
            img_show = frame.copy()
            img_show = draw_skeleton(img_show, keypoints, scores, openpose_skeleton=openpose_skeleton, kpt_thr=kpt_thr)
            
            # X = [kp[:, 0] for kp in keypoints]
            # Y = [kp[:, 1] for kp in keypoints]
            # img_show = draw_bounding_box(X, Y, img_show)

            json_file_path = json_output_dir / f'frame_{frame_count:06d}.json'
            save_to_openpose(json_file_path, keypoints, scores)

            df_angles_list_frame = []
            for person_idx, (person_keypoints, person_scores) in enumerate(zip(keypoints, scores)):
                if np.sum(person_scores >= kpt_thr) < len(person_keypoints) * 0.3:
                    continue  

                df_points = convert_keypoints_to_dataframe(person_keypoints, person_scores)
                
                joint_angle_values = {}
                segment_angle_values = {}
                
                for joint in joint_angles:
                    angle_params = get_joint_angle_params(joint)
                    if angle_params:
                        angle = joint_angles_series_from_points(df_points, angle_params, kpt_thr)
                        if angle is not None:
                            joint_angle_values[joint] = angle[0]
                
                for segment in segment_angles:
                    angle_params = get_segment_angle_params(segment)
                    if angle_params:
                        angle = segment_angles_series_from_points(df_points, angle_params, segment, kpt_thr)
                        if angle is not None:
                            segment_angle_values[segment] = angle[0]
                
                df_angles_list_frame.append({**joint_angle_values, **segment_angle_values})
                
                # Add joint and segment angles to the lists
                joint_angles_data.append(joint_angle_values)
                segment_angles_data.append(segment_angle_values)

            img_show = overlay_angles(img_show, df_angles_list_frame, keypoints, scores, kpt_thr)

            cv2.imshow("Real-time Pose Estimation", img_show)
            
            if save_video and video_writer is not None:
                video_writer.write(img_show)

            if save_images:
                cv2.imwrite(str(img_output_dir / f'frame_{frame_count:06d}.png'), img_show)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"Processed frame {frame_count}", end='\r')

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    print(f"\nTotal frames processed: {frame_count}")

    # Save joint angles and segment angles to CSV files
    joint_angles_df = pd.DataFrame(joint_angles_data)
    segment_angles_df = pd.DataFrame(segment_angles_data)

    joint_angles_df.to_csv(output_dir / 'joint_angles.csv', index=False)
    segment_angles_df.to_csv(output_dir / 'segment_angles.csv', index=False)

    # JSON to CSV conversion
    pose_model = 'HALPE_26'
    
    if filter_options[0]:
        filter_options = list(filter_options)
        filter_options[4] = frame_rate
        filter_options = tuple(filter_options)
    
    json_to_csv(json_output_dir, frame_rate, pose_model, interp_gap_smaller_than, filter_options, show_plots)

    print(f"Output saved to: {output_dir}")
    print(f"JSON files: {json_output_dir}")
    if save_video:
        print(f"Video file: {video_output_path}")
    if save_images:
        print(f"Image files: {img_output_dir}")
    print(f"CSV files: {json_output_dir.parent}")

def compute_joint_angles(df_points, joint_angles, kpt_thr):
    return {joint: joint_angles_series_from_points(df_points, get_joint_angle_params(joint), kpt_thr)[0]
            for joint in joint_angles
            if get_joint_angle_params(joint) and joint_angles_series_from_points(df_points, get_joint_angle_params(joint), kpt_thr) is not None}

def compute_segment_angles(df_points, segment_angles, kpt_thr):
    return {segment: segment_angles_series_from_points(df_points, get_segment_angle_params(segment), segment, kpt_thr)[0]
            for segment in segment_angles
            if get_segment_angle_params(segment) and segment_angles_series_from_points(df_points, get_segment_angle_params(segment), segment, kpt_thr) is not None}



# Convert keypoints to DataFrame
def convert_keypoints_to_dataframe(keypoints, scores):
    data = []
    for kp, score in zip(keypoints, scores):
        data.extend([kp[0], kp[1], score])
    
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel",
        "neck", "left_palm", "right_palm"
    ]
    
    columns = []
    for name in keypoint_names:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_score"])
    
    return pd.DataFrame([data], columns=columns)

# Compute real-time angles
def compute_angles_realtime(df_points, joint_angles, segment_angles):
    joint_angle_values = {}
    segment_angle_values = {}

    angle_dict = get_angle_dict()

    for joint in joint_angles:
        angle_params = angle_dict['joint'].get(joint)
        if angle_params:
            angle = joint_angles_series_from_points(df_points, angle_params)
            joint_angle_values[joint] = angle[0]

    for segment in segment_angles:
        angle_params = angle_dict['segment'].get(segment)
        if angle_params:
            angle = segment_angles_series_from_points(df_points, angle_params, segment)
            segment_angle_values[segment] = angle[0]

    return joint_angle_values, segment_angle_values

# Get angle dictionary
def get_angle_dict():
    return {
        'joint': {
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
        },
        'segment': {
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
    }

# Display angles on frame
# def display_angles(frame, joint_angles, segment_angles, keypoints, scores, kpt_thr, font_scale, thickness):
#     # Calculate positions based on frame size
#     height, width, _ = frame.shape
#     left_x = int(0.01 * width)  # 1% from the left
#     right_x = int(0.83 * width)  # 90% from the left
#     base_y = int(0.03 * height)  # 3% from the top
#     y_step = int(0.05 * height)  # Step size for each line

#     for i, (joint, angle) in enumerate(joint_angles.items()):
#         y_pos = base_y + i * y_step
#         cv2.putText(frame, f"{joint}: {angle:.1f}", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
#         draw_joint_angle(frame, joint, angle, keypoints, scores, kpt_thr)

#     for i, (segment, angle) in enumerate(segment_angles.items()):
#         y_pos = base_y + i * y_step
#         cv2.putText(frame, f"{segment}: {angle:.1f}", (right_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
#         draw_segment_angle(frame, segment, angle, keypoints, scores, kpt_thr)

# # Display angles on frame
# def draw_joint_angle(frame, joint, angle, keypoints, scores, kpt_thr):
#     thickness = 4
#     radius = 30 

#     joint_to_keypoints = {
#         "Right ankle": [22, 20, 16, 14],
#         "Left ankle": [19, 17, 15, 13],
#         "Right knee": [12, 14, 16],
#         "Left knee": [11, 13, 15],
#         "Right hip": [14, 12, 6],
#         "Left hip": [13, 11, 5],
#         "Right shoulder": [12, 6, 8],
#         "Left shoulder": [11, 5, 7],
#         "Right elbow": [6, 8, 10],
#         "Left elbow": [5, 7, 9],
#     }

#     if joint in joint_to_keypoints:
#         pts = [keypoints[i] for i in joint_to_keypoints[joint]]
#         scores_pts = [scores[i] for i in joint_to_keypoints[joint]]
#         if all(score >= kpt_thr for score in scores_pts):
#             pt1, pt2, pt3 = pts[-3:]
#             draw_angle_arc(frame, pt1, pt2, pt3, angle, thickness, radius)

# def visualize_angles(frame, keypoints, scores, joint_angles, segment_angles, kpt_thr=0.5):

#     # 관절 각도 시각화
#     for joint, angle in joint_angles.items():
#         draw_joint_angle(frame, joint, angle, keypoints, scores, kpt_thr)

#     # 세그먼트 각도 시각화
#     for segment, angle in segment_angles.items():
#         draw_segment_angle(frame, segment, abs(angle), keypoints, scores, kpt_thr)

#     return frame

# # Draw angle arc
# def draw_angle_arc(frame, pt1, pt2, pt3, angle, thickness=2, radius=20, color=(0, 255, 0)):
#     pt1 = tuple(map(int, pt1))
#     pt2 = tuple(map(int, pt2))
#     pt3 = tuple(map(int, pt3))

#     # 벡터 계산
#     vec1 = np.array(pt1) - np.array(pt2)
#     vec2 = np.array(pt3) - np.array(pt2)
    
#     # 두 벡터 간의 코사인 각도 계산
#     cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#     angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
#     # 시작 각도와 끝 각도 계산
#     start_angle = np.arctan2(vec1[1], vec1[0])
#     end_angle = np.arctan2(vec2[1], vec2[0])
    
#     # 시작 각도가 끝 각도보다 클 경우 교환
#     if start_angle > end_angle:
#         start_angle, end_angle = end_angle, start_angle
    
#     # 원호 그리기
#     cv2.ellipse(frame, pt2, (radius, radius), 0, 
#                 np.degrees(start_angle), np.degrees(end_angle), 
#                 color, thickness)
    
#     # 각도 값 텍스트로 표시
#     text_pos = (pt2[0] + radius + 5, pt2[1] - radius - 5)
#     cv2.putText(frame, f"{angle:.1f}°", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# # Draw segment angle
# def draw_segment_angle(frame, segment, angle, keypoints, scores, kpt_thr):
#     thickness = 4
#     length = 40

#     segment_to_keypoints = {
#         "Right foot": [22, 20],
#         "Left foot": [19, 17],
#         "Right shank": [14, 16],
#         "Left shank": [13, 15],
#         "Right thigh": [12, 14],
#         "Left thigh": [11, 13],
#         "Trunk": [6, 12],
#         "Right arm": [6, 8],
#         "Left arm": [5, 7],
#         "Right forearm": [8, 10],
#         "Left forearm": [7, 9],
#     }

#     if segment in segment_to_keypoints:
#         pt1, pt2 = [keypoints[i] for i in segment_to_keypoints[segment]]
#         score1, score2 = [scores[i] for i in segment_to_keypoints[segment]]
#         if score1 >= kpt_thr and score2 >= kpt_thr:
#             draw_angle_line(frame, pt1, pt2, angle, thickness, length)

# # Draw angle line
# def draw_angle_line(frame, pt1, pt2, angle, thickness, length):
#     pt1 = tuple(map(int, pt1))
#     pt2 = tuple(map(int, pt2))

#     dx = pt2[0] - pt1[0]
#     dy = pt2[1] - pt1[1]
    
#     end_point = (int(pt1[0] + length), pt1[1])
#     cv2.line(frame, pt1, end_point, (255, 0, 0), thickness)
    
#     segment_end = (int(pt1[0] + length * math.cos(math.radians(angle))), int(pt1[1] + length * math.sin(math.radians(angle))))
#     cv2.line(frame, pt1, segment_end, (255, 0, 0), thickness)

# Main estimator function
def rtm_estimator(config_dict):
    data_type = config_dict['pose_demo']['data_type']
    device = config_dict['pose_demo']['device']
    backend = config_dict['pose_demo']['backend']
    det_frequency = config_dict['pose_demo']['det_frequency']
    mode = config_dict['pose_demo']['mode']
    tracking = config_dict['pose_demo']['tracking']
    openpose_skeleton = config_dict['pose_demo']['to_openpose']
    save_video = config_dict['pose_demo']['save_video']
    print(f"save video: {save_video}")
    save_images = config_dict['pose_demo']['save_images']
    print(f"save images: {save_images}")
    display_detection = config_dict['pose_demo']['display_detection']
    frame_range = config_dict['pose_demo'].get('frame_range', [])
    output_format = config_dict['pose_demo']['output_format']

    joint_angles = config_dict['compute_angles']['joint_angles']
    segment_angles = config_dict['compute_angles']['segment_angles']

    interp_gap_smaller_than = config_dict.get('pose_advanced').get('interp_gap_smaller_than')
    
    show_plots = config_dict.get('pose_advanced').get('show_plots')
    do_filter = config_dict.get('pose_advanced').get('filter')
    filter_type = config_dict.get('pose_advanced').get('filter_type')
    butterworth_filter_order = config_dict.get('pose_advanced').get('butterworth').get('order')
    butterworth_filter_cutoff = config_dict.get('pose_advanced').get('butterworth').get('cut_off_frequency')
    gaussian_filter_kernel = config_dict.get('pose_advanced').get('gaussian').get('sigma_kernel')
    loess_filter_kernel = config_dict.get('pose_advanced').get('loess').get('nb_values_used')
    median_filter_kernel = config_dict.get('pose_advanced').get('median').get('kernel_size')

    # frame rate = none
    filter_options = (do_filter, filter_type, butterworth_filter_order, butterworth_filter_cutoff, None, gaussian_filter_kernel, loess_filter_kernel, median_filter_kernel)
    
    ModelClass = BodyWithFeet  # Model class used in RTMPose

    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=openpose_skeleton)

    if data_type == "webcam":
        process_webcam(pose_tracker, openpose_skeleton, joint_angles, segment_angles, save_video, save_images, interp_gap_smaller_than, filter_options, show_plots)
    elif data_type == "video":
        video_path = config_dict['pose_demo']['video_path']
        process_video(video_path, pose_tracker, tracking, output_format, save_video, save_images, 
                      display_detection, frame_range, interp_gap_smaller_than, filter_options, show_plots)
    elif data_type == "image":
        image_folder_path = config_dict['pose_demo']['image_folder_path']
        vid_img_extension = config_dict['pose_demo']['vid_img_extension']
        fps = config_dict['pose_demo'].get('fps', 60)
        process_images(image_folder_path, vid_img_extension, pose_tracker, tracking, output_format, 
                       fps, save_video, save_images, display_detection, frame_range, 
                       interp_gap_smaller_than, filter_options, show_plots)
    else:
        raise ValueError("Invalid data_type. Must be 'webcam', 'video', or 'image'.")

if __name__ == "__main__":
    import argparse
    import toml

    parser = argparse.ArgumentParser(description='Run RTMPose estimation')
    parser.add_argument('--config', type=str, default='Config_demo.toml', help='Path to the configuration file')
    args = parser.parse_args()

    # Read configuration file
    with open(args.config, 'r') as f:
        config = toml.load(f)

    # Run the estimator
    rtm_estimator(config)
