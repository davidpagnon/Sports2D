#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
#########################################
## SYNCHRONIZE CAMERAS                 ##
#########################################

    Post-synchronize your cameras in case they are not natively synchronized.

    For each camera, computes mean vertical speed for the chosen keypoints, 
    and find the time offset for which their correlation is highest. 

    Depending on the analysed motion, all keypoints can be taken into account, 
    or a list of them, or the right or left side.
    All frames can be considered, or only those around a specific time (typically, 
    the time when there is a single participant in the scene performing a clear vertical motion).
    Has also been successfully tested for synchronizing random walks with random walks.

    Keypoints whose likelihood is too low are filtered out; and the remaining ones are 
    filtered with a butterworth filter.

    INPUTS: 
    - json files from each camera folders
    - a Config.toml file
    - a skeleton model

    OUTPUTS: 
    - synchronized json files for each camera
'''

## INIT
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import patheffects
from scipy import signal
import json
import os
import glob
import fnmatch
import re
import shutil
from anytree import RenderTree
from anytree.importer import DictImporter
from matplotlib.widgets import TextBox, Button
import logging

from Pose2Sim.common import sort_stringlist_by_last_number, bounding_boxes, interpolate_zeros_nans
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# UI FUNCTIONS
# Global matplotlib settings - remove toolbar
plt.rcParams['toolbar'] = 'none'

# Define keypoint UI parameters
TITLE_SIZE = 12
LABEL_SIZE_KEYPOINTS = 8 # defined twice
BTN_WIDTH_KEYPOINTS = 0.16 # defined twice
BTN_HEIGHT = 0.04
BTN_Y = 0.02
CENTER_X = 0.5
SELECTED_COLOR = 'darkorange'
UNSELECTED_COLOR = 'blue'
NONE_COLOR = 'silver'
BTN_COLOR = 'white'  # Light gray
BTN_HOVER_COLOR = '#D3D3D3'  # Darker gray on hover

# Define person UI parameters
BACKGROUND_COLOR = 'white'
TEXT_COLOR = 'black'
CONTROL_COLOR = 'white'
CONTROL_HOVER_COLOR = '#D3D3D3'
SLIDER_COLOR = '#4682B4'
SLIDER_HIGHLIGHT_COLOR = 'moccasin'
SLIDER_EDGE_COLOR = (0.5, 0.5, 0.5, 0.5)
LABEL_SIZE_PERSON = 10
TEXT_SIZE = 9.5
BUTTON_SIZE = 10
TEXTBOX_WIDTH = 0.09
BTN_WIDTH_PERSON = 0.04
CONTROL_HEIGHT = 0.04
Y_POSITION = 0.1


def reset_styles(rect, annotation):
    '''
    Resets the visual style of a bounding box and its annotation to default.
    
    INPUTS:
    - rect: Matplotlib Rectangle object representing a bounding box
    - annotation: Matplotlib Text object containing the label for the bounding box
    '''

    rect.set_linewidth(1)
    rect.set_edgecolor('white')
    rect.set_facecolor((1, 1, 1, 0.1))
    annotation.set_fontsize(7)
    annotation.set_fontweight('normal')


def create_textbox(ax_pos, label, initial, UI_PARAMS):
    '''
    Creates a textbox widget with consistent styling.
    
    INPUTS:
    - ax_pos: List or tuple containing the position of the axes in the figure [left, bottom, width, height]
    - label: String label for the textbox
    - initial: Initial text value
    - UI_PARAMS: Dictionary containing UI parameters with colors and sizes settings
    
    OUTPUTS:
    - textbox: The created TextBox widget
    '''

    ax = plt.axes(ax_pos)
    ax.set_facecolor(UI_PARAMS['colors']['control'])
    textbox = TextBox(
        ax, 
        label,
        initial=initial,
        color=UI_PARAMS['colors']['control'],
        hovercolor=UI_PARAMS['colors']['control_hover'],
        label_pad=0.1
    )
    textbox.label.set_color(UI_PARAMS['colors']['text'])
    textbox.label.set_fontsize(UI_PARAMS['sizes']['label'])
    textbox.text_disp.set_color(UI_PARAMS['colors']['text'])
    textbox.text_disp.set_fontsize(UI_PARAMS['sizes']['text'])

    return textbox


## Handlers
def handle_ok_button(ui):
    '''
    Handles the OK button click event.
    
    INPUTS:
    - ui: Dictionary containing all UI elements and state
    - fps: Frames per second of the video
    - i: Current camera index
    - selected_id_list: List to store the selected person ID for each camera
    - approx_time_maxspeed: List to store the approximate time of maximum speed for each camera
    '''

    try:
        float(ui['controls']['main_time_textbox'].text)
        float(ui['controls']['time_RAM_textbox'].text)
        int(ui['controls']['person_textbox'].text)
        plt.close(ui['fig'])
    except ValueError:
        logging.warning('Invalid input in textboxes.')
        

def handle_person_change(text, selected_idx_container, person_textbox):
    '''
    Handles changes to the person selection text box.
    
    INPUTS:
    - text: Text from the person selection text box
    - selected_idx_container: List with one element to store the selected person's index
    - person_textbox: TextBox widget for displaying and editing the selected person number
    '''

    try:
        selected_idx_container[0] = int(text)
    except ValueError:
        person_textbox.set_val('0')
        selected_idx_container[0] = 0


def handle_frame_navigation(direction, frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                           pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                           time_range_around_maxspeed, fps, ui):
    '''
    Handles frame navigation (previous or next frame).
    
    INPUTS:
    - direction: Integer, -1 for previous frame, 1 for next frame
    - frame_textbox: TextBox widget for displaying and editing the frame number
    - search_around_frames: Frame ranges to search around for each camera
    - i: Current camera index
    - cap: Video capture object
    - ax_video: Axes for video display
    - frame_to_json: Mapping from frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dir_name: Name of the JSON directory for the current camera
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List of bounding boxes for detected persons
    - fig: The figure object to update
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the video
    - ui: Dictionary containing all UI elements and state
    '''

    time_val = float(frame_textbox.text.split(' ±')[0])
    current = round(time_val * fps)
    
    # Check bounds based on direction
    if (direction < 0 and current > search_around_frames[i][0]) or \
       (direction > 0 and current < search_around_frames[i][1]):
        next_frame = current + direction
        handle_frame_change(next_frame, frame_textbox, cap, ax_video, frame_to_json,
                            pose_dir, json_dir_name, rects, annotations, bounding_boxes_list,
                            fig, search_around_frames, i, time_range_around_maxspeed, fps, ui)


def handle_frame_change(frame_number, frame_textbox, cap, ax_video, frame_to_json, 
                        pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, 
                        fig, search_around_frames, i, time_range_around_maxspeed, fps, ui):
    '''
    Handles changes to the frame number text box.
    
    INPUTS:
    - text: Text from the frame number text box
    - frame_number: The current frame number
    - frame_textbox: TextBox widget for displaying and editing the frame number
    - cap: Video capture object
    - ax_video: Axes for video display
    - frame_to_json: Mapping from frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dir_name: Name of the JSON directory for the current camera
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List of bounding boxes for detected persons
    - fig: The figure object to update
    - search_around_frames: Frame ranges to search around for each camera
    - i: Current camera index
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the video
    - ui: Dictionary containing all UI elements and state
    '''

    if search_around_frames[i][0] <= frame_number <= search_around_frames[i][1]:
        # Update video frame first
        update_play(cap, ax_video.images[0], frame_number, frame_to_json, 
                    pose_dir, json_dir_name, rects, annotations, 
                    bounding_boxes_list, ax_video, fig)
        
        # Update UI elements
        frame_textbox.eventson = False
        new_time = frame_number / fps
        frame_textbox.set_val(f"{new_time:.2f} ±{time_range_around_maxspeed}")
        frame_textbox.eventson = True
        
        # Update slider and highlight
        ui['controls']['frame_slider'].set_val(frame_number)
        update_highlight(frame_number, time_range_around_maxspeed, fps, search_around_frames, i, ui['axes']['slider'], ui['controls'])
        fig.canvas.draw_idle()


def handle_key_press(event, frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                     pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                     time_range_around_maxspeed, fps, ui):
    '''
    Handles keyboard navigation through video frames.
    
    INPUTS:
    - event: Matplotlib keyboard event object
    - frame_textbox: TextBox widget for displaying and editing the frame number
    - search_around_frames: Frame ranges to search around for each camera
    - i: Current camera index
    - cap: Video capture object
    - ax_video: Axes for video display
    - frame_to_json: Mapping from frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dir_name: Name of the JSON directory for the current camera
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List of bounding boxes for detected persons
    - fig: The figure object to update
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the video
    - ui: Dictionary containing all UI elements and state
    '''

    direction = 0
    if event.key == 'left':
        direction = -1
    elif event.key == 'right':
        direction = 1
    if direction != 0:
        handle_frame_navigation(direction, frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                              pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                              time_range_around_maxspeed, fps, ui)


def handle_toggle_labels(keypoint_texts, containers, btn_toggle):
    '''
    Handle toggle labels button click.
    
    INPUTS:
    - event: Matplotlib event object
    - keypoint_texts: List of text objects for keypoint labels
    - containers: Dictionary of container objects
    - btn_toggle: Button object for toggling label visibility
    '''

    containers['show_labels'][0] = not containers['show_labels'][0]  # Toggle visibility state
    for text in keypoint_texts:
        text.set_visible(containers['show_labels'][0])
    # Update button text
    btn_toggle.label.set_text('Hide names' if containers['show_labels'][0] else 'Show names')
    plt.draw()


## Highlighters
def highlight_selected_box(rect, annotation):
    '''
    Highlights a selected rectangle and its annotation with bold orange style.
    
    INPUTS:
    - rect: Matplotlib Rectangle object to highlight
    - annotation: Matplotlib Text object to highlight
    '''

    rect.set_linewidth(2)
    rect.set_edgecolor(SELECTED_COLOR)
    rect.set_facecolor((1, 1, 1, 0.1))
    annotation.set_fontsize(8)
    annotation.set_fontweight('bold')


def highlight_hover_box(rect, annotation):
    '''
    Highlights a hovered rectangle and its annotation with yellow-orange style.
    
    INPUTS:
    - rect: Matplotlib Rectangle object to apply hover effect to
    - annotation: Matplotlib Text object to style for hover state
    '''

    rect.set_linewidth(2)
    rect.set_edgecolor(SELECTED_COLOR)
    rect.set_facecolor((1, 1, 0, 0.2))
    annotation.set_fontsize(8)
    annotation.set_fontweight('bold')


## on_family
def on_hover(event, fig, rects, annotations, bounding_boxes_list, selected_idx_container=None):
    '''
    Manages hover effects for bounding boxes in the video frame.
    
    INPUTS:
    - event: Matplotlib event object containing mouse position
    - fig: Matplotlib figure to update
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List of bounding box coordinates (x_min, y_min, x_max, y_max)
    - selected_idx_container: Optional container holding the index of the currently selected box
    '''

    if event.xdata is None or event.ydata is None:
        return

    # First reset all boxes to default style
    for idx, (rect, annotation) in enumerate(zip(rects, annotations)):
        if selected_idx_container and idx == selected_idx_container[0]:
            # Keep the selected box highlighted with bold white style
            highlight_selected_box(rect, annotation)
        else:
            reset_styles(rect, annotation)

    # Then apply hover effect to the box under cursor (even if it's selected)
    bounding_boxes_list = [bbox for bbox in bounding_boxes_list if np.all(np.isfinite(bbox)) and not np.any(np.isnan(bbox))]
    
    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes_list):
        if x_min <= event.xdata <= x_max and y_min <= event.ydata <= y_max:
            highlight_hover_box(rects[idx], annotations[idx])
            break

    fig.canvas.draw_idle()


def on_click(event, ax, bounding_boxes_list, selected_idx_container, person_textbox):
    '''
    Detects clicks on person bounding boxes and updates the selection state.
    
    INPUTS:
    - event: Matplotlib event object containing click information
    - ax: The axes object of the video frame
    - bounding_boxes_list: List of tuples containing bounding box coordinates (x_min, y_min, x_max, y_max)
    - selected_idx_container: List with one element to store the selected person's index
    - person_textbox: TextBox widget for displaying and editing the selected person number
    '''

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes_list):
        if x_min <= event.xdata <= x_max and y_min <= event.ydata <= y_max:
            selected_idx_container[0] = idx
            person_textbox.set_val(str(idx))  # Update the person number text box
            break


def on_slider_change(val, fps, controls, fig, search_around_frames, cam_index, ax_slider):
    '''
    Updates UI elements when the frame slider value changes.
    
    INPUTS:
    - val: The current slider value (frame number)
    - fps: Frames per second of the video
    - controls: Dictionary containing UI control elements
    - fig: Matplotlib figure to update
    - search_around_frames: Frame ranges to search within
    - cam_index: Current camera index
    - ax_slider: The slider axes object
    '''

    frame_number = int(val)
    main_time = frame_number / fps
    controls['main_time_textbox'].set_val(f"{main_time:.2f}")
    try:
        time_RAM = float(controls['time_RAM_textbox'].text)
    except ValueError:
        time_RAM = 0
    update_highlight(frame_number, time_RAM, fps, search_around_frames, cam_index, ax_slider, controls)
    fig.canvas.draw_idle()


def on_key(event, ui, fps, cap, frame_to_json, pose_dir, json_dirs_names, i, search_around_frames, bounding_boxes_list):
    '''
    Handles keyboard navigation through video frames.
    
    INPUTS:
    - event: Matplotlib keyboard event object
    - ui: Dictionary containing all UI elements and state
    - fps: Frames per second of the video
    - cap: Video capture object
    - frame_to_json: Mapping of frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dirs_names: List of JSON directory names
    - i: Current camera index
    - search_around_frames: Frame ranges to search around for each camera
    - bounding_boxes_list: List of bounding boxes for detected persons
    '''

    if event.key == 'left':
        handle_frame_navigation(-1, ui['controls']['main_time_textbox'], search_around_frames, i, cap, ui['ax_video'], frame_to_json,
                              pose_dir, json_dirs_names[i], ui['containers']['rects'], ui['containers']['annotations'], bounding_boxes_list, ui['fig'],
                              float(ui['controls']['time_RAM_textbox'].text), fps, ui)
    elif event.key == 'right':
        handle_frame_navigation(1, ui['controls']['main_time_textbox'], search_around_frames, i, cap, ui['ax_video'], frame_to_json,
                              pose_dir, json_dirs_names[i], ui['containers']['rects'], ui['containers']['annotations'], bounding_boxes_list, ui['fig'],
                              float(ui['controls']['time_RAM_textbox'].text), fps, ui)


## UI Update Functions
def update_highlight(current_frame, time_RAM, fps, search_around_frames, cam_index, ax_slider, controls):
    '''
    Updates the highlighted range on the frame slider.
    
    INPUTS:
    - current_frame: The current frame number
    - time_RAM: Time range in seconds to highlight around the current frame
    - fps: Frames per second of the video
    - search_around_frames: Valid frame range limits for the current camera
    - cam_index: Current camera index
    - ax_slider: The slider axes object
    - controls: Dictionary containing UI controls and state
    '''

    if 'range_highlight' in controls:
        controls['range_highlight'].remove()
    range_start = max(current_frame - time_RAM * fps, search_around_frames[cam_index][0])
    range_end = min(current_frame + time_RAM * fps, search_around_frames[cam_index][1])
    controls['range_highlight'] = ax_slider.axvspan(range_start, range_end, 
                                                  ymin=0.20, ymax=0.80,
                                                  color=SLIDER_HIGHLIGHT_COLOR, alpha=0.5, zorder=4)


def update_main_time(text, fps, search_around_frames, i, ui, cap, frame_to_json, pose_dir, json_dirs_names, bounding_boxes_list):
    '''
    Updates the UI based on changes to the main time textbox.
    
    INPUTS:
    - text: Text from the main time textbox
    - fps: Frames per second of the video
    - search_around_frames: Valid frame range limits for each camera
    - i: Current camera index
    - ui: Dictionary containing all UI elements and state
    - cap: Video capture object
    - frame_to_json: Mapping of frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dirs_names: List of JSON directory names
    - bounding_boxes_list: List of bounding boxes for detected persons
    '''

    try:
        main_time = float(text)
        frame_num = int(round(main_time * fps))
        frame_num = max(search_around_frames[i][0], min(frame_num, search_around_frames[i][1]))
        ui['controls']['frame_slider'].set_val(frame_num)
        update_frame(frame_num, fps, ui, frame_to_json, pose_dir, json_dirs_names, i, search_around_frames, bounding_boxes_list)
    except ValueError:
        pass


def update_time_RAM(text, fps, search_around_frames, i, ui):
    '''
    time_RAM = time_range_around_maxspeed
    Updates the highlight range based on changes to the time_RAM textbox.

    INPUTS:
    - text: Text from the time_RAM textbox
    - fps: Frames per second of the video
    - search_around_frames: Valid frame range limits for each camera
    - i: Current camera index
    - ui: Dictionary containing UI elements and controls
    '''
    
    try:
        time_RAM = float(text)
        if time_RAM < 0:
            time_RAM = 0
        frame_num = int(ui['controls']['frame_slider'].val)
        update_highlight(frame_num, time_RAM, fps, search_around_frames, i, ui['axes']['slider'], ui['controls'])
    except ValueError:
        pass


def draw_bounding_boxes_and_annotations(ax, bounding_boxes_list, rects, annotations):
    '''
    Draws the bounding boxes and annotations on the given axes.

    INPUTS:
    - ax: The axes object to draw on.
    - bounding_boxes_list: list of tuples. Each tuple contains (x_min, y_min, x_max, y_max) of a bounding box.
    - rects: List to store rectangle patches representing bounding boxes.
    - annotations: List to store text annotations for each bounding box.

    OUTPUTS:
    - None. Modifies rects and annotations in place.
    '''

    # Clear existing rectangles and annotations
    for items in [rects, annotations]:
            for item in items:
                item.remove()
            items.clear()

    # Draw bounding boxes and annotations
    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes_list):
        if not np.isfinite([x_min, y_min, x_max, y_max]).all():
            continue  # Skip invalid bounding boxes for solve issue(posx and posy should be finite values)

        rect = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1, edgecolor='white', facecolor=(1, 1, 1, 0.1),
            linestyle='-', path_effects=[patheffects.withSimplePatchShadow()], zorder=2
        ) # add shadow
        ax.add_patch(rect)
        rects.append(rect)

        annotation = ax.text(
            x_min, y_min - 10, f'Person {idx}', color='white', fontsize=7, fontweight='normal',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'), zorder=3
        )
        annotations.append(annotation)


def update_keypoint_selection(selected_keypoints, all_keypoints, keypoints_names, scatter, keypoint_texts, selected_text, btn_all_none,
                              SELECTED_COLOR, UNSELECTED_COLOR, NONE_COLOR):
    '''
    Updates the selected keypoints and their visualization on the scatter plot.
    
    INPUTS:
    - selected_keypoints: List of keypoints that are currently selected
    - all_keypoints: List of all available keypoints
    - keypoints_names: List of valid keypoint names
    - scatter: Scatter plot object to update
    - keypoint_texts: List of text objects for keypoint labels
    - selected_text: Text object displaying the currently selected keypoints
    - btn_all_none: Button object for toggling between "Select All" and "Select None"
    - SELECTED_COLOR: Color to use for selected keypoints
    - UNSELECTED_COLOR: Color to use for unselected keypoints
    - NONE_COLOR: Color to use for non-keypoint elements
    
    OUTPUTS:
    - None. Updates the visualization in place.
    '''
    # Update scatter colors
    colors = [
        SELECTED_COLOR if kp in selected_keypoints else UNSELECTED_COLOR if kp in keypoints_names else NONE_COLOR
        for kp in all_keypoints
    ]
    scatter.set_facecolors(colors)
    
    # Update text weights
    for text, kp in zip(keypoint_texts, all_keypoints):
        text.set_fontweight('bold' if kp in selected_keypoints else 'normal')
    
    # Update selected text and button label
    if selected_keypoints:
        text_parts = ['Selected: '] + [f'$\\bf{{{kp}}}$' if i == 0 else f', $\\bf{{{kp}}}$' for i, kp in enumerate(selected_keypoints)]
        selected_text.set_text(''.join(text_parts))
        btn_all_none.label.set_text('Select None')
    else:
        selected_text.set_text('Selected: None\nClick on keypoints to select them')
        btn_all_none.label.set_text('Select All')
    
    plt.draw()


def load_frame_and_bounding_boxes(cap, frame_number, frame_to_json, pose_dir, json_dir_name):
    '''
    Given a video capture object or a list of image files and a frame number, 
    load the frame (or image) and corresponding bounding boxes.

    INPUTS:
    - cap: cv2.VideoCapture object or list of image file paths.
    - frame_number: int. The frame number to load.
    - frame_to_json: dict. Mapping from frame numbers to JSON file names.
    - pose_dir: str. Path to the directory containing pose data.
    - json_dir_name: str. Name of the JSON directory for the current camera.

    OUTPUTS:
    - frame_rgb: The RGB image of the frame or image.
    - bounding_boxes_list: List of bounding boxes for the frame/image.
    '''

    # Case 1: If input is a video file (cv2.VideoCapture object)
    if isinstance(cap, cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            return None, []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Case 2: If input is a list of image file paths
    elif isinstance(cap, list):
        if frame_number >= len(cap):
            return None, []
        image_path = cap[frame_number]
        frame = cv2.imread(image_path)
        if frame is None:
            return None, []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    else:
        raise ValueError("Input must be either a video capture object or a list of image file paths.")

    # Get the corresponding JSON file for bounding boxes
    json_file_name = frame_to_json.get(frame_number)
    bounding_boxes_list = []
    if json_file_name:
        json_file_path = os.path.join(pose_dir, json_dir_name, json_file_name)
        bounding_boxes_list.extend(bounding_boxes(json_file_path))

    return frame_rgb, bounding_boxes_list


def update_frame(val, fps, ui, frame_to_json, pose_dir, json_dirs_names, i, search_around_frames, bounding_boxes_list):
    '''
    Synchronizes all UI elements when the frame number changes.
    
    INPUTS:
    - val: The current frame value from the slider
    - fps: Frames per second of the video
    - ui: Dictionary containing UI elements and controls
    - cap: Video capture object
    - frame_to_json: Mapping of frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dirs_names: List of JSON directory names
    - i: Current camera index
    - search_around_frames: Frame ranges to search around for each camera
    - bounding_boxes_list: List of bounding boxes for detected persons
    '''

    frame_num = int(val)
    main_time = frame_num / fps
    ui['controls']['main_time_textbox'].set_val(f"{main_time:.2f}")
    
    # Update yellow highlight position
    try:
        time_RAM = float(ui['controls']['time_RAM_textbox'].text)
    except ValueError:
        time_RAM = 0
        
    # Update highlight
    update_highlight(frame_num, time_RAM, fps, search_around_frames, i, ui['axes']['slider'], ui['controls'])
    
    # Update video frame and bounding boxes
    update_play(ui['cap'], ui['ax_video'].images[0], frame_num, frame_to_json, 
            pose_dir, json_dirs_names[i], ui['containers']['rects'], 
            ui['containers']['annotations'], bounding_boxes_list, 
            ui['ax_video'], ui['fig'])
    
    # Update canvas
    ui['fig'].canvas.draw_idle()


def update_play(cap, image, frame_number, frame_to_json, pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, ax, fig):
    '''
    Updates the video frame and bounding boxes for the given frame number.

    INPUTS:
    - cap: Video capture object or list of image file paths
    - image: The image object to update
    - frame_number: The frame number to display
    - frame_to_json: Mapping from frame numbers to JSON file names
    - pose_dir: Directory containing pose data
    - json_dir_name: Name of the JSON directory for the current camera
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List to store bounding box coordinates
    - ax: The axes object to draw on
    - fig: The figure object to update
    '''

    # Store the currently selected box index if any
    selected_idx = None
    for idx, rect in enumerate(rects):
        if rect.get_linewidth() > 1:  # If box is highlighted
            selected_idx = idx
            break

    frame_rgb, bounding_boxes_list_new = load_frame_and_bounding_boxes(cap, frame_number, frame_to_json, pose_dir, json_dir_name)
    if frame_rgb is None:
        return

    # Update image
    image.set_array(frame_rgb)
    
    # Clear existing boxes and annotations
    for rect in rects:
        rect.remove()
    for ann in annotations:
        ann.remove()
    rects.clear()
    annotations.clear()
    
    # Update bounding boxes list
    bounding_boxes_list.clear()
    bounding_boxes_list.extend(bounding_boxes_list_new)
    
    # Draw new boxes and annotations
    draw_bounding_boxes_and_annotations(ax, bounding_boxes_list, rects, annotations)
    
    # Restore highlight on the selected box if it still exists
    if selected_idx is not None and selected_idx < len(rects):
        highlight_selected_box(rects[selected_idx], annotations[selected_idx])
    
    fig.canvas.draw_idle()


def keypoints_ui(keypoints_to_consider, keypoints_names):
    '''
    Step 1: Initializes the UI for selecting keypoints.

    This function creates an interactive GUI for selecting keypoints. It displays
    a human figure with selectable keypoints, allows users to toggle keypoint names,
    select all or none, and confirm their selection. The GUI uses matplotlib for
    visualization and interaction.

    The function performs the following steps:
    1. Sets up the figure and axes for the GUI
    2. Defines keypoint positions and colors
    3. Creates interactive elements (scatter plot, buttons, text)
    4. Sets up event handlers for user interactions
    5. Displays the GUI and waits for user input
    6. Returns the list of selected keypoints

    INPUTS:
    - keypoints_names: List of strings. The names of the keypoints to select.

    OUTPUTS:
    - selected_keypoints: List of strings. The names of the selected keypoints.
    '''
    
    # Create figure
    fig = plt.figure(figsize=(6, 8), num='Synchronizing cameras')
    fig.patch.set_facecolor('white')

    # Keypoint selection area
    ax_keypoints = plt.axes([0.1, 0.2, 0.8, 0.7])
    ax_keypoints.set_facecolor('white')
    ax_keypoints.set_title('Select keypoints to synchronize on', fontsize=TITLE_SIZE, pad=10, color='black')
    
    # Define all keypoints and their positions
    all_keypoints = [
        'Hip', 'Neck', 'Head', 'Nose', 
        'RHip', 'RShoulder', 'RElbow', 'RWrist', 
        'RKnee', 'RAnkle', 'RSmallToe', 'RBigToe', 'RHeel', 
        'LHip', 'LShoulder', 'LElbow', 'LWrist', 
        'LKnee', 'LAnkle', 'LSmallToe', 'LBigToe', 'LHeel',
    ]
    keypoints_positions = {
        'Hip': (0.50, 0.42), 'Neck': (0.50, 0.75), 'Head': (0.50, 0.85), 'Nose': (0.53, 0.82), 
        'RHip': (0.42, 0.42), 'RShoulder': (0.40, 0.75), 'RElbow': (0.35, 0.65), 'RWrist': (0.25, 0.50),
        'LHip': (0.58, 0.42), 'LShoulder': (0.60, 0.75), 'LElbow': (0.65, 0.65), 'LWrist': (0.75, 0.50),
        'RKnee': (0.40, 0.25), 'RAnkle': (0.40, 0.05), 'RSmallToe': (0.35, 0.0), 'RBigToe': (0.42, 0.0), 'RHeel': (0.40, 0.02),
        'LKnee': (0.60, 0.25), 'LAnkle': (0.60, 0.05), 'LSmallToe': (0.65, 0.0), 'LBigToe': (0.58, 0.0), 'LHeel': (0.60, 0.02)
    }
    
    # Generate keypoint coordinates
    keypoints_x, keypoints_y = zip(*[keypoints_positions[name] for name in all_keypoints])
    
    # Set initial colors
    initial_colors = [SELECTED_COLOR if kp in keypoints_to_consider else UNSELECTED_COLOR if kp in keypoints_names else NONE_COLOR for kp in all_keypoints]
    
    # Create scatter plot
    selected_keypoints = keypoints_to_consider
    scatter = ax_keypoints.scatter(keypoints_x, keypoints_y, c=initial_colors, picker=True)
    
    # Add keypoint labels
    keypoint_texts = [ax_keypoints.text(x + 0.02, y, name, va='center', fontsize=LABEL_SIZE_KEYPOINTS, color='black', visible=False)
                      for x, y, name in zip(keypoints_x, keypoints_y, all_keypoints)]
    
    ax_keypoints.set_xlim(0, 1)
    ax_keypoints.set_ylim(-0.1, 1)
    ax_keypoints.axis('off')
    
    # Selected keypoints display area
    ax_selected = plt.axes([0.1, 0.08, 0.8, 0.04])
    ax_selected.axis('off')
    ax_selected.set_facecolor('black')
    text_parts = ['Selected: '] + [f'$\\bf{{{kp}}}$' if i == 0 else f', $\\bf{{{kp}}}$' for i, kp in enumerate(selected_keypoints)]
    selected_text = ax_selected.text(0.0, 0.5, ''.join(text_parts), 
                                    va='center', fontsize=BUTTON_SIZE, wrap=True, color='black')
    
    # Add buttons
    btn_all_none = plt.Button(plt.axes([CENTER_X - 1.5*BTN_WIDTH_KEYPOINTS - 0.01, BTN_Y, BTN_WIDTH_KEYPOINTS, BTN_HEIGHT]), 'Select All')
    btn_toggle = plt.Button(plt.axes([CENTER_X - BTN_WIDTH_KEYPOINTS/2, BTN_Y, BTN_WIDTH_KEYPOINTS, BTN_HEIGHT]), 'Show names')
    btn_ok = plt.Button(plt.axes([CENTER_X + 0.5*BTN_WIDTH_KEYPOINTS + 0.01, BTN_Y, BTN_WIDTH_KEYPOINTS, BTN_HEIGHT]), label='OK')
    btn_ok.label.set_fontweight('bold')
    
    # button colors
    for btn in [btn_all_none, btn_toggle, btn_ok]:
        btn.color = BTN_COLOR
        btn.hovercolor = BTN_HOVER_COLOR
    
    # Define containers for data
    containers = {
        'show_labels': [False],  # Label display status
        'selected_keypoints': selected_keypoints  # List of selected keypoints
    }

    # Connect button events
    btn_toggle.on_clicked(lambda event: handle_toggle_labels(keypoint_texts, containers, btn_toggle))
    btn_ok.on_clicked(lambda event: plt.close())
    btn_all_none.on_clicked(lambda event: (
        selected_keypoints.clear() if selected_keypoints else selected_keypoints.extend(keypoints_names),
        update_keypoint_selection(selected_keypoints, all_keypoints, keypoints_names, scatter, keypoint_texts, selected_text, btn_all_none,
        SELECTED_COLOR, UNSELECTED_COLOR, NONE_COLOR)
    )[-1])
    
    fig.canvas.mpl_connect('pick_event', lambda event: (
        (selected_keypoints.remove(all_keypoints[event.ind[0]]) 
         if all_keypoints[event.ind[0]] in selected_keypoints 
         else selected_keypoints.append(all_keypoints[event.ind[0]])) 
        if all_keypoints[event.ind[0]] in keypoints_names else None,
        update_keypoint_selection(selected_keypoints, all_keypoints, keypoints_names, scatter, keypoint_texts, selected_text, btn_all_none,
        SELECTED_COLOR, UNSELECTED_COLOR, NONE_COLOR)
    )[-1] if all_keypoints[event.ind[0]] in keypoints_names else None)
    
    plt.show()
    
    return selected_keypoints


def person_ui(frame_rgb, cam_name, frame_number, search_around_frames, time_range_around_maxspeed, fps, cam_index, frame_to_json, pose_dir, json_dirs_names):
    '''
    Step 2: Initializes the UI for person and frame selection.
    
    INPUTS:
    - frame_rgb: The initial RGB frame to display
    - cam_name: Name of the current camera
    - frame_number: Initial frame number to display
    - search_around_frames: Frame ranges to search around for each camera
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the video
    - cam_index: Index of the current camera
    - frame_to_json: Mapping from frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dirs_names: Names of JSON directories for each camera
    
    OUTPUTS:
    - ui: Dictionary containing all UI elements and state
    '''
    
    # Set up UI based on frame size and orientation
    frame_height, frame_width = frame_rgb.shape[:2]
    is_vertical = frame_height > frame_width
    
    # Calculate appropriate figure height based on video orientation
    if is_vertical:
        fig_height = frame_height / 250  # For vertical videos
    else:
        fig_height = max(frame_height / 300, 6)  # For horizontal videos
    
    fig = plt.figure(figsize=(8, fig_height), num=f'Synchronizing cameras')
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    # Adjust UI layout based on video orientation
    video_axes_height = 0.7 if is_vertical else 0.6
    slider_y = 0.15 if is_vertical else 0.2
    controls_y = Y_POSITION if is_vertical else 0.1
    lower_controls_y = controls_y - 0.05  # Y-coordinate for lower controls
    
    ax_video = plt.axes([0.1, 0.2, 0.8, video_axes_height])
    ax_video.imshow(frame_rgb)
    ax_video.axis('off')
    ax_video.set_facecolor(BACKGROUND_COLOR)

    # Create frame slider
    ax_slider = plt.axes([ax_video.get_position().x0, slider_y, ax_video.get_position().width, 0.04])
    ax_slider.set_facecolor(BACKGROUND_COLOR)
    frame_slider = Slider(
        ax=ax_slider,
        label='',
        valmin=search_around_frames[cam_index][0],
        valmax=search_around_frames[cam_index][1],
        valinit=frame_number,
        valstep=1,
        valfmt=None 
    )

    frame_slider.poly.set_edgecolor(SLIDER_EDGE_COLOR)
    frame_slider.poly.set_facecolor(SLIDER_COLOR)
    frame_slider.poly.set_linewidth(1)
    frame_slider.valtext.set_visible(False)

    # Add highlight for time range around max speed
    range_start = max(frame_number - time_range_around_maxspeed * fps, search_around_frames[cam_index][0])
    range_end = min(frame_number + time_range_around_maxspeed * fps, search_around_frames[cam_index][1])
    highlight = ax_slider.axvspan(range_start, range_end, 
                                  ymin=0.20, ymax=0.80,
                                  color=SLIDER_HIGHLIGHT_COLOR, alpha=0.5, zorder=4)

    # Save highlight for later updates
    controls = {'range_highlight': highlight}
    controls['frame_slider'] = frame_slider

    # Calculate positions for UI elements
    controls_y = Y_POSITION
    lower_controls_y = controls_y - 0.05  # Y-coordinate for lower controls
    
    # Create person textbox (centered)
    controls['person_textbox'] = create_textbox(
        [0.5 - TEXTBOX_WIDTH/2 + 0.17, controls_y, TEXTBOX_WIDTH, CONTROL_HEIGHT],
        f"{cam_name}: Synchronize on person number",
        '0',
        {'colors': {'background': BACKGROUND_COLOR, 'text': TEXT_COLOR, 'control': CONTROL_COLOR, 'control_hover': CONTROL_HOVER_COLOR},
         'sizes': {'label': LABEL_SIZE_PERSON, 'text': TEXT_SIZE}}
    )

    # Create main time textbox (lower left)
    controls['main_time_textbox'] = create_textbox(
        [0.5 - TEXTBOX_WIDTH/2 - 0.05, lower_controls_y, TEXTBOX_WIDTH, CONTROL_HEIGHT],
        'around time',
        f"{frame_number / fps:.2f}",
        {'colors': {'background': BACKGROUND_COLOR, 'text': TEXT_COLOR, 'control': CONTROL_COLOR, 'control_hover': CONTROL_HOVER_COLOR},
         'sizes': {'label': LABEL_SIZE_PERSON, 'text': TEXT_SIZE}}
    )

    # Create time RAM textbox (lower center)
    controls['time_RAM_textbox'] = create_textbox(
        [0.5 - TEXTBOX_WIDTH/2 + 0.07, lower_controls_y, TEXTBOX_WIDTH, CONTROL_HEIGHT],
        '±',
        f"{time_range_around_maxspeed:.2f}",
        {'colors': {'background': BACKGROUND_COLOR, 'text': TEXT_COLOR, 'control': CONTROL_COLOR, 'control_hover': CONTROL_HOVER_COLOR},
         'sizes': {'label': LABEL_SIZE_PERSON, 'text': TEXT_SIZE}}
    )
    
    # Create OK button (lower right)
    ok_ax = plt.axes([0.5 - TEXTBOX_WIDTH/2 + 0.17, lower_controls_y, BTN_WIDTH_PERSON * 1.5, CONTROL_HEIGHT])
    ok_ax.set_facecolor(CONTROL_COLOR)
    controls['btn_ok'] = Button(
        ok_ax, 
        label='OK', 
        color=CONTROL_COLOR,
        hovercolor=CONTROL_HOVER_COLOR
    )
    controls['btn_ok'].label.set_color(TEXT_COLOR)
    controls['btn_ok'].label.set_fontsize(BUTTON_SIZE)
    controls['btn_ok'].label.set_fontweight('bold')
    
    # Initialize containers for dynamic elements
    containers = {
        'rects': [],
        'annotations': [],
        'bounding_boxes_list': [],
        'selected_idx': [0]
    }

    # Create UI dictionary
    ui = {
        'fig': fig,
        'ax_video': ax_video,
        'controls': controls,
        'containers': containers,
        'axes': {'slider': ax_slider}
    }

    # Connect hover event
    fig.canvas.mpl_connect('motion_notify_event', 
        lambda event: on_hover(event, fig, containers['rects'], 
                             containers['annotations'], 
                             containers['bounding_boxes_list'],
                             containers['selected_idx']))

    # Connect event handlers using lambda
    frame_slider.on_changed(lambda val: on_slider_change(val, fps, controls, fig, search_around_frames, cam_index, ax_slider))
    controls['main_time_textbox'].on_submit(lambda text: update_main_time(text, fps, search_around_frames, cam_index, ui, ui['cap'], frame_to_json, pose_dir, json_dirs_names, containers['bounding_boxes_list']))
    controls['time_RAM_textbox'].on_submit(lambda text: update_time_RAM(text, fps, search_around_frames, cam_index, ui))

    return ui


def select_persons_on_vid(vid_or_img_files, all_pose_coords, nb_persons_to_detect='all'):
    '''
    # This function manages the process of selecting keypoints and persons for each camera.
    # It performs two main steps:
    # 1. Select keypoints to consider for all cameras
    # 2. For each camera, select a person ID and a specific frame

    INPUTS:
    - vid_or_img_files: path or list of paths. The video or image files
    - all_pose_coords: numpy array, of shape (Npersons, Nframes, Nkpts, Ndims). The pose coordinates for all persons in the video

    OUTPUTS:
    - selected_persons: List of selected person IDs
    '''
    
    if nb_persons_to_detect == 'all' or nb_persons_to_detect > all_pose_coords.shape[0]:
        nb_persons_to_detect = all_pose_coords.shape[0]

    try: # video
        cap = cv2.VideoCapture(vid_or_img_files)
        if not cap.isOpened():
            raise
    except: # images
        pass

    frame_count = 0
    while cap.isOpened() or frame_count < len(vid_or_img_files):
        if 'cap' in locals(): # video
            cap.read()
            if frame_count < frame_range[0]:
                frame_count += 1
                continue
        else:
            cap = cv2.VideoCapture(vid_or_img_files[frame_count])
        if not cap.isOpened():
            break





# REMPLIR ICI

    # selected_id_list = []

    # frame_rgb, bounding_boxes_list = load_frame_and_bounding_boxes(cap, frame_number, frame_to_json, pose_dir, json_dirs_names[i])
    # if frame_rgb is None:
    #     logging.warning(f'Cannot read frame {frame_number} from video {vid_or_img_files}')
    #     selected_id_list.append(None)
    #     time_RAM_list.append(time_range_around_maxspeed)  # Use default value for missing cameras
    #     if isinstance(cap, cv2.VideoCapture):
    #         cap.release()
    #     continue
    
    # # Initialize UI for person/frame selection only (no keypoint selection)
    # ui = person_ui(frame_rgb, cam_name, frame_number, search_around_frames, time_range_around_maxspeed, fps, i, frame_to_json, pose_dir, json_dirs_names)
    # ui['cap'] = cap
    
    # # Draw initial bounding boxes
    # draw_bounding_boxes_and_annotations(ui['ax_video'], bounding_boxes_list, 
    #                                     ui['containers']['rects'], 
    #                                     ui['containers']['annotations'])
    # ui['containers']['bounding_boxes_list'] = bounding_boxes_list 
    # ui['controls']['frame_slider'].on_changed(lambda val: update_frame(val, fps, ui, frame_to_json, pose_dir, json_dirs_names, i, search_around_frames, bounding_boxes_list))
    
    # # Update main time textbox to also update slider
    # ui['controls']['main_time_textbox'].on_submit(lambda text: update_main_time(text, fps, search_around_frames, i, ui, ui['cap'], frame_to_json, pose_dir, json_dirs_names, ui['containers']['bounding_boxes_list']))
    
    # # Update time_RAM textbox to update highlight
    # ui['controls']['time_RAM_textbox'].on_submit(lambda text: update_time_RAM(text, fps, search_around_frames, i, ui))

    # # Add click event handler
    # ui['fig'].canvas.mpl_connect('button_press_event', 
    #     lambda event: on_click(event, ui['ax_video'], bounding_boxes_list, 
    #                             ui['containers']['selected_idx'], ui['controls']['person_textbox']))

    # # Event handlers connection
    # ui['controls']['person_textbox'].on_submit(
    #     lambda text: handle_person_change(text, ui['containers']['selected_idx'], ui['controls']['person_textbox']))

    # # OK button
    # btn_ok = ui['controls']['btn_ok']
    # btn_ok.on_clicked(lambda event: handle_ok_button(ui))

    # # Keyboard navigation
    # ui['fig'].canvas.mpl_connect('key_press_event', lambda event: handle_key_press(event, ui['controls']['main_time_textbox'],
    #                         search_around_frames, i, ui['cap'], ui['ax_video'], frame_to_json, pose_dir,
    #                         json_dirs_names[i], ui['containers']['rects'], ui['containers']['annotations'], bounding_boxes_list, ui['fig'],
    #                         time_range_around_maxspeed, fps, ui))

    # # Show plot and wait for user input
    # plt.show()
    # cap.release()

    # # Store selected values after OK button is clicked
    # selected_id_list.append(int(ui['controls']['person_textbox'].text))
    # current_frame = int(round(float(ui['controls']['main_time_textbox'].text) * fps))
    # approx_time_maxspeed.append(current_frame / fps)
    # current_time_RAM = float(ui['controls']['time_RAM_textbox'].text)
    # time_RAM_list.append(current_time_RAM)  # Store the time_RAM for this camera
    # logging.info(f'--> Camera #{i}: selected person #{ui["controls"]["person_textbox"].text} at time {current_frame / fps:.2f} ± {current_time_RAM:.2f} s')

    # return selected_id_list, keypoints_to_consider, approx_time_maxspeed, time_RAM_list





