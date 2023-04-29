<!-- 
[![Continuous integration](https://github.com/davidpagnon/sports2d/actions/workflows/continuous-integration.yml/badge.svg?branch=main)](https://github.com/davidpagnon/sports2d/actions/workflows/continuous-integration.yml)
[![PyPI version](https://badge.fury.io/py/Sports2D.svg)](https://badge.fury.io/py/Sports2D) \
[![Downloads](https://pepy.tech/badge/sports2d)](https://pepy.tech/project/sports2d)
[![Stars](https://badgen.net/github/stars/davidpagnon/sports2d)](https://github.com/davidpagnon/sports2d/stargazers)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub issues](https://img.shields.io/github/issues/davidpagnon/sports2d)](https://github.com/davidpagnon/sports2d/issues)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/davidpagnon/sports2d)](https://GitHub.com/davidpagnon/sports2d/issues?q=is%3Aissue+is%3Aclosed) 
-->

# Sports2D

`Sports2D` lets you compute 2D joint and segment angles from a single video. 

If you need more accurate results and want to analyze the movements of several persons, you can install and use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md) instead of the default BlazePose.

Apps with GUI will be released for Windows, Linux, MacOS, as well as Android and iOS.
Mobile versions will only support exploratory angle computation from BlazePose, hence less accurately and tunable.

<!-- GIF HERE -->


***/!\ Warning /!\***

- The angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect.
- It will only lead to acceptable results if the persons move in the 2D plane (sagittal plane).
- The persons need to be filmed as perpendicularly as possible from their side.
If you need research-grade markerless joint kinematics, consider using several cameras,
and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
https://github.com/perfanalytics/pose2sim


## Contents
1. [Installation and Demonstration](#installation-and-demonstration)
   1. [Installation](#installation)
   2. [Demonstration: Detect pose and compute 2D angles](#demonstration-detect-pose-and-compte-2d-angles)
2. [Go further](#go-further)
   1. [With OpenPose and other models](#with-openpose-and-other-models)
   2. [Advanced-settings: Pose](#advanced-settings-pose)
   3. [Advanced-settings: Angles](#advanced-settings-angles)
3. [How to cite and how to contribute](#how-to-cite-and-how-to-contribute)
   1. [How to cite](#how-to-cite)
   2. [How to contribute](#how-to-contribute)

## Installation and Demonstration

### Installation

1. ***Optional.*** *Install Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). \
   Open an Anaconda terminal and create a virtual environment with typing:*
   <pre><i>conda create -n Sports2D python>=3.10 
   conda activate Sports2D</i></pre>

2. **Install Sports2D**: \
If you don't use Anaconda, type `python -V` in terminal to make sure python '>=3.7 <=3.10' is installed.
   - OPTION 1: **Quick install:** Open a terminal. 
       ```
       pip install sports2d
       ```
     
   - OPTION 2: **Build from source and test the last changes:**
     Open a terminal in the directory of your choice and clone the Sports2D repository.
       ```
       git clone https://github.com/davidpagnon/sports2d.git
       cd sports2d
       pip install .
       ```

1. ***Optional.*** **Install OpenPose** for more accurate and multi-person analysis (instructions [there](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md)). \
*Windows portable demo is enough.*



- `pip show sports2d`
- Adjust your settings (in particular video path) in `Config_demo.toml`
- ```from Sports2D import Sports2D
Sports2D.detect_pose('Sports2D\Demo\Config_demo.toml')
Sports2D.compute_angles('Sports2D\Demo\Config_demo.toml')```



-----
Pose detection:
-----
Detect joint centers from a video with OpenPose or BlazePose.
Save a 2D csv position file per person, and optionally json files, image files, and video files.

If OpenPose is used, multiple persons can be consistently detected across frames.
Interpolates sequences of missing data if they are less than N frames long.
Optionally filters results with Butterworth, gaussian, median, or loess filter.
Optionally displays figures.

If BlazePose is used, only one person can be detected.
No interpolation nor filtering options available. Not plotting available.

-----
Angle computation:
-----
Compute joint and segment angles from csv position files.
Automatically adjust angles when person switches to face the other way.
Save a 2D csv angle file per person. These joint and segment angles can be plotted and processed with any spreadsheet software or programming language.
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

-----
To-do list:
-----

@misc{Pagnon20223,
  author = {Pagnon, David},
  title = {Sports2D - Angles from monocular video},
  year = {2013},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/davidpagnon/Sports2D}},
}



- GUI applications for all platforms (with Kivy: https://kivy.org/)
- Pose refinement: click and move badly estimated 2D points (cf DeepLabCut: https://www.youtube.com/watch?v=bEuBKB7eqmk)
- Include OpenPose in Sports2D (dockerize it cf https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/demo/Dockerfile)
- Constrain points to OpenSim skeletal model for better angle estimation (cf Pose2Sim but in 2D https://github.com/perfanalytics/pose2sim)
