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

`Announcement:` Apps with GUI will be released for Windows, Linux, MacOS, as well as Android and iOS.
Mobile versions will only support exploratory angle computation from BlazePose, hence less accurately and tunable.


<figure><img src='Content/demo_gif.gif' title='Demonstration of Sports2D with OpenPose.'></figure>


`Warning:` Angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect.\
`Warning:` Results are acceptable only if captured persons move in 2D, from right to left or from left to right.\
If you need research-grade markerless joint kinematics, consider using several cameras, and constraining angles to a biomechanically accurate model. See [Pose2Sim](https://github.com/perfanalytics/pose2sim) for example.


## Contents
1. [Installation and Demonstration](#installation-and-demonstration)
   1. [Installation](#installation)
   2. [Demonstration: Detect pose and compute 2D angles](#demonstration-detect-pose-and-compute-2d-angles)
2. [Go further](#go-further)
   1. [With OpenPose and other models](#with-openpose-and-other-models)
   2. [Advanced-settings: Pose](#advanced-settings-pose)
   3. [Advanced-settings: Angles](#advanced-settings-angles)
3. [How to cite and how to contribute](#how-to-cite-and-how-to-contribute)

## Installation and Demonstration

### Installation

1. ***Optional.*** *Install Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). \
   Open an Anaconda prompt and create a virtual environment by typing:*
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
3. ***Optional. Install OpenPose*** for more accurate and multi-person analysis (instructions [there](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md)). \
*Windows portable demo works fine.*


### Demonstration: Detect pose and compute 2D angles

#### Run
Open a terminal, enter `pip show sports2d`, report package location. \
Copy this path and go to the Demo folder with `cd <path>\Sports2D\Demo`. \
Type `ipython`, and test the following code:
```
from Sports2D import Sports2D
Sports2D.detect_pose('Config_demo.toml')
Sports2D.compute_angles('Config_demo.toml')
```

#### Results
You should obtain two .csv files, which can be processed with any spreadsheet software, or with the Python Pandas library:
- `demo_blazepose_points.csv` with 2D joint coordinates
- `demo_blazepose_angles.csv`, with joint and segment angle coordinates

Additionally, you will obtain a a visual output: 
- A video: `demo_blazepose.mp4` with detected joints overlayed on the person, as well as a angles reported in the upper-left corner. 
- The same results as images in the `demo_blazepose_img` folder. 

CSV AND DETECTION IMAGES HERE

*N.B.:* Default parameters have been provided in `Demo\Config_demo.toml` but can be edited.\
*N.B.:* OpenPose-like json coordinates are also stored in the `demo_blazepose_json` folder. A `logs.txt` file lets you recover details about your chosen configuration.

## Go further

Copy, edit, and if you like, rename your `Config_demo.toml` file to alter your settings.

`Project`: 


### Pose detection



-----
Pose detection:
-----
Detect joint centers from a video with OpenPose or BlazePose.
Save a 2D csv position file per person, and optionally json files, image files, and video files.

If OpenPose is used, multiple persons can be consistently detected across frames.
However, it needs to be [installed](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md) separately.
It supports several models: BODY_25 is the standard one, BODY_25B is more accurate but requires manually [downloading the model](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/blob/master/experimental_models/README.md)
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

## How to cite and how to contribute

### How to cite
If you use this code or data, please cite [Pagnon, 2023].

     @misc{Pagnon2023,
       author = {Pagnon, David},
       title = {Sports2D - Angles from monocular video},
       year = {2013},
       publisher = {GitHub},
       journal = {GitHub repository},
       howpublished = {\url{https://github.com/davidpagnon/Sports2D}},
     }

### How to contribute
I would happily welcome any proposal for new features, code improvement, and more!\
If you want to contribute to Sports2D, please follow [this guide](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) on how to fork, modify and push code, and submit a pull request. I would appreciate it if you provided as much useful information as possible about how you modified the code, and a rationale for why you're making this pull request. Please also specify on which operating system and on which python version you have tested the code.

*Here is a to-do list, for general guidance purposes only:*
> <li> <b>GUI applications:</b> For Windows, Mac, and Linux, as well as for Android and iOS (minimal version on mobile device, with only BlazePose). Code with <a href="https://kivy.org">Kivy</a>.</li>
> <li> <b>Pose refinement:</b> Click and move badly estimated 2D points. See <a href="https://www.youtube.com/watch?v=bEuBKB7eqmk">DeepLabCut</a> for inspiration.
> <li> <b>Include OpenPose in Sports2D:</b> For example, <a href="https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/demo/Dockerfile">Dockerize</a> it. Otherwise, run Sports2D in a <a href="https://colab.research.google.com/github/hardik0/AI-basketball-analysis-on-google-colab/blob/master/AI_basketball_analysis_google_colab.ipynb">Colab notebook</a>.
> <li> <b>Constrain points</b> to OpenSim skeletal model for better angle estimation. Cf <a href="https://github.com/perfanalytics/pose2sim">Pose2Sim</a>, but in 2D.
