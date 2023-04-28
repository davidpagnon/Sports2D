
This repository provides a workflow to compute 2D markerless
joint and segment angles from videos. 
These angles can be plotted and processed with any 
spreadsheet software or programming language.

This is a headless version, but apps will be released 
for Windows, Linux, MacOS, as well as Android and iOS.
Mobile versions will only support exploratory joint detection 
from BlazePose, hence less accurately and tunable.

If you need to detect several persons and want more accurate results, 
you can install and use OpenPose: 
https://github.com/CMU-Perceptual-Computing-Lab/openpose

-----
Sports2D installation:
-----
Optional: 
- Install Miniconda
- Open a Anaconda Prompt and type: 
`conda create -n Sports2D python>=3.7`
`conda activate Sports2D`
pip install 

- Open a python prompt and type `pip install sports2d`
- `pip show sports2d`
- Adjust your settings (in particular video path) in `Config_demo.toml`
- ```from Sports2D import Sports2D
Sports2D.detect_pose('Sports2D\Demo\Config_demo.toml')
Sports2D.compute_angles('Sports2D\Demo\Config_demo.toml')```

-----
/!\ Warning /!\
-----
- The angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect.
- It will only lead to acceptable results if the persons move in the 2D plane (sagittal plane).
- The persons need to be filmed as perpendicularly as possible from their side.
If you need research-grade markerless joint kinematics, consider using several cameras,
and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
https://github.com/perfanalytics/pose2sim

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

-----
To-do list:
-----
- GUI applications for all platforms (with Kivy: https://kivy.org/)
- Pose refinement: click and move badly estimated 2D points (cf DeepLabCut: https://www.youtube.com/watch?v=bEuBKB7eqmk)
- Include OpenPose in Sports2D (dockerize it cf https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/demo/Dockerfile)
- Constrain points to OpenSim skeletal model for better angle estimation (cf Pose2Sim but in 2D https://github.com/perfanalytics/pose2sim)
