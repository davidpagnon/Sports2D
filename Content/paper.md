---
title: 'Sports2D: Compute 2D joint and segment angles from a video.'
tags:
  - python
  - markerless kinematics
  - motion capture
  - sports performance analysis
  - openpose
  - clinical gait analysis
authors:
  - name: David Pagnon^[corresponding author] 
    orcid: 0000-0002-6891-8331
    affiliation: 1
affiliations:
 - name: Centre for the Analysis of Motion, Entertainment Research & Applications (CAMERA), University of Bath, Claverton Down, Bath, BA2 7AY, UK
   index: 1
date: February 14 2024
bibliography: paper.bib
---


# Summary
`Sports2D` provides a user-friendly solution for automatic analysis of human movement from videos. This Python package uses markerless 2D pose estimation to detect joint coordinates from videos, and then computes 2D joint and segment angles. It can be installed either locally or on a free server, which makes it possible to run it directly from a smartphone.

It outputs annotated videos and image sequences which display joint locations, joint angles, and segment angles, for each of the detected persons. This information is also stored in .csv files, editable on MS Excel or any other spreadsheet editor for further analysis. 

`Sports2D` may be useful for clinicians as a decision supports system (CDSS) [], as well as for gait analysis [] or ergonomic design []. Sports coaches can also use it to quantify key performance indicators (KPIs) [], or to better understand, correct, or compare athletes' movement patterns. Finally, it can be used by researchers as a simple tool for 2D biomechanical analysis on the fly. One of the multiple use cases would be to evaluate ACL injury risks from deceleration drills [].

![Example results from a Demo video.\label{fig:demo video results}](demo_openpose_results.png)
![Example plot of joint angle evolution.\label{fig:joint angle evolution}](demo_show_plots.png)


# Statement of need

Machine learning has recently accelerated the development and availability of markerless kinematics, which allows for the collection of kinematic data without the use of physical markers or of manual annotation. 

A large part of these tools focus on 2D analysis, such as `OpenPose` [@Cao_2019], `BlazePose` [], or `DeepLabCut` []. Although they bear the advantage of being open-source, they are not easily accessible to people who do not have a programming background, and the output is not directly usable for further kinematic investigation. Yet, clinical acceptance of new technologies is known to be influenced not only by their price value and their performance, but also by their perceived ease-of-use, the social influence around the customer, and other parameters described by the Unified Theory of Acceptance and Use of Technology (UTAUT2) [].

In fact, there is a clear trade-off between accuracy and ease-of-use. Some open-source tools focus on the accuracy of a 3D analysis by using multiple cameras, such as `Pose2Sim` [] or `OpenCap` []. These, however, require either a certain level of programming skills, or a particular hardware setup. Some other tools choose to put more emphasis on user-friendliness, and point out that 2D analysis is often sufficient when the analyzed motion mostly lies in the sagittal or frontal plane. `Kinovea` [], for example, is a widely used software for sports performance analysis which provides multiple additional features. However, it relies on tracking manual labels. This may be time-consuming when analyzing numerous videos, and it can also be lacking robustness when the tracked points of interest are lost. It is also only available on Windows, and requires the user to transfer files prior to analysis.

`Sports2D` is an alternative solution that aims at filling this gap: it is free and open-source, easy to install, can be run from any smartphone or computer, and automatically provides 2D joint and segment angles without the need for manual annotation. It is also robust, and can be used to analyze numerous videos at once. The motion of multiple people can be analyzed in the same video, and the output is directly usable for further statistical analysis. 


# Workflow

`Sports2D` can be installed and run two different ways: locally, or on a Google Colab free server [].
- *If run locally*, it can be installed via `pip install sports2d`. Two options are then offered: either run it with BlazePose as a pose estimation model, or with OpenPose. BlazePose comes preinstalled and is very fast, however it is less accurate and only detects one person per video. OpenPose is more accurate, allows for the detection of multiple people, and comes with more fine-tuning in `Sports2D`, but it is slower and requires the user to install it themselves. 
- *If run on Colab*, it can be installed in one click from any computer or smartphone device, either every time the user needs it, or once for all on Google Drive. In this case, OpenPose can be automatically installed and runs by default, and video and table results are automatically saved on Google Drive. A video tutorial can be found at this address: https://www.youtube.com/watch?v=Er5RpcJ8o1Y.

After installation, the user can choose one or several videos to analize. Then, `Sports2D` goes through two stages:
- **Pose detection:** Joint centers are detected for each video frame. If OpenPose is used, multiple persons can be detected with consistent IDs across frames. A person is associated to another in the next frame when they are at a small distance. Sequences of missing data are interpolated if they are less than N frames long, N being a threshold defined by the user. Resulting coordinates can be filtered with a Butterworth, Gaussian, Median, or LOESS filter. They can also be plotted. Note that locations are in pixels, but can be converted to meters if the user provides the distance between two points in the video.
- **Joint and segment angle estimation:** Specific joint and segment angles can be chosen, and are computed from the previously calculated positions.
If a person suddenly faces the other way, this change of direction is taken into account. The person is considered to go to the left when their toes are to the left of their heels.
Resulting angles can be filtered in the same way as point coordinates, and they can also be plotted.

Joint angle conventions are as follows:
- Ankle dorsiflexion: Between heel and big toe, and ankle and knee
- Knee flexion: Between hip, knee, and ankle
- Hip flexion: Between knee, hip, and shoulder
- Shoulder flexion: Between hip, shoulder, and elbow
- Elbow flexion: Between wrist, elbow, and shoulder

Segment angles are measured anticlockwise between the horizontal and the segment lines:
- Foot: Between heel and big toe
- Shank: Between knee and ankle
- Thigh: Between hip and knee
- Arm: Between shoulder and elbow
- Forearm: Between elbow and wrist
- Trunk: Between shoulder midpoint and hip midpoint

![Joint angle conventions. Adapted from [@Yang2007].\label{fig:joint angle conventions}](Joint_convention.png)


# Limitations

The user of `Sports2D` should be aware of the following limitations:
- Results are acceptable only if the persons move in the 2D plane, from right to left or from left to right.
If you need research-grade markerless joint kinematics, consider using several cameras, and constraining angles to a biomechanically accurate model. See `Pose2Sim` for example.
- Angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect, especially if motion blur is significant such as on some broadcast videos.
- Google Colab does not follow the European GDPR requirements regarding data privacy []. Install locally if this matters.


# Acknowledgements
I would like to acknowledge Rob Olivar, a sports coach who enlightened me about the need for such a tool.\
I also acknowledge the work of the dedicated people involved in the many major software programs and packages used by `Sports2D`, such as `Python`, `OpenPose`, `BlazePose`, `OpenCV` [@Bradski_2000], among others. 


# References


