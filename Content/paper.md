---
title: 'Sports2D: Compute 2D joint and segment angles on your smartphone.'
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
`Sports2D` provides a user-friendly solution for automatic analysis of human movement from a video. This Python package uses 2D markerless pose estimation to detect joint coordinates from videos, and then computes 2D joint and segment angles. It can be installed either locally or on a free server, which makes it possible to run it directly from a smartphone.

The output incorporates annotated videos and image sequences overlaid with joint locations, joint angles, and segment angles, for each of the detected persons. This information is also stored in .csv files for further analysis, editable on MS Excel® or any other spreadsheet editor. 

`Sports2D` may be useful for clinicians as a decision supports system (CDSS) [@Bright_2012], as well as for gait analysis [@Whittle_2014] or ergonomic design [@Patrizi_2016]. Sports coaches can also use it to quantify key performance indicators (KPIs) [@ODonoghue_2008; @Pagnon_2022b], or to better understand, correct, or compare athletes' movement patterns. Finally, it can be used by researchers as a simple tool for 2D biomechanical analysis on the fly. One of the multiple use cases would be to evaluate ACL injury risks from deceleration drills [@Di_2021]. 

![Example results from a demonstration video.\label{fig:demo_video_results}](demo_openpose_results.png)

![Example plot of joint angle evolution.\label{fig:joint_angle_evolution}](demo_show_plots.png)


# Statement of need

Machine learning has recently accelerated the development and availability of markerless kinematics [@Zheng_2023; @Colyer_2018], which allows for the collection of kinematic data without the use of physical markers or manual annotation. 

A large part of these tools focus on 2D analysis, such as `OpenPose` [@Cao_2019], `BlazePose` [@Bazarevsky_2020], or `DeepLabCut` [@Mathis_2018]. Although they bear the advantage of being open-source, they are not easily accessible to people who do not have a programming background, and the output is not directly usable for further kinematic investigation. Yet, clinical acceptance of new technologies is known to be influenced not only by their price value and their performance, but also by their perceived ease-of-use, the social influence around the customer, and other parameters described by the Unified Theory of Acceptance and Use of Technology (UTAUT2) [@Venkatesh_2012].

In fact, there is a clear trade-off between accuracy and ease-of-use. Some open-source tools focus on the accuracy of a 3D analysis by using multiple cameras, such as `Pose2Sim` [@Pagnon_2022a] or `OpenCap` [@Uhlrich_2022]. These, however, require either a certain level of programming skills, or a particular hardware setup. Some other tools choose to put more emphasis on user-friendliness, and point out that 2D analysis is often sufficient when the analyzed motion mostly lies in the sagittal or frontal plane. `Sit2Stand` [@Boswell_2023] and `CP GaitLab` [@Kidzinski_2020] provide such tools, focused however on very specific tasks. `Kinovea` [@Kinovea], on the other hand, is a widely used software for sports performance analysis, which provides multiple additional features. However, it relies on tracking manual labels. This can be time-consuming when analyzing numerous videos, and it may also be lacking robustness when the tracked points are lost. It is also only available on Windows, and requires the user to transfer files prior to analysis.

`Sports2D` is an alternative solution that aims at filling this gap: it is free and open-source, easy to install, can be run from any smartphone or computer, and automatically provides 2D joint and segment angles without the need for manual annotation. It is also robust, and can be used to analyze numerous videos at once. The motion of multiple people can be analyzed in the same video, and the output is directly usable for further statistical analysis. 


# Workflow

`Sports2D` can be installed and run two different ways: locally, or on a Google Colab® free server [@Bisong_2019].

* *If run locally*, it is installed under Python via `pip install sports2d`. Two options are then offered: either run it with BlazePose [@Bazarevsky_2020] as a pose estimation model, or with OpenPose [@Cao_2019]. BlazePose comes preinstalled and is very fast to run, however it is less accurate and only detects one person per video. OpenPose is more accurate, allows for the detection of multiple people, and comes with more fine-tuning, but it is slower and requires the user to install it themselves. 

* *If run on Colab*, it can be installed in one click from any computer or smartphone device, either every time the user needs it, or once for all on Google Drive®. In both cases, OpenPose is automatically installed and runs by default, and video and table results are automatically saved on Google Drive®. A video tutorial can be found at this address: [https://www.youtube.com/watch?v=Er5RpcJ8o1Y](https://www.youtube.com/watch?v=Er5RpcJ8o1Y).

After installation, the user can choose one or several videos to analyze.`Sports2D` then goes through two stages:

* **Pose detection:** Joint centers are detected for each video frame. If OpenPose is used, multiple persons can be detected with consistent IDs across frames. A person is associated to another in the next frame when they are at a small Euclidian distance. Sequences of missing data are interpolated if they are less than N frames long, N being a threshold defined by the user. Resulting coordinates can be filtered with a Butterworth [@Butterworth_1930], Gaussian, Median, or LOESS [@Cleveland_1981] filter. They can also be plotted. Note that locations are in pixels, but can be converted to meters if the user provides the distance between two points in the video.

* **Joint and segment angle estimation:** Specific joint and segment angles can be chosen, and are computed from the previously calculated positions. Angles are consistent regardless of the direction the participant is facing: the participant is considered to go to the left when their toes are to the left of their heels, and to the right otherwise. Resulting angles can be filtered in the same way as point coordinates, and they can also be plotted.

Joint angle conventions are as follows (\autoref{fig:joint_angle_conventions}):

* Ankle dorsiflexion: Between heel and big toe, and ankle and knee;
* Knee flexion: Between hip, knee, and ankle;
* Hip flexion: Between knee, hip, and shoulder;
* Shoulder flexion: Between hip, shoulder, and elbow;
* Elbow flexion: Between wrist, elbow, and shoulder.

Segment angles are measured anticlockwise between the horizontal and the segment lines:

* Foot: Between heel and big toe;
* Shank: Between knee and ankle;
* Thigh: Between hip and knee;
* Arm: Between shoulder and elbow;
* Forearm: Between elbow and wrist;
* Trunk: Between shoulder midpoint and hip midpoint.

![Joint angle conventions. Adapted from [@Yang_2007].\label{fig:joint_angle_conventions}](joint_convention.png)


# Limitations

The user of `Sports2D` should be aware of the following limitations:

* Results are acceptable only if the participants move in the 2D plane, from right to left or from left to right. If you need research-grade markerless joint kinematics, consider using several cameras, and constraining angles to a biomechanically accurate model. See `Pose2Sim` [@Pagnon_2022a] for example.
* Angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect [@Wade_2022], especially if motion blur is significant such as on some broadcast videos.
* Google Colab does not follow the European GDPR requirements regarding data privacy [@Minssen_2020]. Install locally if this matters.


# Acknowledgements

I would like to acknowledge Rob Olivar, a sports coach who enlightened me about the need for such a tool.\
I also acknowledge the work of the dedicated people involved in the many major open-source software programs and packages used by `Sports2D`, such as `Python`, `OpenPose`, `BlazePose`, `OpenCV` [@Bradski_2000], among others. 


# References


