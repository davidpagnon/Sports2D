---
title: 'Sports2D: Compute 2D human pose and angles from a video or a webcam'
tags:
  - python
  - markerless kinematics
  - motion capture
  - sports performance analysis
  - rtmpose
  - clinical gait analysis
authors:
  - name: David Pagnon^[corresponding author] 
    orcid: 0000-0002-6891-8331
    affiliation: 1
  - name: HunMin Kim
    orcid: 0009-0007-7710-8051
    affiliation: 2
affiliations:
 - name: Centre for the Analysis of Motion, Entertainment Research & Applications (CAMERA), University of Bath, Claverton Down, Bath, BA2 7AY, United Kingdom
   index: 1
 - name: Inha University, Yonghyeon Campus, 100 Inha-ro, Michuhol-gu, Incheon 22212, South Korea
   index: 2
date: February 14 2024
bibliography: paper.bib
---


# Summary
`Sports2D` provides a user-friendly solution for automatic and real-time analysis of multi-person human movement from a video or a webcam. This Python package uses 2D markerless pose estimation to detect joint coordinates from videos, and then computes 2D joint and segment angles. 
<!--It can be installed either locally or on a free server, which makes it possible to run it directly from a smartphone.-->

The output incorporates annotated videos and image sequences overlaid with joint locations, joint angles, and segment angles, for each of the detected persons. For further analysis, this information is also stored in files that are editable with MS Excel® or any other spreadsheet editor (.trc for locations, .mot for angles, according to the OpenSim standard [@Delp_2007; @Seth_2018]). 

`Sports2D` may be useful for clinicians as a decision supports system (CDSS) [@Bright_2012], as well as for gait analysis [@Whittle_2014] or ergonomic design [@Patrizi_2016]. Sports coaches can also use it to quantify key performance indicators (KPIs) [@ODonoghue_2008; @Pagnon_2022b], or to better understand, correct, or compare athletes' movement patterns. Finally, it can be used by researchers as a simple tool for 2D biomechanical analysis on the fly. One of the multiple use cases would be to evaluate ACL injury risks from deceleration drills [@Di_2021]. 


# Statement of need

Machine learning has recently accelerated the development and availability of markerless kinematics [@Zheng_2023; @Colyer_2018], which allows for the collection of kinematic data without the use of physical markers or manual annotation. 

A large part of these tools focus on 2D analysis, such as `OpenPose` [@Cao_2019], `BlazePose` [@Bazarevsky_2020], or `DeepLabCut` [@Mathis_2018]. More recently, `RTMPose` [@Jiang_2023] offered a faster, more accurate, and more flexible alternative to the previous solutions. Still, although they bear the advantage of being open-source, none of these options are easily accessible to people who do not have a programming background, and the output is not directly usable for further kinematic investigation. Yet, clinical acceptance of new technologies is known to be influenced not only by their price value and their performance, but also by their perceived ease-of-use, the social influence around the customer, and other parameters described by the Unified Theory of Acceptance and Use of Technology (UTAUT2) [@Venkatesh_2012].

![Example results from a demonstration video.\label{fig:Demo_results}](Demo_results.png)

![Example joint angle output.\label{fig:Demo_plots}](Demo_plots.png)

In fact, there is a clear trade-off between accuracy and ease-of-use. Some open-source tools focus on the accuracy of a 3D analysis by using multiple cameras, such as `Pose2Sim` [@Pagnon_2022a] or `OpenCap` [@Uhlrich_2022]. These, however, require either a certain level of programming skills, a particular hardware setup, or to send data to a server that does not comply with the European rules of data protection (GDPR). Some other tools choose to put more emphasis on user-friendliness, and point out that 2D analysis is often sufficient when the analyzed motion mostly lies in the sagittal or frontal plane. `Sit2Stand` [@Boswell_2023] and `CP GaitLab` [@Kidzinski_2020] provide such tools, although they are focused on very specific tasks. `Kinovea` [@Kinovea], on the other hand, is a widely used software for sports performance analysis, which provides multiple additional features. However, it relies on tracking manual labels. This can be time-consuming when analyzing numerous videos, and it may also be lacking robustness when the tracked points are lost. It is also only available on Windows, and requires the user to transfer files prior to analysis.

`Sports2D` is an alternative solution that aims at filling this gap: it is free and open-source, straightforward to install and to run, can be run on any platform, can be run locally for data protection, and it automatically provides 2D joint and segment angles without the need for manual annotation. It is also robust and flexible, works in real-time, supports multi-person analysis, and can process one video, several videos simultaneously, or a webcam stream. The output is provided as .trc files for locations and .mot files for angles, which makes it compatible with OpenSim [@Delp_2007; @Seth_2018] and readable by any spreadsheet software for further statistical analysis. 


# Workflow

## Installation and usage

`Sports2d` is installed under Python via `pip install sports2d`. If a valid CUDA installation is found, Sports2D uses the GPU, otherwise it uses the CPU with OpenVino acceleration. 

<!-- `Sports2D` can be installed and run two different ways: locally, or on a Google Colab® free server [@Bisong_2019].

* *If run locally*, it is installed under Python via `pip install sports2d`. If a valid CUDA installation is found, Sports2D uses the GPU, otherwise it uses the CPU with OpenVino acceleration. 

* *If run on Colab*, it can be installed in one click from any computer or smartphone device, either every time the user needs it, or once for all on Google Drive®. Results are automatically saved on Google Drive®. The arguments are the same as with the local installation. A full video tutorial can be found at this address: [https://www.youtube.com/watch?v=Er5RpcJ8o1Y](https://www.youtube.com/watch?v=Er5RpcJ8o1Y).-->

A detailed installation and usage guide can be found on the repository: https://github.com/davidpagnon/Sports2D.

## Sports2D method details

[Sports2D]{.ul}: 

1. Reads stream from a webcam, from one video, or from a list of videos. It selects an optional specified time range to process.
2. Sets up the RTMLib pose tracker with specified parameters. It can be run in lightweight, balanced, or performance mode, and for faster inference, keypoints can be tracked for a certain number of frames instead of detected. Any RTMPose model can be used. 
3. Tracks people so that their IDs are consistent across frames. A person is associated to another in the next frame when they are at a small distance. IDs remain consistent even if the person disappears for a few frames. This carefully crafted `sports2d` tracker runs at a comparable speed as the RTMlib one but is much more robust. The user can still choose the RTMLib method if they need it by using the `tracking_mode` argument.
4. Retrieves the keypoints with high enough confidence, and only keeps the persons with enough average high-confidence.
5. Computes the selected joint and segment angles, and flips them on the left/right side if the respective foot is pointing to the left/right. The user can select which angles they want to compute, display, and save.
5. Draws bounding boxes around each person and writes their IDs\
   Draws the skeleton and the keypoints, with a green to red color scale to account for their confidence\
   Draws joint and segment angles on the body, and writes the values either near the joint/segment, or on the upper-left of the image with a progress bar
6. Interpolates missing pose and angle sequences if gaps are not too large. Filters them with the selected filter (among `Butterworth`, `Gaussian`, `LOESS`, or `Median`) and their parameters
7. Optionally shows processed images, saves them, or saves them as a video\
   Optionally plots pose and angle data before and after processing for comparison\
   Optionally saves poses for each person as a TRC file, and angles as a MOT file 

<br>

[The Demo video]{.ul} that Sports2D is tested on is voluntarily challenging, in order to demonstrate the robustness of the process after sorting, interpolation and filtering. It contains:

* One person walking in the sagittal plane
*  One person in the frontal plane. This person then performs a flip while being backlit, both of which are challenging for the pose detection algorithm
* One tiny person flickering in the background who needs to be ignored

<br>

[Joint and segment angle estimation]{.ul}:

Specific joint and segment angles can be chosen. They are consistent regardless of the direction the participant is facing: the participant is considered to look to the left when their toes are to the left of their heels, and to the right otherwise. Resulting angles can be filtered in the same way as point coordinates, and they can also be plotted.

Joint angle conventions are as follows (\autoref{fig:joint_angle_conventions}):

* Ankle dorsiflexion: Between heel and big toe, and ankle and knee.\
  *-90° when the foot is aligned with the shank.*
* Knee flexion: Between hip, knee, and ankle.\
  *0° when the shank is aligned with the thigh.*
* Hip flexion: Between knee, hip, and shoulder.\
  *0° when the trunk is aligned with the thigh.* 
* Shoulder flexion: Between hip, shoulder, and elbow.\
  *180° when the arm is aligned with the trunk.*
* Elbow flexion: Between wrist, elbow, and shoulder.\
  *0° when the forearm is aligned with the arm.*

Segment angles are measured anticlockwise between the horizontal and the segment lines:

* Foot: Between heel and big toe.
* Shank: Between knee and ankle.
* Thigh: Between hip and knee.
* Pelvis: Between left and right hip
* Trunk: Between hip midpoint and shoulder midpoint
* Shoulders: Between left and right shoulder
* Head: Between neck and top of the head
* Arm: Between shoulder and elbow.
* Forearm: Between elbow and wrist.

![Joint angle conventions\label{fig:joint_angle_conventions}](joint_convention.png)


# Limitations

The user of `Sports2D` should be aware of the following limitations:

* Results are acceptable only if the participants move in the 2D plane, either in the frontal plane or in the sagittal one. If you need research-grade markerless joint kinematics, consider using several cameras, and constraining angles to a biomechanically accurate model. See `Pose2Sim` [@Pagnon_2022a] for example.
* Angle estimation is only as good as the pose estimation algorithm, i.e., it is not perfect [@Wade_2022], especially if motion blur is significant such as on some broadcast videos.
<!--* Google Colab does not follow the European GDPR requirements regarding data privacy [@Minssen_2020]. Install locally if this matters.-->


# Acknowledgements

I would like to acknowledge Rob Olivar, a sports coach who enlightened me about the need for such a tool.\
I also acknowledge the work of the dedicated people involved in the many major open-source software programs and packages used by `Sports2D`, such as `Python`, `RTMPPose`, `OpenCV` [@Bradski_2000], among others. 


# References


