#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## SKELETONS DEFINITIONS                                                 ##
    ###########################################################################
    
    The definition and hierarchy of the following skeletons are available: 
    - CUSTOM (e.g.., from DeepLabCut),
    - OpenPose BODY_25B, BODY_25, BODY_135, COCO, MPII
    - Mediapipe BLAZEPOSE
    - AlphaPose HALPE_26, HALPE_68, HALPE_136, COCO_133, COCO, MPII 
    (for COCO and MPII, AlphaPose must be run with the flag "--format cmu")
    
    N.B.: Not all face and hand keypoints are reported in the skeleton architecture, 
    since some are redundant for the orientation of some bodies.

    N.B.: The corresponding OpenSim model files are provided in the "Pose2Sim\Empty project" folder.
    If you wish to use any other, you will need to adjust the markerset in the .osim model file, 
    as well as in the scaling and IK setup files.
    
    N.B.: In case you built a custom skeleton, you can check its structure by typing: 
    from anytree import Node, RenderTree
    for pre, _, node in RenderTree(CUSTOM): 
            print(f'{pre}{node.name} id={node.id}')
    If you build it from a DeepLabCut model, make sure the node ids 
    correspond to the column numbers, starting from zero.
'''

## INIT
from anytree import Node, RenderTree, PreOrderIter


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.3.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


'''
SKELETONS
'''

'''CUSTOM SKELETON (e.g., from DeepLabCut detection)'''
CUSTOM = Node("Root", id=0, children=[
    Node("Child1", id=1),
    Node("Child2", id=2),
    ])


'''BODY_25B (full-body without hands, experimental, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/blob/master/experimental_models/README.md'''
BODY_25B = Node("CHip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=17, children=[
        Node("Head", id=18, children=[
            Node("Nose", id=0, children=[
                Node("REye", id=2, children=[
                    Node("REar", id=4)
                    ]),
                Node("LEye", id=1, children=[
                    Node("LEar", id=3)
                    ]),
            ]),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])


'''BODY_25 (full-body without hands, standard, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
BODY_25 = Node("CHip", id=8, children=[
    Node("RHip", id=9, children=[
        Node("RKnee", id=10, children=[
            Node("RAnkle", id=11, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=12, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=14, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0, children=[
                Node("REye", id=15, children=[
                    Node("REar", id=17)
                    ]),
                Node("LEye", id=16, children=[
                    Node("LEar", id=18)
                    ]),
            ]),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])


'''BODY_135 (full-body with hands and face, experimental, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/blob/master/experimental_models/README.md)'''
BODY_135 = Node("CHip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=17, children=[
        Node("Head", id=18, children=[
            Node("Nose", id=0),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=48),
                    Node("RIndex", id=51),
                    Node("RPinky", id=63),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=27),
                    Node("LIndex", id=30),
                    Node("LPinky", id=42),
                ]),
            ]),
        ]),
    ]),
])


'''BLAZEPOSE (full-body with simplified hand and foot, from mediapipe) 
https://google.github.io/mediapipe/solutions/pose'''
BLAZEPOSE = Node("CHip", id=None, children=[
    Node("RHip", id=24, children=[
        Node("RKnee", id=26, children=[
            Node("RAnkle", id=28, children=[
                Node("RHeel", id=30),
                Node("RBigToe", id=32),
            ]),
        ]),
    ]),
    Node("LHip", id=23, children=[
        Node("LKnee", id=25, children=[
            Node("LAnkle", id=27, children=[
                Node("LHeel", id=29),
                Node("LBigToe", id=31),
            ]),
        ]),
    ]),
    Node("Nose", id=0, children=[ 
        Node("LEyeInner", id=1),
        Node("LEye", id=2),
        Node("LEyeOuter", id=3),
        Node("REyeInner", id=4),
        Node("REye", id=5),
        Node("REyeOuter", id=6),
        Node("LEar", id=7),
        Node("REar", id=8),
        Node("LMouth", id=9),
        Node("RMouth", id=10),
    ]),
    Node("RShoulder", id=12, children=[
        Node("RElbow", id=14, children=[
            Node("RWrist", id=16, children=[
                Node("RPinky", id=18),
                Node("RIndex", id=20),
                Node("RThumb", id=22),
            ]),
        ]),
    ]),
    Node("LShoulder", id=11, children=[
        Node("LElbow", id=13, children=[
            Node("LWrist", id=15, children=[
                Node("LPinky", id=17),
                Node("LIndex", id=19),
                Node("LThumb", id=21),
            ]),
        ]),
    ]),
])


'''HALPE_26 (full-body without hands, from AlphaPose)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md'''
# HALPE_26 = Node("Hip", id=19, children=[
#     Node("RHip", id=12, children=[
#         Node("RKnee", id=14, children=[
#             Node("RAnkle", id=16, children=[
#                 Node("RBigToe", id=21, children=[
#                     Node("RSmallToe", id=23),
#                 ]),
#                 Node("RHeel", id=25),
#             ]),
#         ]),
#     ]),
#     Node("LHip", id=11, children=[
#         Node("LKnee", id=13, children=[
#             Node("LAnkle", id=15, children=[
#                 Node("LBigToe", id=20, children=[
#                     Node("LSmallToe", id=22),
#                 ]),
#                 Node("LHeel", id=24),
#             ]),
#         ]),
#     ]),
#     Node("Neck", id=18, children=[
#         Node("Head", id=17, children=[
#             Node("Nose", id=0),
#         ]),
#         Node("RShoulder", id=6, children=[
#             Node("RElbow", id=8, children=[
#                 Node("RWrist", id=10),
#             ]),
#         ]),
#         Node("LShoulder", id=5, children=[
#             Node("LElbow", id=7, children=[
#                 Node("LWrist", id=9),
#             ]),
#         ]),
#     ]),
# ])


'''HALPE_68 (full-body with hands without face, from AlphaPose)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md'''
HALPE_68 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=49),
                    Node("RIndex", id=52),
                    Node("RPinky", id=64),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=28),
                    Node("LIndex", id=31),
                    Node("LPinky", id=43),
                ])
            ]),
        ]),
    ]),
])


'''HALPE_136 (full-body with hands and face, from AlphaPose)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md'''
HALPE_136 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=117),
                    Node("RIndex", id=120),
                    Node("RPinky", id=132),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=96),
                    Node("LIndex", id=99),
                    Node("LPinky", id=111),
                ])
            ]),
        ]),
    ]),
])


'''COCO_133 (full-body with hands and face, from AlphaPose)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md'''
COCO_133 = Node("Hip", id=None, children=[
    Node("RHip", id=13, children=[
        Node("RKnee", id=15, children=[
            Node("RAnkle", id=17, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=22),
                ]),
                Node("RHeel", id=23),
            ]),
        ]),
    ]),
    Node("LHip", id=12, children=[
        Node("LKnee", id=14, children=[
            Node("LAnkle", id=16, children=[
                Node("LBigToe", id=18, children=[
                    Node("LSmallToe", id=19),
                ]),
                Node("LHeel", id=20),
            ]),
        ]),
    ]),
    Node("Neck", id=None, children=[
        Node("Nose", id=1, children=[
            Node("right_eye", id=3),
            Node("left_eye", id=2),
        ]),
        Node("RShoulder", id=7, children=[
            Node("RElbow", id=9, children=[
                Node("RWrist", id=11, children=[
                    Node("RThumb", id=115),
                    Node("RIndex", id=118),
                    Node("RPinky", id=130),
                ]),
            ]),
        ]),
        Node("LShoulder", id=6, children=[
            Node("LElbow", id=8, children=[
                Node("LWrist", id=10, children=[
                    Node("LThumb", id=94),
                    Node("LIndex", id=97),
                    Node("LPinky", id=109),
                ])
            ]),
        ]),
    ]),
])


'''COCO (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, etc)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
COCO = Node("CHip", id=None, children=[
    Node("RHip", id=8, children=[
        Node("RKnee", id=9, children=[
            Node("RAnkle", id=10),
        ]),
    ]),
    Node("LHipJ", id=11, children=[
        Node("LKnee", id=12, children=[
            Node("LAnkle", id=13),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])


'''MPII (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, etc)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
MPII = Node("CHip", id=14, children=[
    Node("RHip", id=8, children=[
        Node("RKnee", id=9, children=[
            Node("RAnkle", id=10),
        ]),
    ]),
    Node("LHipJ", id=11, children=[
        Node("LKnee", id=12, children=[
            Node("LAnkle", id=13),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])

# Halpe26 for RTMPose
halpe26_rtm = dict(name='halpe26',         
    keypoint_info={
        0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper', swap='left_elbow'),
        9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
        10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper', swap='left_wrist'),
        11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower', swap='right_hip'),
        12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
        13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower', swap='right_knee'),
        14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
        15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower', swap='right_ankle'),
        16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle'),
        17: dict(name='head', id=17, color=[255, 128, 0], type='upper', swap=''),
        18: dict(name='neck', id=18, color=[255, 128, 0], type='upper', swap=''),
        19: dict(name='hip', id=19, color=[255, 128, 0], type='lower', swap=''),
        20: dict(name='left_big_toe', id=20, color=[255, 128, 0], type='lower', swap='right_big_toe'),
        21: dict(name='right_big_toe', id=21, color=[255, 128, 0], type='lower', swap='left_big_toe'),
        22: dict(name='left_small_toe', id=22, color=[255, 128, 0], type='lower', swap='right_small_toe'),
        23: dict(name='right_small_toe', id=23, color=[255, 128, 0], type='lower', swap='left_small_toe'),
        24: dict(name='left_heel', id=24, color=[255, 128, 0], type='lower', swap='right_heel'),
        25: dict(name='right_heel', id=25, color=[255, 128, 0], type='lower', swap='left_heel')
    },
    skeleton_info={
        0: dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2: dict(link=('left_hip', 'hip'), id=2, color=[0, 255, 0]),
        3: dict(link=('right_ankle', 'right_knee'), id=3, color=[255, 128, 0]),
        4: dict(link=('right_knee', 'right_hip'), id=4, color=[255, 128, 0]),
        5: dict(link=('right_hip', 'hip'), id=5, color=[255, 128, 0]),
        6: dict(link=('head', 'neck'), id=6, color=[51, 153, 255]),
        # 7: dict(link=('neck', 'hip'), id=7, color=[51, 153, 255]),
        8: dict(link=('neck', 'left_shoulder'), id=8, color=[0, 255, 0]),
        9: dict(link=('left_shoulder', 'left_elbow'), id=9, color=[0, 255, 0]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11: dict(link=('neck', 'right_shoulder'), id=11, color=[255, 128, 0]),
        12: dict(link=('right_shoulder', 'right_elbow'), id=12, color=[255, 128, 0]),
        13: dict(link=('right_elbow', 'right_wrist'), id=13, color=[255, 128, 0]),
        14: dict(link=('left_eye', 'right_eye'), id=14, color=[51, 153, 255]),
        15: dict(link=('nose', 'left_eye'), id=15, color=[51, 153, 255]),
        16: dict(link=('nose', 'right_eye'), id=16, color=[51, 153, 255]),
        17: dict(link=('left_eye', 'left_ear'), id=17, color=[51, 153, 255]),
        18: dict(link=('right_eye', 'right_ear'), id=18, color=[51, 153, 255]),
        19: dict(link=('left_ear', 'left_shoulder'), id=19, color=[51, 153, 255]),
        20: dict(link=('right_ear', 'right_shoulder'), id=20, color=[51, 153, 255]),
        21: dict(link=('left_ankle', 'left_big_toe'), id=21, color=[0, 255, 0]),
        22: dict(link=('left_ankle', 'left_small_toe'), id=22, color=[0, 255, 0]),
        23: dict(link=('left_ankle', 'left_heel'), id=23, color=[0, 255, 0]),
        24: dict(link=('right_ankle', 'right_big_toe'), id=24, color=[255, 128, 0]),
        25: dict(link=('right_ankle', 'right_small_toe'), id=25, color=[255, 128, 0]),
        26: dict(link=('right_ankle', 'right_heel'), id=26, color=[255, 128, 0]),
    }
)