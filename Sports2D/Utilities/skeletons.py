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
__version__ = "0.4.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## SKELETONS
'''CUSTOM SKELETON (e.g., from DeepLabCut detection)'''
CUSTOM = Node("Root", id=0, children=[
    Node("Child1", id=1),
    Node("Child2", id=2),
    ])


'''BodyWithFeet from HALPE_26 (full-body without hands, for RTMPose, AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
body_with_feet = Node("Hip", id=19, children=[
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
        Node("Head", id=17, children=[
            Node("Nose", id=0),
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
