import os

# For continuous integration (check whether Sports2D still works at each code modification) 
def test_workflow():
    from Sports2D import Sports2D
    Sports2D.detect_pose('test_with_RTMPose.toml')
    Sports2D.compute_angles('test_with_RTMPose.toml')
