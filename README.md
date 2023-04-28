Right click in folder parent to Sports2D -> Open PowerShell here

conda create -n Sports2D python[version='<3.11,>=3.7']

pip install ipython toml numpy pandas scipy anytree opencv-python mediapipe PyQt5
ipython

from Sports2D import Sports2D
Sports2D.detect_pose('Sports2D\Demo\Config_demo.toml')
Sports2D.compute_angles('Sports2D\Demo\Config_demo.toml')


Copy, edit, and if you like, rename your Config_demo.toml file