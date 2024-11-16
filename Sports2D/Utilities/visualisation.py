#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Common classes and functions                 ##
    ##################################################
    
    - A class for displaying several matplotlib figures in tabs.
    - A function for interpolating sequences with missing data. 
    It does not interpolate sequences of more than N contiguous missing data.
'''


## INIT
import sys
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## CLASSES
class plotWindow():
    '''
    Display several figures in tabs
    Taken from https://github.com/superjax/plotWindow/blob/master/plotWindow.py

    USAGE:
    pw = plotWindow()
    f = plt.figure()
    plt.plot(x1, y1)
    pw.addPlot("1", f)
    f = plt.figure()
    plt.plot(x2, y2)
    pw.addPlot("2", f)
    '''
    def __init__(self, parent=None):
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.setWindowTitle("Multitabs figure")
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(1280, 720)
        self.MainWindow.show()

    def addPlot(self, title, figure):
        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        figure.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.91, wspace=0.2, hspace=0.2)
        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        self.app.exec_()


## FUNCTIONS
def pose_plots(trc_data_unfiltered, trc_data, person_id):
    '''
    Displays trc filtered and unfiltered data for comparison
    /!\ Often crashes on the third window...

    INPUTS:
    - trc_data_unfiltered: pd.DataFrame. The unfiltered trc data
    - trc_data: pd.DataFrame. The filtered trc data

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''
    
    mpl.use('qt5agg')
    mpl.rc('figure', max_open_warning=0)

    keypoints_names = trc_data.columns[1::3]
    
    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person'+ str(person_id) + ' coordinates') # Main title

    for id, keypoint in enumerate(keypoints_names):
        f = plt.figure()
        f.canvas.manager.window.setWindowTitle(keypoint + ' Plot')

        axX = plt.subplot(211)
        plt.plot(trc_data_unfiltered.iloc[:,0], trc_data_unfiltered.iloc[:,id*3+1], label='unfiltered')
        plt.plot(trc_data.iloc[:,0], trc_data.iloc[:,id*3+1], label='filtered')
        plt.setp(axX.get_xticklabels(), visible=False)
        axX.set_ylabel(keypoint+' X')
        plt.legend()

        axY = plt.subplot(212)
        plt.plot(trc_data_unfiltered.iloc[:,0], trc_data_unfiltered.iloc[:,id*3+2], label='unfiltered')
        plt.plot(trc_data.iloc[:,0], trc_data.iloc[:,id*3+2], label='filtered')
        axY.set_xlabel('Time (seconds)')
        axY.set_ylabel(keypoint+' Y')

        pw.addPlot(keypoint, f)
    
    pw.show()


def angle_plots(angle_data_unfiltered, angle_data, person_id):
    '''
    Displays angle filtered and unfiltered data for comparison
    /!\ Often crashes on the third window...

    INPUTS:
    - angle_data_unfiltered: pd.DataFrame. The unfiltered angle data
    - angle_data: pd.DataFrame. The filtered angle data

    OUTPUT:
    - matplotlib window with tabbed figures for each angle
    '''

    mpl.use('qt5agg')
    mpl.rc('figure', max_open_warning=0)

    angles_names = angle_data.columns[1:]

    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person'+ str(person_id) + ' angles') # Main title

    for id, angle in enumerate(angles_names):
        f = plt.figure()
        
        ax = plt.subplot(111)
        plt.plot(angle_data_unfiltered.iloc[:,0], angle_data_unfiltered.iloc[:,id+1], label='unfiltered')
        plt.plot(angle_data.iloc[:,0], angle_data.iloc[:,id+1], label='filtered')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(angle+' (°)')
        plt.legend()

        pw.addPlot(angle, f)

    pw.show()


def display_quit_message(frame, cam_width, cam_height, fontSize, thickness):
    message = "Press 'q' to quit"
    position = (cam_width - int(400 * fontSize), cam_height - 20)
    cv2.putText(
        frame, message, position,
        cv2.FONT_HERSHEY_SIMPLEX, fontSize + 0.2,
        (0, 0, 255), thickness, cv2.LINE_AA
    )