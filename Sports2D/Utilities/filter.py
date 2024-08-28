#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Filter TRC files                             ##
    ##################################################
    
    Filters pandans columns or numpy arrays.
    Available filters: Butterworth, Gaussian, LOESS, Median.
    
    Usage: 
    col_filtered = filter1d(col, *filter_options)
    filter_options = (do_filter, filter_type, butterworth_filter_order, butterworth_filter_cutoff, frame_rate, gaussian_filter_kernel, loess_filter_kernel, median_filter_kernel)
                        bool        str             int                         int                    int         int                     int                 int
    
'''


## INIT
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"



## FUNCTIONS
def butterworth_filter_1d(col, args):
    '''
    1D Zero-phase Butterworth filter (dual pass)
    Deals with nans

    INPUT:
    - col: numpy array
    - order: int
    - cutoff: int
    - framerate: int

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    order, cutoff, framerate = args

    # Filter
    b, a = signal.butter(order/2, cutoff/(framerate/2), 'low', analog = False)
    padlen = 3 * max(len(a), len(b))
    
    # split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered) | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    if idx_sequences[0].size > 0:
        idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > padlen]
    
        # Filter each of the selected sequences
        for seq_f in idx_sequences_to_filter:
            col_filtered[seq_f] = signal.filtfilt(b, a, col_filtered[seq_f])
    
    return col_filtered


def gaussian_filter_1d(col, kernel):
    '''
    1D Gaussian filter

    INPUT:
    - col: numpy array
    - kernel: Sigma kernel value (int)

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    col_filtered = gaussian_filter1d(col, kernel)

    return col_filtered
    

def loess_filter_1d(col, kernel):
    '''
    1D LOWESS filter (Locally Weighted Scatterplot Smoothing)

    INPUT:
    - col: numpy array
    - kernel: Kernel value: window length used for smoothing (int)
    NB: frac = kernel / frames_number

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''
    
    # split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered) 
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    if idx_sequences[0].size > 0:
        idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > kernel]
    
        # Filter each of the selected sequences
        for seq_f in idx_sequences_to_filter:
            col_filtered[seq_f] = lowess(col_filtered[seq_f], seq_f, is_sorted=True, frac=kernel/len(seq_f), it=0)[:,1]
    
    return col_filtered
    

def median_filter_1d(col, kernel):
    '''
    1D median filter

    INPUT:
    - col: numpy array
    - kernel: window size (int)
    
    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''
    
    col_filtered = signal.medfilt(col, kernel_size=kernel)

    return col_filtered
    

def filter1d(col, *filter_options):
    '''
    Choose filter type and filter column

    INPUT:
    - col: Pandas dataframe column
    - filter_options = (do_filter, filter_type, butterworth_filter_order, butterworth_filter_cutoff, frame_rate, gaussian_filter_kernel, loess_filter_kernel, median_filter_kernel)
    
    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''
    
    filter_type = filter_options[1]
    if filter_type == 'butterworth':
        args = (filter_options[2], filter_options[3], filter_options[4])
    if filter_type == 'gaussian':
        args = (filter_options[5])
    if filter_type == 'loess':
        args = (filter_options[6])
    if filter_type == 'median':
        args = (filter_options[7])
        
    # Choose filter
    filter_mapping = {
        'butterworth': butterworth_filter_1d, 
        'gaussian': gaussian_filter_1d, 
        'loess': loess_filter_1d, 
        'median': median_filter_1d
        }
    filter_fun = filter_mapping[filter_type]
    
    # Filter column
    col_filtered = filter_fun(col, args)

    return col_filtered
