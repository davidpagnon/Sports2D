#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Data Processing and Calculations             ##
    ##################################################

    - A function for interpolating zeros and NaNs in datasets.
    - A function to resample video frames to a target frequency.
    - Converts 2D points into angle measurements.
    - Computes the Euclidean distance between two points.
    - A function to find the minimum values with indices in a dataset.
    - Processes coordinates and angles for biomechanical analysis.
    - Function to flip direction from left to right in datasets.
    - Calculates angles based on coordinate data.
'''


## INIT
import re
import logging
import argparse
import numpy as np


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def make_homogeneous(list_of_arrays):
    '''
    Make a list of arrays (or a list of lists) homogeneous by padding with nans

    Example: foo = [[array([nan, 656.02643776]), array([nan, nan])],
                    [array([1, 2, 3]), array([1, 2])]]
    becomes foo_updated = array([[[nan, 656.02643776, nan], [nan, nan, nan]],
                                [[1., 2., 3.], [1., 2., nan]]])
    Or foo = [[1, 2, 3], [1, 2], [3, 4, 5]]
    becomes foo_updated = array([[1., 2., 3.], [1., 2., nan], [3., 4., 5.]])

    INPUTS:
    - list_of_arrays: list of arrays or list of lists

    OUTPUT:
    - np.array(list_of_arrays): numpy array of padded arrays
    '''
    
    def get_max_shape(list_of_arrays):
        if isinstance(list_of_arrays[0], list):
            # Maximum length at the current level plus the max shape at the next level
            return [max(len(arr) for arr in list_of_arrays)] + get_max_shape(
                [item for sublist in list_of_arrays for item in sublist])
        else:
            # Determine the maximum shape across all list_of_arrays at this level
            return [len(list_of_arrays)] + [max(arr.shape[i] for arr in list_of_arrays) for i in range(list_of_arrays[0].ndim)]

    def pad_with_nans(list_of_arrays, target_shape):
        '''
        Recursively pad list_of_arrays with nans to match the target shape.
        '''
        if isinstance(list_of_arrays, np.ndarray):
            # Pad the current array to the target shape
            pad_width = [(0, max_dim - curr_dim) for curr_dim, max_dim in zip(list_of_arrays.shape, target_shape)]
            return np.pad(list_of_arrays.astype(float), pad_width, constant_values=np.nan)
        # Recursively pad each array in the list
        return [pad_with_nans(array, target_shape[1:]) for array in list_of_arrays]

    # Pad all missing dimensions of arrays with nans
    list_of_arrays = [np.array(arr, dtype=float) if not isinstance(arr, np.ndarray) else arr for arr in list_of_arrays]
    max_shape = get_max_shape(list_of_arrays)
    list_of_arrays = pad_with_nans(list_of_arrays, max_shape)

    return np.array(list_of_arrays)


def read_frame(cap, frame_idx):
    success, frame = cap.read()
    if not success:
        logging.warning(f"Failed to grab frame {frame_idx}.")
        return None
    return frame


def get_leaf_keys(config, prefix=''):
    '''
    Flatten configuration to map leaf keys to their full path
    '''

    leaf_keys = {}
    for key, value in config.items():
        if isinstance(value, dict):
            leaf_keys.update(get_leaf_keys(value, prefix=prefix + key + '.'))
        else:
            leaf_keys[prefix + key] = value
    return leaf_keys


def set_nested_value(config, flat_key, value):
    '''
    Update the nested dictionary based on flattened keys
    '''

    keys = flat_key.split('.')
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def str2bool(v):
    '''a
    Convert a string to a boolean value.
    '''
    
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')