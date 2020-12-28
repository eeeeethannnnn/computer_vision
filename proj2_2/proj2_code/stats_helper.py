import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    '''
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    '''
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    #print(dir_name)
    jpg = [f for f in glob.glob(dir_name + "**/*.jpg", recursive = True)]
    input = np.array([])
    #print(len(jpg))
    i = 1
    for f in jpg:
        tmpImage = Image.open(f)
        tmpImage = ImageOps.grayscale(tmpImage)
        tmpNumpy = np.array(tmpImage)
        tmpNumpy = tmpNumpy.flatten()
        tmpNumpy = tmpNumpy / 255
        input = np.append(input, tmpNumpy)
        i = i + 1
    scaler = StandardScaler()
    scaler.partial_fit(input.reshape(-1, 1))
    mean = scaler.mean_
    std = scaler.scale_
# raise NotImplementedError('compute_mean_and_std not implemented')

############################################################################
# Student code end
############################################################################
    return mean, std
