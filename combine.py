import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    
    if im_A is None or im_B is None:
        print('Failed to read')
        return
        
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)

# Directory where your images are stored
image_directory = "/Users/deniztekin/Documents/Programme/Masterarbeit/images/"

# Loop through all files in the 'A/train' directory
for filename in os.listdir(os.path.join(image_directory, "A/train")):
    if filename.endswith(".tiff"):
        path_A = os.path.join(image_directory, "A/train", filename)
        path_B = os.path.join(image_directory, "B/train", filename)
        path_AB = os.path.join(image_directory, "AB", filename)

        image_write(path_A, path_B, path_AB)

