import os
from typing import Any

import pandas as pd

from torchvision.transforms import v2

import cv2
import numpy as np

import random 
from PIL import Image

from common import load_single_image
import sys

def ran_number(length):
    list = []
    for _ in range(length):
        list.append(random.randint(1,100))
    return list

def rotation(image):
    '''
    Created 10 random rotations on a single image
    '''
    rotater = v2.RandomRotation(degrees=(0, 180))
    rotated_images = [rotater(image) for _ in range(10)]
    return rotated_images

def rotation_multi(images):
    '''
    Created 10 random rotations on a single image
    '''
    try:
        rotated_images = []
        for image in images:
            rotater = v2.RandomRotation(degrees=(0, 180))
            rotated_images += [rotater(image) for _ in range(10)]
    except:
        rotated_images = rotation(images)
    return rotated_images

def pad(image):
    '''
    Created 10 randomly padded image
    '''
    padded_images = [v2.Pad(padding=padding)(image) for padding in ran_number(10)]
    return padded_images

def pers(image):
    '''
    Creates 10 random perspectives
    '''
    perspective_transformer = v2.RandomPerspective(distortion_scale=0.5, p=1.0)
    perspective_images = [perspective_transformer(image) for _ in range(10)]
    return perspective_images

def aff(image):
    '''
    Creates 10 random affine transformations
    '''
    affine_transfomer = v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
    affine_images = [affine_transfomer(image) for _ in range(10)]
    return affine_images

def blur(image):
    '''
    Creates 10 gaussian blurs of original image
    '''
    blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
    blurred_images = [blurrer(image) for _ in range(10)]
    return blurred_images 

def elasticTransform(image):
    elastic_transformer = v2.ElasticTransform(alpha=250.0)
    return [elastic_transformer(image) for _ in range(10)]

def colourJitter(image):
    jitter = v2.ColorJitter(brightness=.5, hue=.3)
    return [jitter(image) for _ in range(10)]

def randomInvert(image):
    inverter = v2.RandomInvert()
    return [inverter(image) for _ in range(4)]

def randomPosterize(image):
    posterizer = v2.RandomPosterize(bits=2)
    return [posterizer(image) for _ in range(4)]

def randomSolarize(image):
    solarizer = v2.RandomSolarize(threshold=10.0)
    return [solarizer(image) for _ in range(4)]


def generate_data(image):
    '''
    Generate augmented data from single image and return list of images with corresponding list of labels
    '''
    # convert_tensor = v2.ToTensor()
    # image = convert_tensor(image)
    expanded = []
    expanded += rotation(image)
    # for i in expanded[1:10]:
    #     expanded += pad(i)
    
    expanded +=(pad(image))
    expanded +=(pers(image))
    expanded +=(aff(image))
    expanded +=(blur(image))

    # expanded += randomSolarize(image)
    
    return expanded

def generate_labels(label, len):
    '''
    Generates list of labels corresponding to augmented data
    '''
    return [label] * len

if __name__ == '__main__':
    '''
    preprocess.py testing
    python3 preprocess.py "Data - Is Epic Intro 2024-03-25" "apocalypse_v3_59sec-178285.mp3.png" 
    '''
    image_file_path = os.path.join(sys.argv[1], sys.argv[2])
    image = load_single_image(image_file_path)
    aug_data = generate_data(image)
    print(len(aug_data))
    # aug_data[len(aug_data) - 1].show()
    # for i in aug_data[len(aug_data) - 1]:
    #     i.show()
    # im = aug_data[2].convert["RGB"]
    # im.show()