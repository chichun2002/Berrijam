import os
from typing import Any

import pandas as pd

import torch
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import save_image

import cv2
import numpy as np

import random 
from PIL import Image

from common import load_image_labels, load_predict_image_names, load_single_image

def ran_number(length):
    list = []
    for i in range(length):
        list.append(random.randint(1,100))
    return list

def load_train_test(image_dir: str, image_list: str):
    '''
    Loads the images and splits them into the test and train
    '''
    image_list_file = os.path.join(image_dir, image_list)
    image_label = load_image_labels(image_list_file)

    pos = []
    neg = []
    for filename, label in zip(list(image_label['Filename']),list(image_label['Is Epic'])):
        if (label == "Yes"):
            pos.append(filename)
        else:
            neg.append(filename)

    pos_test = pos.pop(random.randrange(len(pos)))
    neg_test = neg.pop(random.randrange(len(neg)))

    return pos, [pos_test], neg, [neg_test]

def rotation(image):
    '''
    Created 10 random rotations on a single image
    '''
    rotater = v2.RandomRotation(degrees=(0, 180))
    rotated_images = [rotater(image) for _ in range(10)]
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

def generate_data(image_dir: str, image_list: str):
    '''
    Generate augmented data and return list of images with corresponding list of labels
    '''
    data = load_train_test(image_dir, image_list)
    expanded = []
    for set in data:
        for image_name in set:
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path)
            expanded.append(image)
            expanded.append(rotation(image))
            expanded.append(pad(image))
            expanded.append(pers(image))
            expanded.append(aff(image))
            expanded.append(blur(image))
    
    return expanded[0], expanded[1], expanded[2], expanded[3]

