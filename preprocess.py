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

def rotation(images):
    '''
    Created 10 random rotations from a single or list of image
    '''
    rotater = v2.RandomRotation(degrees=(0, 180))
    try:
        rotated_images = []
        for image in images:
            rotated_images.extend([rotater(image) for _ in range(10)])
    except:
        rotated_images = [rotater(image) for _ in range(10)]
    return rotated_images

def pad(images):
    '''
    Created 10 randomly padded image from a single or list of image
    '''
    try:
        padded_images = []
        for image in images:
            padded_images.extend([v2.Pad(padding=padding)(image) for padding in ran_number(10)])
    except:
        padded_images = [v2.Pad(padding=padding)(image) for padding in ran_number(10)]
    return padded_images

def pers(images):
    '''
    Creates 10 random perspectives from a single or list of image
    '''
    perspective_transformer = v2.RandomPerspective(distortion_scale=0.5, p=1.0)
    try:
        perspective_images = []
        for image in images:
            perspective_images.extend([perspective_transformer(image) for _ in range(10)])
    except:
        perspective_images = [perspective_transformer(image) for _ in range(10)]
    return perspective_images

def aff(images):
    '''
    Creates 10 random affine transformations from a single or list of image
    '''
    affine_transfomer = v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
    try:
        affine_images = []
        for image in images:
            affine_images.extend([affine_transfomer(image) for _ in range(10)])
    except:
        affine_images = [affine_transfomer(image) for _ in range(10)]
    return affine_images

def blur(images):
    '''
    Creates 10 gaussian blurs of original image from a single or list of image
    '''
    blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
    try:
        blurred_images = []
        for image in images:
            blurred_images.extend([blurrer(image) for _ in range(10)])
    except:
        blurred_images = [blurrer(image) for _ in range(10)]
    return blurred_images 

def elasticTransform(images):
    '''
    Creates 10 elastic transforms from a single or list of image
    '''
    elastic_transformer = v2.ElasticTransform(alpha=250.0)
    try: 
        elastic_images = []
        for image in images:
            elastic_images.extend([elastic_transformer(image) for _ in range(10)])
    except:
        elastic_images = [elastic_transformer(image) for _ in range(10)]
    return elastic_images

def colourJitter(images):
    '''
    Creates 10 colour jitters from a single or list of image
    '''
    jitter = v2.ColorJitter(brightness=.5, hue=.3)
    try:
        jitter_images = []
        for image in images:
            jitter_images.extend([jitter(image) for _ in range(10)])
    except:
        jitter_images = [jitter(image) for _ in range(10)]
    return jitter_images

def randomInvert(images):
    '''
    Creates 10 random inverts from a single or list of image
    '''
    inverter = v2.RandomInvert()
    try:
        inverter_images = []
        for image in images:
            inverter_images.extend([inverter(image) for _ in range(10)])
    except:
        inverter_images = [inverter(image) for _ in range(10)]
    return inverter_images

def randomPosterize(images):
    '''
    Creates 10 random posterize from a single or list of image
    '''
    posterizer = v2.RandomPosterize(bits=2)
    try:
        posterize_images = []
        for image in images:
            posterize_images.extend([posterizer(image) for _ in range(10)])
    except:
        posterize_images = [posterizer(image) for _ in range(10)]
    return posterize_images

def randomSolarize(images):
    '''
    Creates 10 random solarize from a single or list of image
    '''
    solarizer = v2.RandomSolarize(threshold=10.0)
    try:
        solarize_images = []
        for image in images:
            solarize_images.extend([solarizer(image) for _ in range(10)])
    except:
        solarize_images = [solarizer(image) for _ in range(10)]
    return solarize_images

def grayscale(images):
    '''
    Randomly Grayscales an image with 50% chance
    '''
    try:
        random_Gray_images = []
        for image in images:
            gray = v2.RandomGrayscale([0.5])
            random_Gray_images += gray(image)
    except:
        random_Gray_images = gray(images)
    return random_Gray_images

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
    
    expanded.extend(pad(image))
    expanded.extend(pers(image))
    expanded.extend(aff(image))
    expanded.extend(blur(image))

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