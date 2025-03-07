from typing import Any
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image

def create_dataset(images, labels, transform):
    class CustomImageDataset(Dataset):
        def __init__(self, images, labels, transform):
            self.imgs = images
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, index) -> Any:
            image = self.transform(self.imgs[index].convert("RGB"))
            label = self.labels[index]
            return image, label
    
    return CustomImageDataset(images, labels, transform)

def create_dataloader(dataset):
    dl = torch.utils.data.DataLoader(dataset, batch_size = 8, shuffle=True, num_workers=0)
    return dl

def create_data_transform(input_size):
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size,scale=(0.001, 1)),
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                
                transforms.TrivialAugmentWide(),
                transforms.AutoAugment(),

                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),

                # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                # transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.3,0.3)),
                

                
                
                # transforms.GaussianBlur(5, sigma=(0.1, 2.0)),

                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    return data_transforms

########################################################################################################################
# Data Loading functions
########################################################################################################################
def load_image_labels(labels_file_path: str):
    """
    Loads the labels from CSV file.

    :param labels_file_path: CSV file containing the image and labels.
    :return: Pandas DataFrame
    """
    df = pd.read_csv(labels_file_path)
    return df


def load_predict_image_names(predict_image_list_file: str) -> [str]:
    """
    Reads a text file with one image file name per line and returns a list of files
    :param predict_image_list_file: text file containing the image names
    :return list of file names:
    """
    with open(predict_image_list_file, 'r') as file:
        lines = file.readlines()
    # Remove trailing newline characters if needed
    lines = [line.rstrip('\n') for line in lines]
    return lines


def load_single_image(image_file_path: str) -> Image:
    """
    Load the image.

    NOTE: you can optionally do some initial image manipulation or transformation here.

    :param image_file_path: the path to image file.
    :return: Image (or other type you want to use)
    """
    # Load the image
    image = Image.open(image_file_path)

    # The following are examples on how you might manipulate the image.
    # See full documentation on Pillow (PIL): https://pillow.readthedocs.io/en/stable/

    # To make the image 50% smaller
    # Determine image dimensions
    # width, height = image.size
    # new_width = int(width * 0.50)
    # new_height = int(height * 0.50)
    # image = image.resize((new_width, new_height))

    # To crop the image
    # (left, upper, right, lower) = (20, 20, 100, 100)
    # image = image.crop((left, upper, right, lower))

    # To view an image
    # image.show()

    # Return either the pixels as array - image_array
    # To convert to a NumPy array
    # image_array = np.asarray(image)
    # return image_array

    # or return the image
    return image


########################################################################################################################
# Model Loading and Saving Functions
########################################################################################################################

def save_model(model: Any, model_name: str, target: str, output_dir: str):
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation.

    Common Deep Learning Model File Formats are:

        SavedModel (TensorFlow)
        Pros: Framework-agnostic format, can be deployed in various environments. Contains a complete model representation.
        Cons: Can be somewhat larger in file size.

        HDF5 (.h5) (Keras)
        Pros: Hierarchical structure, good for storing model architecture and weights. Common in Keras.
        Cons: Primarily tied to the Keras/TensorFlow ecosystem.

        ONNX (Open Neural Network Exchange)
        Pros: Framework-agnostic format aimed at improving model portability.
        Cons: May not support all operations for every framework.

        Pickle (.pkl) (Python)
        Pros: Easy to save and load Python objects (including models).
        Cons: Less portable across languages and environments. Potential security concerns.

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param model: the model that you want to save.
    :param target: the target value - can be useful to name the model file for the target it is intended for
    :param output_dir: the output directory to same one or more model files.
    """
    # TODO: implement your model saving code here
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file path for saving model
    model_file_path = os.path.join(output_dir, f"{model_name}_{target}_model.pth")

    # Save the model
    torch.save(model, model_file_path)

    print(f"Model '{model_name}' saved successfully at {model_file_path}")


def load_model(trained_model_dir: str, target_column_name: str) -> Any:
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation and should mirror save_model()

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param trained_model_dir: the directory where the model file(s) are saved.
    :param target_column_name: the target value - can be useful to name the model file for the target it is intended for
    :returns: the model
    """
    models = []
    for item in os.listdir(trained_model_dir):
        path = os.path.join(trained_model_dir, item)
        models.append(torch.load(path))
        # return torch.load(path)
    return models