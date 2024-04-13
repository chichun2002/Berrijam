import argparse
import os
import time
from typing import Any

from PIL import Image
import io
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.model_selection import StratifiedKFold

from common import load_image_labels, load_single_image, save_model, create_dataloader, create_dataset
from preprocess import generate_data, generate_labels
from sklearn.model_selection import train_test_split

########################################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv', required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True, help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir', required=True, help='Output directory for trained model')
    args = parser.parse_args()
    return args


def load_train_resources(resource_dir: str = 'resources') -> List[Tuple[str, Any]]:
    """
    Load any resources (i.e. pre-trained models, data files, etc) here.
    Make sure to submit the resources required for your algorithms in the sub-folder 'resources'
    :param resource_dir: the relative directory from train.py where resources are kept.
    :return: TBD
    """
    models = []
    
    # # Load the first model (Swin Transformer)
    # model_swin = torchvision.models.swin_v2_b(weights='IMAGENET1K_V1')
    # for param in model_swin.parameters():
    #     param.requires_grad = False 
    
    # num_features_swin = model_swin.head.in_features
    # model_swin.head = nn.Linear(num_features_swin, 2)
    # models.append(("Swin Transformer", model_swin))

    # # Load the second model (Vit)
    # model_vit = torchvision.models.vit_l_16(weights='IMAGENET1K_V1')
    # for param in model_vit.parameters():
    #     param.requires_grad = False
    # num_features_vit = model_vit.hidden_dim
    # model_vit.heads = nn.Linear(num_features_vit, 2)
    # models.append(("Vit_l", model_vit))
    
    # # Load the second model (Vit)
    # model_vit = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
    # for param in model_vit.parameters():
    #     param.requires_grad = False
    # num_features_vit = model_vit.hidden_dim
    # model_vit.heads = nn.Linear(num_features_vit, 2)
    # models.append(("Vit_b", model_vit))

    # # # Load the third model (EfficientNet)
    # model_efficientnet = torchvision.models.efficientnet_b4(weights='IMAGENET1K_V1')
    # for param in model_efficientnet.parameters():
    #     param.requires_grad = False
    # num_features_efficientnet = model_efficientnet.classifier[-1].in_features
    # model_efficientnet.classifier = nn.Linear(num_features_efficientnet, 2)
    # models.append(("EfficientNet", model_efficientnet))
    

    model_mnasnet = torch.load('Pretrained_Models/mnasnet')
    for param in model_mnasnet.parameters():
        param.requires_grad = False 
    num_features_mnasnet = model_mnasnet.classifier[-1].in_features
    model_mnasnet.classifier[-1]  = nn.Linear(num_features_mnasnet, 2)
    models.append(("MnasNet", model_mnasnet))

    # model_efficientnet = torch.load('Pretrained_Models/efficientnet')
    # for param in model_efficientnet.parameters():
    #     param.requires_grad = False
    # num_features_efficientnet = model_efficientnet.classifier[-1].in_features
    # model_efficientnet.classifier = nn.Linear(num_features_efficientnet, 2)
    # models.append(("EfficientNet", model_efficientnet))
    
    # model_shufflenet = torch.load('Pretrained_Models/shufflenet')
    # for param in model_shufflenet.parameters():
    #     param.requires_grad = False
    # num_features_shufflenet = model_shufflenet.fc.in_features
    # model_shufflenet.fc = nn.Linear(num_features_shufflenet, 2)
    # models.append(("ShuffleNet", model_shufflenet))
    
    # model_regnet = torch.load('Pretrained_Models/regnet')
    # for param in model_regnet.parameters():
    #     param.requires_grad = False
    # num_features_regnet = model_regnet.fc.in_features
    # model_regnet.fc = nn.Linear(num_features_regnet, 2)
    # models.append(("RegNet", model_regnet))
    return models
    

def train(output_dir: str, model, name: str, num_epochs, dataloader, size, optimizer, scheduler, criterion) -> Any:
    """
    Trains a classification model using the training images and corresponding labels.

    :param images: the list of image (or array data)
    :param labels: the list of training labels (str or 0,1)
    :param output_dir: the directory to write logs, stats, etc to along the way
    :return: model: model file(s) trained.
    """
    # TODO: Implement your logic to train a problem specific model here
    # Along the way you might want to save training stats, logs, etc in the output_dir
    # The output from train can be one or more model files that will be saved in save_model function.
    print(size)
    #since = time.time()
    best_acc = 0.0
    best_loss = float('inf')


    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
         

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()
        
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / size[phase]
            epoch_acc = running_corrects.double() / size[phase]

            if phase == 'train':
                scheduler.step()
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and (epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss)):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model = model

        print()

    # time_elapsed = time.time() - since
    # print(f'Model: {name}')
    # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print(f'Best val Acc: {best_acc:.4f}')
    # print(f'Corresponding Loss: {best_loss:.4f}')

    # # # Plotting training and validation loss/accuracy curves
    
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label=f'Training Loss ({name})')
    # plt.plot(val_losses, label=f'Validation Loss ({name})')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(train_accs, label=f'Training Accuracy ({name})')
    # plt.plot(val_accs, label=f'Validation Accuracy ({name})')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.tight_layout()
    # # Save the plot in the current working directory
    # plot_file_path = f"{name}_training_curves.png"
    # plt.savefig(plot_file_path)
    # print(f"Training curves saved successfully at {plot_file_path}")

    model = best_model
    
    return model

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str):
    """
    The main body of the train.py responsible for
     1. loading resources
     2. loading labels
     3. loading data
     4. transforming data
     5. training model
     6. saving trained model

    :param train_input_dir: the folder with the CSV and training images.
    :param train_labels_file_name: the CSV file name
    :param target_column_name: Name of the target column within the CSV file
    :param train_output_dir: the folder to save training output.
    """
    since = time.time()


    # load label file
    labels_file_path = os.path.join(train_input_dir, train_labels_file_name)
    df_labels = load_image_labels(labels_file_path)

    # load in images and labels
    train_images = []
    train_labels = []

    # Now iterate through every record and load in the image data files
    # Given the small number of data samples, iterrows is not a big performance issue.
    for index, row in df_labels.iterrows():
        try:
            filename = row['Filename']
            label = row[target_column_name]
            if label == 'Yes':
                label = 1
            elif label == 'No':
                label = 0

            print(f"Loading image file: {filename}")
            image_file_path = os.path.join(train_input_dir, filename)
            image = load_single_image(image_file_path)

            train_labels.append(label)
            train_images.append(image)
            
        except Exception as ex:
            print(f"Error loading {index}: {filename} due to {ex}")

    print(f"Loaded {len(train_labels)} training images and labels")
    os.makedirs(train_output_dir, exist_ok=True)

    augmented_images = []
    augemented_labels = []
    for image in train_images:
        aug_images = generate_data(image)  
        for augmented in aug_images:
            augmented_images.append(augmented)

    # Convert PIL.Image.Image objects to PIL.PngImagePlugin.PngImageFile objects
    for i in range(len(augmented_images)):
        # Convert the image to bytes
        img_byte_array = io.BytesIO()
        augmented_images[i].save(img_byte_array, format='PNG')
        
        # Open the image from the bytes
        augmented_images[i] = Image.open(img_byte_array)
        
    train_images.extend(augmented_images)
    
    for label in train_labels:
        aug_labels = generate_labels(label, 50)
        for augmented in aug_labels:
            augemented_labels.append(augmented)
            
    train_labels.extend(augemented_labels)
    
    print(f"Loaded {len(train_images)} training images and labels with augmented data")
    
    # load pre-trained models or resources at this stage.
    models = load_train_resources()
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
    for name, model_obj in models:
        model = model_obj.to(device)
        
        # Define a range of image sizes for evaluation during inference
        resize_sizes = [224, 256, 288]

        # Perform inference on validation data with different image sizes
        best_accuracy = 0.0
        optimal_resize_size = None

        for size in resize_sizes:
            print(f"Size: {size}")
            # Define data transforms with resizing
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(size,scale=(0.001, 1)),
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                    transforms.AutoAugment(policy= transforms.AutoAugmentPolicy.IMAGENET),
                    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
                'val': transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
            
            # Now create datasets for training and validation
            dataset_train = create_dataset(train_images, train_labels, data_transforms['train'])
            dataset_val = create_dataset(val_images, val_labels, data_transforms['val'])

            dataloaders = {'train' : create_dataloader(dataset_train), 'val' : create_dataloader(dataset_val)}
            
            # Uncomment TO SHOW IMAGES

            # def imshow(inp, title=None):
            #     """Display image for Tensor."""
            #     inp = inp.numpy().transpose((1, 2, 0))
            #     mean = np.array([0.485, 0.456, 0.406])
            #     std = np.array([0.229, 0.224, 0.225])
            #     inp = std * inp + mean
            #     inp = np.clip(inp, 0, 1)
            #     plt.imshow(inp)
            #     if title is not None:
            #         plt.title(title)
            #     plt.pause(0.001)  # pause a bit so that plots are updated


            # # Get a batch of training data
            # inputs, classes = next(iter(dataloaders['train']))

            # # Make a grid from batch
            # out = torchvision.utils.make_grid(inputs)

            # imshow(out, title="kek")
            # plt.show()
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            model = train('output', model, name, 10, dataloaders, {'train': len(dataset_train), 'val':len(dataset_val)}, optimizer, exp_lr_scheduler, criterion)
            
            # Evaluate model accuracy on validation data
            accuracy = evaluate_model(model, dataloaders['val'])

            # Record the best accuracy and optimal resize size
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                optimal_resize_size = size
            
        print(f"optimal_resize_size = {optimal_resize_size}")
        # Define data transforms with resizing
        optimal_data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(optimal_resize_size,scale=(0.001, 1)),
                    transforms.Resize(optimal_resize_size),
                    transforms.CenterCrop(optimal_resize_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                    transforms.AutoAugment(policy= transforms.AutoAugmentPolicy.IMAGENET),
                    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(optimal_resize_size),
                transforms.CenterCrop(optimal_resize_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # Now create datasets for training and validation
        dataset_train = create_dataset(train_images, train_labels, optimal_data_transforms['train'])
        dataset_val = create_dataset(val_images, val_labels, optimal_data_transforms['val'])
        dataloaders = {'train' : create_dataloader(dataset_train), 'val' : create_dataloader(dataset_val)}
        model = train('output', model, name, 50, dataloaders, {'train': len(dataset_train), 'val':len(dataset_val)}, optimizer, exp_lr_scheduler, criterion)
        print(f"Model = {name}, Best Accuracy = {best_accuracy}")
        
        save_model(model, name, target_column_name, train_output_dir)
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
if __name__ == '__main__':
    """
    Example usage:
    
    python train.py -d "path/to/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models"
     
    """
    args = parse_args()
    train_data_image_dir = args.train_data_image_dir
    train_data_labels_csv = args.train_data_labels_csv
    target_column_name = args.target_column_name
    trained_model_output_dir = args.trained_model_output_dir

    main(train_data_image_dir, train_data_labels_csv, target_column_name, trained_model_output_dir)

########################################################################################################################
