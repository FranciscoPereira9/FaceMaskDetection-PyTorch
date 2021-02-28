
# Face Mask Detection Model Using PyTorch

# Import Libraries

# File interpretation
import os
# Others
import time

import albumentations as A
# Data management
import numpy as np
# PyTorch
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import DatasetClass
import utils

# OpenCV
# Image Plots

# Find Annotation Files
# Annotations directory path
ann_directory = '.\\annotations'
# Listed directory
ann_files = os.listdir(ann_directory)


# Find Image Files
# Image directory path
img_directory = '.\\images'
# Listed directory
img_files = os.listdir(img_directory)


# Composer Tranformations to pass into Dataset Class -> needs to be Albumentation transform
transform = A.Compose(
    [
    A.HorizontalFlip(p=0.1),
    A.RandomBrightnessContrast(p=0.4),
    #A.Rotate(limit=40, p=0.5, border_mode = cv.BORDER_CONSTANT)
    ],
    bbox_params= A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=[])
)


# Create Data Pipeline
# Training Data
dataset_train = DatasetClass.Dataset(ann_directory,img_directory, mode = 'train')
loader = DataLoader(dataset_train, batch_size=3, shuffle=True, collate_fn=utils.collate_fn)
# Validation Data
dataset_validation = DatasetClass.Dataset(ann_directory,img_directory, mode = 'validation')
loader_val = DataLoader(dataset_validation, batch_size=3, shuffle=True, collate_fn=utils.collate_fn)

# Set up Faster R-CNN
# Setting up GPU device if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Nº of classes: without_mask, with_mask, mask_weared_incorrect
num_classes = 3

# Load a pre-trained model on COCO Dataset
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Get number of input features
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)


# Set Hyper-parameters
# Network params
params = [p for p in model.parameters() if p.requires_grad]

# Optimizers
# optimizer = torch.optim.Adam(params, lr=0.01)
optimizer = torch.optim.SGD(params, lr=0.01,momentum=0.9, weight_decay=0.0005)

# Learning Rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

epochs=1

# Train the model
loss_list = []

utils.train_model(model,loader,optimizer,lr_scheduler,epochs,device)


# Make predictions
