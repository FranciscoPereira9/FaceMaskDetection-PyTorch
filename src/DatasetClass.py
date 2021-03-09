import utils
# PyTorch
import torch
import torchvision
from torch.utils.data import Dataset
# Data management
import numpy as np
import pandas as pd
from IPython.display import display
# File interpretation
import os
import xml.etree.ElementTree as ET
# Image
import cv2


# Create dataset object
class Dataset(Dataset):

    # Constructor
    def __init__(self, ann_dir, img_dir, transform=None, mode='train'):

        # Image directories
        self.ann_dir = ann_dir
        self.img_dir = img_dir

        # The transform is goint to be used on image
        self.transform = transform

        # Create dataframe to hold info
        self.data = pd.DataFrame(columns=['Filename', 'BoundingBoxes', 'Labels'])

        # Append rows with image filename and respective bounding boxes to the df
        for file in enumerate(os.listdir(img_dir)):

            # Find image annotation file
            ann_file_path = os.path.join(ann_dir, file[1][:-4]) + '.xml'

            # Read XML file and return bounding boxes and class attributes
            objects = self.read_XML_classf(ann_file_path)

            # Create list of labels in an image
            list_labels = utils.encoded_labels(objects[0]['labels'])

            # Create list of bounding boxes in an image
            list_bb = []
            for i in objects[0]['objects']:
                list = [i['xmin'], i['ymin'], i['xmax'], i['ymax']]
                list_bb.append(list)

            # Create dataframe object with row containing [(Image file name),(Bounding Box List)]
            df = pd.DataFrame([[file[1], list_bb, list_labels]], columns=['Filename', 'BoundingBoxes', 'Labels'])
            self.data = self.data.append(df)

        if mode == 'train':
            self.data = self.data[:9]
        elif mode == 'validation':
            self.data = self.data[21:26]

        # Number of images in dataset
        self.len = self.data.shape[0]

        # Get the length

    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):

        # Image file path
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])

        # Open image file and tranform to tensor
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get labels
        labels = torch.tensor(self.data.iloc[idx, 2])

        # Get bounding box coordinates
        bbox = torch.tensor(self.data.iloc[idx, 1])

        # If any, aplly tranformations to image and bounding box mask
        if self.transform:
            transformed = self.transform(image=img, bboxes=bbox)
            img = transformed['image']
            bbox = torch.tensor(transformed['bboxes'])

        # Transform img to tensor
        img = torch.tensor(img.transpose())

        # Build Targer dict
        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        return img, target

    # XML reader -> returns dictionary with image bounding boxes sizes
    def read_XML_classf(self, ann_file_path):
        bboxes = [{
            'file': ann_file_path,
            'labels': [],
            'objects': []
        }]

        # Reading XML file objects and print Bounding Boxes
        tree = ET.parse(ann_file_path)
        root = tree.getroot()
        objects = root.findall('object')

        for obj in objects:
            # label
            label = obj.find('name').text
            bboxes[0]['labels'].append(label)

            # bbox dimensions
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bboxes[0]['objects'].append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

        return bboxes

