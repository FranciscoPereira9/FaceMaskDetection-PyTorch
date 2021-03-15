# Imports
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import argparse
from DatasetClass import MyDataset
from datetime import datetime
from utils.engine import train_one_epoch, evaluate
import utils.helper as helper


# ----------------------------------------------- Default Arguments ----------------------------------------------------

batch_size = 1
epochs = 1
optimizer_type = 'sgd'
lr = 0.1


# ----------------------------------------------- Parsed Arguments -----------------------------------------------------

# Initiate the parser
parser = argparse.ArgumentParser()

# Add long and short argument
parser.add_argument("--batch_size", help="Set batch size.")
parser.add_argument("--epochs", help="Set number of epochs.")
parser.add_argument("--optimizer", help="Set optimizer. Can be 'sgd' or 'adam'.")
parser.add_argument("--learning_rate", help="Set learning rate.")

# Read arguments from the command line
args = parser.parse_args()

# Check arguments
print(33*"-")
if args.batch_size:
    batch_size = int(args.batch_size)
out = "| Batch size: " + str(batch_size)
print(out, (30 - len(out))*' ', '|')
if args.epochs:
    epochs = int(args.epochs)
out = "| Number of epochs: " + str(epochs)
print(out, (30 - len(out))*' ', '|')
if args.optimizer:
    optimizer_type = args.optimizer
out = '| Optimizer type: ' + optimizer_type
print(out, (30 - len(out))*' ', '|')
if args.learning_rate:
    lr = float(args.learning_rate)
out = '| Learning rate: ' + str(lr)
print(out, (30 - len(out))*' ', '|')
print(33*"-")


# ----------------------------------------------- Dataset Files --------------------------------------------------------

# Annotations directory path
ann_directory = '../annotations'
# Listed directory
ann_files = os.listdir(ann_directory)

# Image directory path
img_directory = '../images'
# Listed directory
img_files = os.listdir(img_directory)


# ----------------------------------------------- Create Data Pipeline -------------------------------------------------

# Training Data
dataset_train = MyDataset(ann_directory, img_directory, mode='train')
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=helper.collate_fn)

# Validation Data
dataset_validation = MyDataset(ann_directory, img_directory, mode='validation')
loader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, collate_fn=helper.collate_fn)


# ----------------------------------------------- Set Up the Model -----------------------------------------------------

# Setting up GPU device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Nº of classes: background, with_mask, mask_weared_incorrect, without_mask and build model (faster r-cnn)
num_classes = 4
model = helper.build_model(num_classes)
model = model.to(device)

# Network params
params = [p for p in model.parameters() if p.requires_grad]

# Optimizers
if optimizer_type == 'adam':
    optimizer = torch.optim.Adam(params, lr=lr)
else:
    optimizer = torch.optim.SGD(params, lr=lr)

if epochs > 10:
    step_size = round(epochs/10)
else:
    step_size = 1

# Learning Rate, lr decreases by half every step_size
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)


# ----------------------------------------------- Train the Model ------------------------------------------------------

# train_log = pd.DataFrame(columns=['Epoch', 'Learning_Rate', 'Loss', 'Loss_Classifier', 'Loss_BBox_Regression',
#                                  'Loss_RPN_BBox_Regression', 'Time'])

for epoch in range(1, epochs+1):
    # train for one epoch, printing every 2 iterations
    training_results = train_one_epoch(model, optimizer, loader_train, device, epoch, print_freq=1)
    # evaluate on the validation data set
    evaluate(model, loader_validation, device=device)
    # update the learning rate
    lr_scheduler.step() 


# ----------------------------------------------- Save the Model -------------------------------------------------------

# Save model with current date
now = datetime.now()
d = now.strftime("%Y_%b_%d")
PATH = '../models/model_ces_'+d+'.pt'
torch.save(model.state_dict(), PATH)




