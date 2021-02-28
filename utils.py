# Image Plots
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# Data managements
import numpy as np
import time

def draw_bounding_boxes(img, bboxes):
    """Draws bounding boxes in given images.

        Args:
          img:
            PIL image.
          bboxes:
            list of lists with bounding boxes coordinates (xmin, ymin, xmax, ymax)

        Returns:
          None
        """

    # fetching the dimensions
    wid, hgt = img.size
    print(str(wid) + "x" + str(hgt))

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    for coordinates in bboxes:
        x = coordinates[0]
        y = coordinates[1]
        width = coordinates[2] - coordinates[0]
        height = coordinates[3] - coordinates[1]

        # Create Rectangle patches and add the patches to the axes
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none', fill=False)
        ax.add_patch(rect)

        # Show cropped bb
        # image_croped = img.crop((x, y, x+width, y+height))
        # fig = plt.figure()
        # plt.imshow(image_croped)

    plt.show(img)



def encoded_labels(lst_labels):
    """Encodes label classes from string to integers.

        Labels are encoded accordingly:
            - with_mask => 1
            - mask_weared_incorrect => 2
            - without_mask => 3

            Args:
              lst_labels:
                A list with classes in string format (e.g. ['with_mask', 'mask_weared_incorrect'...]).

            Returns:
              encoded:
                A list with integers that represent each class.
            """

    encoded=[]
    for label in lst_labels:
        if label == "with_mask":
            code = 1
        elif label == "mask_weared_incorrect":
            code = 2
        else:
            code = 3
        encoded.append(code)
    return encoded


def train_model(model, loader, optimizer, scheduler, epochs, device):
    # Train the model
    loss_list = []

    for epoch in range(epochs):
        print('Starting epoch...... {}/{} '.format(epoch + 1, epochs))
        iter = 0
        loss_sub_list = []
        start = time.time()
        for images, targets in loader:
            # Agregate images in batch loader
            images = list(image.to(device) for image in images)

            # Agregate targets in batch loader
            targets = [{key: val.to(device) for key, val in target.items()} for target in targets]

            # Sets model to train mode (just a flag)
            model.train()

            # Output of model returns loss and detections
            output = model(images, targets)

            # Calculate Cost
            losses = sum(loss for loss in output.values())
            loss_value = losses.item()
            loss_sub_list.append(loss_value)
            print(' --> Loss: {:.3f}'.format(loss_value))

            # Update optimizer and learning rate
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()
            iter += 1
            print('Iteration: ', iter)
        end = time.time()

        # print the loss of epoch
        epoch_loss = np.mean(loss_sub_list)
        loss_list.append(epoch_loss)
        print('Epoch loss: {:.3f} , time used: ({:.1f}s)'.format(epoch_loss, end - start))


def collate_fn(batch):
    return tuple(zip(*batch))