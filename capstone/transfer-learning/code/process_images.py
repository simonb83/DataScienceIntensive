"""
Pre-process all images to rescale to 256 x 256, split into training, validation and test data and create a text file of image names and classes.
"""

from skimage import io
from skimage.transform import resize
import pandas as pd
import glob
import os
import numpy as np
import json

classes = pd.read_csv('../data/top_classes/top_classes.csv', index_col=0)
class_list = list(classes['class'].unique())
train_image_list = []
val_image_list = []
test_image_list = []

class_mapping = dict()

for j, c in enumerate(class_list):
    class_mapping[c] = j

    # Make the new directory for storing resized images
    if not os.path.exists(os.path.join("../data/resized", c)):
        os.makedirs(os.path.join("../data/resized", c))
    images = glob.glob(os.path.join("../data/top_classes", c, '*.jpg'))
    class_images = []
    for i in images:
        image = io.imread(i)
        resized = resize(image, (256, 256))
        image_name = i.split("/")[-1]
        image_path = os.path.join("../data/resized", c, image_name)
        io.imsave(image_path, resized)
        class_images.append(c + "/" + image_name + " " + str(j))
    np.random.shuffle(class_images)
    train_split = int(0.8 * len(images)) #E.g., 800 per class
    num_train = int(0.83 * num_train) # E.g., 667 per class

    train_image_list.append(class_images[:num_train])
    val_image_list.append(class_images[num_train:train_split])
    test_image_list.append(class_images[train_split:])

training = [item for sublist in train_image_list for item in sublist]
validation = [item for sublist in val_image_list for item in sublist]
test = [item for sublist in test_image_list for item in sublist]

with open("../data/alexnet_2/train.txt", "w") as f:
    f.write("\n".join(training))

with open("../data/alexnet_2/val.txt", "w") as f:
    f.write("\n".join(validation))

with open("../data/alexnet_2/test.txt", "w") as f:
    f.write("\n".join(test))

json.dump(class_mapping, open("../data/alexnet_2/class_mapping.txt", "w"))

