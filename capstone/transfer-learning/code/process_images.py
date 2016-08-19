"""
Pre-process all images to rescale to 227 x 227, split into training and test data and create a text file of image names and classes.
"""

import cv2
import pandas as pd
import glob
import os
import numpy as np

classes = pd.read_csv('../data/top_classes/top_classes.csv', index_col=0)
class_list = list(classes['class'].unique())
train_image_list = []
test_image_list = []

for c in class_list:
    images = glob.glob(os.path.join("../data/top_classes", c, '*.jpg'))
    class_images = []
    for i in images:
        image = cv2.imread(i)
        resized = cv2.resize(image, (227, 227), interpolation = cv2.INTER_AREA)
        image_name = i.split("/")[-1]
        image_path = os.path.join("../data/resized", c, image_name)
        cv2.imwrite(image_path, resized)
        class_images.append(c + "/" + image_name + " " + c)
    np.random.shuffle(class_images)
    num_train = int(0.8 * len(images))
    train_image_list.append(class_images[:num_train])
    test_image_list.append(class_images[num_train:])

training = [item for sublist in train_image_list for item in sublist]
test = [item for sublist in test_image_list for item in sublist]

with open("../data/alexnet_2/train.txt", "w") as f:
    f.write("\n".join(training))

with open("../data/alexnet_2/test.txt", "w") as f:
    f.write("\n".join(test))


