"""
Make predictions using pre-trained caffe models.

Make simple predictions using single test images, or averaged predictions using image crops and / or augmented image strategies.

:output: Save per class predictions for each test image and prediction method.
"""

import caffe
import argparse
import os
import numpy as np
import skimage.exposure as exposure
import skimage.transform as transform
import timeit

DEPLOY = "/home/ubuntu/sb/capstone/transfer-learning/models/alexnet_4/deploy.prototxt"
caffe.set_mode_gpu()


def initialize_model(model_def, model_weights, mean_image):
    """
    Initialize a caffe model and data transformer
    :params model_def: Path to model prototxt file
    :params model_weights: Path to model weights
    :params mean_image: Path to mean image file
    :return net: Net object
    :return transformer: Transformer object
    """
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    mu = np.load(mean_image)
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    net.blobs['data'].reshape(1, 3, 227, 227)

    return net, transformer

def augment_image(img):
    """
    Augment an image using a combination of lightening, darkening, rotation and mirror images.
    :params img: Image as numpy array
    :return: array of augmented images 
    """
    augmented_images = []
    augmented_images.append(np.fliplr(img))
    for g in [0.45, 0.65, 0.85, 1.25, 1.5, 2]:
        new_img = exposure.adjust_gamma(img, gamma=g)
        augmented_images.append(new_img)
        augmented_images.append(np.fliplr(new_img))
    new_img = transform.rotate(img, 180)
    augmented_images.append(new_img)
    augmented_images.append(np.fliplr(new_img))
    return np.array(augmented_images)

def simple_predict(img, net, transformer):
    """
    Perform simple prediction on a single image by making a forward pass through the network
    :params img: Image as numpy array
    :params net: Initialized Net object
    :params transformer: Initialized Transformer object
    :return: array of predicted probabilities for each class
    """
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    p = net.forward()
    predictions = p['prob'][0].copy()
    return predictions

def predict_with_crops(img, net, transformer):
    """
    Perform predictions on original image + 5 crops + mirror images and average across all predictions.
    :params img: Image as numpy array
    :params net: Initialized Net object
    :params transformer: Initialized Transformer object
    :return: array of predicted probabilities for each class
    """
    predictions = []
    predictions.append(simple_predict(img, net, transformer))
    crops = caffe.io.oversample([img], (227, 227))
    for c in crops:
        predictions.append(simple_predict(c, net, transformer))
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0)

def predict_with_augment(img, net, transformer):
    """
    Perform predictions on original image plus a series of augmented images, and average across all predictions.
    :params img: Image as numpy array
    :params net: Initialized Net object
    :params transformer: Initialized Transformer object
    :return: array of predicted probabilities for each class
    """
    predictions = []
    predictions.append(simple_predict(img, net, transformer))
    aug_images = augment_image(img)
    for c in aug_images:
        predictions.append(simple_predict(c, net, transformer))
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0)

def predict_images(image_list, net, transformer, predictor):
    """
    Iterate over a list of images and make predictions for each class based upon a specified prediction approach.
    :params image_list: Iterable of paths to images
    :params net: Initialized Net object
    :params transformer: Initialized Transformer object
    :params predictor: Method for making predictions: 
        simple_predict = simple prediction on single images
        predict_with_crops = predictions on original image + 5 crops + mirror images and average across all predictions.
        predict_with_augment = predictions on original image plus a series of augmented images, and average across all predictions.
    :return: array of predicted class probabilities for each image of shape (num_images, num_classes)
    """
    predictions = []
    for i in image_list:
        path = os.path.join('../data/resized', i)
        img = caffe.io.load_image(path)
        predictions.append(predictor(img, net, transformer))
    return np.array(predictions)


print("Reading image list.")
image_list = []
with open("../data/alexnet_2/test.txt", "r") as f:
    test_images = f.read().splitlines()

for im in test_images:
    image_list.append(im.split(" ")[0])

# print("Initializing net...")
# net, transformer = initialize_model(DEPLOY, 
#         os.path.join("/data/models", "v" + str(7), "_iter_5000.caffemodel"), 
#         os.path.join("/home/ubuntu/sb/capstone/transfer-learning/data/alexnet_4/alexnet_4_mean.npy"))

# print("Making simple predictions")

# #Make one prediction
# tic = timeit.default_timer()
# predict_images([image_list[0]], net, transformer, simple_predict)
# toc = timeit.default_timer()

# print("Time for simple prediction on 1 image is {}".format(toc - tic))

# tic = timeit.default_timer()
# predict_images(image_list, net, transformer, simple_predict)
# toc = timeit.default_timer()

# print("Time for simple predictions on {} images is {}".format(len(image_list), toc - tic))

# # Make augmented predictions
# #Make one prediction

# print("Making augmented predictions")

# tic = timeit.default_timer()
# predict_images([image_list[0]], net, transformer, predict_with_augment)
# toc = timeit.default_timer()

# print("Time for augmented prediction on 1 image is {}".format(toc - tic))

# tic = timeit.default_timer()
# predict_images(image_list, net, transformer, predict_with_augment)
# toc = timeit.default_timer()

# print("Time for augmented predictions on {} images is {}".format(len(image_list), toc - tic))

print("Done")
