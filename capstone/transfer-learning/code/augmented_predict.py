"""
Script for making predictions using a pre-trained caffe model.

:args gpu: Flag to indicate if CPU or GPU mode should be used; 0 = CPU, 1 = GPU
:args weights: Path to caffe model weights

:output: dumps numpy array of class predictions to ../data/alexnet_3/ directory.

"""

import caffe
import argparse
import os
import numpy as np
import skimage.exposure as exposure
import skimage.transform as transform

DEPLOY = "/home/ubuntu/sb/capstone/transfer-learning/models/alexnet_4/deploy.prototxt"
caffe.set_mode_gpu()


def initialize_model(model_def, model_weights, mean_image):
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
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    p = net.forward()
    predictions = p['prob'][0].copy()
    return predictions

def predict_with_crops(img, net, transformer):
    predictions = []
    predictions.append(simple_predict(img, net, transformer))
    crops = caffe.io.oversample([img], (227, 227))
    for c in crops:
        predictions.append(simple_predict(c, net, transformer))
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0)

def predict_with_augment(img, net, transformer):
    predictions = []
    predictions.append(simple_predict(img, net, transformer))
    aug_images = augment_image(img)
    for c in aug_images:
        predictions.append(simple_predict(c, net, transformer))
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0)

def predict_images(image_list, net, transformer, predictor):
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

print("Starting predictions.")
# Models 3 and 4
for i in [3, 4]:
    print("Initializing net...")
    net, transformer = initialize_model(DEPLOY, 
        os.path.join("/data/models", "v" + str(i), "_iter_5000.caffemodel"), 
        os.path.join("/home/ubuntu/sb/capstone/transfer-learning/data/alexnet_2/alexnet_2_mean.npy"))
    # Make simple predictions
    print("Making simple predictions {}".format(i))
    #predictions = predict_images(image_list, net, transformer, simple_predict)
    #predictions.dump(os.path.join("/data/predictions", "simple_" + str(i)))
    # Make predictions with crops
    print("Making predictions with crops {}".format(i))
    predictions = predict_images(image_list, net, transformer, predict_with_crops)
    predictions.dump(os.path.join("/data/predictions", "crops_" + str(i)))

# Models 7 and 8
for i in [7, 8]:
    print("Initializing net...")
    net, transformer = initialize_model(DEPLOY, 
        os.path.join("/data/models", "v" + str(i), "_iter_5000.caffemodel"), 
        os.path.join("/home/ubuntu/sb/capstone/transfer-learning/data/alexnet_4/alexnet_4_mean.npy"))
    # Make simple predictions
    print("Making simple predictions {}".format(i))
    #predictions = predict_images(image_list, net, transformer, simple_predict)
    #predictions.dump(os.path.join("/data/predictions", "simple_" + str(i)))
    # Make predictions with crops
    print("Making predictions with augmentation {}".format(i))
    predictions = predict_images(image_list, net, transformer, predict_with_augment)
    predictions.dump(os.path.join("/data/predictions", "augmented_" + str(i)))

print("Done")
