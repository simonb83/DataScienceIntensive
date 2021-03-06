"""
Script to extract features from pre-trained CNN and optimize and train a Linear SVC on top of extracted features.

:args model_name: Name of the pre-trained CNN to use; one of alexnet, vggnet
:args gpu: Flag to indicate if CPU or GPU mode should be used; 0 = CPU, 1 = GPU
:args mode: Indicator for running script in (d)evelopment or (f)ull mode; in development mode only a small subset of data is 
            used in order to validate that the script is running correctly. In full mode, the entire dataset is used.

The script performs the following steps:

1. Initialize a Caffe Net object using the deploy.prototext and .caffemodel files for the specified model
2. Set-up a Transformer for pre-processing the data
3. Load the image data and extract 3 sets of features from fully-connected layers 6, 7 and 8
4. Split the features into training data and test data (training proportion = 0.8)
5. For each set of features:
    i. Run a grid search optimization to find the optimal C parameter for a Linear SVC model
    ii. Fit a Linear SVC to the training data
    iii. Make predictions on the test data
    iv. Dump the relevant test and training data for download
"""


import numpy as np
import pandas as pd
import caffe
import os, sys
import argparse
import linear_svm as ls

caffe_file = {
    'alexnet': 'bvlc_alexnet.caffemodel',
    'vggnet': 'VGG_CNN_S.caffemodel'
}

mean_image = {
    'alexnet': 'ilsvrc_2012_mean.npy',
    'vggnet': 'VGG_mean.npy'
}


parser = argparse.ArgumentParser(description='download model binary')
parser.add_argument('model_name', metavar='N', type=str, nargs=1,
    help='Name of model')
parser.add_argument('gpu', metavar='G', type=int, nargs=1,
    help='Indicator for GPU or CPU processing')
parser.add_argument('mode', metavar='M', type=str, nargs=1,
    help='Development or full mode')


args = parser.parse_args()
model = args.model_name[0]
gpu = args.gpu[0]
mode = args.mode[0]

if model not in ['alexnet', 'vggnet']:
    print ("Model name not valid. Must be one of 'alexnet' or 'vggnet'.")
    sys.exit(0)

elif mode not in ['d', 'f']:
    print("Mode must be d (development) or f (full).")
    sys.exit(0)

else:
    # Initialize Net object using the relevant deploy.prototxt and model-weights
    filename = caffe_file[model]
    model_filename = os.path.join('../models', model, filename)

    if gpu == 1:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    model_def = os.path.join('../models', model, "deploy.prototxt")
    model_weights = model_filename
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # load the mean ImageNet image
    mu = np.load(os.path.join("../data", mean_image[model]))
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    if model == 'alexnet':
        net.blobs['data'].reshape(1, 3, 227, 227)
    else:
        net.blobs['data'].reshape(1, 3, 224, 224)

    classes = pd.read_csv('../data/top_classes/top_classes.csv', index_col=0)
    class_list = list(classes['class'].unique())

    # Maintain a list of the image names used
    image_names = {}
    for c in class_list:
        image_names[c] = np.array(classes[classes['class'] == c]['name'])

    if mode == 'd':
        for c in class_list:
            image_names[c] = image_names[c][:30]

    features_8 = [] #Last FC layer
    features_7 = [] #Second last FC layer
    features_6 = [] #Third last FC layer
    class_names = []
    image_list = []

    # Load the images and extract features from FC layers 6, 7 and 8
    print("Obtaining features")
    for c in class_list:
        for im in image_names[c]:
            img = caffe.io.load_image(os.path.join("../data/top_classes", c, im))
            img = caffe.io.resize_image(img, (256, 256))
            net.blobs['data'].data[...] = transformer.preprocess('data', img)
            p = net.forward()
            features_8.append(net.blobs['fc8'].data[0].copy())
            features_7.append(net.blobs['fc7'].data[0].copy())
            features_6.append(net.blobs['fc6'].data[0].copy())
            class_names.append(c)
            image_list.append(c + "/" + im)
    print("Done obtaining features")

    features_8 = np.array(features_8) #Last FC layer
    features_7 = np.array(features_7) #Second last FC layer
    features_6 = np.array(features_6) #Third last FC layer
    class_names = np.array(class_names)
    image_list = np.array(image_list)

    train_mask, test_mask = ls.split_indices(class_names.shape[0], 0.8)

    test_images = image_list[test_mask]
    train_images = image_list[train_mask]
    fname = str(model) + "_test_images"
    test_images.dump(os.path.join("../models/", fname))
    fname = str(model) + "_train_images"
    test_images.dump(os.path.join("../models/", fname))

    for l in [6, 7, 8]:
        # Split into training and test data
        X_train, y_train, X_test, y_test = ls.split_data(eval("features_" + str(l)), class_names, train_mask, test_mask)
        # Run grid search
        print("Running grid-search for layer {}".format(l))
        c = ls.get_best_params(X_train, y_train)
        # Fit the model, make predictions and dump the data
        print("Training model for layer {}".format(l))
        ls.run_model(X_train, y_train, X_test, y_test, model, str(l), c)
        print("Layer {} complete".format(l))
    print("Done")