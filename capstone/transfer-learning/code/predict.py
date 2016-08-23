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


parser = argparse.ArgumentParser(description='make predictions')
parser.add_argument('gpu', metavar='G', type=int, nargs=1,
    help='Indicator for GPU or CPU processing')
parser.add_argument('weights', metavar='W', type=str, nargs=1,
    help='Path to model weights')

args = parser.parse_args()
gpu = args.gpu[0]
weights = args.weights[0]

if not os.path.exists(weights):
    print ("Model weights not valid. Please pass path to valid .caffemodel file.")
    sys.exit(0)

if gpu == 1:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

# Initialize the model
print("Initializing model")
model_def = os.path.join('../models/alexnet_3', "deploy.prototxt")
model_weights = weights
net = caffe.Net(model_def, model_weights, caffe.TEST)

# Check if the mean image exists in numpy format, otherwise convert it
if not os.path.exists("../data/alexnet_2/alexnet_2_mean.npy"):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open('../data/alexnet_2/alexnet_2_mean.binaryproto', "rb").read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob))
    np.save("../data/alexnet_2/alexnet_2_mean.npy", arr[0])

# load the mean ImageNet image
mu = np.load("../data/alexnet_2/alexnet_2_mean.npy")
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1, 3, 227, 227)

with open("../transfer-learning/data/alexnet_2/test.txt", "rb") as f:
    test_images = f.read().splitlines()

predictions = []

print("Making predictions")
for x in test_images:
    path = os.path.join('../data/resized', x.decode().split(" ")[0])
    img = caffe.io.load_image(os.path.join("../data/top_classes", c, im))
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    p = net.forward()
    predictions.append(p['prob'][0])

predictions = np.array(predictions)
predictions_file = os.path.join("../data/alexnet_3", "predictions" + weights)
predictions.dump(predictions_file)

print("Done")



