"""
Download pre-trained model weights for either alexnet or googlenet

"""

import os
import sys
import argparse
import urllib.request
import time
import hashlib


def reporthook(count, block_size, total_size):
    """
    From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = (time.time() - start_time) or 0.01
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def check_model(filename, sha1):
    """
    Check downloaded model vs provided sha1
    """
    with open(filename, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest() == sha1


model_urls = {
    'alexnet': 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
    'googlenet': 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
}

model_sha1 = {
    'alexnet': '9116a64c0fbe4459d18f4bb6b56d647b63920377',
    'googlenet': '405fc5acd08a3bb12de8ee5e23a96bec22f08204'
}

parser = argparse.ArgumentParser(description='download model binary')
parser.add_argument('model_name', metavar='name', type=str, nargs=1,
    help='Name of model')

args = parser.parse_args()
model = args.model_name[0]

if model not in ['alexnet', 'googlenet']:
    print ("Model name not valid. Must be one of 'alexnet' or 'googlenet'.")
else:
    # Check if model already exists
    filename = "bvlc_" + model + ".caffemodel"
    model_filename = os.path.join('../models', model, filename)
    if os.path.exists(model_filename) and check_model(model_filename, model_sha1[model]):
        print("Model already exists.")
        sys.exit(0)
    # Else download model
    else:
        urllib.request.urlretrieve(model_urls[model], model_filename, reporthook)
        if not check_model(model_filename, model_sha1[model]):
            print("Model did not download correctly. Try again.")
            sys.exit(1)
