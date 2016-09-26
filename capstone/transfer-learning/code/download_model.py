"""
Script for downloading pre-trained model weights.
"""

import os
import sys
import argparse
import urllib
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

# URL for downloading the model weights
model_urls = {
    'alexnet': 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
    'vggnet': 'http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_S.caffemodel'
}

# For ensuring weights corretly downloaded
model_sha1 = {
    'alexnet': '9116a64c0fbe4459d18f4bb6b56d647b63920377',
    'vggnet': '862b3744bce69b7ba90d29b8099aed3b00c8580b'
}

# Name of model weights file
caffe_file = {
    'alexnet': 'bvlc_alexnet.caffemodel',
    'vggnet': 'VGG_CNN_S.caffemodel'
}

parser = argparse.ArgumentParser(description='download model binary')
parser.add_argument('model_name', metavar='name', type=str, nargs=1,
    help='Name of model')

args = parser.parse_args()
model = args.model_name[0]

if model not in ['alexnet', 'vggnet']:
    print ("Model name not valid. Must be one of 'alexnet' or 'vggnet'.")
else:
    # Check if model already exists
    filename = caffe_file[model]
    model_filename = os.path.join('../models', model, filename)
    if os.path.exists(model_filename) and check_model(model_filename, model_sha1[model]):
        print("Model already exists.")
        sys.exit(0)
    # Else download model
    else:
        urllib.urlretrieve(model_urls[model], model_filename, reporthook)
        if not check_model(model_filename, model_sha1[model]):
            print("Model did not download correctly. Try again.")
            sys.exit(1)
