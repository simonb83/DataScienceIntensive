import argparse
import os
import numpy as np
import re


parser = argparse.ArgumentParser(description='make predictions')
parser.add_argument('mean_file', metavar='M', type=str, nargs=1,
    help='Path to mean image binary proto')

args = parser.parse_args()
mean_file = args.mean_file[0]

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(mean_file, "rb").read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob))

file_name = re.findall(r'\/(\w+)\.binaryproto', mean_file)[0]
file_path = re.findall(r'(.+)\/\w+.binaryproto', mean_file)[0]

np.save(os.path.join(file_path, file_name + ".npy"), arr[0])