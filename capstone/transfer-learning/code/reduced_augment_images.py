import argparse
import os
import numpy as np
from skimage import io
import skimage.exposure as exposure
import skimage.transform as transform
import re


parser = argparse.ArgumentParser(description='augment image dataset')
parser.add_argument('images', metavar='I', type=str, nargs=1,
    help='Text file containing list of images')

args = parser.parse_args()
images = args.images[0]

if not os.path.exists(images):
    print ("Image file not valid. Please pass path to valid txt file.")
    sys.exit(0)

with open(images, "r") as f:
    image_list = f.read().splitlines()

new_image_list = []

for x in image_list:
    pair = x.split(" ")
    label = pair[1]
    im_path = pair[0]

    full_im_name = re.findall(r'\/(.+)', im_path)[0]
    im_name = re.findall(r'(\d+)\.', full_im_name)[0]

    # Add original image to list of augmented images
    #Load the image
    img = io.imread(os.path.join("../data/resized", im_path))
    #Save the original image
    io.imsave(os.path.join("/data/images", full_im_name), img)
    new_image_list.append(full_im_name + " " + label)
    io.imsave(os.path.join("/data/images", "m_" + full_im_name), np.fliplr(img))
    new_image_list.append("m_" + full_im_name + " " + label)
    #Perform transformations
    # Lighten 1 & mirror image
    new_img = exposure.adjust_gamma(img, gamma=0.45)
    new_name = im_name + "_p1" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), new_img)
    new_name = im_name + "_p1m" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), np.fliplr(new_img))

    # Lighten 2 & mirror image
    new_img = exposure.adjust_gamma(img, gamma=0.65)
    new_name = im_name + "_p2" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), new_img)
    new_name = im_name + "_p2m" + ".jpg"
    io.imsave(os.path.join("/data/images", new_name), np.fliplr(new_img))

    # Lighten 3 & mirror image
    new_img = exposure.adjust_gamma(img, gamma=0.85)
    new_name = im_name + "_p3" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), new_img)
    new_name = im_name + "_p3m" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), np.fliplr(new_img))

    # Darken 1 & mirror image
    new_img = exposure.adjust_gamma(img, gamma=1.25)
    new_name = im_name + "_p4" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), new_img)
    new_name = im_name + "_p4m" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), np.fliplr(new_img))

    # Darken 2 & mirror image
    new_img = exposure.adjust_gamma(img, gamma=1.5)
    new_name = im_name + "_p5" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), new_img)
    new_name = im_name + "_p5m" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), np.fliplr(new_img))

    # Darken 3 & mirror image
    new_img = exposure.adjust_gamma(img, gamma=2)
    new_name = im_name + "_p6" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), new_img)
    new_name = im_name + "_p6m" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), np.fliplr(new_img))

    # Rotate 180 degrees & mirror image
    new_img = transform.rotate(img, 180)
    new_name = im_name + "_p7" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), new_img)
    new_name = im_name + "_p7m" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name), np.fliplr(new_img))


with open(os.path.join("../data/alexnet_4", "augmented_" + re.findall(r'\/(\w+\.txt)$', images)[0]), "w") as f:
    f.write("\n".join(new_image_list))