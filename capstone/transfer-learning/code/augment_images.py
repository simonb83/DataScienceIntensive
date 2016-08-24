import argparse
import os
import numpy as np
from skimage import io
import skimage.exposure as exposure
import skimage.transform as transform
import re


def jitter_rgb(img, delta):
    """
    Jitter RGB channels of image by fixed amount
    :params img: numpy array representation of image of type uint8
    :params delta: amount to perturb each channel
    :return: numpy array representation of perturbed image
    """
    j = np.zeros_like(img)
    j[:,:,0] = delta
    j[:,:,1] = delta
    j[:,:,2] = delta
    return np.add(img, j)

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
    new_image_list.append(full_im_name + " " + label)
    #Load the image
    img = io.imread(os.path.join("../data/resized", im_path))
    #Save the original image
    io.imsave(os.path.join("/data/images", full_im_name))
    #Perform transformations
    # Jitter by +5 in RGB
    new_img = jitter_rgb(img, 5)
    new_name = im_name + "_p1" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Jitter by -5 in RGB
    new_img = jitter_rgb(img, -5)
    new_name = im_name + "_p2" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Jitter by +10 in RGB
    new_img = jitter_rgb(img, 10)
    new_name = im_name + "_p3" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

        # Jitter by -10 in RGB
    new_img = jitter_rgb(img, -10)
    new_name = im_name + "_p4" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Equalize histogram
    new_img = exposure.equalize_hist(img)
    new_name = im_name + "_p5" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Lighten 1
    new_img = exposure.adjust_gamma(img, gamma=0.5)
    new_name = im_name + "_p6" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Lighten 2
    new_img = exposure.adjust_gamma(img, gamma=0.65)
    new_name = im_name + "_p7" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Lighten 3
    new_img = exposure.adjust_gamma(img, gamma=0.85)
    new_name = im_name + "_p8" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Darken 1
    new_img = exposure.adjust_gamma(img, gamma=1.25)
    new_name = im_name + "_p9" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Darken 2
    new_img = exposure.adjust_gamma(img, gamma=1.5)
    new_name = im_name + "_p10" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Darken 3
    new_img = exposure.adjust_gamma(img, gamma=2)
    new_name = im_name + "_p11" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Rotation 1
    new_img = transform.rotate(img, 45)
    new_name = im_name + "_p12" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Rotation 2
    new_img = transform.rotate(img, 90)
    new_name = im_name + "_p13" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Rotation 3
    new_img = transform.rotate(img, 135)
    new_name = im_name + "_p14" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Rotation 4
    new_img = transform.rotate(img, 180)
    new_name = im_name + "_p15" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Rotation 5
    new_img = transform.rotate(img, 225)
    new_name = im_name + "_p16" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Rotation 6
    new_img = transform.rotate(img, 270)
    new_name = im_name + "_p17" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

    # Rotation 7
    new_img = transform.rotate(img, 315)
    new_name = im_name + "_p17" + ".jpg"
    new_image_list.append(new_name + " " + label)
    io.imsave(os.path.join("/data/images", new_name))

with open(os.path.join("../data/alexnet_4", "augmented_" + re.findall(r'\/(\w+\.txt)$', images)[0]), "w") as f:
    f.write("\n".join(new_image_list))