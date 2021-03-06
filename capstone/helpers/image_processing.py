import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.transform as transform
import skimage.exposure as exposure
import glob
import os


def print_columns(data, cols):
    """
    Split a list into sublists and print it out in columns
    :params data: list of data to be printed
    :params cols: number of columns
    :return: print the data and return none
    """
    num_per_col = int(np.ceil(len(data) / cols))
    split = [data[i:i + num_per_col] for i in range(0, len(data), num_per_col)]
    for row in zip(*split):
        print("".join(str.ljust(i, 30) for i in row))


def standardize(image_name, size, path):
    """
    Standardize an image to a particular size and rotation
    :params image_name: name of new image
    :params size: standard image size as tuple
    :params path: path to images directory
    :return: standardized image as numpy array
    """
    image = io.imread(path + image_name)
    existing_shape = image.shape
    if existing_shape[1] > existing_shape[0]:
        image = transform.rotate(image, 90, resize=True)
    if image.shape != size:
        image = transform.resize(image, size)
    return image


def mean_image(image_names, shape, path):
    """
    Calculate the average image from a list of image names
    :params image_names: list of image names
    :params shape: standard image shape as tuple
    :params path: path to images directory
    :return: average image as numpy array
    """
    num_images = len(image_names)
    avg_image = np.zeros(shape)
    for name in image_names:
        image = standardize(name, shape, path)
        avg_image += image / num_images
    return avg_image


def get_image_names(name):
    base = '/Users/simonbedford/Documents/Coding/datascience/capstone/data/food-101/images/'
    images = []
    for f in glob.glob(os.path.join(base, name, "*.jpg")):
        images.append(f)
    return images


def get_histogram(image, channel, bins=None):
    """
    Calculate histogram and bins for an image based on specification of Red, Green, Blue or Greyscale
    :params image: image for calculating histogram
    :params channel: 0 = Red, 1 = Green, 2 = Blue, 3 = Greyscale
    :return: array of bin counts, array of bins
    """
    if channel == 3:
        return expoure.histogram(image)
    else:
        return exposure.histogram(image[:, :, channel])


def plot_histogram(images, names, channel):
    """
    Plot side-by-side histograms for a set of 12 images
    and a specified channel
    :params images: set of images
    :params names: image names
    :params channel: 0 = Red, 1 = Green, 2 = Blue, 3 = Greyscale
    """
    colors = ('#e41a1c', '#4daf4a', '#377eb8', '#878787')
    fig, axes = plt.subplots(3, 4, figsize=(20, 8))
    for ax, image, name in zip(axes.flat, images, names):
        hist, bins = get_histogram(image, channel)
        width = bins[1] - bins[0]
        ax.bar(bins, hist, align='center', width=width, alpha=0.5, color=colors[channel], axes=ax)
        ax.set_title("{}".format(name), weight='bold', size=14)
    fig.tight_layout()
    plt.show()


def nearest_neighbors(image, image_names, n, path):
    """
    For a specified image represented as array find the nearest neighbor
    in a separate set of images using Frobenius norm
    :params image: array representing the specified image
    :params image_names: list of image names for comparison
    :params n: closet n neighbors
    :params path: path to images directory
    return: names of nearest neighbors
    """
    distances = {}
    s = image.shape
    for im in image_names:
        temp_image = standardize(im, s, path)
        distances[im] = np.linalg.norm(image - temp_image)
    return sorted(distances, key=distances.get)[:n]
