import numpy as np
from skimage.color import rgb2gray


def intersection(arr1, arr2):
    """
    Calculate the intersection between two histograms
    """
    arr1 = arr1 / sum(arr1)
    arr2 = arr2 / sum(arr2)
    return sum(np.minimum(arr1, arr2))


def l1_norm(arr1, arr2):
    """
    Calculate the L-1 norm between two histograms (arrays)
    """
    arr1 = arr1 / sum(arr1)
    arr2 = arr2 / sum(arr2)
    return np.linalg.norm(arr2 - arr1, ord=1)


def euclid(arr1, arr2):
    """
    Calculate the Euclidean norm between two histograms (arrays)
    """
    arr1 = arr1 / sum(arr1)
    arr2 = arr2 / sum(arr2)
    return np.linalg.norm(arr2 - arr1)


def classify(image, model, distance):
    """
    Classify a set of images using histogram comparison
    :params image: histogram of image to classify as numpy array
    :params model: dictionary of classes and corresponding representative histogram
    :params distance: method for comparing histograms; can be one of: Intersection, L-1 Norm, Euclidean
    :return: predicated image class as string
    """
    if distance == 'intersection':
        current_max = 0
        image_class = ''
        for k, v in model.items():
            i = intersection(v, image)
            if i > current_max:
                current_max = i
                image_class = k
        return image_class
    else:
        current_min = 2
        image_class = ''
        for k, v in model.items():
            i = eval(distance)(v, image)
            if i < current_min:
                current_min = i
                image_class = k
        return image_class


def greyscale_histogram(image):
    return np.histogram(rgb2gray(image), bins=256)


def color_histogram(image):
    """
    Calculate chained histogram of all color channels in image
    :params image: image as numpy array
    :return: numpy array of shape (768, )
    """
    red_hist, bins = np.histogram(image[:, :, 0], bins=256)
    green_hist, bins = np.histogram(image[:, :, 1], bins=256)
    blue_hist, bins = np.histogram(image[:, :, 2], bins=256)
    return np.concatenate((red_hist, green_hist, blue_hist))


def complete_histogram(image):
    """
    Calculate chained histogram of Red, Green, Blue and Greyscale in image
    :params image: image as numpy array
    :return: numpy array of shape (1024, )
    """
    colors = color_histogram(image)
    grey_hist, bins = np.histogram(rgb2gray(image), bins=256)
    return np.concatenate((colors, grey_hist))
