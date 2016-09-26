"""
Functions for running training and optiizing a Linear SVC classifier
"""

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import datetime
import os

def split_indices(array_length, train_proportion):
    """
    Create masks of training and test indices
    :params array_length: length of array to be split, integer
    :params train_proportion: proportion of data to be used for training, decimal in [0, 1.0]
    :return: array of train_indices, array of test_indices
    """
    num_train = int(train_proportion * array_length)
    indices = np.arange(array_length)
    np.random.shuffle(indices)
    return indices[:num_train], indices[num_train:]

def split_data(X, y, train_mask, test_mask):
    """
    Split features and labels into Training and Tests subsets
    :params X: array of features of shape (n_samples, n_features)
    :params y: array of labels of shape (n_samples,)
    :params train_mask: mask of indices for selecting training data
    :params test_mask: mask of indices for selecting test data
    :return: X_train: array of training features of shape (len(train_mask), n_features)
    :return: y_train: array of training labels of shape (len(train_mask),)
    :return: X_test: array of test features of shape (len(test_mask), n_features)
    :return: y_test: array of test labels of shape (len(test_mask),)
    """
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    return X_train, y_train, X_test, y_test

def get_best_params(X, y):
    """
    Implement sklearn GridSearch on LinearSVC model to find best C parameter
    :params X: Array of features
    :params y: Array of labels
    :return: float, best C value
    """
    c_vals = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
    model = LinearSVC()
    grid = GridSearchCV(estimator=model, param_grid=dict(C=c_vals), n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_.C

def run_model(X_train, y_train, X_test, y_test, model, layer, C=0):
    """
    Implement sklearn LinearSVC model and fit to training data.
    Predict labels of test data.
    Dump the training data, training labels, test features, test labels, predictions and pickled model.
    :params X_train: array of training features
    :params y_train: array of training labels
    :params X_test: array of test features
    :params y_test: array of test labels
    :params model: Name of the pre-trained CNN used for extracting features
    :params layer: Name of the layer used for extracting features
    :params C: Optimized C value from grid search
    """
    svc = LinearSVC(C=C)
    svc.fit(X_train, y_train)
    predicted_labels = svc.predict(X_test)
    directory_name = "svm_" + model + "_" + "layer_" + layer + "_" + str(datetime.date.today()).replace("-","_")
    directory_path = os.path.join("../models", directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    X_train.dump(os.path.join(directory_path, "train_data"))
    y_train.dump(os.path.join(directory_path, "train_labels"))
    X_test.dump(os.path.join(directory_path, "test_data"))
    y_test.dump(os.path.join(directory_path, "test_labels"))
    predicted_labels.dump(os.path.join(directory_path, "predicted_labels"))
    joblib.dump(svc, os.path.join(directory_path, "model.pkl"))
    return

