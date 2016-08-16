import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import datetime
import os

def split_indices(array_length, train_proportion):
    num_train = int(train_proportion * array_length)
    indices = np.arange(array_length)
    np.random.shuffle(indices)
    return indices[:num_train], indices[num_train:]

def split_data(X, y, train_proportion):
    train_mask, test_mask = split_indices(y.shape[0], train_proportion)
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    return X_train, y_train, X_test, y_test

def get_best_params(X, y):
    c_vals = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
    model = LinearSVC()
    grid = GridSearchCV(estimator=model, param_grid=dict(C=c_vals), n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_.C

def run_model(X_train, y_train, X_test, y_test, model, layer, C=0):
    svc = LinearSVC(C=C)
    svc.fit(X_train, y_train)
    predicted_labels = svc.predict(X_test)
    directory_name = "svm_" + model + "_" + "layer_" + layer + "_" + str(datetime.date.today()).replace("-","_")
    directory_path = os.path.join("../models", directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    y_test.dump(os.path.join(directory_path, "test_labels"))
    predicted_labels.dump(os.path.join(directory_path, "predicted_labels"))
    joblib.dump(svc, os.path.join(directory_path, "model.pkl"))
    return

