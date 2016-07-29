import numpy as np
from nn.loss_functions import *

class LinearClassifier(object):

    """
    Implement Linear Classifier model of form WX, where weight
    matrix W is updated using stochastic gradient descent. Based on assignment
    1 from Stanford CS231n Convolutional Neural Networks course.
    """

    def __init__(self):
        self.W = None

    def fit(self, X, y, learning_rate=1e-4, mini_batch_size=256, lam=1, num_iterations=200, verbose=False):
        """
        Fit linear classifier to data
        Params:
        X: Data matrix of shape (N, M) where N is number of samples, and each sample has vector length M
        y: Array of integer labels of length N
        learning_rat: learning rate or step size of updating weights matrix
        mini_batch_size: batch size for calculating loss and gradient
        lam: weight regularization factor
        num_iterations: number of iterations to use in gradient descent
        verbose: if True, then prints out loss every 100 iterations

        Return:
        List of losses of length num_iterations
        """
        n_samples = X.shape[0]
        vector_length = X.shape[1]
        n_classes = np.max(y) + 1
        losses = []
        # Initialize weights vector W, if not already done
        if self.W is None:
            self.W = 0.001 * np.random.randn(vector_length, n_classes)

        self.losses = []
        for n in range(num_iterations):
            mask = np.random.choice(n_samples, size=mini_batch_size)
            X_batch = X[mask]
            y_batch = y[mask]

            loss, gradient = self.loss(X_batch, y_batch, lam)
            losses.append(loss)
            self.W += -1 * learning_rate * gradient

            # Print out the current loss if called with verbose=True
            if verbose and n % 100 == 0:
                print("Iteration {:,} of {:,}: {:.3f}".format(n, num_iterations, loss))

        return losses

    def predict(self, X):
        """
        Predict class for images
        Params:
        X: Data matrix of shape (N, M) where N is number of samples, and each sample has vector length M

        Return:
        List of predicted classes of length N
        """
        y_predict = np.zeros(X.shape[0])
        scores = np.dot(X, self.W)
        y_predict = np.argmax(scores, axis=1)

        return y_predict

    def loss(self, X, y, lam):
        """
        Implement loss function. Subsequently defined in sub-classes.
        """
        pass


class LinearSVM(LinearClassifier):

    def loss(self, X, y, lam):
        """
        Loss function for LinearSVM
        Params:
        X: Data matrix of shape (N, M) where N is number of samples, and each sample has vector length M
        y: Array of integer labels of length N
        lam: weight regularization factor

        Return:
        Average loss (float)
        Gradient matrix of dimension (num_classes, M)
        """
        return svm_loss(self.W, X, y, lam)


class LinearSoftmax(LinearClassifier):

    def loss(self, X, y, lam):
        """TODO: Implement Softmax"""
        pass
