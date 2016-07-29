import numpy as np


def svm_loss(W, X, y, lam):
    """
    Calculate average SVM loss and gradient

    Params:
    W: Weight matrix of size (num_classes, M) where M is size of vectors of X
    X: Data matrix of shape (N, M) where N is number of samples, and each sample has vector length M
    y: Array of integer labels of length N
    lam: weight regularization factor

    Return:
    Average loss (float)
    Gradient matrix with same shape as W
    """
    n_samples = X.shape[0]

    # Initialize loss and gradient matrix
    loss = 0
    dW = np.zeros(W.shape)
    # Calculate full matrix of scores from W and X
    scores = np.dot(X, W)
    # Get score for true class from each row in matrix
    actual_scores = np.choose(y, scores.T)
    # Subtract score for true class from each score and add 1
    diffs = (scores.T - actual_scores).T + np.ones(scores.shape)
    # We can ignore all differences < 0 due to max in loss function
    diffs[diffs < 0] = 0
    # Sum all of the differences; each true class score adds one, so subtract number of samples from result
    loss = np.sum(diffs) - n_samples
    loss /= n_samples
    # Add the L2 weight regularization
    loss += lam * np.sum(W * W)

    # For calculating gradient, once again use the score - true class score + 1
    l_scores = (scores.T - actual_scores).T + np.ones(scores.shape)
    # All differences < 1 contribute 0 to the gradient
    l_scores[l_scores < 0] = 0
    # All differences > 1 contribute 1 to the gradient
    l_scores[l_scores > 0] = 1
    # For true class, contibute -1 * count of difference > 0
    l_scores[np.arange(0, scores.shape[0]), y] = 0
    l_scores[np.arange(0, scores.shape[0]), y] = -1 * np.sum(l_scores, axis=1)
    # Dot product of contributions with X and normalize by number of samples
    dW = np.dot(X.T, l_scores)
    dW /= n_samples
    return loss, dW


def softmax_loss(W, X, y, lam):
    """Function documentation"""
    loss = 0.0
    dW = np.zeros(W.shape)
    n_samples = X.shape[0]
    scores = np.dot(X, W)
    scores -= np.max(scores)
    actual_scores = np.choose(y, scores.T)
    loss_m = np.exp(actual_scores) / np.sum(np.exp(scores), axis=1)
    loss = np.sum(loss_m) / n_samples
    # Add the L2 weight regularization
    loss += lam * np.sum(W * W)

    ## TODO: Implement gradient for Softmax ##

    return loss, dW
