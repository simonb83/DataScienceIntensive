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
    actual_scores = np.choose(y, scores.T).reshape(n_samples, 1)
    # Calculate the margins
    margins = np.maximum(((scores - actual_scores) + np.ones((n_samples, 1))), 0)
    # Calculate the overall data loss
    data_loss = (np.sum(margins) - n_samples) / n_samples
    # Calculate the regularization loss
    reg_loss = lam * np.sum(W * W)
    # Overall loss
    loss = data_loss + reg_loss

    ## GRADIENT CALCS ##

    # d_margins is 0 where the margin is less than 0, 1 for all score not equal
    # to true class score and -1 * number of margins > 0 for true class score
    d_margins = margins
    d_margins[d_margins > 0] = 1
    d_margins[np.arange(0, scores.shape[0]), y] = 0
    d_margins[np.arange(0, scores.shape[0]), y] = -1 * np.sum(d_margins, axis=1)
    # d_scores is d_margins dot X
    d_scores = np.dot(X.T, d_margins) / n_samples
    d_reg = 2 * lam * W
    dW = d_scores + d_reg

    return loss, dW


def softmax_loss(W, X, y, lam):
    """
    Calculate average Softmax loss and gradient

    Params:
    W: Weight matrix of size (num_classes, M) where M is size of vectors of X
    X: Data matrix of shape (N, M) where N is number of samples, and each sample has vector length M
    y: Array of integer labels of length N
    lam: weight regularization factor

    Return:
    Average loss (float)
    Gradient matrix with same shape as W
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    n_samples = X.shape[0]

    scores = np.dot(X, W)
    # Subtract max score to avoid numeric instability
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    actual_scores = np.choose(y, scores.T)
    exp_actual_scores = np.exp(actual_scores)
    # Sum over all scores
    exp_sum = np.sum(exp_scores, axis=1)
    norm_exp_sum = exp_actual_scores / exp_sum
    margins = -np.log(norm_exp_sum)
    data_loss = np.sum(margins) / n_samples
    reg_loss = lam * np.sum(W * W)
    loss = data_loss + reg_loss

    # Derivate of -log(x) = -1 / x
    d_margins = (-1 / norm_exp_sum).reshape(norm_exp_sum.shape[0], 1)

    # For each score normalized score Sj (except normalized true class score Sy), 
    # gradient is Sj * Sj;
    # For true class score Sy, gradient = Sy * (1 - Sy)
    d_exp_sum = exp_scores / exp_sum.reshape(n_samples, 1)
    d_exp_sum[np.arange(n_samples), y] *= -1
    d_exp_sum[np.arange(n_samples), y] += 1
    d_exp_sum = d_exp_sum * norm_exp_sum.reshape(norm_exp_sum.shape[0], 1)
    # Backpropogation through normalized exponential
    d_exp_sum *= d_margins

    d_scores = np.dot(X.T, d_exp_sum) / n_samples
    d_reg = 2 * lam * W
    dW = d_scores + d_reg

    return loss, dW
