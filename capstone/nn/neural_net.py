import numpy as np


class TwoLayerNet(object):
    """Class Documentation"""

    def __init__(self, input_size, output_size, hidden_size, std=1e-4):
        """Docs"""
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * std
        self.params['W2'] = np.random.randn(hidden_size, output_size) * std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Calculate scores from forward pass
        scores = None
        layer_1 = np.dot(X, W1) + b1
        layer_2 = np.maximum(layer_1, 0)
        scores = np.dot(layer_2, W2) + b2

        # If no true classes given, then terminate
        if y is None:
            return scores

        # Calculate loss
        loss = 0.0
        # Subtract maximum to avoid numeric instability
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        actual_scores = scores[np.arange(scores.shape[0]), y]
        exp_actual_scores = np.exp(actual_scores)
        # Sum over all scores
        exp_sum = np.sum(exp_scores, axis=1)
        norm_exp_sum = exp_actual_scores / (exp_sum + 1e-15)
        margins = -np.log(norm_exp_sum + 1e-15)
        data_loss = np.sum(margins) / N
        w1_reg_loss = 0.5 * reg * np.sum(W1 * W1)
        w2_reg_loss = 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + w1_reg_loss + w2_reg_loss

        # Compute the gradients at each of W1, W2, b1, b2
        grads = {}

        # Calculate softmax gradient
        d_softmax = exp_scores / exp_sum.reshape(exp_scores.shape[0], 1)
        y_true = np.zeros(d_softmax.shape)
        y_true[np.arange(y_true.shape[0]), y] = 1
        d_softmax -= y_true

        # Gradient on b2
        d_b2 = np.sum(d_softmax, axis=0) / N
        grads['b2'] = d_b2

        # Gradient on W2
        d_w2 = np.dot(d_softmax.T, layer_2).T / N
        d_w2 += reg * W2
        grads['W2'] = d_w2

        # Backprop gradient through ReLU
        d_layer2 = np.dot(W2, d_softmax.T).T
        d_layer2[layer_1 <= 0] = 0

        # Gradient on b1
        d_b1 = np.sum(d_layer2, axis=0) / N
        grads['b1'] = d_b1

        # Gradient on W1
        d_w1 = np.dot(X.T, d_layer2) / N
        d_w1 += reg * W1
        grads['W1'] = d_w1

        return loss, grads

    def train(self, X, y, X_val, y_val,
        learning_rate=1e-3, learning_rate_decay=0.95,
        reg=1e-5, num_iterations=100, batch_size=256,
        verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = int(max(num_train / batch_size, 1))

        # SGD for optimizing params in model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iterations):
            mask = np.random.choice(num_train, size=batch_size)
            X_batch = X[mask]
            y_batch = y[mask]

            # Compute loss and gradients
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] += -1 * learning_rate * grads['W1']
            self.params['W2'] += -1 * learning_rate * grads['W2']
            self.params['b1'] += -1 * learning_rate * grads['b1']
            self.params['b2'] += -1 * learning_rate * grads['b2']

            #if verbose and it % 100 == 0:
                #print('iteration {} / {}: loss {:.8f}'.format(it, num_iterations, loss))
            # Every epoch, check train and val accuracy and decay learning rate
            if it % iterations_per_epoch == 0:
                train_accuracy = (self.predict(X_batch) == y_batch).mean()
                val_accuracy = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_accuracy)
                val_acc_history.append(val_accuracy)

                learning_rate *= learning_rate_decay
                if verbose:
                    print("Loss: {}, Train: {}, Val: {}".format(loss, train_accuracy, val_accuracy))

        return {'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history}

    def predict(self, X):

        y_pred = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        layer_1 = np.dot(X, W1) + b1
        layer_2 = np.maximum(layer_1, 0)
        scores = np.dot(layer_2, W2) + b2
        y_pred = np.argmax(scores, axis = 1)

        return y_pred
