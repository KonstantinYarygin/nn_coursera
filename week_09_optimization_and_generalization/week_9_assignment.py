import pandas as pd
import numpy as np
import scipy.io

TINY = np.finfo('f').tiny
HUGE = np.finfo('f').max

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def load_data():
    mat_data = scipy.io.loadmat('datasets/data.mat')['data']
    data = {}
    data['X_train'] = mat_data[0][0][0][0][0][0].T
    data['y_train'] = mat_data[0][0][0][0][0][1].T
    data['X_valid'] = mat_data[0][0][1][0][0][1].T
    data['y_valid'] = mat_data[0][0][1][0][0][0].T
    data['X_test'] = mat_data[0][0][2][0][0][0].T
    data['y_test'] = mat_data[0][0][2][0][0][1].T
    return data

class NeuralNet(object):
    def __init__(self, seed=0,
                 r_mean=0, r_std=0.01,
                 n_hidden=200):
        self.N_IN = 256
        self.N_HIDDEN = n_hidden
        self.N_OUT = 10

        np.random.seed(seed)

        self.w1 = np.random.normal(r_mean, r_std, (self.N_IN, self.N_HIDDEN))
        self.b1 = np.random.normal(r_mean, r_std, self.N_HIDDEN)
        self.w2 = np.random.normal(r_mean, r_std, (self.N_HIDDEN, self.N_OUT))
        self.b2 = np.random.normal(r_mean, r_std, self.N_OUT)

        self.input = np.zeros(self.N_IN)
        self.hidden = np.zeros(self.N_HIDDEN)
        self.output = np.zeros(self.N_OUT)

        self.w1_delta = np.zeros((self.N_IN, self.N_HIDDEN))
        self.b1_delta = np.zeros(self.N_HIDDEN)
        self.w2_delta = np.zeros((self.N_HIDDEN, self.N_OUT))
        self.b2_delta = np.zeros(self.N_OUT)

    def f_prop(self, X):
        n_samples = X.shape[0]
        self.input = X
        self.hidden = np.dot(self.input, self.w1) + self.b1
        self.hidden = sigmoid(self.hidden)
        self.output = np.dot(self.hidden, self.w2) + self.b2
        self.output = np.exp(self.output)
        self.output[np.isposinf(self.output)] = HUGE
        self.output = self.output / self.output.sum(axis=1).reshape((n_samples, 1))

    def loss(self, w_decay, y_true):
        classification_loss = -np.mean(np.sum(np.log(self.output) * y_true, axis=1))
        sum_weights = np.sum(self.w1**2) + np.sum(self.b1**2) + np.sum(self.w2**2) + np.sum(self.b2**2)
        sum_weights = np.sum(self.w1**2) + np.sum(self.w2**2)
        wd_loss = 1/2 * w_decay * sum_weights
        return classification_loss + wd_loss

    def b_prop(self, X, y, w_decay, learning_rate, momentum):
        n_samples = y.shape[0]

        self.f_prop(X)
        loss = self.loss(w_decay, y)

        loss_deriv_output_in = self.output - y
        loss_deriv_hidden_in = np.dot(loss_deriv_output_in, self.w2.T) * \
                               self.hidden * (1 - self.hidden)
        loss_deriv_w2 = np.dot(self.hidden.T, loss_deriv_output_in) + self.w2*w_decay
        loss_deriv_b2 = loss_deriv_output_in.sum(0) + self.b2*w_decay
        loss_deriv_w1 = np.dot(self.input.T, loss_deriv_hidden_in) + self.w1*w_decay
        loss_deriv_b1 = loss_deriv_hidden_in.sum(0) + self.b1*w_decay

        self.w2_delta = momentum*self.w2_delta + loss_deriv_w2/n_samples
        self.b2_delta = momentum*self.b2_delta + loss_deriv_b2/n_samples
        self.w1_delta = momentum*self.w1_delta + loss_deriv_w1/n_samples
        self.b1_delta = momentum*self.b1_delta + loss_deriv_b1/n_samples

        self.w2 -= learning_rate * self.w2_delta
        self.b2 -= learning_rate * self.b2_delta
        self.w1 -= learning_rate * self.w1_delta
        self.b1 -= learning_rate * self.b1_delta

        return loss

    def train(self, X_train, y_train, X_valid, y_valid,
              n_iter=100, w_decay=0,
              learning_rate=0.1, momentum=0.9,
              batchsize=100, do_early_stop=False,
              verbose=False):
        n_batches = (X_train.shape[0] // batchsize) + int(X_train.shape[0] % batchsize)

        if do_early_stop:
            best_valid_loss_so_far = np.inf

        for i_iter in range(n_iter):
            i = i_iter % n_batches
            X_batch = X_train[(batchsize*i):(batchsize*(i+1)), :]
            y_batch = y_train[(batchsize*i):(batchsize*(i+1))]
            batch_loss = self.b_prop(X_batch, y_batch, w_decay, learning_rate, momentum)
            if verbose:
                print('Iteration {}, batch loss: {:.5f}'.format(i_iter, batch_loss))
            if do_early_stop:
                self.f_prop(X_valid)
                if self.loss(w_decay, y_valid) < best_valid_loss_so_far:
                    best_valid_loss_so_far = self.loss(w_decay, y_valid)

        self.f_prop(X_train)
        train_loss = self.loss(w_decay, y_train)
        self.f_prop(X_valid)
        valid_loss = self.loss(w_decay, y_valid)
        acc = np.mean(np.argmax(self.output, axis=1) == np.argmax(y_valid, axis=1))

        print('Train loss: {:.5f}; Valid loss: {:.5f}; accuracy: {:.3f}'.format(train_loss, valid_loss, acc))
        if do_early_stop:
            print('Best Valid loss: {:.5f}'.format(best_valid_loss_so_far))

if __name__ == '__main__':
    data = load_data()

    # nn = NeuralNet(r_mean=0, r_std=0.1, n_hidden=10)
    # nn.train(X_train=data['X_train'], y_train=data['y_train'],
    #          X_valid=data['X_valid'], y_valid=data['y_valid'],
    #          w_decay=0, n_iter=70, learning_rate=0.005, momentum=0, batchsize=4)

    # for m in [0, 0.9]:
    #     for lr in [0.002, 0.01, 0.05, 0.2, 1.0, 5.0, 20.0]:
    #         nn = NeuralNet(r_mean=0, r_std=0.1, n_hidden=10)
    #         print(m, lr, end='|')
    #         nn.train(X_train=data['X_train'], y_train=data['y_train'],
    #                  X_valid=data['X_valid'], y_valid=data['y_valid'],
    #                  w_decay=0, n_iter=70, learning_rate=lr, momentum=m, batchsize=4)

    # nn = NeuralNet(r_mean=0, r_std=0.1, n_hidden=200)
    # nn.train(X_train=data['X_train'], y_train=data['y_train'],
    #          X_valid=data['X_valid'], y_valid=data['y_valid'],
    #          w_decay=0, n_iter=1000, learning_rate=0.35, momentum=0.9, batchsize=100)

    # nn = NeuralNet(r_mean=0, r_std=0.1, n_hidden=200)
    # nn.train(X_train=data['X_train'], y_train=data['y_train'],
    #          X_valid=data['X_valid'], y_valid=data['y_valid'],
    #          w_decay=0, n_iter=1000, learning_rate=0.35, momentum=0.9,
    #          batchsize=100, do_early_stop=True, verbose=True)

    # for wd in [0, 0.0001, 0.001, 0.01, 1, 5]:
    #     print(wd, end=' | ')
    #     nn = NeuralNet(r_mean=0, r_std=0.1, n_hidden=200)
    #     nn.train(X_train=data['X_train'], y_train=data['y_train'],
    #              X_valid=data['X_valid'], y_valid=data['y_valid'],
    #              w_decay=wd, n_iter=1000, learning_rate=0.35, momentum=0.9, batchsize=100)
    #     nn.f_prop(data['X_valid'])
    #     print(nn.loss(0, data['y_valid']))

    # for nh in [10, 30, 100, 130, 170]:
    #     print(nh, end=' | ')
    #     nn = NeuralNet(r_mean=0, r_std=0.1, n_hidden=nh)
    #     nn.train(X_train=data['X_train'], y_train=data['y_train'],
    #              X_valid=data['X_valid'], y_valid=data['y_valid'],
    #              w_decay=0, n_iter=1000, learning_rate=0.35, momentum=0.9, batchsize=100)

    # for nh in [18, 37, 113, 189, 236]:
    #     print(nh, end=' | ')
    #     nn = NeuralNet(r_mean=0, r_std=0.1, n_hidden=nh)
    #     nn.train(X_train=data['X_train'], y_train=data['y_train'],
    #              X_valid=data['X_valid'], y_valid=data['y_valid'],
    #              w_decay=0.01, n_iter=1000, learning_rate=0.35, momentum=0.9, batchsize=100)

