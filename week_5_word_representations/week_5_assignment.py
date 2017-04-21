from scipy.special import expit
import pandas as pd
import numpy as np
import pickle
import random
import re

TINY = np.finfo('f').tiny
HUGE = np.finfo('f').max

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def load_data(raw_sentences_path, N=4):
    with open(raw_sentences_path, 'r') as f:
        data = (line.strip().lower() for line in f)
        data = (re.sub(' +', ' ', line) for line in data)
        data = (line.split(' ') for line in data)
        n_grams = [sentence[i:i+N] for sentence in data for i in range(len(sentence)-N+1)]
    vocab = sorted(list({word for n_gram in n_grams for word in n_gram}))
    word2index = {word: i for i, word in enumerate(vocab)}

    random.seed(0)
    random.shuffle(n_grams)
    # The training set consists of 372,550 4-grams.
    # The validation and test sets have 46,568 4-grams each.
    train_n_grams = n_grams[:372550]
    valid_n_grams = n_grams[372550:372550+46568]
    test_n_grams = n_grams[372550+46568:]

    train_set = np.array([[word2index[word] for word in n_gram] for n_gram in train_n_grams])
    valid_set = np.array([[word2index[word] for word in n_gram] for n_gram in valid_n_grams])
    test_set = np.array([[word2index[word] for word in n_gram] for n_gram in test_n_grams])

    return vocab, train_set, valid_set, test_set

class NeuralNet(object):
    def __init__(self, seed=0,
                 r_mean=0, r_std=0.01,
                 vocab_size=250,
                 n_words_in=3,
                 n_embed=50,
                 n_hidden=200):
        self.VOCAB_SIZE = vocab_size
        self.N_WORDS_IN = n_words_in
        self.N_IN = vocab_size
        self.N_EMBED = n_embed
        self.N_HIDDEN = n_hidden
        self.N_OUT = vocab_size

        np.random.seed(seed)

        self.w1 = np.random.normal(r_mean, r_std, (self.N_IN, self.N_EMBED))
        self.w2 = np.random.normal(r_mean, r_std, (self.N_WORDS_IN * self.N_EMBED, self.N_HIDDEN))
        self.b2 = np.random.normal(r_mean, r_std, self.N_HIDDEN)
        self.w3 = np.random.normal(r_mean, r_std, (self.N_HIDDEN, self.N_OUT))
        self.b3 = np.random.normal(r_mean, r_std, self.N_OUT)

        self.input = np.zeros(self.N_WORDS_IN)
        self.embed = [np.zeros(self.N_EMBED) for i in range(self.N_WORDS_IN)]
        self.embed_u = np.zeros(self.N_WORDS_IN * self.N_EMBED)
        self.hidden = np.zeros(self.N_HIDDEN)
        self.output = np.zeros(self.N_OUT)

        self.w1_delta = np.zeros((self.N_IN, self.N_EMBED))
        self.w2_delta = np.zeros((self.N_WORDS_IN * self.N_EMBED, self.N_HIDDEN))
        self.b2_delta = np.zeros(self.N_HIDDEN)
        self.w3_delta = np.zeros((self.N_HIDDEN, self.N_OUT))
        self.b3_delta = np.zeros(self.N_OUT)

    def f_prop(self, X):
        n_samples = X.shape[0]
        X_sparse = [np.zeros((n_samples, self.N_IN)) for i in range(self.N_WORDS_IN)]
        for i in range(self.N_WORDS_IN):
            X_sparse[i][np.arange(n_samples), X[:, i]] = 1

        self.embed = [np.dot(X_sparse[i], self.w1) for i in range(self.N_WORDS_IN)]
        self.embed_u = np.hstack(self.embed)
        self.hidden = np.dot(self.embed_u, self.w2) + self.b2
        self.hidden = sigmoid(self.hidden)
        self.output = np.dot(self.hidden, self.w3) + self.b3
        self.output = np.exp(self.output)
        self.output[np.isposinf(self.output)] = HUGE
        self.output = self.output / self.output.sum(axis=1).reshape((n_samples, 1))

    def error_CE(self, y_true):
        n_samples = y_true.shape[0]
        y_true_sparse = np.zeros((n_samples, self.N_OUT))
        y_true_sparse[np.arange(n_samples), y_true] = 1
        err_CE = -np.sum(y_true_sparse * np.log(TINY + self.output)) / n_samples
        return err_CE


    def b_prop(self, X, y, learning_rate, momentum):
        n_samples = y.shape[0]

        y_sparse = np.zeros((n_samples, self.N_OUT))
        y_sparse[np.arange(n_samples), y] = 1
        X_sparse = [np.zeros((n_samples, self.N_IN)) for i in range(self.N_WORDS_IN)]
        for i in range(self.N_WORDS_IN):
            X_sparse[i][np.arange(n_samples), X[:, i]] = 1


        self.f_prop(X)
        CE = self.error_CE(y)

        loss_deriv_output_in = self.output - y_sparse
        loss_deriv_hidden_in = np.dot(loss_deriv_output_in, self.w3.T) * \
                               self.hidden * (1 - self.hidden)
        loss_deriv_embed_u = np.dot(loss_deriv_hidden_in, self.w2.T)
        loss_deriv_embed = [loss_deriv_embed_u[:, :self.N_EMBED],
                            loss_deriv_embed_u[:, self.N_EMBED:self.N_EMBED * 2],
                            loss_deriv_embed_u[:, self.N_EMBED * 2:]]

        loss_deriv_w3 = np.dot(self.hidden.T, loss_deriv_output_in)
        loss_deriv_b3 = loss_deriv_output_in.sum(0)
        loss_deriv_w2 = np.dot(self.embed_u.T, loss_deriv_hidden_in)
        loss_deriv_b2 = loss_deriv_hidden_in.sum(0)
        loss_deriv_w1 = sum([np.dot(X_sparse[i].T, loss_deriv_embed[i]) for i in range(self.N_WORDS_IN)])

        self.w3_delta = momentum*self.w3_delta + loss_deriv_w3/n_samples
        self.b3_delta = momentum*self.b3_delta + loss_deriv_b3/n_samples
        self.w2_delta = momentum*self.w2_delta + loss_deriv_w2/n_samples
        self.b2_delta = momentum*self.b2_delta + loss_deriv_b2/n_samples
        self.w1_delta = momentum*self.w1_delta + loss_deriv_w1/n_samples

        self.w3 -= learning_rate * self.w3_delta
        self.b3 -= learning_rate * self.b3_delta
        self.w2 -= learning_rate * self.w2_delta
        self.b2 -= learning_rate * self.b2_delta
        self.w1 -= learning_rate * self.w1_delta

        return CE


    def train(self, X_train, y_train, X_valid, y_valid,
              batchsize=100, n_epoch=10, save_epoch=[1, 10],
              learning_rate=0.1, momentum=0.9, verbose=True):
        n_samples_train = X_train.shape[0]
        n_batches = n_samples_train // batchsize + 1

        for i_epoch in range(1, n_epoch+1):
            if verbose:
                print('=== EPOCH {} ==='.format(i_epoch))

            chunk_CE = 0
            for i in range(n_batches):
                X_batch = X_train[(batchsize*i):(batchsize*(i+1)), :]
                y_batch = y_train[(batchsize*i):(batchsize*(i+1))]
                chunk_CE += self.b_prop(X_batch, y_batch, learning_rate, momentum)
                if verbose and i % 100 == 0 and i != 0:
                    print('Batch {}, train CE: {:.3f}'.format(i, chunk_CE / batchsize))
                    chunk_CE = 0

            valid_CE = self.get_CE(X=X_valid, y=y_valid)
            if verbose:
                print('Validation CE: {:.3f}'.format(valid_CE))

            if i_epoch in save_epoch:
                self.dump_weights('weights/epoch_{}.n_embed_{}.n_hidden_{}.lr_{:.0e}.m_{:.0e}'.format(
                                  i_epoch, self.N_EMBED, self.N_HIDDEN, learning_rate, momentum))

    def get_CE(self, X, y, batchsize=100, verbose=False):
        n_samples = X.shape[0]
        n_batches = n_samples // batchsize + 1

        total_CE = 0
        for i in range(n_batches):
            X_batch = X[(batchsize*i):(batchsize*(i+1)), :]
            y_batch = y[(batchsize*i):(batchsize*(i+1))]
            self.f_prop(X_batch)
            batch_CE = self.error_CE(y_batch) * X_batch.shape[0]
            total_CE += batch_CE
            if verbose and i % 100 == 0 and i != 0:
                print('Batch {}/{}'.format(i, n_batches))

        mean_CE = total_CE / X.shape[0]
        return mean_CE

    def dump_weights(self, path):
        weights = [self.w1, self.w2, self.b2, self.w3, self.b3]
        np.savez(path, weights)

    def load_weights(self, path):
        weights = np.load(path)
        self.w1, self.w2, self.b2, self.w3, self.b3 = weights['arr_0']

if __name__ == '__main__':
    vocab, train_set, valid_set, test_set = load_data('data/raw_sentences.txt')
    nn = NeuralNet()
