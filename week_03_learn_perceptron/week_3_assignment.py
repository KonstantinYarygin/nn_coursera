import pandas as pd
import numpy as np
import scipy.io


def load_dataset(dataset_number):
    dataset = {}
    mat = scipy.io.loadmat('datasets/dataset{}.mat'.format(dataset_number))
    dataset['w_init'] = mat['w_init']
    dataset['w_gen_feas'] = mat['w_gen_feas']
    dataset['pos_examples_nobias'] = mat['pos_examples_nobias']
    dataset['neg_examples_nobias'] = mat['neg_examples_nobias']
    return dataset

def train_perceptron(w_init, pos_examples_nobias, neg_examples_nobias):
    num_pos_examples = pos_examples_nobias.shape[0]
    num_neg_examples = neg_examples_nobias.shape[0]
    pos_examples = np.hstack((pos_examples_nobias, np.ones((num_pos_examples, 1))))
    neg_examples = np.hstack((neg_examples_nobias, np.ones((num_neg_examples, 1))))
    weights = np.copy(w_init).reshape(3, )
    for i_iter in range(1000):
        get_better = False
        for sample in pos_examples:
            out = np.dot(sample, weights)
            out = 1 * (out >= 0)
            if out == 0:
                get_better = True
                weights = weights + sample.reshape(3, )
        for sample in neg_examples:
            out = np.dot(sample, weights)
            out = 1 * (out >= 0)
            if out == 1:
                get_better = True
                weights = weights - sample.reshape(3, )
        if not get_better:
            break
    print(i_iter)

if __name__ == '__main__':
    for dataset_number in range(1, 5):
        dataset = load_dataset(dataset_number)
        train_perceptron(
            w_init=dataset['w_init'],
            pos_examples_nobias=dataset['pos_examples_nobias'],
            neg_examples_nobias=dataset['neg_examples_nobias']
        )
