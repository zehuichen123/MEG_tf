import scipy.io as sio
from sklearn.model_selection import train_test_split
import data.kddcup10 as kddcup10

import numpy as np
import pandas as pd

class Data:
    def __init__(self, data, _type=1):
        if _type == 1:
            x = data['X']
            y = data['y']

        elif _type == 2:
            x = data['x'][0][0][0]
            y = data['x'][0][0][2]
            #y = np.where(y == 1, 0, y)
            y = np.where(y == 2, 0, y)
        elif _type == 3:
            x = data[:, :-1]
            y = data[:, -1].reshape(-1, 1)

        self.data, self.test_data, self.labels, self.test_labels = \
            train_test_split(x, y, test_size=0.5)
        anomaly_index = np.where(self.labels == 1)[0]
        self.data = np.delete(self.data, anomaly_index, axis=0)
        self.labels = np.delete(self.labels, anomaly_index, axis=0)

def minmax_normalization(x, base):
    min_val = np.min(base, axis=0)
    max_val = np.max(base, axis=0)
    norm_x = (x - min_val) / (max_val - min_val + 1e-10)
    return norm_x


def load_data(opts):
    if opts['dataset'] == 'kdd99':
        data = kddcup10.Kddcup('./demo/kddcup99-10.data.pp.csv.gz')
        data.get_clean_training_testing_data(0.5)

    elif opts['dataset'] == 'pageblock':
        data = sio.loadmat('demo/pageblocks.mat')
        data = Data(data, 2)

    elif opts['dataset'] == 'mushroom':
        data = np.array(pd.read_csv('demo/mushroom.csv'))
        data = Data(data, 3)

    else:
        data = sio.loadmat('./demo/%s.mat' % opts['dataset'])
        data = Data(data)
    train_x = data.data
    test_x = data.test_data
    base = np.concatenate([train_x, test_x], 0)

    data.data = minmax_normalization(train_x, base)
    data.test_data = minmax_normalization(test_x, base)

    return data

dataset_config = {}

dataset_config['kdd99'] = {
    'dataset': 'kdd99',
    'batch_size': 256,
    'data_shape': 120,
    'epoch_num': 100,
    'lr': 1e-4,
    'anomaly_ratio': 0.2,
    'zdim': 32,
    'energy_model_iters': 1,
    'lambda': 100,
    'g_net': [64, 128, 120],
    'e_net': [256, 128, 1],
    's_net': [256, 128, 1],
    'print_every': 100,
}
dataset_config['pageblock'] = {
    'dataset': 'pageblock',
    'batch_size': 256,
    'data_shape': 10,
    'epoch_num': 500,
    'lr': 1e-4,
    'anomaly_ratio': 0.1,
    'zdim': 2,
    'energy_model_iters': 1,
    'lambda': 100,
    'g_net': [6, 10],
    'e_net': [16, 8, 1],
    's_net': [16, 8, 1],
    'print_every': 3,
}