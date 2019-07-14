from core.meg import MEG
import tensorflow as tf
import data.dataHandler as DataHandler
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    num_runs = 1
    sum_f1 = 0
    sum_pre = 0
    sum_rec = 0
    dataset_name = 'kdd99'
    for i in range(num_runs):
        tf.reset_default_graph()
        opts = DataHandler.dataset_config[dataset_name]
        data = DataHandler.load_data(opts)
        meg = MEG(opts)
        pre, rec, f1 = meg.train(data)
        sum_f1 += f1
        sum_pre += pre
        sum_rec += rec
    print("Precision: %g, Recall: %g, F1: %g" % (sum_pre/num_runs, sum_rec/num_runs, sum_f1/num_runs))
