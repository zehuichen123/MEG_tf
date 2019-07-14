import math
import common.metrics_lib as metrics_lib
import tensorflow as tf
import numpy as np

class MEG():
    def __init__(self, opts):
        self.config = opts
        self.x = tf.placeholder(tf.float64, shape=(None, opts['data_shape']), name='x')
        self.z = tf.placeholder(tf.float64, shape=(opts['batch_size'], opts['zdim']), name='z')

        x_fake = self.Generator(self.z)

        self.each_E_real = self.EnergyModel(self.x)
        self.each_E_fake = self.EnergyModel(x_fake)

        self.E_real = tf.reduce_mean(self.each_E_real)
        self.E_fake = tf.reduce_mean(self.each_E_fake)

        self.each_penalty = tf.reshape(self.score_penalty(self.each_E_real, self.x), (-1, 1))
        self.penalty = tf.reduce_mean(self.each_penalty)

        ## to-do Mutual Informaiton
        _zeros = tf.zeros((opts['batch_size'],1), tf.float64)
        _ones = tf.ones((opts['batch_size'], 1), tf.float64)
        label = tf.concat([_ones, _zeros], axis=0)

        z_bar = tf.random.shuffle(self.z)
        concat_x = tf.concat([x_fake, x_fake], axis=0)
        concat_z = tf.concat([self.z, z_bar], axis=0)
        concat_all = tf.concat([concat_x, concat_z], axis=-1)
        statistics_pred = self.StatisticsNet(concat_all)
        self.mi_estimate = self.MI_estimate(statistics_pred, label)

        self.generator_obj = self.E_fake + self.mi_estimate
        self.energy_obj = self.E_real - self.E_fake + opts['lambda'] * self.penalty

        self.add_optimizers()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def MI_estimate(self, y_pred, y_label):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_label, logits=y_pred))

    def train(self, data):
        batch_size = self.config['batch_size']
        epoch_num = self.config['epoch_num']
        energy_model_iters = self.config['energy_model_iters']
        num_points = data.data.shape[0]
        batch_num = math.ceil(num_points/batch_size)
        steps = 1
        for epoch in range(epoch_num):
            sum_e_loss = []
            sum_e_fake = []
            sum_e_real = []
            sum_penalty = []
            sum_g_loss = []
            sum_mi_loss = []
            for ii in range(batch_num):
                steps += 1
                batch_index = np.random.choice(num_points, batch_size, replace=False)
                batch_data = data.data[batch_index]
                sample_z = np.random.normal(0., 1., (batch_size, self.config['zdim']))
                feed_d = {self.x: batch_data, self.z: sample_z}
                [_, e_loss, e_fake, e_real, penalty] = self.sess.run(
                    [
                        self.opt_e,
                        self.energy_obj,
                        self.E_fake,
                        self.E_real,
                        self.penalty,
                    ],
                    feed_dict=feed_d
                )
                sum_e_loss.append(e_loss)
                sum_e_fake.append(e_fake)
                sum_e_real.append(e_real)
                sum_penalty.append(penalty)
                if steps % energy_model_iters == 0:
                    sample_z = np.random.normal(0., 1., (batch_size, self.config['zdim']))
                    feed_d[self.z] = sample_z
                    [_, _, g_loss, mi_estimate] = self.sess.run(
                        [
                            self.opt_g,
                            self.opt_s,
                            self.generator_obj,
                            self.mi_estimate
                        ],
                        feed_dict=feed_d
                    )
                    sum_g_loss.append(g_loss)
                    sum_mi_loss.append(mi_estimate)
                if steps % self.config['print_every'] == 0:
                    print("Step %d, Energy loss: %g, E_fake: %g, E_real: %g, "
                          "penalty: %g, G loss: %g, MI estimation: %g" % (
                        ii, np.mean(sum_e_loss), np.mean(sum_e_fake),
                        np.mean(sum_e_real), np.mean(sum_penalty),
                        np.mean(sum_g_loss), np.mean(sum_mi_loss)
                    ))
                    sum_e_loss = []
                    sum_e_fake = []
                    sum_e_real = []
                    sum_penalty = []
                    sum_g_loss = []
                    sum_mi_loss = []
            self.eval(data, epoch)

    def eval(self, data, epoch):
        batch_size = self.config['batch_size']
        num_points = data.test_data.shape[0]
        batch_num = math.ceil(num_points / batch_size)
        energy_loss = np.zeros((1, 1))
        penalty_loss = np.zeros((1, 1))
        test_y = np.zeros((1, 1))
        for ii in range(batch_num):
            batch_index = np.random.choice(num_points, batch_size, replace=False)
            batch_data = data.test_data[batch_index]
            batch_label = data.test_labels[batch_index]
            sample_z = np.random.normal(0., 1., (batch_size, self.config['zdim']))
            feed_d = {self.x: batch_data, self.z: sample_z}
            [e_loss, penalty] = self.sess.run([self.each_E_real, self.each_penalty], feed_dict=feed_d)
            energy_loss = np.concatenate([energy_loss, e_loss], axis=0)
            penalty_loss = np.concatenate([penalty_loss, penalty], axis=0)
            test_y = np.concatenate([test_y, batch_label], axis=0)
        print("Epoch %d" % epoch)
        print(penalty_loss)
        metrics_lib.compute_score(energy_loss[1:], test_y[1:], 'Energy', self.config['anomaly_ratio'])
        metrics_lib.compute_score(penalty_loss[1:], test_y[1:], 'Penalty', self.config['anomaly_ratio'])

    def add_optimizers(self):
        learning_rate = self.config['lr']
        with tf.variable_scope('optimizers', reuse=tf.AUTO_REUSE):
            g_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_net')
            e_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='e_net')
            s_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='s_net')
            opt_g = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9)
            opt_e = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9)
            opt_s = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9)
            self.opt_g = opt_g.minimize(self.generator_obj, var_list=g_ops)
            self.opt_s = opt_s.minimize(self.generator_obj, var_list=s_ops)
            self.opt_e = opt_e.minimize(self.energy_obj, var_list=e_ops)

    def Generator(self, inputs):
        g_net = self.config['g_net']
        hi = inputs
        with tf.variable_scope('g_net', reuse=tf.AUTO_REUSE):
            for each_units in g_net[:-1]:
                hi = tf.keras.layers.Dense(each_units, use_bias=True, activation=tf.nn.relu)(hi)
            hi = tf.keras.layers.Dense(g_net[-1], use_bias=True)(hi)
        return hi

    def EnergyModel(self, inputs):
        e_net = self.config['e_net']
        hi = inputs
        with tf.variable_scope('e_net', reuse=tf.AUTO_REUSE):
            for each_units in e_net[:-1]:
                hi = tf.keras.layers.Dense(each_units, use_bias=True, activation=tf.nn.leaky_relu)(hi)
            hi = tf.keras.layers.Dense(e_net[-1], use_bias=True)(hi)
        return hi

    def StatisticsNet(self, inputs):
        s_net = self.config['s_net']
        hi = inputs
        with tf.variable_scope('s_net', reuse=tf.AUTO_REUSE):
            for each_units in s_net[:-1]:
                hi = tf.keras.layers.Dense(each_units, use_bias=True, activation=tf.nn.leaky_relu)(hi)
            hi = tf.keras.layers.Dense(s_net[-1], use_bias=True)(hi)
        return hi

    def score_penalty(self, E_real, x_real, beta=1.):
        score = tf.gradients(E_real * beta, x_real)[0]
        norm_score = tf.norm(score, ord=2, axis=1) ** 2
        return norm_score