#encoding:utf-8
import tensorflow as tf
import numpy as np
import os

# 这个类就是我们要建立的神经网络模型
# 做音频的项目，首先就得想到将时域转到频域，再做分析
# 而神经网络的套路也基本是下面的几步：
# 1. 创建占位符、变量
# 2. 设置学习率和batch size等参数
# 3. 构建神经网络
# 4. 设置损失函数
# 5. 设置优化器
# 6. 创建会话
# 7. 向网络喂数据开始训练，一般都是mini-batch的方法
class SVMRNN(object):

    # num_features:音频特征数
    # num_hidden_units:rnn 神经元数
    # tensorboard_dir: tensorboard保存的路径
    def __init__(self, num_features, num_hidden_units = [256, 256, 256]):
        # 保存传入的参数
        self.num_features = num_features
        self.num_rnn_layer = len(num_hidden_units)
        self.num_hidden_units = num_hidden_units


        # 设置变量
        # 训练了多少步
        self.g_step = tf.Variable(0, dtype=tf.int32, name='g_step')

        # 设置占位符
        # 学习率
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        #混合了背景音乐和人声的数据
        self.x_mixed_src = tf.placeholder(tf.float32, shape=[None, None, num_features], name='x_mixed_src')

        #背景音乐数据
        self.y_music_src = tf.placeholder(tf.float32, shape=[None, None, num_features], name='y_music_src')
        #人声数据
        self.y_voice_src = tf.placeholder(tf.float32, shape=[None, None, num_features], name='y_voice_src')

        #keep dropout，用于RNN网络的droupout
        self.dropout_rate = tf.placeholder(tf.float32)

        #初始化神经网络
        self.y_pred_music_src, self.y_pred_voice_src = self.network_init()

        # 设置损失函数
        self.loss = self.loss_init()

        # 设置优化器
        self.optimizer = self.optimizer_init()

        #创建会话
        self.sess = tf.Session()

        #需要保存模型，所以获取saver
        self.saver = tf.train.Saver(max_to_keep=1)



    #损失函数
    def loss_init(self):
        with tf.variable_scope('loss') as scope:
            #求方差
            loss = tf.reduce_mean(
                tf.square(self.y_music_src - self.y_pred_music_src)
                + tf.square(self.y_voice_src - self.y_pred_voice_src), name='loss')
        return loss

    #优化器
    def optimizer_init(self):
        ottimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        return ottimizer

    #构建神经网络
    def network_init(self):
        rnn_layer = []

        #根据num_hidden_units的长度来决定创建几层RNN，每个RNN长度为size
        for size in self.num_hidden_units:
            #使用GRU，同时，加上dropout
            layer_cell = tf.nn.rnn_cell.GRUCell(size)
            layer_cell = tf.contrib.rnn.DropoutWrapper(layer_cell, input_keep_prob=self.dropout_rate)
            rnn_layer.append(layer_cell)

        #创建多层RNN
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = tf.nn.dynamic_rnn(cell = multi_rnn_cell, inputs = self.x_mixed_src, dtype = tf.float32)

        #全连接层
        y_dense_music_src = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_dense_music_src')

        y_dense_voice_src = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_dense_voice_src')

        y_music_src = y_dense_music_src / (y_dense_music_src + y_dense_voice_src + np.finfo(float).eps) * self.x_mixed_src
        y_voice_src = y_dense_voice_src / (y_dense_music_src + y_dense_voice_src + np.finfo(float).eps) * self.x_mixed_src

        return y_music_src, y_voice_src

    #保存模型
    def save(self, directory, filename, global_step):
        #如果目录不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.saver.save(self.sess, os.path.join(directory, filename), global_step=global_step)
        return os.path.join(directory, filename)

    # 加载模型，如果没有模型，则初始化所有变量
    def load(self, file_dir):
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        # 没有模型的话，就重新初始化
        kpt = tf.train.latest_checkpoint(file_dir)
        print("kpt:", kpt)
        startepo = 0
        if kpt != None:
            self.saver.restore(self.sess, kpt)
            ind = kpt.find("-")
            startepo = int(kpt[ind + 1:])

        return startepo


    #开始训练
    def train(self, x_mixed_src, y_music_src, y_voice_src, learning_rate, dropout_rate):
        #已经训练了多少步
        # step = self.sess.run(self.g_step)

        _, train_loss = self.sess.run([self.optimizer, self.loss],
            feed_dict = {self.x_mixed_src: x_mixed_src, self.y_music_src: y_music_src, self.y_voice_src: y_voice_src,
                         self.learning_rate: learning_rate, self.dropout_rate: dropout_rate})
        return train_loss

    #验证
    def validate(self, x_mixed_src, y_music_src, y_voice_src, dropout_rate):
        y_music_src_pred, y_voice_src_pred, validate_loss = self.sess.run([self.y_pred_music_src, self.y_pred_voice_src, self.loss],
            feed_dict = {self.x_mixed_src: x_mixed_src, self.y_music_src: y_music_src, self.y_voice_src: y_voice_src, self.dropout_rate: dropout_rate})
        return y_music_src_pred, y_voice_src_pred, validate_loss

    #测试
    def test(self, x_mixed_src, dropout_rate):
        y_music_src_pred, y_voice_src_pred = self.sess.run([self.y_pred_music_src, self.y_pred_voice_src],
                                         feed_dict = {self.x_mixed_src: x_mixed_src, self.dropout_rate: dropout_rate})

        return y_music_src_pred, y_voice_src_pred