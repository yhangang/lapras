import math
import os

import numpy as np
import pandas as pd
from tensorflow.python.keras.metrics import AUC, Precision
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.python.keras.callbacks import Callback
import tensorflow as tf
from collections import OrderedDict
import pickle
import datetime


class BUNDLE_DSSM():
    def __init__(self, u_continue_cols=[], u_discrete_cols=[], u_discrete_col_input_dim={},
                 i_bundle_id_col="", i_bundle_id_param=(3,100), i_bundle_weight_col="",
                 i_continue_cols=[], i_discrete_cols=[], i_discrete_col_input_dim={},
                 u_history_cols=[], u_history_col_ts_step={}, u_lstm_cols=[], u_lstm_col_ts_step={}):
        """
        Column names doesn't matter, but you should know what are you modeling.
        Args:
            u_continue_cols: 用户连续特征列名
            u_discrete_cols: 用户离散特征列名
            u_discrete_col_input_dim: 实数，用户离散特征词表大小
            i_continue_cols: 物品连续特征列名
            i_discrete_cols: 物品离散特征列名
            i_discrete_col_input_dim: 实数，物品离散特征词表大小
            i_bundle_id_param: (10,100) 第一个参数：序列长度  第二个参数：词表大小
            u_history_cols: 用户关于物品历史记录特征列名
            u_history_col_ts_step: 用户历史行为对应的序列长度  (10,100) 第一个参数：序列长度  第二个参数：词表大小
            u_lstm_cols: 用户时序型历史数据特征列名
            u_lstm_col_ts_step: 用户时序型历史数据对应的序列长度  (10,100) LSTM数据的长和宽
        """
        self.u_continue_cols = u_continue_cols
        self.u_discrete_cols = u_discrete_cols
        self.u_discrete_col_input_dim = u_discrete_col_input_dim

        self.u_history_cols = u_history_cols
        self.u_history_col_ts_step = u_history_col_ts_step
        self.u_lstm_cols = u_lstm_cols
        self.u_lstm_col_ts_step = u_lstm_col_ts_step

        self.i_continue_cols = i_continue_cols
        self.i_discrete_cols = i_discrete_cols
        self.i_bundle_id_col = i_bundle_id_col
        self.i_bundle_weight_col = i_bundle_weight_col
        self.i_bundle_id_param = i_bundle_id_param
        self.i_discrete_col_input_dim = i_discrete_col_input_dim

        self.model = self.u_hidden_units = self.i_hidden_units = self.activation = self.dropout = self.embedding_dim = \
            self.user_input_len = self.item_input_len = None

    def compile(self, embedding_dim=4, u_hidden_units=[128,64,16], i_hidden_units=[64,32,16], activation='relu',
                 dropout=0.3, loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(1e-4),  metrics=[Precision(), AUC()], summary=True, **kwargs):
        """
            Args:
                embedding_dim: 静态离散特征embedding参数,embedding[0]表示输入值离散空间上限,embedding[1]表示输出向量维度
                u_hidden_units: 用户塔MLP层神经元参数，最少2层
                i_hidden_units： 物品塔MLP层神经元参数，最少2层
                activation: MLP层激活函数
                dropout: dropout系数
                loss: 损失函数
                optimizer: 优化器
                metrics: 效果度量函数
                summary: 是否输出summary信息
        """
        self.embedding_dim = embedding_dim
        self.u_hidden_units = u_hidden_units
        self.i_hidden_units = i_hidden_units
        self.activation = activation
        self.dropout = dropout

        # 定义输入格式
        user_input_features = OrderedDict()
        item_input_features = OrderedDict()
        if self.u_continue_cols:
            user_input_features['u_continue_cols'] = tf.keras.layers.Input(shape=len(self.u_continue_cols), name='u_continue_cols_input')  # 用户数值特征
        for col in self.u_discrete_cols:
            user_input_features[col] = tf.keras.layers.Input(shape=1, name=col+'_input')  # 用户离散特征
        for col in self.u_history_cols:
            user_input_features[col] = tf.keras.layers.Input(shape=(self.u_history_col_ts_step[col][0], 1), name=col+'_input')  # 用户关于物品的历史序列特征
        for col in self.u_lstm_cols:
            user_input_features[col] = tf.keras.layers.Input(shape=(self.u_lstm_col_ts_step[col][0], self.u_lstm_col_ts_step[col][1]), name=col+'_input')  # 用户LSTM特征

        if self.i_continue_cols:
            item_input_features['i_continue_cols'] = tf.keras.layers.Input(shape=len(self.i_continue_cols), name='i_continue_cols_input')  # 物品数值特征
        for col in self.i_discrete_cols:
            item_input_features[col] = tf.keras.layers.Input(shape=1, name=col+'_input')  # 物品离散特征
        item_input_features['i_bundle_id_col'] = tf.keras.layers.Input(shape=(self.i_bundle_id_param[0], 1),
                                                                       name='i_bundle_id_col_input')  # 礼包id嵌入向量
        item_input_features['i_bundle_weight_col'] = tf.keras.layers.Input(shape=(self.i_bundle_id_param[0], 1),
                                                                       name='i_bundle_weight_col_input')  # 礼包id合并权重

        # 构造双塔结构
        user_vector_list = []
        item_vector_list = []

        if self.u_continue_cols:
            u_dense = user_input_features['u_continue_cols']
            user_vector_list.append(u_dense)
        if self.i_continue_cols:
            i_dense = item_input_features['i_continue_cols']
            item_vector_list.append(i_dense)

        for col in self.u_discrete_cols:
            user_vector_list.append(tf.reshape(tf.keras.layers.Embedding(self.u_discrete_col_input_dim[col],
                self.embedding_dim, name=col+'_embedding')(user_input_features[col]), [-1, self.embedding_dim]))

        for col in self.i_discrete_cols:
            item_vector_list.append(tf.reshape(tf.keras.layers.Embedding(self.i_discrete_col_input_dim[col],
                self.embedding_dim, name=col+'_embedding')(item_input_features[col]), [-1, self.embedding_dim]))

        bundle_pooling_embedding = tf.keras.layers.Embedding(self.i_bundle_id_param[1],
                                                      self.embedding_dim, name=self.i_bundle_id_col + '_embedding')(item_input_features['i_bundle_id_col'])
        bundle_pooling_embedding = tf.reshape(bundle_pooling_embedding, [-1, self.i_bundle_id_param[0], self.embedding_dim])
        bundle_pooling_embedding = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last',
                name=self.i_bundle_id_col + '_pooling')(bundle_pooling_embedding * item_input_features['i_bundle_weight_col'])
        # bundle_pooling_embedding = tf.reduce_mean(bundle_pooling_embedding * item_input_features['i_bundle_weight_col'],
        #                                           axis=1, name=self.i_bundle_id_col + '_pooling')

        item_vector_list.append(bundle_pooling_embedding)

        for col in self.u_history_cols:
            pooling_embedding = tf.keras.layers.Embedding(self.u_history_col_ts_step[col][1],
                                                          self.embedding_dim, name=col + '_embedding')(user_input_features[col])
            pooling_embedding = tf.reshape(pooling_embedding, [-1, self.u_history_col_ts_step[col][0], self.embedding_dim])
            pooling_embedding = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last',
                                                                       name=col+'_pooling')(pooling_embedding)
            user_vector_list.append(pooling_embedding)

        for col in self.u_lstm_cols:
            # 动态计算LSTM输出维度
            lstm_out_dim = int(self.u_lstm_col_ts_step[col][0] * self.u_lstm_col_ts_step[col][1] / 4)
            lstm_vector = tf.keras.layers.LSTM(units=lstm_out_dim, name=col+'_LSTM')(user_input_features[col])
            user_vector_list.append(lstm_vector)

        user_embedding = tf.keras.layers.concatenate(user_vector_list, axis=1)
        item_embedding = tf.keras.layers.concatenate(item_vector_list, axis=1)

        for i in range(len(self.u_hidden_units[:-1])):
            # user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
            user_embedding = tf.keras.layers.Dense(self.u_hidden_units[i], activation=self.activation)(user_embedding)
            user_embedding = tf.keras.layers.Dropout(self.dropout)(user_embedding)
        user_embedding = tf.keras.layers.Dense(self.u_hidden_units[-1], name='user_embedding', activation='tanh')(user_embedding)

        for i in range(len(self.i_hidden_units[:-1])):
            # item_embedding = tf.keras.layers.BatchNormalization()(item_embedding)
            item_embedding = tf.keras.layers.Dense(self.i_hidden_units[i], activation=self.activation)(item_embedding)
            item_embedding = tf.keras.layers.Dropout(self.dropout)(item_embedding)
        item_embedding = tf.keras.layers.Dense(self.i_hidden_units[-1], name='item_embedding', activation='tanh')(item_embedding)

        # 双塔向量做内积输出
        output = tf.expand_dims(tf.reduce_sum(user_embedding * item_embedding, axis=1), 1)
        output = tf.sigmoid(output)

        user_input = list(user_input_features.values())
        self.user_input_len = len(user_input)
        item_input = list(item_input_features.values())
        self.item_input_len = len(item_input)
        inputs_list = user_input + item_input
        self.model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)
        if summary:
            self.model.summary()

    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=10, batch_size=256, validation_split=0,
            callback=True,
            **kwargs):
        """
            通过贝叶斯调参寻找最优参数
            Args:
                X: 训练集X数据集
                y: 训练集y数据集
                X_test: 测试集X数据集
                y_test: 测试集y数据集
                epochs: 指定epochs
                batch_size: 指定batch_size
                validation_split: 验证集比例
                callback: epoch结束后是否调用回调函数，计算模型效果指标
        """

        class AUC_KS(Callback):
            def on_epoch_end(self, epoch, logs=None):
                print()
                predictions = self.model.predict(X_train)
                label = y_train
                auc_score = roc_auc_score(label, predictions)
                fpr, tpr, _ = roc_curve(label, predictions)
                ks = np.max(np.abs(tpr - fpr))
                print(' train_auc {:4f} train_ks {:4f}'.format(auc_score, ks))

                if X_test is not None and y_test is not None:
                    predictions = self.model.predict(X_test)
                    label = y_test
                    auc_score = roc_auc_score(label, predictions)
                    fpr, tpr, _ = roc_curve(label, predictions)
                    ks = np.max(np.abs(tpr - fpr))
                    print(' test_auc {:4f} test_ks {:4f}'.format(auc_score, ks))

        if callback:
            metrit = [AUC_KS()]
        else:
            metrit = []

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                       callbacks=metrit, **kwargs)

    def predict(self, X, **kwargs):
        predictions = self.model.predict(X, **kwargs)
        return predictions

    def get_user_model(self):
        inputs = self.model.inputs[:self.user_input_len]
        output = self.model.get_layer(name="user_embedding").output
        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        return model

    def get_item_model(self):
        inputs = self.model.inputs[self.user_input_len:]
        output = self.model.get_layer(name="item_embedding").output
        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        return model

    def get_bundle_id_model(self):
        inputs = self.model.inputs[-2:]
        output = self.model.get_layer(name=self.i_bundle_id_col + '_pooling').output
        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        return model

    def get_item_id_model(self):
        inputs = self.model.inputs[-2:]
        output = self.model.get_layer(name=self.i_bundle_id_col + '_embedding').output
        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        return model

    def embedding_inner_product(self, user_embedding: np.ndarray, item_embedding: np.ndarray):
        result = self.sigmoid(np.sum((user_embedding * item_embedding), axis=1))
        return result

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_params(self):
        params_dict = {'u_continue_cols': self.u_continue_cols, 'u_discrete_cols': self.u_discrete_cols,
                       'u_discrete_col_input_dim': self.u_discrete_col_input_dim, 'u_history_cols': self.u_history_cols,
                       'u_history_col_ts_step': self.u_history_col_ts_step, 'u_lstm_cols': self.u_lstm_cols,
                       'u_lstm_col_ts_step': self.u_lstm_col_ts_step, 'i_continue_cols': self.i_continue_cols,
                       'i_discrete_cols': self.i_discrete_cols,
                       'i_discrete_col_input_dim': self.i_discrete_col_input_dim, 'u_hidden_units': self.u_hidden_units,
                       'i_hidden_units': self.i_hidden_units, 'activation': self.activation, 'dropout': self.dropout,
                       'embedding_dim': self.embedding_dim, 'user_input_len': self.user_input_len,
                       'item_input_len': self.item_input_len, 'i_bundle_id_col': self.i_bundle_id_col,
                       'i_bundle_id_param': self.i_bundle_id_param, 'i_bundle_weight_col': self.i_bundle_weight_col
                       }
        return params_dict

    def _load_params(self, params_dict: dict):
        self.u_continue_cols = params_dict['u_continue_cols']
        self.u_discrete_cols = params_dict['u_discrete_cols']
        self.u_discrete_col_input_dim = params_dict['u_discrete_col_input_dim']

        self.u_history_cols = params_dict['u_history_cols']
        self.u_history_col_ts_step = params_dict['u_history_col_ts_step']
        self.u_lstm_cols = params_dict['u_lstm_cols']
        self.u_lstm_col_ts_step = params_dict['u_lstm_col_ts_step']

        self.i_continue_cols = params_dict['i_continue_cols']
        self.i_discrete_cols = params_dict['i_discrete_cols']
        self.i_discrete_col_input_dim = params_dict['i_discrete_col_input_dim']

        self.u_hidden_units = params_dict['u_hidden_units']
        self.i_hidden_units = params_dict['i_hidden_units']
        self.activation = params_dict['activation']
        self.dropout = params_dict['dropout']
        self.embedding_dim = params_dict['embedding_dim']
        self.user_input_len = params_dict['user_input_len']
        self.item_input_len = params_dict['item_input_len']
        self.i_bundle_id_col = params_dict['i_bundle_id_col']
        self.i_bundle_id_param = params_dict['i_bundle_id_param']
        self.i_bundle_weight_col = params_dict['i_bundle_weight_col']

    def save(self, path='./model_file', name='sbcnm_model'):
        if not os.path.exists(path):
            os.makedirs(path)
        tf_model_name = path + "/" + name
        param_file_name = tf_model_name + ".param"
        f = open(param_file_name, 'wb')  # pickle只能以二进制格式存储数据到文件
        f.write(pickle.dumps(self.get_params()))  # dumps序列化源数据后写入文件
        f.close()
        tf.saved_model.save(self.model, tf_model_name)

    @classmethod
    def load(cls, path='./model_file', name='sbcnm_model'):
        model = BUNDLE_DSSM()

        tf_model_name = path + "/" + name
        param_file_name = tf_model_name + ".param"

        f = open(param_file_name, 'rb')
        params_dict = pickle.loads(f.read())
        f.close()
        model._load_params(params_dict)
        model.model = tf.keras.models.load_model(tf_model_name)
        return model
