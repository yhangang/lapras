import math
import os

import numpy as np
import pandas as pd
from sklearn import model_selection
from tensorflow.python.keras.metrics import AUC, Precision
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.python.keras.callbacks import Callback
import tensorflow as tf


class WideDeepBinary():
    def __init__(self, static_continue_X_cols:list, static_discrete_X_cols:list, rnn_continue_X_cols:list, ts_step:int,
                 embedding=(1000, 8), discrete_cells=20, rnn_cells=64, hidden_units=[64,16], activation='swish',
                 dropout=0.3, network_reinforce=False):
        """
        Column names doesn't matter, but you should know what are you modeling.
        Args:
            static_continue_X_cols: 静态连续特征列名
            static_discrete_X_cols: 静态离散特征列名
            rnn_continue_X_cols: 时序连续特征列名
            ts_step: 时间序列步长
            embedding: 静态离散特征embedding参数,embedding[0]表示输入值离散空间上限,embedding[1]表示输出向量维度
            discrete_cells: 静态离散数据输出神经元个数
            rnn_cells: 时序连续特征输出神经元个数
            hidden_units： MLP层神经元参数，最少2层
            activation: MLP层激活函数
            dropout: dropout系数
            network_reinforce: 是否使用网络增强

        """

        self.static_continue_X_cols = static_continue_X_cols
        self.static_discrete_X_cols = static_discrete_X_cols
        self.rnn_continue_X_cols = rnn_continue_X_cols
        self.ts_step = ts_step
        self.embedding = embedding
        self.discrete_cells = discrete_cells
        self.rnn_cells = rnn_cells
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.network_reinforce = network_reinforce

        input1 = tf.keras.layers.Input(shape=len(static_continue_X_cols)) # 连续静态数据
        input2 = tf.keras.layers.Input(shape=len(static_discrete_X_cols)) # 离散静态数据
        input3 = tf.keras.layers.Input(shape=(ts_step,len(rnn_continue_X_cols))) # 连续时间序列数据

        x1 = tf.keras.layers.BatchNormalization()(input1)

        x2 = tf.keras.layers.Embedding(embedding[0], embedding[1])(input2)
        x2 = tf.keras.layers.GRU(discrete_cells, recurrent_initializer='glorot_uniform')(x2)

        x3 = tf.keras.layers.BatchNormalization()(input3)
        x3 = tf.keras.layers.LSTM(units=rnn_cells, recurrent_initializer='glorot_uniform')(x3)

        x = tf.keras.layers.concatenate([x1, x2, x3], axis=1)

        #######################################################################
        # 网络增强
        if network_reinforce:
            x0 = tf.keras.layers.BatchNormalization()(x)
            encoder = tf.keras.layers.GaussianNoise(dropout)(x0)
            encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
            encoder = tf.keras.layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Activation('swish')(encoder)

            decoder = tf.keras.layers.Dropout(dropout)(encoder)
            decoder = tf.keras.layers.Dense(len(static_continue_X_cols)+discrete_cells+rnn_cells, name='decoder')(decoder)

            x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
            x_ae = tf.keras.layers.BatchNormalization()(x_ae)
            x_ae = tf.keras.layers.Activation('swish')(x_ae)
            x_ae = tf.keras.layers.Dropout(dropout)(x_ae)
            out_ae = tf.keras.layers.Dense(1, activation='sigmoid', name='ae_action')(x_ae)

            x = tf.keras.layers.Concatenate()([x0, encoder])
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout)(x)

        #######################################################################

        for i in range(len(hidden_units)):
            x = tf.keras.layers.Dense(hidden_units[i])(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation)(x)
            x = tf.keras.layers.Dropout(dropout)(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid', name='action')(x)

        if network_reinforce:
            self.model = tf.keras.models.Model(inputs=[input1, input2, input3], outputs=[out_ae, output])
        else:
            self.model = tf.keras.models.Model(inputs=[input1, input2, input3], outputs=output)

    def compile(self, loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=[Precision(), AUC()], summary=True, **kwargs):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)
        if summary:
            self.model.summary()

    def pre_processing(self, basic_df:pd.DataFrame, rnn_df:pd.DataFrame, id_label:str, ts_label:str, training=True,
                       y_label='y', test_size=0.2, fill_na=0, **kwargs):
        """
        对原始数据进行预处理，输入为pandas dataframe，输出为直接入模的numpy array
        Args:
            basic_df: 静态数据的Dataframe
            rnn_df: 时序数据的Dataframe
            id_label: id的列名
            ts_label: 时序标签的列名
            training: 是否为训练步骤,True or False
            y_label: Y标签列名  training为True时起作用
            test_size: 测试集比例  training为True时起作
            fill_na: 缺失值填充
        """
        if type(basic_df) != pd.DataFrame or type(rnn_df) != pd.DataFrame:
            print("Error: Input X data must be Pandas.DataFrame format.")
            return

        if len(basic_df)*self.ts_step != len(rnn_df):
            print("Error: Some of the train data size is different from others, please check it again.")
            return

        try:
            basic_df[self.static_continue_X_cols]
            basic_df[self.static_discrete_X_cols]
            rnn_df[self.rnn_continue_X_cols]
        except:
            print("Error: Some of the declared columns is not in the input data, please check it again.")
            return

        # 对连续型数据填充缺失值
        basic_df = basic_df.fillna(fill_na)
        rnn_df = rnn_df.fillna(fill_na)

        # 区分训练和预测场景
        if training:
            # 划分训练集和验证集
            train_id, test_id, _, _ = model_selection.train_test_split(basic_df[[id_label, y_label]],
                basic_df[y_label], test_size=test_size, random_state=2020)

            # 划分训练集和测试集——静态和时序宽表
            train_basic_df = basic_df[basic_df[id_label].isin(train_id[id_label])]
            test_basic_df = basic_df[basic_df[id_label].isin(test_id[id_label])]
            train_rnn_df = rnn_df[rnn_df[id_label].isin(train_id[id_label])]
            test_rnn_df = rnn_df[rnn_df[id_label].isin(test_id[id_label])]

            # 排序
            train_basic_df = train_basic_df.sort_values(id_label)
            test_basic_df = test_basic_df.sort_values(id_label)
            train_rnn_df = train_rnn_df.sort_values([id_label, ts_label])
            test_rnn_df = test_rnn_df.sort_values([id_label, ts_label])

            # 构造Y标向量
            y_train = np.array(train_basic_df[[y_label]])
            y_test = np.array(test_basic_df[[y_label]])

            # 按特征类型构造入模向量X
            static_continue_X_train = np.array(train_basic_df[self.static_continue_X_cols])
            static_continue_X_test = np.array(test_basic_df[self.static_continue_X_cols])

            static_discrete_X_train = np.array(train_basic_df[self.static_discrete_X_cols])
            static_discrete_X_test = np.array(test_basic_df[self.static_discrete_X_cols])

            rnn_continue_X_train = self._create_dataset(train_rnn_df[self.rnn_continue_X_cols], self.ts_step)
            rnn_continue_X_test = self._create_dataset(test_rnn_df[self.rnn_continue_X_cols], self.ts_step)

            return [static_continue_X_train, static_discrete_X_train, rnn_continue_X_train], y_train,\
                    [static_continue_X_test, static_discrete_X_test, rnn_continue_X_test], y_test

        else:
            train_basic_df = basic_df
            train_rnn_df = rnn_df

            train_basic_df = train_basic_df.sort_values(id_label)
            train_rnn_df = train_rnn_df.sort_values([id_label, ts_label])

            # 按特征类型构造入模向量X
            static_continue_X_train = np.array(train_basic_df[self.static_continue_X_cols])
            static_discrete_X_train = np.array(train_basic_df[self.static_discrete_X_cols])
            rnn_continue_X_train = self._create_dataset(train_rnn_df[self.rnn_continue_X_cols], self.ts_step)

            return [static_continue_X_train, static_discrete_X_train, rnn_continue_X_train], train_basic_df[[id_label]]

    def fit(self, X, y, X_test=None, y_test=None, epochs=10,batch_size=256,validation_split=0,**kwargs):
        network_reinforce = self.network_reinforce
        class AUC_KS(Callback):
            def on_epoch_end(self, epoch, logs=None):
                print()
                predictions = self.model.predict(X)
                if network_reinforce:
                    predict = predictions[1]
                else:
                    predict = predictions
                auc_score = roc_auc_score(y, predict)
                fpr, tpr, _ = roc_curve(y, predict)
                ks = np.max(np.abs(tpr - fpr))
                print(' train_auc {:4f} train_ks {:4f}'.format(auc_score, ks))

                if X_test is not None and y_test is not None:
                    predictions = self.model.predict(X_test)
                    if network_reinforce:
                        predict = predictions[1]
                    else:
                        predict = predictions
                    auc_score = roc_auc_score(y_test, predict)
                    fpr, tpr, _ = roc_curve(y_test, predict)
                    ks = np.max(np.abs(tpr - fpr))
                    print(' test_auc {:4f} test_ks {:4f}'.format(auc_score, ks))

        if self.network_reinforce:
            y = [y,y]
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                       callbacks=[AUC_KS()], **kwargs)

    def predict(self, x, **kwargs):
        predictions = self.model.predict(x, **kwargs)
        if self.network_reinforce:
            return predictions[1]
        else:
            return predictions

    def _create_dataset(self, X, ts_step=1):
        """
        将二维时间序列数据reshape成三维
        """
        Xs= []
        for i in range(0, len(X), ts_step):
            v = X.iloc[i:(i + ts_step)].values
            Xs.append(v)

        return np.array(Xs)

    def get_params(self):
        params_dict = {'static_continue_X_cols': self.static_continue_X_cols,
                       'static_discrete_X_cols': self.static_discrete_X_cols,
                       'rnn_continue_X_cols': self.rnn_continue_X_cols, 'ts_step': self.ts_step,
                       'embedding': self.embedding, 'discrete_cells': self.discrete_cells, 'rnn_cells': self.rnn_cells,
                       'hidden_units': self.hidden_units, 'activation': self.activation, 'dropout': self.dropout,
                       'network_reinforce': self.network_reinforce}
        return params_dict

    def _load_params(self, params_dict: dict):
        self.static_continue_X_cols = params_dict['static_continue_X_cols']
        self.static_discrete_X_cols = params_dict['static_discrete_X_cols']
        self.rnn_continue_X_cols = params_dict['rnn_continue_X_cols']
        self.ts_step = params_dict['ts_step']
        self.embedding = params_dict['embedding']
        self.discrete_cells = params_dict['discrete_cells']
        self.rnn_cells = params_dict['rnn_cells']
        self.hidden_units = params_dict['hidden_units']
        self.activation = params_dict['activation']
        self.dropout = params_dict['dropout']
        self.network_reinforce = params_dict['network_reinforce']

