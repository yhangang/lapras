import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from tensorflow.python.keras.metrics import AUC,Precision
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
import random
import lapras
from datetime import datetime,timedelta
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.python.keras.callbacks import Callback
import tensorflow as tf


class WideDeepModel():
    def __init__(self, static_continue_X_cols:list, static_discrete_X_cols:list, rnn_continue_X_cols:list, ts_step:int, task='binary',
                 embedding=(1000,8,20),rnn_cells=64, hidden_units=[64,16], activation='swish', dropout=0.3, **kwargs):
        """
        Column names doesn't matter, but you should know what are you modeling.
        Args:
            ts_step: 时间序列步长
            task: binary:0-1二分类; regression:回归
        """
        self.ts_step = ts_step
        self.task = task
        self.static_continue_X_cols = static_continue_X_cols
        self.static_discrete_X_cols = static_discrete_X_cols
        self.rnn_continue_X_cols = rnn_continue_X_cols


        input1 = tf.keras.layers.Input(shape=len(static_continue_X_cols)) # 连续静态数据
        input2 = tf.keras.layers.Input(shape=len(static_discrete_X_cols)) # 离散静态数据
        input3 = tf.keras.layers.Input(shape=(ts_step,len(rnn_continue_X_cols))) # 连续时间序列数据
        x1 = tf.keras.layers.BatchNormalization()(input1)
        x3 = tf.keras.layers.BatchNormalization()(input3)

        x2 = tf.keras.layers.Embedding(embedding[0], embedding[1])(input2)
        x2 = tf.keras.layers.GRU(embedding[2])(x2)

        x3 = tf.keras.layers.LSTM(units=rnn_cells)(x3)

        x = tf.keras.layers.concatenate([x1, x2, x3], axis=1)

        for i in range(len(hidden_units)):
            x = tf.keras.layers.Dense(hidden_units[i])(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation)(x)
            x = tf.keras.layers.Dropout(dropout)(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.models.Model(inputs=[input1,input2,input3], outputs=output)


    def compile(self, loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=[Precision(), AUC()],**kwargs):
        self.model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics,**kwargs)
        self.model.summary()

    def pre_processing(self, static_continue_X, static_discrete_X, rnn_continue_X, y, **kwargs):
        """
        对原始数据进行预处理，输出直接可以入模的格式
        """
        return [static_continue_X, static_discrete_X, rnn_continue_X], y


    def fit(self, x, y, x_test, y_test, epochs=10,batch_size=256,validation_split=0,**kwargs):
        class TrainAUC(Callback):
            def on_epoch_end(self, epoch, logs=None):
                print()
                predictions = self.model.predict(x)
                auc_score = roc_auc_score(y, predictions)
                fpr, tpr, _ = roc_curve(y, predictions)
                ks = np.max(np.abs(tpr - fpr))
                print(' train_auc {:4f} train_ks {:4f}'.format(auc_score, ks))

                predictions = self.model.predict(x_test)
                auc_score = roc_auc_score(y_test, predictions)
                fpr, tpr, _ = roc_curve(y_test, predictions)
                ks = np.max(np.abs(tpr - fpr))
                print(' test_auc {:4f} test_ks {:4f}'.format(auc_score, ks))

        self.model.fit(x,y, epochs=epochs,batch_size=batch_size,validation_split=validation_split,
                       callbacks=[TrainAUC()], **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def _create_dataset(X, time_steps=1, step=1):
        """
        将二维时间序列数据reshape成三维
        """
        Xs= []
        for i in range(0, len(X), step):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)

        return np.array(Xs)





if __name__ == "__main__":
    model = WideDeepModel(['a'],['b'],['c'],ts_step=3)
    model.compile()