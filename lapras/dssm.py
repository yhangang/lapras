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


class DSSM():
    def __init__(self, u_continue_cols: list, u_discrete_cols: list, i_continue_cols: list, i_discrete_cols: list,
                 u_history_cols: list, u_history_col_names: list, u_history_col_ts_step: dict):
        """
        Column names doesn't matter, but you should know what are you modeling.
        Args:
            u_continue_cols: 用户连续特征列名
            u_discrete_cols: 用户离散特征列名
            i_continue_cols: 物品连续特征列名
            i_discrete_cols: 物品离散特征列名
            u_history_cols: 用户关于物品历史记录特征列名
            u_history_col_names: 用户历史记录列对应的物品特征列名
            u_history_col_ts_step: 用户历史行为对应的序列长度
        """
        self.u_continue_cols = u_continue_cols
        self.u_discrete_cols = u_discrete_cols
        self.u_history_cols = u_history_cols
        self.u_history_col_names = u_history_col_names
        self.i_continue_cols = i_continue_cols
        self.i_discrete_cols = i_discrete_cols
        self.u_history_col_ts_step = u_history_col_ts_step
        self.le_dict = {}
        self.scalar_dict = {}
        self.data_fitted = False

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
        user_input_features['u_continue_cols'] = tf.keras.layers.Input(shape=len(self.u_continue_cols), name='u_continue_cols_input')  # 用户数值特征
        for col in self.u_discrete_cols:
            user_input_features[col] = tf.keras.layers.Input(shape=1, name=col+'_input')  # 用户离散特征
        item_input_features['i_continue_cols'] = tf.keras.layers.Input(shape=len(self.i_continue_cols), name='i_continue_cols_input')  # 物品数值特征
        for col in self.i_discrete_cols:
            item_input_features[col] = tf.keras.layers.Input(shape=1, name=col+'_input')  # 物品离散特征
        for col in self.u_history_cols:
            user_input_features[col] = tf.keras.layers.Input(shape=self.u_history_col_ts_step[col], name=col+'_input')  # 用户关于物品的历史序列特征

        # 构造双塔结构
        user_vector_list = []
        item_vector_list = []

        # u_dense = tf.keras.layers.BatchNormalization()(user_input_features['u_continue_cols'])
        u_dense = user_input_features['u_continue_cols']
        user_vector_list.append(u_dense)

        # i_dense = tf.keras.layers.BatchNormalization()(item_input_features['i_continue_cols'])
        i_dense = item_input_features['i_continue_cols']
        item_vector_list.append(i_dense)

        for col in self.u_discrete_cols:
            le = self.le_dict[col]
            user_vector_list.append(tf.reshape(tf.keras.layers.Embedding(len(le.classes_)+10,
                self.embedding_dim, name=col+'_embedding')(user_input_features[col]), [-1, self.embedding_dim]))
        share_embedding_dict = {}
        for col in self.i_discrete_cols:
            le = self.le_dict[col]
            if col not in self.u_history_col_names:  # 该特征不在用户历史记录中
                item_vector_list.append(tf.reshape(tf.keras.layers.Embedding(len(le.classes_)+10,
                self.embedding_dim, name=col+'_embedding')(item_input_features[col]), [-1, self.embedding_dim]))
            else:
                embedding_dim = int(len(le.classes_) ** 0.25) + 1  # 动态确定维度
                embedding_layer = tf.keras.layers.Embedding(len(le.classes_)+10, embedding_dim, name=col+'_embedding')  # ItemId的embedding层用户和物品塔共用
                share_embedding_dict[col] = embedding_layer
                item_vector_list.append(tf.reshape(embedding_layer(item_input_features[col]), [-1, embedding_dim]))

        for i in range(len(self.u_history_cols)):
            item_col_name = self.u_history_col_names[i]
            le = self.le_dict[item_col_name]
            embedding_dim = int(len(le.classes_) ** 0.25) + 1  # 动态确定维度
            lstm_out_dim = int((embedding_dim * self.u_history_col_ts_step[self.u_history_cols[i]])/2)

            embedding_series = share_embedding_dict[item_col_name](user_input_features[self.u_history_cols[i]])
            # embedding_series = tf.keras.layers.BatchNormalization()(embedding_series)
            user_vector_list.append(tf.keras.layers.LSTM(units=lstm_out_dim)(embedding_series))

        user_embedding = tf.keras.layers.concatenate(user_vector_list, axis=1)
        item_embedding = tf.keras.layers.concatenate(item_vector_list, axis=1)

        for i in range(len(self.u_hidden_units[:-1])):
            # user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
            user_embedding = tf.keras.layers.Dense(self.u_hidden_units[i], activation=self.activation)(user_embedding)
            user_embedding = tf.keras.layers.Dropout(self.dropout)(user_embedding)
        user_embedding = tf.keras.layers.Dense(self.u_hidden_units[-1], name='user_embedding', activation='tanh')(user_embedding)
        # user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)

        for i in range(len(self.i_hidden_units[:-1])):
            # item_embedding = tf.keras.layers.BatchNormalization()(item_embedding)
            item_embedding = tf.keras.layers.Dense(self.i_hidden_units[i], activation=self.activation)(item_embedding)
            item_embedding = tf.keras.layers.Dropout(self.dropout)(item_embedding)
        item_embedding = tf.keras.layers.Dense(self.i_hidden_units[-1], name='item_embedding', activation='tanh')(item_embedding)
        # item_embedding = tf.keras.layers.BatchNormalization()(item_embedding)

        # 双塔向量做内积输出
        output = tf.expand_dims(tf.reduce_sum(user_embedding * item_embedding, axis=1), 1)
        # output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(output)
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
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def fit_data(self, user_df: pd.DataFrame, item_df: pd.DataFrame, fill_na=0):
        """
        对全量训练数据进行编码和归一化适配，保存编码器
        Args:
            user_df: 用户数据的Dataframe
            item_df: 物品数据的Dataframe
        """

        scalar_user = StandardScaler()
        scalar_user.fit(user_df[self.u_continue_cols])
        self.scalar_dict['scalar_user'] = scalar_user

        scalar_item = StandardScaler()
        scalar_item.fit(item_df[self.i_continue_cols])
        self.scalar_dict['scalar_item'] = scalar_item

        for col in self.u_discrete_cols:
            le = LabelEncoder()
            le.fit(user_df[col])
            le.classes_ = np.append(le.classes_, fill_na)  # 将不在训练集的枚举值统一变为 fill_na
            self.le_dict[col] = le

        for col in self.i_discrete_cols:
            le = LabelEncoder()
            le.fit(item_df[col])
            le.classes_ = np.append(le.classes_, fill_na)
            self.le_dict[col] = le

        self.data_fitted = True

    def user_processing(self, user_df: pd.DataFrame, user_his_df: pd.DataFrame, u_id_col: str,
                        start_index=0, end_index=-1, fill_na=0, **kwargs):
        """
        对原始数据进行预处理，输入为pandas dataframe，输出为直接入模的numpy array
        Args:
            user_df: 静态数据的Dataframe
            user_his_df: 时序数据的Dataframe
            u_id_col: 用户id列名
        """
        if type(user_df) != pd.DataFrame or type(user_his_df) != pd.DataFrame:
            raise ValueError("Error: Input X data must be Pandas.DataFrame format.\n输入数据必须是Pandas DataFrame格式！")
        # 填充缺失值
        user_df = user_df.fillna(fill_na)
        user_his_df = user_his_df.fillna(fill_na)

        if end_index == -1:
            end_index = len(user_df)
        u_ids = user_df.iloc[start_index:end_index][[u_id_col]]
        # 取出本次batch的数据
        user_batch_df = user_df[user_df[u_id_col].isin(u_ids[u_id_col])]
        user_batch_his_df = pd.merge(user_batch_df[[u_id_col]], user_his_df, how='left', on=[u_id_col])
        user_batch_his_df = user_batch_his_df.fillna(fill_na)  # 对没有任何历史记录的用户填充

        # 拼接用户端输入特征
        user_input = [self.scalar_dict['scalar_user'].transform(user_batch_df[self.u_continue_cols])]

        for col in self.u_discrete_cols:
            le = self.le_dict[col]
            user_batch_df_tmp = user_batch_df[col].map(lambda s: fill_na if s not in le.classes_ else s)
            user_input.append(le.transform(user_batch_df_tmp))
        for col in self.u_history_cols:
            item_col_name = self.u_history_col_names[self.u_history_cols.index(col)]
            le = self.le_dict[item_col_name]
            constant_fill_na = dict(zip(le.classes_, le.transform(le.classes_)))[fill_na]  # 取出fill_na在编码后的值

            user_batch_his_df_tmp = user_batch_his_df[col].map(lambda s: fill_na if s not in le.classes_ else s)
            user_batch_his_df[col] = le.transform(user_batch_his_df_tmp)

            user_batch_col_tmp = user_batch_his_df[[u_id_col, col]].groupby([u_id_col]) \
                .apply(lambda x: np.pad(x[col].values, (self.u_history_col_ts_step[col] - len(x[col].values), 0), 'constant',
                        constant_values=constant_fill_na) if len(x[col].values) < self.u_history_col_ts_step[col] else x[col].values)
            user_batch_col_tmp = np.stack(user_batch_col_tmp.values)
            user_input.append(user_batch_col_tmp)

        return user_input, user_batch_df[[u_id_col]].values

    def item_processing(self, item_df: pd.DataFrame, i_id_col: str, start_index=0, end_index=-1,
                       fill_na=0, **kwargs):
        """
        对原始数据进行预处理，输入为pandas dataframe，输出为直接入模的numpy array
        Args:
            item_df: 时序数据的Dataframe
            i_id_col: 物品id列名
        """
        if type(item_df) != pd.DataFrame:
            raise ValueError("Error: Input X data must be Pandas.DataFrame format.\n输入数据必须是Pandas DataFrame格式！")
        # 填充缺失值
        item_df = item_df.fillna(fill_na)

        if end_index == -1:
            end_index = len(item_df)
        i_ids = item_df.iloc[start_index:end_index][[i_id_col]]
        # 取出本次batch的数据
        item_batch_df = item_df[item_df[i_id_col].isin(i_ids[i_id_col])]

        # 拼接物品端输入特征
        item_input = [self.scalar_dict['scalar_item'].transform(item_batch_df[self.i_continue_cols])]
        for col in self.i_discrete_cols:
            le = self.le_dict[col]
            item_batch_df_tmp = item_batch_df[col].map(lambda s: fill_na if s not in le.classes_ else s)
            item_input.append(le.transform(item_batch_df_tmp))

        return item_input, item_batch_df[[i_id_col]].values

    def label_processing(self, user_df: pd.DataFrame, item_df: pd.DataFrame, user_his_df: pd.DataFrame, label_df: pd.DataFrame,
                       u_id_col: str, i_id_col: str, y_label: str, start_index=0, end_index=-1,
                       fill_na=0, **kwargs):
        """
        对原始数据进行预处理，输入为pandas dataframe，输出为直接入模的numpy array
        Args:
            user_df: 静态数据的Dataframe
            item_df: 时序数据的Dataframe
            user_his_df: 时序数据的Dataframe
            label_df: Y标签的Dataframe training为True时起作用
            u_id_col: user_id的列名
            i_id_col: item_id的列名
            start_index: 本次批量的开始位置
            end_index: 本次批量的结束位置
            y_label: Y标签列名  training为True时起作用
            fill_na: 缺失值填充
        """
        if type(user_df) != pd.DataFrame or type(item_df) != pd.DataFrame or type(user_his_df) != pd.DataFrame\
                or type(label_df) != pd.DataFrame:
            raise ValueError("Error: Input X data must be Pandas.DataFrame format.\n输入数据必须是Pandas DataFrame格式！")
        # 填充缺失值
        user_df = user_df.fillna(fill_na)
        item_df = item_df.fillna(fill_na)
        user_his_df = user_his_df.fillna(fill_na)

        # 先对离散字段做编码
        for col in self.i_discrete_cols:
            le = self.le_dict[col]
            tmp = item_df[col].map(lambda s: fill_na if s not in le.classes_ else s)
            item_df[col+str('_transformed')] = le.transform(tmp)

        if end_index == -1:
            end_index = len(user_df)
        u_ids = user_df.iloc[start_index:end_index][[u_id_col]]
        # 取出本次batch的数据
        user_batch_df = user_df[user_df[u_id_col].isin(u_ids[u_id_col])]
        # 笛卡尔积之前先做编码
        for col in self.u_discrete_cols:
            le = self.le_dict[col]
            tmp = user_batch_df[col].map(lambda s: fill_na if s not in le.classes_ else s)
            user_batch_df[col+str('_transformed')] = le.transform(tmp)

        user_batch_his_df = user_his_df[user_his_df[u_id_col].isin(u_ids[u_id_col])]
        # 先对离散字段做编码
        for col in self.u_history_cols:
            item_col_name = self.u_history_col_names[self.u_history_cols.index(col)]
            le = self.le_dict[item_col_name]
            tmp = user_batch_his_df[col].map(lambda s: fill_na if s not in le.classes_ else s)
            user_batch_his_df[col+str('_transformed')] = le.transform(tmp)

        # 获取用户侧输入特征
        label_df = label_df.drop_duplicates([u_id_col, i_id_col], keep='first')  # 防止出现脏数据
        print("label_df长度：" + str(len(label_df)))
        print("用户长度："+str(len(user_batch_df)))
        basic_df = pd.merge(user_batch_df, label_df, how='inner', on=[u_id_col])
        print("和用户拼接后长度：" + str(len(basic_df)))
        basic_df = pd.merge(basic_df, item_df, how='inner', on=[i_id_col])
        print("和图片拼接后长度：" + str(len(basic_df)))
        basic_df['id'] = basic_df[u_id_col] + "@" + basic_df[i_id_col]
        user_batch_his_df = pd.merge(basic_df[['id', u_id_col]], user_batch_his_df, how='left', on=[u_id_col])
        user_batch_his_df = user_batch_his_df.fillna(fill_na)  # 对没有任何历史记录的用户填充
        print("数据预处理完成")

        # 拼接输入特征
        X_input = [self.scalar_dict['scalar_user'].transform(basic_df[self.u_continue_cols])]
        for col in self.u_discrete_cols:
            X_input.append(basic_df[col+str('_transformed')])
        print("用户基础特征处理完成")

        for col in self.u_history_cols:
            constant_fill_na = dict(zip(le.classes_, le.transform(le.classes_)))[fill_na]  # 取出fill_na在编码后的值
            user_batch_col_tmp = user_batch_his_df[['id', col+str('_transformed')]].groupby(['id'])\
                .apply(lambda x: np.pad(x[col+str('_transformed')].values, (self.u_history_col_ts_step[col]-len(x[col+str('_transformed')].values), 0),'constant',
                constant_values=constant_fill_na) if len(x[col+str('_transformed')].values) < self.u_history_col_ts_step[col] else x[col+str('_transformed')].values)
            user_batch_col_tmp = np.stack(user_batch_col_tmp.values)
            X_input.append(user_batch_col_tmp)
        print("用户历史记录特征处理完成")

        # 拼接物品端输入特征
        X_input.append(self.scalar_dict['scalar_item'].transform(basic_df[self.i_continue_cols]))
        for col in self.i_discrete_cols:
            X_input.append(basic_df[col+str('_transformed')])
        print("物品特征处理完成")
        return X_input, basic_df[y_label].values, basic_df['id'].values

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
            metrit = [AUC_KS(), self.tensorboard_callback]
        else:
            metrit = [self.tensorboard_callback]
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

    def embedding_inner_product(self, user_embedding: np.ndarray, item_embedding: np.ndarray):
        result = self.sigmoid(np.sum((user_embedding * item_embedding), axis=1))
        return result

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_params(self):
        params_dict = {'u_continue_cols': self.u_continue_cols, 'u_discrete_cols': self.u_discrete_cols,
                       'u_history_cols': self.u_history_cols, 'u_history_col_names': self.u_history_col_names,
                       'i_continue_cols': self.i_continue_cols, 'i_discrete_cols': self.i_discrete_cols,
                       'u_history_col_ts_step': self.u_history_col_ts_step, 'le_dict': self.le_dict, 'scalar_dict': self.scalar_dict,
                       'data_fitted': self.data_fitted, 'u_hidden_units': self.u_hidden_units,
                       'i_hidden_units': self.i_hidden_units, 'activation': self.activation, 'dropout': self.dropout,
                       'embedding_dim': self.embedding_dim, 'user_input_len': self.user_input_len,
                       'item_input_len': self.item_input_len}
        return params_dict

    def _load_params(self, params_dict: dict):
        self.u_continue_cols = params_dict['u_continue_cols']
        self.u_discrete_cols = params_dict['u_discrete_cols']
        self.u_history_cols = params_dict['u_history_cols']
        self.u_history_col_names = params_dict['u_history_col_names']
        self.i_continue_cols = params_dict['i_continue_cols']
        self.i_discrete_cols = params_dict['i_discrete_cols']
        self.u_history_col_ts_step = params_dict['u_history_col_ts_step']
        self.le_dict = params_dict['le_dict']
        self.scalar_dict = params_dict['scalar_dict']
        self.data_fitted = params_dict['data_fitted']
        self.u_hidden_units = params_dict['u_hidden_units']
        self.i_hidden_units = params_dict['i_hidden_units']
        self.activation = params_dict['activation']
        self.dropout = params_dict['dropout']
        self.embedding_dim = params_dict['embedding_dim']
        self.user_input_len = params_dict['user_input_len']
        self.item_input_len = params_dict['item_input_len']

    def save(self, path='./model_file', name='dssm_model'):
        if not os.path.exists(path):
            os.makedirs(path)
        tf_model_name = path + "/" + name
        param_file_name = tf_model_name + ".param"
        f = open(param_file_name, 'wb')  # pickle只能以二进制格式存储数据到文件
        f.write(pickle.dumps(self.get_params()))  # dumps序列化源数据后写入文件
        f.close()
        tf.saved_model.save(self.model, tf_model_name)

    @classmethod
    def load(cls, path='./model_file', name='dssm_model'):
        model = DSSM([], [], [], [], [], [], 0)

        tf_model_name = path + "/" + name
        param_file_name = tf_model_name + ".param"

        f = open(param_file_name, 'rb')
        params_dict = pickle.loads(f.read())
        f.close()
        model._load_params(params_dict)

        model.model = tf.keras.models.load_model(tf_model_name)
        return model
