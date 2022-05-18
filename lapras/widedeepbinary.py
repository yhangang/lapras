import math
import os

import lapras
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.metrics import AUC, Precision
import tensorflow as tf
from hyperopt import fmin, tpe, hp
import hyperopt
import pickle


class WideDeepBinary():
    def __init__(self, static_continue_X_cols:list, static_discrete_X_cols:list, rnn_continue_X_cols:list, ts_step: int):
        """
        Column names don't matter, but you should know what are you modeling.
        Args:
            static_continue_X_cols: 静态连续特征列名
            static_discrete_X_cols: 静态离散特征列名
            rnn_continue_X_cols: 时序连续特征列名
            ts_step: 时间序列步长

        """
        self.static_continue_X_cols = static_continue_X_cols
        self.static_discrete_X_cols = static_discrete_X_cols
        self.rnn_continue_X_cols = rnn_continue_X_cols
        self.ts_step = ts_step
        self.embedding_dim = self.rnn_cells = self.activation = self.dropout = \
            self.hidden_units = self.model = None
        self.le_dict = {}
        self.scalar_basic = None
        self.scalar_rnn = None

    def compile(self, embedding_dim=4, rnn_cells=64, hidden_units=[64,16], activation='relu',
                 dropout=0.3, loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(1e-4),  metrics=[Precision(), AUC()], summary=True, **kwargs):
        """
                Args:
                    embedding_dim: 静态离散特征embedding参数,表示输出向量维度，输入词典维度由训练数据自动计算产生
                    rnn_cells: 时序连续特征输出神经元个数
                    hidden_units： MLP层神经元参数，最少2层
                    activation: MLP层激活函数
                    dropout: dropout系数
                    loss: 损失函数
                    optimizer: 优化器
                    metrics: 效果度量函数
                    summary: 是否输出summary信息
                """
        self.embedding_dim = embedding_dim
        self.rnn_cells = rnn_cells
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout

        if not self.scalar_rnn:
            print("数据Scalar尚未初始化，请先调用pre_processing方法进行数据预处理，然后才能编译模型！")
            return

        # 定义输入格式
        input_features = OrderedDict()

        input_features['input_rnn_continue'] = tf.keras.layers.Input(shape=(self.ts_step, len(self.rnn_continue_X_cols)), name='input_rnn_continue')  # 连续时间序列数据
        if self.static_continue_X_cols:
            input_features['input_static_continue'] = tf.keras.layers.Input(shape=len(self.static_continue_X_cols), name='input_static_continue')  # 连续静态数据
        for col in self.static_discrete_X_cols:
            input_features[col] = tf.keras.layers.Input(shape=1, name=col)  # 静态离散特征

        # 构造网络结构
        rnn_dense = [tf.keras.layers.LSTM(units=self.rnn_cells)(input_features['input_rnn_continue'])]
        static_dense = []
        if self.static_continue_X_cols:
            static_dense = [input_features['input_static_continue']]
        static_discrete = []
        for col in self.static_discrete_X_cols:
            vol_size = len(self.le_dict[col].classes_) + 1
            vec = tf.keras.layers.Embedding(vol_size, self.embedding_dim)(input_features[col])
            static_discrete.append(tf.reshape(vec, [-1, self.embedding_dim]))

        concated_vec = rnn_dense + static_dense + static_discrete
        if len(concated_vec) == 1:
            x = concated_vec[0]
        else:
            x = tf.keras.layers.concatenate(concated_vec, axis=1)

        # 特征拼接后加入全连接层
        for i in range(len(hidden_units)):
            x = tf.keras.layers.Dense(hidden_units[i], activation=activation)(x)
            x = tf.keras.layers.Dropout(dropout)(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid', name='action')(x)

        inputs_list = list(input_features.values())
        self.model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

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
            raise ValueError("Error: Input X data must be Pandas.DataFrame format.\n输入数据必须是Pandas DataFrame格式！")

        if len(basic_df)*self.ts_step != len(rnn_df):
            raise ValueError("Error: Some of the train data size is different from others, please check it again."
                             "\n时间序列数据记录数没对齐！")

        try:
            basic_df[self.static_continue_X_cols]
            basic_df[self.static_discrete_X_cols]
            rnn_df[self.rnn_continue_X_cols]
        except:
            raise ValueError("Error: Some of the declared columns is not in the input data, please check it again."
                             "\n声明的列名在数据中不存在！")

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

            if self.static_continue_X_cols:
                self.scalar_basic = StandardScaler()
                self.scalar_basic.fit(train_basic_df[self.static_continue_X_cols])
                basic_X_train_transform = self.scalar_basic.transform(train_basic_df[self.static_continue_X_cols])
                basic_X_test_transform = self.scalar_basic.transform(test_basic_df[self.static_continue_X_cols])

            self.scalar_rnn = StandardScaler()
            self.scalar_rnn.fit(train_rnn_df[self.rnn_continue_X_cols])
            rnn_X_train_transform = self.scalar_rnn.transform(train_rnn_df[self.rnn_continue_X_cols])
            rnn_X_test_transform = self.scalar_rnn.transform(test_rnn_df[self.rnn_continue_X_cols])

            # 对离散特征进行LabelEncoder编码
            if self.static_discrete_X_cols:
                category_X_train = pd.DataFrame(columns=self.static_discrete_X_cols)
                category_X_test = pd.DataFrame(columns=self.static_discrete_X_cols)

            for col in self.static_discrete_X_cols:
                le = LabelEncoder()
                le.fit(train_basic_df[col])

                test_basic_col_tmp = test_basic_df[col].map(lambda s: -99 if s not in le.classes_ else s)
                le.classes_ = np.append(le.classes_, -99)

                category_X_train[col] = le.transform(train_basic_df[col])
                category_X_test[col] = le.transform(test_basic_col_tmp)

                self.le_dict[col] = le

            # 构造Y标向量
            y_train = np.array(train_basic_df[[y_label]])
            y_test = np.array(test_basic_df[[y_label]])

            # 按特征类型构造入模向量X
            rnn_continue_X_train = self._create_dataset(rnn_X_train_transform, self.ts_step)
            rnn_continue_X_test = self._create_dataset(rnn_X_test_transform, self.ts_step)

            X_train = [rnn_continue_X_train]
            if self.static_continue_X_cols:
                X_train.append(np.array(basic_X_train_transform))
            for col in self.static_discrete_X_cols:
                X_train.append(np.array(category_X_train[col]))

            X_test = [rnn_continue_X_test]
            if self.static_continue_X_cols:
                X_test.append(np.array(basic_X_test_transform))
            for col in self.static_discrete_X_cols:
                X_test.append(np.array(category_X_test[col]))

            return X_train, y_train, X_test, y_test

        else:
            all_basic_df = basic_df
            all_rnn_df = rnn_df

            # 排序
            all_basic_df = all_basic_df.sort_values(id_label)
            all_rnn_df = all_rnn_df.sort_values([id_label, ts_label])

            if self.static_continue_X_cols:
                basic_X_all_transform = self.scalar_basic.transform(all_basic_df[self.static_continue_X_cols])

            rnn_X_all_transform = self.scalar_rnn.transform(all_rnn_df[self.rnn_continue_X_cols])

            # 对离散特征进行LabelEncoder编码
            if self.static_discrete_X_cols and not self.le_dict:
                print("请首先生成训练集样本，LabelEncoder还未初始化！")
                return

            if self.static_discrete_X_cols:
                category_X_all = pd.DataFrame(columns=self.static_discrete_X_cols)
            for col in self.static_discrete_X_cols:
                le = self.le_dict[col]
                test_basic_col_tmp = all_basic_df[col].map(lambda s: -99 if s not in le.classes_ else s)
                category_X_all[col] = le.transform(test_basic_col_tmp)

            # 按特征类型构造入模向量X
            rnn_continue_X_train = self._create_dataset(rnn_X_all_transform, self.ts_step)

            X_predict = [rnn_continue_X_train]
            if self.static_continue_X_cols:
                X_predict.append(np.array(basic_X_all_transform))
            for col in self.static_discrete_X_cols:
                X_predict.append(np.array(category_X_all[col]))

            return X_predict, all_basic_df[[id_label]].values

    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=10, batch_size=256, validation_split=0, callback=True,
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

    def _create_dataset(self, X, ts_step=1):
        """
        将二维时间序列数据reshape成三维
        """
        n = int(X.shape[0] / ts_step)
        return X.reshape(n, ts_step, X.shape[1])

    def get_params(self):
        params_dict = {'static_continue_X_cols': self.static_continue_X_cols,
                       'static_discrete_X_cols': self.static_discrete_X_cols,
                       'rnn_continue_X_cols': self.rnn_continue_X_cols, 'ts_step': self.ts_step,
                       'embedding_dim': self.embedding_dim, 'rnn_cells': self.rnn_cells,
                       'hidden_units': self.hidden_units, 'activation': self.activation, 'dropout': self.dropout,
                       'le_dict': self.le_dict,
                       'scalar_basic': self.scalar_basic, 'scalar_rnn': self.scalar_rnn}
        return params_dict

    def _load_params(self, params_dict: dict):
        self.static_continue_X_cols = params_dict['static_continue_X_cols']
        self.static_discrete_X_cols = params_dict['static_discrete_X_cols']
        self.rnn_continue_X_cols = params_dict['rnn_continue_X_cols']
        self.ts_step = params_dict['ts_step']
        self.embedding_dim = params_dict['embedding_dim']
        self.rnn_cells = params_dict['rnn_cells']
        self.hidden_units = params_dict['hidden_units']
        self.activation = params_dict['activation']
        self.dropout = params_dict['dropout']
        self.le_dict = params_dict['le_dict']
        self.scalar_basic = params_dict['scalar_basic']
        self.scalar_rnn = params_dict['scalar_rnn']

    def param_optimize(self, X_train, y_train, X_test, y_test, embedding_dim=(3, 6),
                        rnn_cells=[64, 128, 256, 512], hidden_units_layers=(2, 4),
                        hidden_units_cells1=[64, 128, 256, 512], hidden_units_cells2=[16, 32, 64, 128],
                        hidden_units_cells3=[4, 8, 16, 32], hidden_units_cells4=[2, 4, 8],
                        activation=['swish', 'tanh', 'sigmoid', 'relu'], dropout=(0.1, 0.8),
                        epochs=10, batch_size=256, max_evals=100):
        """
        通过贝叶斯调参寻找最优参数
        Args:
            X: 训练集X数据集
            y: 训练集y数据集
            X_test: 测试集X数据集
            y_test: 测试集y数据集
            embedding_dim: 离散数据embedding层输出维度，tuple类型,表示最小最大整数范围
            rnn_cells: 序列数据最终输出层神经元个数，list类型,指定枚举值
            hidden_units_layers: MLP层的层数,最多支持4层
            hidden_units_cells[i]: MLP每层神经元数
            activation: 激活函数枚举值
            dropout: dropout的调参范围，必须是0-1之间的实数
            #######以下非调优参数#######
            epochs: 指定epochs，不对该参数进行调优
            batch_size: 指定batch_size，不对该参数进行调优
            max_evals: 优化器最大迭代次数
        """
        # space = {
        #     'x': hp.uniform('x', 0, 1),  # 0-1的均匀分布
        #     'y': hp.normal('y', 0, 1),  # 0-1正态分布
        #     'name': hp.choice('name', ['alice', 'bob']), }  # 枚举值

        # 根据入参构造待调优参数列表
        space = {
            'embedding_dim': hp.choice('embedding_dim', list(range(embedding_dim[0], embedding_dim[1]+1))),
            'rnn_cells': hp.choice('rnn_cells', rnn_cells),
            'activation': hp.choice('activation', activation),
            'dropout': hp.uniform('dropout', dropout[0], dropout[1]),  # 0-1的均匀分布
            'hidden_units_cells1': hp.choice('hidden_units_cells1', hidden_units_cells1),
            'hidden_units_cells2': hp.choice('hidden_units_cells2', hidden_units_cells2),
            'hidden_units_cells3': hp.choice('hidden_units_cells3', hidden_units_cells3),
            'hidden_units_cells4': hp.choice('hidden_units_cells4', hidden_units_cells4),
            'hidden_units_layers': hp.choice('hidden_units_layers', list(range(hidden_units_layers[0],
                                                                               hidden_units_layers[1]+1))),
        }

        # define an objective function
        def objective(args):
            print("开始优化迭代，本次参数为：")
            print(args)

            hidden_units_layers = []
            for i in range(1, args['hidden_units_layers']+1):
                hidden_units_layers.append(args['hidden_units_cells'+str(i)])

            self.compile(embedding_dim=args['embedding_dim'], rnn_cells=args['rnn_cells'],
                         activation=args['activation'], dropout=args['dropout'], hidden_units=hidden_units_layers,
                         summary=False)

            self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callback=False)
            result = self.predict(X_test)
            auc = lapras.AUC(result.reshape(-1,), y_test.reshape(-1,))
            print("本次验证集AUC值为：" + str(auc))
            print("===================================================================")

            return -auc  # 优化目标AUC最大，也就是-AUC最小

        # minimize the objective over the space
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals
        )

        best_params = hyperopt.space_eval(space, best)
        print("优化完成，最优参数为：" + str(best_params))
        print("\n")
        print("开始按照最优参数重新训练模型……")

        hidden_units_layers = []
        for i in range(1, best_params['hidden_units_layers'] + 1):
            hidden_units_layers.append(best_params['hidden_units_cells' + str(i)])

        self.compile(embedding_dim=best_params['embedding_dim'], rnn_cells=best_params['rnn_cells'],
                     activation=best_params['activation'], dropout=best_params['dropout'], hidden_units=hidden_units_layers,
                     summary=False)

        self.fit(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size, verbose=0)
        print("模型已按照最优参数重新训练，可直接使用")

        return best_params

    def save(self, path='./model_file', name='widedeepbinary_model'):
        if not os.path.exists(path):
            os.makedirs(path)
        tf_model_name = path + "/" + name
        param_file_name = tf_model_name + ".param"
        f = open(param_file_name, 'wb')  # pickle只能以二进制格式存储数据到文件
        f.write(pickle.dumps(self.get_params()))  # dumps序列化源数据后写入文件
        f.close()
        tf.saved_model.save(self.model, tf_model_name)

    @classmethod
    def load(cls, path='./model_file', name='widedeepbinary_model'):
        model = WideDeepBinary([], [], [], 0)

        tf_model_name = path + "/" + name
        param_file_name = tf_model_name + ".param"

        f = open(param_file_name, 'rb')
        params_dict = pickle.loads(f.read())
        f.close()
        model._load_params(params_dict)

        model.model = tf.keras.models.load_model(tf_model_name)
        return model
