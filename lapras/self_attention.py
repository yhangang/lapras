from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape",WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)


        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (self.output_dim**0.5)

        QK = K.softmax(QK)

        print("QK.shape",QK.shape)

        V = K.batch_dot(QK,WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)



max_features = 20000

print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#标签转换为独热码
y_train, y_test = pd.get_dummies(y_train),pd.get_dummies(y_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#%%数据归一化处理

maxlen = 64


print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)

#%%

batch_size = 128
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import *



S_inputs = Input(shape=(64,), dtype='int32')

embeddings = Embedding(max_features, 128)(S_inputs)


O_seq = Self_Attention(128)(embeddings)


O_seq = GlobalAveragePooling1D()(O_seq)

O_seq = Dropout(0.5)(O_seq)

outputs = Dense(2, activation='softmax')(O_seq)


model = Model(inputs=S_inputs, outputs=outputs)

print(model.summary())
# try using different optimizers and different optimizer configs
opt = Adam(lr=0.0002,decay=0.00001)
loss = 'categorical_crossentropy'
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

#%%
print('Train...')

h = model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test))

plt.plot(h.history["loss"],label="train_loss")
plt.plot(h.history["val_loss"],label="val_loss")
# plt.plot(h.history["acc"],label="train_acc")
# plt.plot(h.history["val_acc"],label="val_acc")
plt.legend()
plt.show()

