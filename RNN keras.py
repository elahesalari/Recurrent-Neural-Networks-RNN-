import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class RNN:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def read_data(self):
        annotates = []
        samples = []
        for i in range(1, 74):
            if i == 15:
                continue
            dt_annotate = pd.read_csv(f'dataset/annotation_{i}.csv')
            dt_sample = pd.read_csv(f'dataset/samples_{i}.csv')

            annotates.append(np.array(dt_annotate.iloc[:, -1], dtype=int))
            list_sm = dt_sample.iloc[1:, -1].tolist()
            sam = np.array([float(item) for item in list_sm])
            samples.append(sam)

        return annotates, samples

    def splitter(self, annotation, sample, size):

        Xtrain = np.zeros((1, (size * 2 - 1)))
        Xtest = np.zeros((1, (size * 2 - 1)))
        Ytrain = np.zeros(1)
        Ytest = np.zeros(1)

        for i in range(72):
            pattern = self.window_sliding(annotation[i], sample[i], size)
            x = pattern[:, :-1]
            y = pattern[:, -1]

            idx_split = int(x.shape[0] * 0.8)

            x_train = x[:idx_split]
            x_test = x[idx_split:]
            y_train = y[:idx_split]
            y_test = y[idx_split:]

            Xtrain = np.concatenate((Xtrain, x_train))
            Xtest = np.concatenate((Xtest, x_test))
            Ytrain = np.concatenate((Ytrain, y_train))
            Ytest = np.concatenate((Ytest, y_test))

        self.x_train = Xtrain[1:]
        self.x_test = Xtest[1:]
        self.y_train = Ytrain[1:]
        self.y_test = Ytest[1:]

        self.y_train = self.one_hot(self.y_train, 4)
        self.y_test = self.one_hot(self.y_test, 4)

    def window_sliding(self, annotation, sample, size):
        slide_size = len(sample) - size + 1
        data_1 = np.zeros((slide_size, size * 2))
        for s in range(slide_size):
            sm = sample[s:s + size]
            an = annotation[s:s + size]
            pair = zip(sm, an)
            signal = []
            for i, j in pair:
                signal.append(i)
                signal.append(j)
            data_1[s] = signal

        return data_1

    def one_hot(self, y, c):
        Ty = np.zeros((max(y.shape), c))
        for i in range(c):
            Ty[y.reshape(max(y.shape)) == i, i] = 1

        return Ty

    def elman(self):
        x_train = self.x_train
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # data convert to 3D

        x_test = self.x_test
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        y_train = self.y_train
        y_test = self.y_test

        model = keras.Sequential()
        # Adding hidden layer
        model.add(keras.layers.SimpleRNN(units=y_train.shape[1], return_sequences=False, activation=tf.nn.tanh))

        # Adding output layer
        model.add(keras.layers.Dense(units=y_train.shape[1], activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5, batch_size=64)

        resualt_test = model.evaluate(x_test, y_test, verbose=0)[1]
        resualt_train = model.evaluate(x_train, y_train, verbose=0)[1]

        return resualt_test, resualt_train

    def narx(self):
        x_train = self.x_train
        y_train = self.y_train

        net = NARX()

        optimizer = tf.keras.optimizers.Adam(learning_rate=.002)
        net.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        net.fit(x_train, y_train, epochs=5, batch_size=64)

        resualt_test = net.evaluate(self.x_test, self.y_test, verbose=0)[1]
        resualt_train = net.evaluate(x_train, y_train, verbose=0)[1]

        return resualt_test, resualt_train


class NARX(keras.Model):

    def __init__(self):
        super(NARX, self).__init__(name='narx')
        self.hidden = keras.layers.Dense(4, activation=keras.activations.tanh)
        self.output_layer = keras.layers.Dense(4, activation=keras.activations.softmax)

    def call(self, inputs):
        x = self.hidden(inputs)
        return self.output_layer(x)


if __name__ == '__main__':
    rnn = RNN()
    annotate, sample = rnn.read_data()
    sliding_size = [5, 11, 21]
    acc_narx = []
    acc_elman = []
    for s in sliding_size:
        rnn.splitter(annotate, sample, s)
        print('Elman network, window size=', s)
        acc_test_elman, acc_train_elman = rnn.elman()
        acc_elman.append([acc_test_elman, acc_train_elman])
        print('Finish Elman')

        print('NARX network, window size=', s)
        acc_test_narx, acc_train_narx = rnn.narx()
        acc_narx.append([acc_test_narx, acc_train_narx])
        print('Finish NARX')

    for i in range(len(sliding_size)):
        print(
            f'Elman -->  Sliding size:{sliding_size[i]} - Test Accuracy: {"%.3f" % acc_elman[i][0]} & Train Accuracy: {"%.3f" % acc_elman[i][1]}')
        print(
            f'NARX  -->  Sliding size:{sliding_size[i]} - Test Accuracy: {"%.3f" % acc_narx[i][0]} & Train Accuracy: {"%.3f" % acc_narx[i][1]}')
