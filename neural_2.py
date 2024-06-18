import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from keras.layers import Embedding, Bidirectional, LSTM, SimpleRNN, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

# Загрузка датасета IMDB
num_words = 10000
maxlen = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Построение модели с обычной RNN
model_rnn = Sequential()
model_rnn.add(Embedding(num_words, 64))
model_rnn.add(SimpleRNN(64))
model_rnn.add(Dense(1, activation='sigmoid'))

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_rnn.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Построение модели с Bidirectional RNN
model_brnn = Sequential()
model_brnn.add(Embedding(num_words, 64))
model_brnn.add(Bidirectional(LSTM(64)))
model_brnn.add(Dense(1, activation='sigmoid'))

model_brnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_brnn.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))