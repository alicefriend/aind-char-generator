import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

from keras.layers import Activation

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    assert len(series) > window_size
    X = [series[i:i+window_size] for i in range(len(series)-window_size)]
    y = [series[i+window_size] for i in range(len(series)-window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
	model = Sequential()
	model.add(LSTM(5, input_shape=(window_size,1)))
	model.add(Dense(1))
	return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
	# regard any punctuation as normal. 
	punctuation = ['!', ',', '.', ':', ';', '?']

	# 561688 index is where story ends and postface start.
	text = text[:561688]

	# All variations of alphabet is replaced by origin.
	text = text.replace('à', 'a')
	text = text.replace('â', 'a')
	text = text.replace('è', 'e')
	text = text.replace('é', 'e')
	return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    text_idx = np.arange(len(text)-window_size)
    iterator = text_idx[text_idx % step_size == 0]

    inputs = [text[i:i+window_size] for i in range(len(text)-window_size)]
    outputs = [text[i+window_size] for i in range(len(text)-window_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
	model = Sequential()
	model.add(LSTM(200, input_shape=(window_size, num_chars)))
	model.add(Dense(num_chars))
	model.add(Activation('softmax'))
	return model
