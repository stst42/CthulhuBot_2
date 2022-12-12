# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# https://forums.developer.nvidia.com/t/does-gtx-1050ti-or-1650-for-notebook-support-tensorflow-gpu/77384/9
# https://stackoverflow.com/questions/47125723/keras-lstm-for-text-generation-keeps-repeating-a-line-or-a-sequence


# Load what is needed
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import requests

# load and preprocess
# load data
url = 'https://raw.githubusercontent.com/urschrei/lovecraft/master/lovecraft.txt'
url = requests.get(url)

raw_text = url.content
raw_text = raw_text.decode("latin-1")

# to lower and replace double a capo
# raw_text = raw_text.lower()
raw_text = raw_text.replace('\n\n', '')

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
execfile("/home/user01/CthulhuBot_2/scripts/models/second_lstm.py")

# define the checkpoint
filepath="weights/weights-improvement-second_lstm-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=50, batch_size=150, callbacks=callbacks_list)
