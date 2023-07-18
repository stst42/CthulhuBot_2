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
raw_text = raw_text.lower()
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
    seq_out = raw_text[i + seq_length:i + seq_length + 1]
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
model = Sequential()
model.add(LSTM(24, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(150))
model.add(Dropout(0.1))
model.add(Dense(n_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weigths/weights-improvement-v4-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=80, batch_size=64, callbacks=callbacks_list)

# generate text
seed = "The"
generated_text = []
for _ in range(100):
    prediction = model.predict([[seed]])[0]
    next_word = int_to_char[np.argmax(prediction)]
    generated_text.append(next_word)
    seed = seed + " " + next_word

print("".join(generated_text))
