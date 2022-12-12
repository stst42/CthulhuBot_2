
# load the network weights
filename = "weights-improvement-v3-40-1.4905.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# try it

# helper function to sample an index from a probability array
def sample(preds,temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(y.shape[1],preds[0],1)
    return np.argmax(probas)

# test with random start in training
# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
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

# test with invented test
text='the creature in the darkness'
pattern = [char_to_int[char] for char in text]

print(text)
temperature = 1.8
new_text = []
# predict
for i in range(200):
    # reshape input
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    # make preds
    prediction = model.predict(x, verbose=0)
    # add randomness
    index = sample(prediction, temperature)
    # transform into characters
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # writeout result
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    # save in a vector
    new_text.append(result)
''.join(new_text)

