"""Stacked LSTM architecture used for the v2 Lovecraft generator."""

# Build the sequential container once the training script defines X / y.
model = Sequential()

# First layer captures short-term patterns while keeping sequence context.
model.add(LSTM(24, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

# Middle layer increases capacity for longer motifs and keeps sequences alive.
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.3))

# Final recurrent layer compresses to a fixed-size thought vector.
model.add(LSTM(150))
model.add(Dropout(0.1))

# Softmax head returns a probability for each token in the vocabulary.
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
