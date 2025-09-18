"""Variant of the v2 architecture retained for quick A/B comparisons."""

# Sequential container binds layer order for Keras to process.
model = Sequential()

# Smaller first layer trades width for slightly faster convergence.
model.add(LSTM(24, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

# High-capacity middle block keeps rich temporal representations.
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.3))

# Final recurrent compression before handing off to the classifier.
model.add(LSTM(150))
model.add(Dropout(0.1))

# Softmax layer converts hidden state into a probability distribution.
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
