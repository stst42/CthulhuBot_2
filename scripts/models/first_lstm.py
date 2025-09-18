"""Older, shallower LSTM baseline kept for experimentation."""

# Construct the model with the globals provided by the training script.
model = Sequential()

# Single stacked setup with larger hidden width to capture structure.
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

# Collapse sequence to a single state before the classifier head.
model.add(LSTM(256))
model.add(Dropout(0.1))

# Predict the next character as a categorical distribution.
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
