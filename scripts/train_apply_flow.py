"""End-to-end training + inference flow for the Lovecraft LSTM text generator."""

# Core scientific stack required both here and inside the model definition
import logging
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils
import requests
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("cthulhubot")


# ``execfile`` is python2-only; add a tiny shim so we can run this script under
# python3 without touching the legacy model definition files.
try:  # pragma: no cover - compatibility branch
    execfile
except NameError:  # pragma: no cover - python3 fallback
    def execfile(path, globals_=None, locals_=None):
        """Execute the given file in the provided global/local scope."""
        with open(path, "r", encoding="utf-8") as handle:
            code = compile(handle.read(), path, "exec")
        exec(code, globals_ if globals_ is not None else globals(), locals_ if locals_ is not None else locals())


# ---------------------------------------------------------------------------
# Data loading and preprocessing pipeline
# ---------------------------------------------------------------------------
# Fetch the Lovecraft corpus straight from GitHub (small enough for each run).
raw_corpus_url = "https://raw.githubusercontent.com/urschrei/lovecraft/master/lovecraft.txt"
logger.info("Downloading Lovecraft corpus from %s", raw_corpus_url)
try:
    raw_corpus = requests.get(raw_corpus_url, timeout=30)
    raw_corpus.raise_for_status()
except requests.RequestException as exc:
    logger.error("Failed to download corpus", exc_info=exc)
    raise SystemExit(1)
logger.info("Corpus downloaded: %d bytes", len(raw_corpus.content))

# Decode the payload with the encoding used inside the source text.
raw_text = raw_corpus.content
raw_text = raw_text.decode("latin-1")

# Basic cleanup: lowercase everything and collapse double line breaks.
raw_text = raw_text.lower()
raw_text = raw_text.replace("\n\n", "")

# Build forward/backward lookup tables between characters and integer ids.
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Summaries help when experimenting with sequence lengths and vocab size.
n_chars = len(raw_text)
n_vocab = len(chars)

# Prepare sliding-window sequences of characters and the next char to predict.
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i : i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

# Reshape inputs to ``(samples, timesteps, features)`` as required by Keras.
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# Normalise inputs and convert the categorical targets to one-hot vectors.
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)


# ---------------------------------------------------------------------------
# Model definition and training setup
# ---------------------------------------------------------------------------
# Execute the chosen architecture to populate the global ``model`` variable.
try:
    execfile("/home/user01/CthulhuBot_2/scripts/models/second_lstm.py")
except Exception as exc:  # pragma: no cover - defensive guard
    logger.error("Model definition crashed", exc_info=exc)
    raise SystemExit(1)

# Persist the best weights during the run so we can resume/improve later.
filepath = "weigths/weights-improvement-v4-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks_list = [checkpoint]

# Train the network end-to-end on the prepared character windows.
logger.info("Starting training: patterns=%d, seq_length=%d", n_patterns, seq_length)
try:
    model.fit(X, y, epochs=80, batch_size=64, callbacks=callbacks_list)
except Exception as exc:
    logger.error("Training failed", exc_info=exc)
    raise SystemExit(1)
logger.info("Training finished successfully")

# Reload a pre-trained checkpoint for inference to save time at demo time.
filename = "weights-improvement-v4-34-1.6287.hdf5"
try:
    model.load_weights(filename)
except OSError as exc:
    logger.warning("Could not load weights '%s': %s", filename, exc)
else:
    logger.info("Loaded pre-trained weights from %s", filename)

model.compile(loss="categorical_crossentropy", optimizer="adam")


# ---------------------------------------------------------------------------
# Text generation helpers and inference
# ---------------------------------------------------------------------------
# Provide an explicit seed instead of the historical random selection above.
seed_text = "the creature in the darkness"
pattern = [char_to_int[char] for char in seed_text]


def sample(preds, temperature):
    """Sample an index from the model's probability distribution."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(y.shape[1], preds[0], 1)
    return np.argmax(probas)


# Generate a few hundred characters while annealing through the temperature.
temperature = 1.7
new_text = []
for _ in range(400):
    # Prepare the current rolling window of characters for the LSTM.
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)

    # Ask the model for the distribution over the next character.
    try:
        prediction = model.predict(x, verbose=0)
    except Exception as exc:
        logger.error("Prediction failed", exc_info=exc)
        break

    # Sample a new character id and convert it back to a readable symbol.
    index = sample(prediction, temperature)
    result = int_to_char[index]

    # Emit the generated character immediately for interactive demos.
    sys.stdout.write(result)

    # Roll the window forward by one character and track the output.
    pattern.append(index)
    pattern = pattern[1 : len(pattern)]
    new_text.append(result)


# Expose the generated artefact in case the caller wants to use it programmatically.
generated_text = "".join(new_text)
if not generated_text:
    logger.warning("No text generated. Check earlier warnings for context.")
