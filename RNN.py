from tensorflow.keras.models import Sequential  # Import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense  # Also import Dense layer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load and preprocess IMDB data
max_words = 10000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Build the RNN model
model = Sequential([
    Embedding(max_words, 32),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
