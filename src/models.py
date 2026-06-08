import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Bidirectional, LSTM, GRU

def create_mlp_model(input_dim):
    """
    Creates an optimized MLP model with Batch Normalization and Dropout layers.
    """
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def create_lstm_model(vocab_size, embedding_dim=64, lstm_units=64, max_len=50):
    """
    Creates a Bidirectional LSTM model with Embedding, Batch Normalization, and Dropout.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True),
        Bidirectional(LSTM(lstm_units)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def create_gru_model(vocab_size, embedding_dim=64, gru_units=64, max_len=50):
    """
    Creates a Bidirectional GRU model with Embedding, Batch Normalization, and Dropout.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True),
        Bidirectional(GRU(gru_units)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
