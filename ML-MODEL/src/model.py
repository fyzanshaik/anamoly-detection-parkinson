import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_autoencoder(input_dim, hidden_dims, activation="relu"):
    encoder_layers = []
    for i, h in enumerate(hidden_dims):
        if i == 0:
            encoder_layers.append(keras.layers.Dense(h, activation=activation, input_shape=(input_dim,), name=f"encoder_{i}"))
        else:
            encoder_layers.append(keras.layers.Dense(h, activation=activation, name=f"encoder_{i}"))

    decoder_layers = []
    for i, h in enumerate(reversed(hidden_dims[:-1])):
        decoder_layers.append(keras.layers.Dense(h, activation=activation, name=f"decoder_{i}"))
    decoder_layers.append(keras.layers.Dense(input_dim, activation="linear", name="decoder_output"))

    encoder = keras.Sequential(encoder_layers, name="encoder")
    decoder = keras.Sequential(decoder_layers, name="decoder")

    inputs = keras.Input(shape=(input_dim,), name="input")
    encoded = encoder(inputs)
    decoded = decoder(encoded)

    autoencoder = keras.Model(inputs, decoded, name="autoencoder")

    return autoencoder, encoder, decoder


MODEL_CONFIGS = {
    "statistical": {
        "feature_group": "statistical",
        "hidden_dims": [6, 3],
        "learning_rate": 0.001,
        "epochs": 150,
        "batch_size": 64,
    },
    "mfcc": {
        "feature_group": "mfcc",
        "hidden_dims": [24, 12],
        "learning_rate": 0.001,
        "epochs": 150,
        "batch_size": 64,
    },
    "frequency": {
        "feature_group": "frequency",
        "hidden_dims": [6, 3],
        "learning_rate": 0.001,
        "epochs": 150,
        "batch_size": 64,
    },
    "fused": {
        "feature_group": "all",
        "hidden_dims": [32, 16, 8],
        "learning_rate": 0.0005,
        "epochs": 200,
        "batch_size": 64,
    },
    "stat_freq": {
        "feature_group": "stat_freq",
        "hidden_dims": [12, 6],
        "learning_rate": 0.001,
        "epochs": 150,
        "batch_size": 64,
    },
    "esp32_deploy": {
        "feature_group": "stat_freq",
        "hidden_dims": [8],
        "learning_rate": 0.001,
        "epochs": 150,
        "batch_size": 64,
    },
}
