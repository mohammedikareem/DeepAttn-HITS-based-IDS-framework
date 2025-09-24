
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, MultiHeadAttention,
                                     Flatten, LayerNormalization, Reshape, Add)
from tensorflow.keras.models import Model

def build_attn_autoencoder(input_dim, latent_dim=8, heads=2, dp=0.15):
    inp = Input(shape=(input_dim,))
    seq = Reshape((input_dim, 1))(inp)
    attn = MultiHeadAttention(num_heads=heads, key_dim=8)(seq, seq)
    attn = Dense(1)(attn)
    attn = Flatten()(attn)
    attn = LayerNormalization()(attn)
    x = Add()([inp, attn])
    x = Dense(16, activation='relu')(x)
    x = Dropout(dp)(x)
    bottleneck = Dense(latent_dim, activation='relu', name='latent')(x)
    x = Dense(16, activation='relu')(bottleneck)
    x = Dropout(dp)(x)
    out = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inp, out)
    encoder = Model(inp, bottleneck)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder
