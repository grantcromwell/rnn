import numpy as np 
import tensorflow as tf 
import keras
from keras import layers

print("hello")

model = keras.Sequential([
    keras.Input(shape=(None,), dtype="int64"),
    layers.Embedding(input_dim=1000, output_dim=64),
    layers.LSTM(128),
    layers.Dense(10),
])
model.summary()