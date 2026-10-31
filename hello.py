import numpy as np 
import tensorflow as tf 
import tensorflow_datasets as tfds
import keras
from keras import layers

print("hello")

(mtrain, mtest), info = tfds.load(
    "emnist/balanced", split=["train", "test"],
    as_supervised=True, with_info=True

)
classes = info.features["label"].num_classes

def prep(img, label):
    img = tf.cast(img, tf.float64) / 255.0
    img = tf.transpose(tf.squeeze(img, -1))
    return img, label

mtrain = mtrain.map(prep).batch(128).prefetch(tf.data.AUTOTUNE)
mtest = mtest.map(prep).batch(128).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    keras.Input(shape=(28,28), dtype="float64"),
    layers.GRU(256, return_sequences=True),
    layers.SimpleRNN(128),
    layers.Dense(classes),
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="Adam",
    metrics=["accuracy"])
model.fit(mtrain, epochs=10, validation_data=mtest)
model.evaluate(mtest)