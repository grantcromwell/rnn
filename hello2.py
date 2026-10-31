import numpy as np 
import tensorflow as tf 
import tensorflow_datasets as tfds
import keras
from keras import layers
import os

print("hello")

(mtrain, mval), info = tfds.load(
    "emnist/balanced", split=["train[:90%]", "train[90%:]"],
    as_supervised=True, with_info=True

)

mtest = tfds.load(
    "emnist/balanced", 
    split="test",
    as_supervised=True

)
classes = info.features["label"].num_classes

def prep(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.transpose(tf.squeeze(img, -1))
    return img, label




mtrain =(
    mtrain.map(prep)
    .shuffle(10000)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)

)

mval =(
    mval.map(prep)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)

)

mtest =(
    mtest.map(prep)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)

)




model = keras.Sequential([
    keras.Input(shape=(28,28), dtype="float32"),
    layers.GRU(256, return_sequences=True),
    layers.SimpleRNN(128),
    layers.Dense(classes),
])

model.compile(
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
optimizer="Adam",
metrics=["accuracy"])


checkpoint = keras.callbacks.ModelCheckpoint(
    filepath = "emnist_model.keras",
    save_freq = "epoch",
    save_best_only = True,
    monitor = "val_loss",
    mode = "min"
)

earlyStop = keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 5,
    restore_best_weights = True
)

backup = keras.callbacks.BackupAndRestore(
    backup_dir = "./trainingCheckpoint",
    save_freq = "epoch"
)


model.fit(
    mtrain,
    epochs=30,
    validation_data=mval,
    callbacks = [checkpoint, earlyStop, backup]
)
testLoss, testAcc = model.evaluate(mtest)
print(f"Accuracy: {testAcc:.2%}")