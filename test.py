import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))


print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0"):
    a = tf.random.normal((1000, 1000))
    b = tf.matmul(a, a)

print("Matrix operation device:", b.device)

with tf.device("/GPU:0"):
    gru = tf.keras.layers.GRU(256)
    x = tf.random.normal((128, 28, 28))
    y = gru(x)

print(y.shape)
print(y.device)