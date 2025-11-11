# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:24:10 2022

@author: Francis
"""

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

GOOGLE_COLAB = False

if GOOGLE_COLAB:
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
else:
    npzfile = np.load('fashion_mnist_file.npz')
    train_images = npzfile['arr_0']
    train_labels = npzfile['arr_1']
    test_images = npzfile['arr_2']
    test_labels = npzfile['arr_3']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#ex2
hidden_sizes = [2**k for k in range(4, 14)]
test_accs = []
print(hidden_sizes)
for size in hidden_sizes:
    print(size)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(size, activation='sigmoid'),
        tf.keras.layers.Dense(size, activation='sigmoid'),
        tf.keras.layers.Dense(10)
    ])
    model.summary()
    GD = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=GD,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=32, epochs=1)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    test_accs.append(test_acc)
    print('\nTest accuracy:', test_acc)
plt.plot(hidden_sizes, test_accs)
plt.xscale('log', base=2)
plt.xlabel('Number of perceptrons')
plt.ylabel('Test accuracy')
plt.show()

