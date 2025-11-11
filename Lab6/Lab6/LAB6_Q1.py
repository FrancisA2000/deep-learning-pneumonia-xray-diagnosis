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

GOOGLE_COLAB=False

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
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])
model.summary()
#1
test_accs = []

learning_rates = [0.1, 0.01, 0.001]
for rate in learning_rates:
    print(f"Training with learning rate: {rate}")
    GD=tf.keras.optimizers.SGD(learning_rate=rate)
    model.compile(optimizer=GD,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=32, epochs=15)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    test_accs.append(test_acc)

print(test_accs)