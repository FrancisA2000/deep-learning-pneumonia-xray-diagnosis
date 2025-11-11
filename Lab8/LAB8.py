# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

train_labels = train_labels.flatten()
test_labels = test_labels.flatten()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

learning_rates = [0.1, 0.01, 0.001]
# List of optimizer configurations to test
optimizer_configs = [
    # a. SGD (no decay, no Momentum, no Nesterov)
    {"name": "SGD-basic", "optimizer": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr)},

    # b. SGD (with decay, no Momentum, no Nesterov)
    {"name": "SGD-decay", "optimizer": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr, weight_decay=10e-6)},

    # c. SGD (no decay, with Momentum, no Nesterov)
    {"name": "SGD-momentum", "optimizer": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.6)},

    # d. SGD (no decay, with Momentum, with Nesterov)
    {"name": "SGD-nesterov", "optimizer": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.6, nesterov=True)},

    # e. AdaGrad
    {"name": "AdaGrad", "optimizer": lambda lr: tf.keras.optimizers.Adagrad(learning_rate=lr)},

    # f. RMSprop
    {"name": "RMSprop", "optimizer": lambda lr: tf.keras.optimizers.RMSprop(learning_rate=lr)},

    # g. ADAM
    {"name": "Adam", "optimizer": lambda lr: tf.keras.optimizers.Adam(learning_rate=lr)}
]

# Loop through each optimizer configuration
for config in optimizer_configs:
    print(f"\nTesting {config['name']} optimizer:")

    for lr in learning_rates:
        print(f"\nLearning rate: {lr}")

        # Create CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        # Compile with the current optimizer
        optimizer = config["optimizer"](lr)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train the model
        history = model.fit(
            train_images, train_labels,
            epochs=25,
            validation_split=0.025,
            verbose=1
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print(f'Test accuracy: {test_acc:.4f}')

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label=f'Training Accuracy (lr={lr})')
        plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy (lr={lr})')
        plt.title(f"{config['name']} - Accuracy and validation acc vs. epochs")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.3, 1])
        plt.legend()
        plt.show()
