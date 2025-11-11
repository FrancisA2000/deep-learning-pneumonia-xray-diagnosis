# CNN Implementation for CIFAR-10 Image Classification
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Q1: Data Preparation
# Load CIFAR-10 dataset
(training_images, training_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

# Q2: Visualize Dataset Samples
category_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

# Create visualization grid of sample images
def display_dataset_samples(images, labels, num_samples=25):
    plt.figure(figsize=(10, 10))
    grid_size = int(np.sqrt(num_samples))
    for i in range(num_samples):
        plt.subplot(grid_size, grid_size, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(category_names[labels[i][0]])
    plt.tight_layout()
    plt.show()

display_dataset_samples(training_images, training_labels)

# Q3: Network Architecture
def create_cnn(conv_configs=None, input_kernel=(3,3), hidden_units=64):
    """
    Creates a CNN model with customizable architecture
    
    Args:
        conv_configs: List of tuples [(filters, use_pooling), ...] for additional conv layers
        input_kernel: Kernel size for first convolutional layer
        hidden_units: Number of neurons in the dense layer
    """
    if conv_configs is None:
        conv_configs = []
        
    # Initialize sequential model
    network = models.Sequential()
    
    # First convolutional layer - always 32 filters with specified kernel size
    network.add(layers.Conv2D(32, input_kernel, activation='relu', input_shape=(32, 32, 3)))
    network.add(layers.MaxPooling2D((2, 2)))
    
    # Add additional convolutional layers based on configuration
    for filters, use_pooling in conv_configs:
        network.add(layers.Conv2D(filters, (3, 3), activation='relu'))
        if use_pooling:
            network.add(layers.MaxPooling2D((2, 2)))
    
    # Add final layers
    network.add(layers.Flatten())
    network.add(layers.Dense(hidden_units, activation='relu'))
    network.add(layers.Dense(10))  # Output layer - 10 classes
    
    return network

# Q4: Model Training and Evaluation
def train_and_analyze(conv_config, kernel_size=(3,3), dense_size=64, 
                     training_epochs=20, model_name='Model'):
    """
    Train a CNN model and analyze its performance
    
    Args:
        conv_config: Configuration for convolutional layers
        kernel_size: Kernel size for the first convolutional layer
        dense_size: Size of the dense layer
        training_epochs: Number of epochs to train
        model_name: Name for the model (used in plots and output)
    
    Returns:
        Test accuracy
    """
    # Create the model
    model = create_cnn(conv_config, kernel_size, dense_size)
    
    # Q4.a: Compile model with SGD optimizer
    model.compile(
        optimizer='sgd',  # Stochastic Gradient Descent
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Q4.b: Train model with 2.5% validation split
    training_history = model.fit(
        training_images, 
        training_labels, 
        epochs=training_epochs,
        validation_split=0.025,  # Use 2.5% of training data for validation
        batch_size=64,
        verbose=2
    )
    
    # Evaluate model on test data
    _, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    
    # Q4.c: Visualize training progress
    plt.figure()
    plt.plot(training_history.history['accuracy'], label='Training')
    plt.plot(training_history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Q4.d: Report test accuracy
    print(f'{model_name} - Test accuracy: {test_accuracy:.4f}')
    
    return test_accuracy

# Q4: Original CNN Architecture
print("\n===== Q4: Original CNN Model =====")
# Original architecture: Conv(32)-Pool → Conv(64)-Pool → Conv(64) → Flatten → Dense(64) → Output
original_architecture = [(64, True), (64, False)]
train_and_analyze(
    conv_config=original_architecture,
    kernel_size=(3,3),
    dense_size=64,
    training_epochs=20,
    model_name='Original CNN'
)

# Q5: Model Architecture Variations
print("\n===== Q5: CNN Architecture Variations =====")

# Q5.a: Remove second convolutional layer
print("\n--- Q5.a: Without Second Convolutional Layer ---")
# Architecture: Conv(32)-Pool → Conv(64) → Flatten → Dense(64) → Output
q5a_architecture = [(64, False)]
train_and_analyze(
    conv_config=q5a_architecture,
    kernel_size=(3,3),
    dense_size=64,
    model_name='Q5.a - No Second Conv Layer'
)

# Q5.b: Remove third convolutional layer
print("\n--- Q5.b: Without Third Convolutional Layer ---")
# Architecture: Conv(32)-Pool → Conv(64)-Pool → Flatten → Dense(64) → Output
q5b_architecture = [(64, True)]
train_and_analyze(
    conv_config=q5b_architecture,
    kernel_size=(3,3),
    dense_size=64,
    model_name='Q5.b - No Third Conv Layer'
)

# Q5.c: Remove second and third convolutional layers
print("\n--- Q5.c: Without Second and Third Convolutional Layers ---")
# Architecture: Conv(32)-Pool → Flatten → Dense(64) → Output
q5c_architecture = []
train_and_analyze(
    conv_config=q5c_architecture,
    kernel_size=(3,3),
    dense_size=64,
    model_name='Q5.c - Only First Conv Layer'
)

# Q5.d: Increase kernel size of first convolutional layer
print("\n--- Q5.d: Larger Kernel Size (5x5) ---")
# Architecture: Conv(32, 5x5)-Pool → Conv(64)-Pool → Conv(64) → Flatten → Dense(64) → Output
train_and_analyze(
    conv_config=original_architecture,
    kernel_size=(5,5),
    dense_size=64,
    model_name='Q5.d - 5x5 Kernel'
)

# Q5.e: Decrease kernel size of first convolutional layer
print("\n--- Q5.e: Smaller Kernel Size (2x2) ---")
# Architecture: Conv(32, 2x2)-Pool → Conv(64)-Pool → Conv(64) → Flatten → Dense(64) → Output
train_and_analyze(
    conv_config=original_architecture,
    kernel_size=(2,2),
    dense_size=64,
    model_name='Q5.e - 2x2 Kernel'
)

# Q5.f: Modify dense layer size
print("\n--- Q5.f: Modified Dense Layer Size ---")
# Increase dense layer size to 128
print("Increasing Dense Layer to 128 neurons:")
train_and_analyze(
    conv_config=original_architecture,
    kernel_size=(3,3),
    dense_size=128,
    model_name='Q5.f - 128 Neurons'
)

# Decrease dense layer size to 32
print("Decreasing Dense Layer to 32 neurons:")
train_and_analyze(
    conv_config=original_architecture,
    kernel_size=(3,3),
    dense_size=32,
    model_name='Q5.f - 32 Neurons'
)