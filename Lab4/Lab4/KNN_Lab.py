# -*- coding: utf-8 -*-

"""
K-Nearest Neighbors (K-NN) Experiment:
- Evaluate the accuracy of K-NN for different values of K (1 to 20).
- Compare two weighting strategies: 'uniform' and 'distance'.
- Visualize the accuracy trends for both strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset and extract the first two features
data = load_iris()
features = data.data[:, :2]  # Only the first two features
labels = data.target         # Target labels

# Define the range of K values (number of neighbors)
k_range = range(1, 21)

# Initialize dictionary to store accuracy scores for each weighting method
accuracies = {
    'uniform': [],
    'distance': []
}

# Create a mapping for plot styles
style_map = {
    'uniform': {'line': 'b*-', 'label': 'Uniform Weights'},
    'distance': {'line': 'm--', 'label': 'Distance Weights'}
}

# Iterate over each weighting method
for weights in ['uniform', 'distance']:
    # Iterate over each K value
    for k in k_range:
        # Create and fit K-NN model with the current weighting method
        model = KNeighborsClassifier(n_neighbors=k, weights=weights)
        model.fit(features, labels)
        
        # Compute and store the accuracy score
        accuracy = model.score(features, labels)
        accuracies[weights].append(accuracy)

# Plot the accuracy results
plt.figure(figsize=(10, 6))

# Plot each weighting method's accuracy
for weights in accuracies:
    plt.plot(k_range, accuracies[weights], style_map[weights]['line'], 
             label=style_map[weights]['label'])

plt.title('K-NN Accuracy vs Number of Neighbors (K)', fontsize=14)
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(k_range)
plt.legend(loc='best')
plt.grid(alpha=0.5)
plt.show()
