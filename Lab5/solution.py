# -*- coding: utf-8 -*-

"""
@author: Maroon
"""

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

print(__doc__)

# Turn down for faster convergence
t0 = time.time()
k = 0
softmax_accuracies = np.zeros((10, 1))
softmax_cv_accuracies = np.zeros((10, 1))
if 'X' not in globals():
    MNIST = np.load('mnist.npz', allow_pickle=True)
    X = MNIST['data']
    y = MNIST['label']

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))
# Ex 1.a Classification accuracy
train_samples = np.arange(100, 1100, 100)
score = np.zeros(10)
for indx in range(len(train_samples)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples[indx], test_size=1000)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Turn up tolerance for faster convergence
    softmax = LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', tol=0.01)
    softmax.fit(X_train, y_train)
    score[indx] = softmax.score(X_test, y_test)
print("Accuracy on number of Test data: %\n", train_samples, "\n", (score * 100))
plt.figure(1)
plt.plot(train_samples, score)
plt.grid()
plt.title("Classification accuracy")
plt.xlabel("Number of train samples")
plt.ylabel("Accuracy")
# Ex 1.b K-fold cross-validation
cv = cross_val_score(softmax, X_train, y_train, cv=5)
softmax_accuracies = score
softmax_cv_accuracies = np.mean(cv)
plt.figure(2)
plt.plot(np.arange(1, 6), cv, 'r')
plt.grid()
plt.title("Cross-validation accuracy")
plt.xlabel("Number of test")
plt.ylabel("Accuracy")
plt.show()
print("Accuracy of Cross-validation Test data: %\n", (cv * 100), "\nAvarage: % ", (softmax_cv_accuracies * 100))
# Ex 1.c
predicitons = softmax.predict(X_test)
prediction_errors = np.equal(predicitons, y_test)
errors = 0
for i in range(0, 1000):
    if not prediction_errors[i]:
        print('Multinomial Regression Example:')
        plt.imshow(X_test[i, :].reshape(28, 28))
        plt.show()
        print('Correct Label:', y_test[i])
        print('Predcited Label:', predicitons[i])
        softmax_out = softmax.predict_proba(X_test[i, :].reshape(1, -1)).T
        plt.plot(np.arange(10), softmax_out, '*')
        plt.xticks(np.arange(10), ('0', '1', '2', '3,', '4', '5', '6', '7', '8', '9'))
        plt.title('Softmax Output')
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.show()
        plt.figure()
        errors += 1
        if errors > 5:
            break

# Ex 2.a
KX_train, KX_test, Ky_train, Ky_test = train_test_split(X, y, train_size=500, test_size=1000)
Knn_score = np.zeros(3)
kValue = np.random.randint(3, 11, 3)
for indx in range(0, 3):
    # we create an instance of Neighbours Classifier and fit the data.
    clf = KNeighborsClassifier(n_neighbors=kValue[indx], weights='uniform')
    clf.fit(KX_train, Ky_train)
    Knn_score[indx] = clf.score(KX_train, Ky_train)
max_k = kValue[np.argmax(Knn_score)]
print("The KNN results: \n", kValue, "\n", (Knn_score * 100), "\n Maximum for k = ", max_k)
# Ex 2.b
Ktrain_samples = np.arange(100, 1100, 100)
Kscore = np.zeros([3, 10])
for indx1 in range(0, 3):
    for indx2 in range(0, 10):
        KX_train, KX_test, Ky_train, Ky_test = train_test_split(X, y, train_size=Ktrain_samples[indx2], test_size=1000)
        clf = KNeighborsClassifier(n_neighbors=kValue[indx1], weights='uniform')
        clf.fit(KX_train, Ky_train)
        Kscore[indx1, indx2] = clf.score(KX_train, Ky_train)
    plt.figure(4)
    plt.plot(train_samples, Kscore[indx1, :], label=str(kValue[indx1]))
plt.figure(4)
plt.plot(train_samples, score, label="MSR")
plt.grid()
plt.title("Classification accuracy")
plt.xlabel("Number of train samples")
plt.ylabel("Accuracy")
plt.legend()
plt.show()