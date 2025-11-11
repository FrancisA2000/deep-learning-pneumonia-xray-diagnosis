# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:43:12 2019

@author: User


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
k=0
softmax_accuracies=np.zeros((10,1))
softmax_cv_accuracies=np.zeros((10,1))
if 'X' not in globals():
    MNIST=np.load('mnist.npz',allow_pickle=True)
    X=MNIST['data']
    y=MNIST['label']
    
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))


train_samples = 1000
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=1000)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Turn up tolerance for faster convergence
softmax = LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', tol=0.01)
softmax.fit(X_train, y_train)
score = softmax.score(X_test, y_test)
cv=cross_val_score(softmax, X_train, y_train, cv=5)
softmax_accuracies[k]=score
softmax_cv_accuracies[k]=np.mean(cv)
k=k+1
print("Accuracy on Test data: %", (score*100))
test_pred=softmax.predict(X_test)


# Display 5 correct predcitions
predicitons = softmax.predict(X_test)
prediction_errors = np.equal(predicitons, y_test)
errors=0
for i in range(0,1000):
    if prediction_errors[i]==True:
        print('Multinomial Regression Example:')
        plt.imshow(X_test[i,:].reshape(28,28))
        plt.show()
        print('Correct Label:',y_test[i])
        print('Predcited Label:',predicitons[i])
        softmax_out=softmax.predict_proba(X_test[i,:].reshape(1,-1)).T
        plt.plot(np.arange(10),softmax_out, '*')
        plt.xticks(np.arange(10), ('0', '1', '2', '3,', '4', '5', '6', '7', '8', '9'))
        plt.title('Softmax Output')
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.show()
        errors+=1
        if errors>5:
            break
        

