# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:23:22 2019

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

my_data = genfromtxt('advertising.csv', delimiter=',')
X = my_data[1:201,1:4]
Y = my_data[1:201,4:5]
plt.plot(X[:,0],Y,'b.')
plt.xlabel('X = TV Advertising Budget [K$]')
plt.ylabel('Y = Sales [M$]')
 
plt.show()