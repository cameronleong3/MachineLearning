#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:30:47 2021

@author: cameronleong
"""

#Step 2: Linear Regression using 100 data points


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import pandas as pd
df_train = pd.read_csv("crime-train.csv")
df_train_np = pd.DataFrame(df_train).to_numpy()
df_test = pd.read_csv("crime-test.csv")
df_test_np = pd.DataFrame(df_test).to_numpy()


print("\n\n\nSTARTING")

#Number of training and test samples
Ntrain = len(df_train_np)
Ntest = len(df_test_np)


N = 100
Ntest = len(df_test_np)
M =len(df_train_np[0])
new_train =[ [ 0 for i in range(len(df_train_np[0]) ) ] for j in range(N) ]

for i in range(100):
    for j in range(M):
        new_train[i][j] = df_train_np[i][j]

    

#column vector for t_train and t_test made from 1st column of data matrix
t_train =[ [ 0 for i in range(1) ] for j in range(N) ]
t_test =[ [ 0 for i in range(1) ] for j in range(len(df_test_np)) ]

for i in range(N):
    t_train[i][0] = new_train[i][0]     #save target value and replace with 1s to add bias
    new_train[i][0] = 1
    
for i in range(len(df_test_np)):
    t_test[i][0] = df_test_np[i][0]
    df_test_np[i][0] = 1

#Set lambda value
A = 100
I = np.identity(96)

#define the tensors
X = tf.placeholder(tf.float64, shape = (None, len(df_train_np[0])), name = 'X') #each row is a sample
t = tf.placeholder(tf.float64, shape = (None, 1), name = 't') #target values are in the t vector
n = tf.placeholder(tf.float64, name = 'n')                      #number of samples
XT = tf.transpose(X)
w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),t)   #w = inv(X'*X)*X'*t

#predicted values: a column vector y = [y1, y2, y3,...,yn]' where yn = xn'*w
y = tf.matmul(X,w)

#Mean squared error
MSE = tf.div(tf.matmul(tf.transpose(y-t),y-t),n)

with tf.Session() as sess:
    MSE_train, w_star, y_train = \
    sess.run([MSE,w,y], feed_dict={X:new_train, t:t_train, n: N})
    
    MSE_test, y_test = \
        sess.run([MSE,y],feed_dict={X:df_test_np, t:t_test, n:len(df_test_np), w: w_star})


print("RESULTS:")
print("MSE_train:")
print(MSE_train)
print("MSE_test")
print(MSE_test)

    
               

        
