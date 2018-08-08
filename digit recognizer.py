# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:34:17 2018

@author: Abhishek
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder

X=tf.placeholder(tf.float32,shape=[None,784])

Y=tf.placeholder(tf.float32,shape=[None,10])

data=pd.read_csv('train.csv')

x=data.iloc[:,1:].values
y=data.iloc[:,0].values

x = x / 255.0
y=np.reshape(y,(-1,1))
onehotencoder= OneHotEncoder(categorical_features=[0])
y= onehotencoder.fit_transform(y).toarray()
 
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.2,random_state=1)

def multi_layer_perceptron (x,weights,baises):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['w1']),baises['b1']))
    
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['w2']),baises['b2']))
    
   # layer_3=tf.nn.sigmoid(tf.add(tf.matmul(layer_2,weights['w3']),baises['b3']))
    
    #layer_4=tf.nn.relu(tf.add(tf.matmul(layer_3,weights['w4']),baises['b4']))
    
    output= tf.matmul(layer_2,weights['out'])+baises['out']
    
    return output

weights = {
        'w1':tf.Variable(tf.truncated_normal([784,512])),
        'w2':tf.Variable(tf.truncated_normal([512,256])),
       # 'w3':tf.Variable(tf.truncated_normal([128,128])),
      # 'w4':tf.Variable(tf.truncated_normal([128,128])),
        'out':tf.Variable(tf.truncated_normal([256,10]))
        } 


baises = {
        'b1':tf.Variable(tf.truncated_normal([512])),
        'b2':tf.Variable(tf.truncated_normal([256])),
      #  'b3':tf.Variable(tf.truncated_normal([128])),
      # 'b4':tf.Variable(tf.truncated_normal([128,128])),
        'out':tf.Variable(tf.truncated_normal([10]))
        } 



yy=multi_layer_perceptron(X,weights,baises)

cost_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yy,labels=Y))

training_steps=tf.train.GradientDescentOptimizer(.1).minimize(cost_function)




batchSize=128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range (25):
        loss=0
        for i in range(0, len(X_train), batchSize):
            c, _ = sess.run([cost_function,training_steps], feed_dict = {
                X: X_train[i: i+batchSize],
                Y: Y_train[i: i+batchSize]})
            loss +=c 
        print('Epoch', epoch, 'completed out of',25,'loss:',loss)    
        e= sess.run(cost_function, feed_dict = {
                X: X_test,
                Y: Y_test})
        print('Epoch', epoch, 'completed out of',25,'loss_test:',e)    
        correct = tf.equal(tf.argmax(yy, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy_train:',accuracy.eval({X:X_train, Y:Y_train}))

        print('Accuracy_test:',accuracy.eval({X:X_test, Y:Y_test}))
        