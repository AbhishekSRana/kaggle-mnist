# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:55:48 2018

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
def conv_2d(X,w):
    return tf.nn.conv2d(X,w,strides=[1,1,1,1],padding='SAME')
def maxpooling(X):
    return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def convolution_layer(X):
   weights={ 'w_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
         'w_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
         'w_fc1':tf.Variable(tf.random_normal([7*7*64,512])),
         'out':tf.Variable(tf.random_normal([512,10]))}
   baises={ 'b_conv1':tf.Variable(tf.random_normal([32])),
         'b_conv2':tf.Variable(tf.random_normal([64])),
         'b_fc':tf.Variable(tf.random_normal([512])),
         'out':tf.Variable(tf.random_normal([10]))}
  
   X=tf.reshape(X, shape=[-1,28,28,1])

   conv1=tf.nn.relu(conv_2d(X,weights['w_conv1'])+baises['b_conv1'])
   conv1=maxpooling(conv1)

   
   conv2=tf.nn.relu(conv_2d(conv1,weights['w_conv2'])+baises['b_conv2'])
   conv2=maxpooling(conv2)
     
   fc=tf.reshape(conv2,shape=[-1,7*7*64])
   fc=tf.nn.relu(tf.matmul(fc,weights['w_fc1'])+baises['b_fc'])
 
   output=tf.matmul(fc,weights['out'])+baises['out']

   return output


#?def train_neural_network(x):
y_ = convolution_layer(X)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) )
optimizer = tf.train.AdamOptimizer().minimize(cost)
    
 hm_epochs = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range (25):
        loss=0
        for i in range(0, len(X_train), batchSize):
            c, _ = sess.run([cost,optimizer], feed_dict = {
                X: X_train[i: i+batchSize],
                Y: Y_train[i: i+batchSize]})
            loss +=c 
        print('Epoch', epoch, 'completed out of',25,'loss:',loss)    
        e= sess.run(cost_function, feed_dict = {
                X: X_test,
                Y: Y_test})
        print('Epoch', epoch, 'completed out of',25,'loss_test:',e)    
        correct = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy_train:',accuracy.eval({X:X_train, Y:Y_train}))

        print('Accuracy_test:',accuracy.eval({X:X_test, Y:Y_test}))





            '''for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)'''