

# importing libraries

import tensorflow as tf
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder

#placeholders
X=tf.placeholder(tf.float32,shape=[None,784])

Y=tf.placeholder(tf.float32,shape=[None,10])

#loading dataset
data1=pd.read_csv('train.csv')
data2=pd.read_csv('test.csv')



X_train=data1.iloc[:,1:].values
Y_train=data1.iloc[:,0].values

X_test=data2.iloc[:,:].values


X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train=np.reshape(Y_train,(-1,1))
onehotencoder= OneHotEncoder(categorical_features=[0])
Y_train= onehotencoder.fit_transform(Y_train).toarray()
 
#X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.2,random_state=1)

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
    saver = tf.train.Saver()
    for epoch in range (25):
        loss=0
        for i in range(0, len(X_train), batchSize):
            c, _ = sess.run([cost_function,training_steps], feed_dict = {
                X: X_train[i*batchSize: (i+1)*batchSize],
                Y: Y_train[i*batchSize: (i+1)*batchSize]})
            loss +=c 
        print('Epoch', epoch, 'completed out of',25,'loss:',loss)    
        
        correct = tf.equal(tf.argmax(yy, 1), tf.argmax(Y_train, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy_train:',accuracy.eval({X:X_train, Y:Y_train}))
    
    
    
    classification = sess.run(tf.argmax(yy, 1), feed_dict={X: X_test})
    