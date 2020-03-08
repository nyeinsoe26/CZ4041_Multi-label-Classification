import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras as ks
import math
import pylab

#creating figures folder
current_working_dir = os.getcwd()
folderDir = os.path.join(current_working_dir,'figures')

if not os.path.isdir(folderDir):
    print('creating the figures folder')
    os.makedirs(folderDir)
    
#loading dataset
emotions = pd.read_csv("emotions.csv")
print("Shape of dataset: {}".format(emotions.shape))
emotions.head(5)


#train test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(emotions, random_state=42, test_size=0.20, shuffle=True)
x_train = train.iloc[:,0:72]
y_train = train.iloc[:,-6:]
x_test = test.iloc[:,0:72]
y_test = test.iloc[:,-6:]

#extracting labels
output_labels = y_train.columns

#converting to one.hot vector form
y_train = y_train.values
y_test = y_test.values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
x_scaler = MinMaxScaler(feature_range=(0,1))
x_scaled_train =  x_scaler.fit_transform(x_train)
x_scaled_test = x_scaler.fit_transform(x_test)

#Model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5
threshold = 0.4
beta = pow(10.0,-6)
num_inputs = 72
num_outputs = 6
batch_size = 32
layer1_nodes = 50
layer2_nodes = 100
layer3_nodes = 50
keep_prob = 0.7

N = len(x_scaled_train)
idx = np.arange(N)

#input layer
with tf.variable_scope('input',reuse = tf.AUTO_REUSE):
    X = tf.placeholder(tf.float32, shape=(None,  num_inputs))

#layer 1
with tf.variable_scope('layer_1',reuse = tf.AUTO_REUSE):
    weights = tf.get_variable(name = "weights1", shape = [num_inputs,layer1_nodes], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1",shape = [layer1_nodes],initializer = tf.zeros_initializer())
    layer1_output = tf.nn.relu(tf.matmul(X,weights) +biases)
    layer1_output_drop = tf.nn.dropout(layer1_output, keep_prob)


#layer 2
with tf.variable_scope('layer_2',reuse = tf.AUTO_REUSE):
    weights = tf.get_variable(name = "weights2", shape = [layer1_nodes,layer2_nodes], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2",shape = [layer2_nodes],initializer = tf.zeros_initializer())
    layer2_output = tf.nn.relu(tf.matmul(layer1_output_drop,weights) +biases)
    layer2_output_drop = tf.nn.dropout(layer2_output, keep_prob)

#layer 3
with tf.variable_scope('layer_3',reuse = tf.AUTO_REUSE):
    weights = tf.get_variable(name = "weights3", shape = [layer2_nodes,layer3_nodes], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3",shape = [layer3_nodes],initializer = tf.zeros_initializer())
    layer3_output = tf.nn.relu(tf.matmul(layer2_output_drop,weights) +biases)
    layer3_output_drop = tf.nn.dropout(layer3_output, 0.5)

#output
with tf.variable_scope('output',reuse = tf.AUTO_REUSE):
    weights = tf.get_variable(name = "weights4", shape = [layer3_nodes,num_outputs], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4",shape = [num_outputs],initializer = tf.zeros_initializer())
    logit = tf.sigmoid(tf.matmul(layer3_output_drop,weights) +biases)

#cost function
with tf.variable_scope('cost',reuse = tf.AUTO_REUSE):
    Y = tf.placeholder(tf.float32, shape=(None,6))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logit) + beta
    cost = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))

#accuracy function
with tf.variable_scope('accuracy',reuse = tf.AUTO_REUSE):
    prediction = tf.cast(logit > threshold, tf.float32)
    correct_prediction = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.variable_scope('train',reuse = tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as session:
    #run global initializer to initialise all variables within the graph to default state
    session.run(tf.global_variables_initializer())
    train_cost_list = []
    test_cost_list = []
    train_acc_list = []
    test_acc_list = []
    
    for epoch in range(training_epochs):
        #shuffle data 
        np.random.shuffle(idx)
        x_scaled_train, y_train = x_scaled_train[idx],y_train[idx]
        for start,end in zip(range(0,N,batch_size),range(batch_size,N,batch_size)):
            optimizer.run(feed_dict = {X:x_scaled_train[start:end], Y:y_train[start:end]})
            
        #evaluating performance
        train_cost = cost.eval(feed_dict= {X: x_scaled_train, Y: y_train})
        test_cost = cost.eval(feed_dict= {X: x_scaled_test, Y: y_test})
        
        train_acc = accuracy.eval(feed_dict= {X: x_scaled_train, Y: y_train})
        test_acc = accuracy.eval(feed_dict= {X: x_scaled_test, Y: y_test})
        
        train_cost_list.append(train_cost)
        test_cost_list.append(test_cost)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        if epoch % 5==0:
            print("Epoch: %d, Train Cost: %g, Test Cost: %g, Train Acc: %g, Test Acc: %g"%(epoch, train_cost_list[epoch],test_cost_list[epoch],train_acc_list[epoch],test_acc_list[epoch]))
    #plotting results
    pylab.figure(1)
    pylab.plot(np.arange(training_epochs),train_cost_list,label = 'Train Cost Vs epochs')
    pylab.plot(np.arange(training_epochs),test_cost_list,label = 'Test Cost Vs epochs')
    pylab.xlabel('epochs')
    pylab.ylabel('cost')
    pylab.title('Cost Vs epochs')
    pylab.legend()
    pylab.savefig('./figures/Cost_vs_epochs.png')        
    
    pylab.figure(2)
    pylab.plot(np.arange(training_epochs),train_acc_list,label = 'Train Accuracy Vs epochs')
    pylab.plot(np.arange(training_epochs),test_acc_list,label = 'Test Accuracy Vs epochs')
    pylab.xlabel('epochs')
    pylab.ylabel('accuracy')
    pylab.title('accuracy Vs epochs')
    pylab.legend()
    pylab.savefig('./figures/Accuracy_vs_epochs.png')