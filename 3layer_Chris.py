import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.utils import class_weight
from keras import backend as K
import datetime
import matplotlib.pyplot as plt

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

def propensityLoss(p):
    def pLoss(y_true, y_pred):
        loss = tf.reduce_sum((1.0 / p) * tf.math.abs(2.0 * tf.cast(tf.equal(y_true, y_pred), tf.float32) - 1.0) * tf.math.pow(y_true-y_pred, 2))
        ones = tf.ones_like(y_pred)
        zeros = tf.zeros_like(y_pred)
        y_pred = tf.where(tf.greater_equal(y_pred, 0.5), ones, zeros)
        loss = (1.0 - tf.cast(tf.reduce_all(tf.equal(y_true, y_pred)), tf.float32)) * loss
        return loss
    return pLoss

#creating figures folder
current_working_dir = os.getcwd()
folderDir = os.path.join(current_working_dir,'figures')

if not os.path.isdir(folderDir):
    print('creating the figures folder')
    os.makedirs(folderDir)
    
#loading dataset
#emotions = pd.read_csv("dataset/emotions.csv")
emotions = pd.read_csv("dataset/bookmarks.csv")
print("Shape of dataset: {}".format(emotions.shape))


#Model parameters
learning_rate = 0.0001 #emotions:0.00005, bookmarks: 0.0001 
training_epochs = 100 #emotions:5000, bookmarks: 50

num_inputs = 2150 #emotions: 72, bookmarks: 2150
num_outputs = 208 #emotions:6, bookmarks: 208
batch_size = 32
layer1_nodes = 50
layer2_nodes = 50
layer3_nodes = 100


#train test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(emotions, random_state=42, test_size=0.20, shuffle=True)

x_train = train.iloc[:,:num_inputs].to_numpy()
y_train = train.iloc[:,-num_outputs:].to_numpy()
x_test = test.iloc[:,:num_inputs].to_numpy()
y_test = test.iloc[:,-num_outputs:].to_numpy()


#converting to one.hot vector form
y_train = np.float32(y_train)
y_test = np.float32(y_test)

#Calculate propensity
N = len(x_train)
L = num_outputs
propensity = []
for i in range(L):
    Nl = np.count_nonzero(y_train[:,i])
    propensity.append(1.0 / (1.0 + tf.math.log(N-1.0)*(1.183)*tf.math.exp(-0.5*tf.math.log(Nl + 0.4))))
propensity = np.array(propensity)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=layer1_nodes, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(units=layer2_nodes, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(units=layer3_nodes, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(units=num_outputs, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss=propensityLoss(p=propensity),
      metrics=[precision, recall, f1])

# Calculate the weights for each class so that we can balance the data
weights = class_weight.compute_sample_weight('balanced', y_train)

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.isdir(log_dir):
    print('creating the log folder:', log_dir)
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

history = model.fit(x=x_train, 
          y=y_train, 
          batch_size=batch_size,
          epochs=training_epochs,  
          class_weight=weights,
          callbacks=[tensorboard_callback])


y_pred = model.predict(x_test)

y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

#Calculate score
exact_match_score = np.all(y_pred == y_test, axis=1).mean()
hamming_score = (y_pred == y_test).mean()
print('Exact match score (Whole row must match):', exact_match_score)
print('Hamming score (Individual label predictions):', hamming_score)


#plotting results
plt.plot(history.history['precision'])       
plt.plot(history.history['recall'])  
plt.plot(history.history['f1'])  
plt.title('Model Metrics')
plt.ylabel('Score')
plt.xlabel('Epoch')
plt.legend(['Precision', 'Recall', 'F1'], loc='upper left')
plt.savefig(os.path.join('figures', 'Metrics.png'))

plt.clf()
plt.plot(history.history['loss']) 
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')
plt.savefig(os.path.join('figures', 'Loss.png'))