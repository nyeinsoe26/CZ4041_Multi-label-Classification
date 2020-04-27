# -*- coding: utf-8 -*-
"""3_layerNN_multilabel_individual_thres(Tuning).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SHS5f4bRYrh7qGvbkDT6hqPW0GTRz4_S
"""

# from google.colab import drive 
# drive.mount('/content/drive', force_remount=True)

# import os
# cwd = os.getcwd()
# if str(cwd) !='drive/My Drive/CZ4041':
    # print("Changing directory")
    # os.chdir('drive/My Drive/CZ4041')
# else:
    # print("cwd already set to CZ4041")

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.15


import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

#import dataset
emotions = pd.read_csv("emotions.csv")
emotions.head(5)

#train test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(emotions, random_state=42, test_size=0.20, shuffle=True)
x_train = train.iloc[:,0:72]
y_train = train.iloc[:,-6:]
x_test = test.iloc[:,0:72]
y_test = test.iloc[:,-6:]

print("x_train shape: {}, x_test shape: {}".format(x_train.shape,x_test.shape))
print("x_train type: {}, x_test type: {}".format(type(x_train),type(x_test)))
print("y_train_shape: {}, y_test shape: {}".format(y_train.shape,y_test.shape))
print("y_train type: {}, y_test type: {}".format(type(y_train),type(y_test)))
print("y_train[0]: {}".format(y_train.iloc[0]))

#feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
x_scaler = MinMaxScaler(feature_range=(0,1))
#x_scaler = MinMaxScaler(feature_range=(-1,1))
x_scaled_train =  x_scaler.fit_transform(x_train)
x_scaled_test = x_scaler.fit_transform(x_test)
x_scaled_train = x_scaled_train.astype('float32')
x_scaled_test = x_scaled_test.astype('float32')

print("x_scaled_train shape: {}, x_scaled_test shape: {}".format(x_scaled_train.shape,x_scaled_test.shape))
print("x_scaled_train type: {}, x_scaled_test type: {}".format(type(x_scaled_train),type(x_scaled_test)))

#extracting labels
output_labels = y_train.columns.values
print("output_labels type: {}".format(type(output_labels)))
print("output_labels: {}".format(output_labels))

#convert labels to one hot vector
y_train = np.array(y_train)
y_test = np.array(y_test)
print("y_train shape: {}, y_test shape: {}".format(y_train.shape,y_test.shape))
print("y_train type: {}, y_test type: {}".format(type(y_train),type(y_test)))
print("y_train[0]: {}".format(y_train[0]))

epochs = 500

"""# 1. Identifying the optimiser"""

def build_model(optimizer_,shape_,num_output):
#=========setting up computation graph==========
  model = Sequential()
  model.add(Dense(50,input_shape=(shape_,),activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Dense(100,activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Dense(50,activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(num_output, activation='sigmoid'))

  model.compile(optimizer=optimizer_, loss = 'binary_crossentropy', metrics=['accuracy'])
  return model

#making 3 models that uses SGD,SGD_momentum, Adam
optimizers_ = {'sgd': SGD(lr=0.001, nesterov=True),
        'sgd_momentum': SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
        'adam': Adam(lr=0.001)} 
models = []     
for i,j in optimizers_.items():
  temp = build_model(j,72,6)
  models.append(temp)

#fitting respective models
import time
histories = []
execution_time = []
for i in range(len(models)):
  start_time = time.time()
  histories.append(models[i].fit(x_scaled_train, y_train, batch_size=20,epochs=500,shuffle=True, verbose=0,validation_data=(x_scaled_test, y_test)))
  stop_time = time.time()
  time_lapsed = stop_time-start_time
  execution_time.append(time_lapsed)
  print("Fitting done for model {}, execution time: {:0.3f} seconds".format(i+1, time_lapsed))

def calculate_scores(model,x_test,y_test,output_labels):
  pred_full_testdata = model.predict_proba(x_test)
  pred_full_testdata = np.array(pred_full_testdata)
  lower_bound = np.amin(pred_full_testdata)
  upper_bound = np.amax(pred_full_testdata)
  threshold = np.arange(0.1,0.9,0.05)
  acc = []
  accuracies = []
  best_threshold = np.zeros(output_labels.shape)
  for i in range(pred_full_testdata.shape[1]):
      y_prob = np.array(pred_full_testdata[:,i])
      for j in threshold:
          y_pred = [1 if prob>=j else 0 for prob in y_prob]
          acc.append( matthews_corrcoef(y_test[:,i],y_pred))
      acc   = np.array(acc)
      index = np.where(acc==acc.max()) 
      accuracies.append(acc.max()) 
      best_threshold[i] = threshold[index[0][0]]
      acc = []
  #print(best_threshold)
  y_pred = np.array([[1 if pred_full_testdata[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
  #print(y_pred[0:6])
  hamm_loss = hamming_loss(y_test,y_pred)
  acc_score = accuracy_score(y_test, y_pred)
  Log_Loss = log_loss(y_test,y_pred)
  class_report = classification_report(y_test,y_pred)
  #print(class_report)
  total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == 6])
  #print(total_correctly_predicted)
  scores = []
  scores.append(hamm_loss)
  scores.append(acc_score)
  scores.append(Log_Loss)
  scores.append(class_report)
  scores.append(total_correctly_predicted)
  return scores

#calculate each model score
model_1_scores = calculate_scores(models[0],x_scaled_test,y_test,output_labels)
model_2_scores = calculate_scores(models[1],x_scaled_test,y_test,output_labels)
model_3_scores = calculate_scores(models[2],x_scaled_test,y_test,output_labels)
print("model1 acc: {}, model2 acc:{}, model3 acc: {}".format(model_1_scores[1],model_2_scores[1],model_3_scores[1]))

objects = ('sgd', 'sgd_momentum', 'adam')
y_pos = np.arange(len(models))
hamm_loss_list = [model_1_scores[0],model_2_scores[0],model_3_scores[0]]

#plot hamming lost comparison
fig, ax = plt.subplots()    
width = 0.35 
hamm_plot = ax.bar(y_pos, hamm_loss_list, width, color='r')
ax.set_ylabel('Hamming loss')
ax.set_title('Hamming loss comparison')
ax.set_xticks(y_pos + width / 7)
ax.set_xticklabels(objects)
for rect in hamm_plot:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%0.3f' %(height),
                ha='center', va='bottom')
plt.show()

#plot exact match score comparison
acc_list = [model_1_scores[1],model_2_scores[1],model_3_scores[1]]
fig, ax = plt.subplots()    
width = 0.35 
acc_plot = ax.bar(y_pos, acc_list, width, color='b')
ax.set_ylabel('Exact match')
ax.set_title('Exact match comparison')
ax.set_xticks(y_pos + width / 7)
ax.set_xticklabels(objects)
for rect in acc_plot:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
plt.show()

#plot execution time comparison
fig, ax = plt.subplots()    
width = 0.35 
exe_time_plot = ax.bar(y_pos, execution_time, width, color='g')
ax.set_ylabel('Execution time')
ax.set_title('Execution time comparison')
ax.set_xticks(y_pos + width / 7)
ax.set_xticklabels(objects)
for rect in exe_time_plot:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
plt.show()

#plot training loss comparison
epoch_range = range(1, epochs+1)
plt.plot(epoch_range, histories[0].history['loss'])
plt.plot(epoch_range, histories[1].history['loss'])
plt.plot(epoch_range, histories[2].history['loss'])
plt.title('Training loss comparison')
plt.ylabel('Training loss')
plt.xlabel('Epoch')
plt.legend(['sgd', 'sgd_momentum','adam'], loc='upper right')
plt.show()

#plot validation loss comparison
epoch_range = range(1, epochs+1)
plt.plot(epoch_range, histories[0].history['val_loss'])
plt.plot(epoch_range, histories[1].history['val_loss'])
plt.plot(epoch_range, histories[2].history['val_loss'])
plt.title('Validation loss comparison')
plt.ylabel('Validation loss')
plt.xlabel('Epoch')
plt.legend(['sgd', 'sgd_momentum','adam'], loc='upper right')
plt.show()

#plot training accuracy comparison
plt.plot(epoch_range, histories[0].history['acc'])
plt.plot(epoch_range, histories[1].history['acc'])
plt.plot(epoch_range, histories[2].history['acc'])
plt.title('Training accuracy comparison')
plt.ylabel('Training accuracy')
plt.xlabel('Epoch')
plt.legend(['sgd', 'sgd_momentum','adam'], loc='lower right')
plt.show()

#plot validation accuracy comparison
plt.plot(epoch_range, histories[0].history['val_acc'])
plt.plot(epoch_range, histories[1].history['val_acc'])
plt.plot(epoch_range, histories[2].history['val_acc'])
plt.title('Validation accuracy comparison')
plt.ylabel('Validation accuracy')
plt.xlabel('Epoch')
plt.legend(['sgd', 'sgd_momentum','adam'], loc='lower right')
plt.show()

"""# 2. Dimensionality reduction"""

#re-establish the optimum optimiser
optimum_optimiser = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

"""## 2.1 Feature Selection_ Variance threshold"""
#calculate threshold for variance threshold
variance = np.var(x_scaled_train,axis=0)
thres = np.percentile(variance,25)
#print(thres)

#apply threshold on features
temp_x = np.concatenate((x_scaled_train, x_scaled_test), axis=0)
print("temp_x shape: {}".format(temp_x.shape))
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
features_selector = VarianceThreshold(thres)
x_fs = features_selector.fit_transform(temp_x)
print(x_fs.shape)
x_train_fs = x_fs[0:474,:]
x_test_fs = x_fs[474:593,:]
print("x_train_fs shape: {}, x_test_fs shape: {}".format(x_train_fs.shape,x_test_fs.shape))
num_features = np.size(x_train_fs,1)

#traing variance_threshold model
var_thres_model = build_model(optimum_optimiser,num_features,6)
start_time = time.time()
var_thres_history = var_thres_model.fit(x_train_fs, y_train, batch_size=20,epochs=500,shuffle=True, verbose=0,validation_data=(x_test_fs, y_test))
stop_time = time.time()
var_thres_exeTime = stop_time-start_time
print("fitting done for var thres model, execution time: {}".format(var_thres_exeTime))

#calculate variance threshold scores
var_thres_scores = calculate_scores(var_thres_model,x_test_fs,y_test,output_labels)
print("hamming loss: {}, acc score: {}, log loss: {}".format(var_thres_scores[0],var_thres_scores[1],var_thres_scores[2]))

"""## 2.2 Feature Extraction_PCA"""
#set the PCA information retainment threshold
from sklearn.decomposition import PCA
pca = PCA(0.99)
pca.fit(x_scaled_train)
pca.n_components_

#apply threshold on features
train_x = pca.transform(x_scaled_train)
test_x = pca.transform(x_scaled_test)
print("train_x shape: {}, test_x: {}".format(train_x.shape,test_x.shape))

#train PCA model
pca_model = build_model(optimum_optimiser,pca.n_components_,6)
start_time = time.time()
pca_history = pca_model.fit(train_x, y_train, batch_size=20,epochs=500,shuffle=True, verbose=0,validation_data=(test_x, y_test))
stop_time = time.time()
pca_exeTime = stop_time-start_time
print("fitting done for pca model, execution time: {}".format(pca_exeTime))

#calculate PCA score
pca_scores = calculate_scores(pca_model,test_x,y_test,output_labels)
print("hamming loss: {}, acc score: {}, log loss: {}".format(pca_scores[0],pca_scores[1],pca_scores[2]))

"""## 2.3 Plotting the results"""
x_labels = ["original","Var_thres","PCA"]
exe_time_list = [execution_time[1],var_thres_exeTime,pca_exeTime]
hamm_loss_list_2 = [model_2_scores[0],var_thres_scores[0],pca_scores[0]]
acc_list_2 = [model_2_scores[1],var_thres_scores[1],pca_scores[1]]

#plot exe time comparison
fig, ax = plt.subplots()    
width = 0.35 
exe_time_plot_2 = ax.bar(y_pos, exe_time_list, width, color='g')
ax.set_ylabel('Execution time')
ax.set_title('Execution time comparison')
ax.set_xticks(y_pos + width / 7)
ax.set_xticklabels(x_labels)
for rect in exe_time_plot_2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
plt.show()

#plot hamming loss comparison
fig, ax = plt.subplots()    
width = 0.35 
hamm_loss_plot_2 = ax.bar(y_pos, hamm_loss_list_2, width, color='g')
ax.set_ylabel('Hamming loss ')
ax.set_title('Hamming loss comparison')
ax.set_xticks(y_pos + width / 7)
ax.set_xticklabels(x_labels)
for rect in hamm_loss_plot_2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
plt.show()

#plot exact match accuracy comparison
fig, ax = plt.subplots()    
width = 0.35 
acc_plot_2 = ax.bar(y_pos, acc_list_2, width, color='g')
ax.set_ylabel('Exact match accuracy ')
ax.set_title('Exact match accuracy comparison')
ax.set_xticks(y_pos + width / 7)
ax.set_xticklabels(x_labels)
for rect in acc_plot_2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
plt.show()

"""# 3. Bookmarks dataset"""

#import dataset
bookmarks = pd.read_csv("bookmarks.csv") #number of labels = 208
bookmarks.head(5)

#split x and y
train, test = train_test_split(bookmarks, random_state=42, test_size=0.20, shuffle=True)
x_train_bm = train.iloc[:,0:2150]
y_train_bm = train.iloc[:,-208:]
x_test_bm = test.iloc[:,0:2150]
y_test_bm = test.iloc[:,-208:]
print("x_train shape: {}, x_test shape: {}".format(x_train_bm.shape,x_test_bm.shape))
print("y_train_shape: {}, y_test shape: {}".format(y_train_bm.shape,y_test_bm.shape))

#x_scaler = MaxAbsScaler()
#scale data
x_scaler = MinMaxScaler(feature_range=(0,1))
x_scaled_train_bm =  x_scaler.fit_transform(x_train_bm)
x_scaled_test_bm = x_scaler.fit_transform(x_test_bm)
x_scaled_train_bm = x_scaled_train_bm.astype('float32')
x_scaled_test_bm = x_scaled_test_bm.astype('float32')
print("x_scaled_train shape: {}, x_scaled_test shape: {}".format(x_scaled_train_bm.shape,x_scaled_test_bm.shape))
print("x_scaled_train type: {}, x_scaled_test type: {}".format(type(x_scaled_train_bm),type(x_scaled_test_bm)))

#perform PCA
pca.fit(x_scaled_train_bm)
pca.n_components_
train_x_bm = pca.transform(x_scaled_train_bm)
test_x_bm = pca.transform(x_scaled_test_bm)
print("train_x shape: {}, test_x: {}".format(train_x_bm.shape,test_x_bm.shape))

#extract labels
output_labels_bm = y_train_bm.columns.values
print("bookmark output_labels shape: {}".format(output_labels_bm.shape))
y_train_bm = np.array(y_train_bm)
y_test_bm = np.array(y_test_bm)
print("y_train shape: {}, y_test shape: {}".format(y_train_bm.shape,y_test_bm.shape))
print("y_train type: {}, y_test type: {}".format(type(y_train_bm),type(y_test_bm)))

#build bookmarks model
bookmark_model = build_model(optimum_optimiser,pca.n_components_,208)
start_time = time.time()
bookmark_history = bookmark_model.fit(train_x_bm, y_train_bm, batch_size=20,epochs=500,shuffle=True, verbose=0,validation_data=(test_x_bm, y_test_bm))
stop_time = time.time()
bookmark_exeTime = stop_time-start_time
print("fitting done for bookmark model, execution time: {}".format(bookmark_exeTime))

#calculate bookmarks score
bookmark_scores = calculate_scores(bookmark_model,test_x_bm,y_test_bm,output_labels_bm)
print("hamming loss: {}, acc score: {}, log loss: {}".format(bookmark_scores[0],bookmark_scores[1],bookmark_scores[2]))
