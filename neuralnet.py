import random
import time
import numpy as np
import pandas as pd
import os
from keras import backend as K
import datetime
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, hamming_loss, accuracy_score, log_loss, classification_report
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

#Fix random seed for keras reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

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

def calculate_scores(model, x_test, y_test):
    y_prob = model.predict_proba(x_test)
    y_prob = np.array(y_prob)
    y_pred = np.where(y_prob >= 0.5, 1, 0)

    hamm_loss = hamming_loss(y_test,y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    Log_Loss = log_loss(y_test,y_pred)
    scores = []
    scores.append(hamm_loss)
    scores.append(acc_score)
    scores.append(Log_Loss)
    return scores


def build_model(optimizer_, shape_, num_output, propensity=None):
    def propensityLoss(p):
        def pLoss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            ones = tf.ones_like(y_pred, dtype=tf.float32)
            zeros = tf.zeros_like(y_pred, dtype=tf.float32)
            y_pred_rounded = tf.where(tf.greater_equal(y_pred, 0.5), ones, zeros)

            loss = tf.reduce_mean((1.0 / p) * (1.0 - (y_true * y_pred_rounded)) * tf.math.pow(y_true-y_pred, 2))
            
            loss = (1.0 - tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_rounded), tf.float32))) * loss
            return loss
        return pLoss
    
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

    if propensity is None:
        model.compile(optimizer=optimizer_, loss = 'binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer_, loss = propensityLoss(p=propensity), metrics=['accuracy'])
    return model

#creating figures folder
current_working_dir = os.getcwd()
folderDir = os.path.join(current_working_dir,'figures')

if not os.path.isdir(folderDir):
    print('creating the figures folder')
    os.makedirs(folderDir)

datasets = ['emotions', 'bookmarks', 'yeast']
numinputoutputs = [[72, 6], [2150, 208], [103, 14]]

epochs = 500
batch_size= 20
x_label_width = 0.35

for (dataset, [num_in, num_out]) in zip(datasets, numinputoutputs):
    print("Dataset name:", dataset)
    #import dataset
    ds = pd.read_csv(os.path.join("datasets", dataset+".csv"))
    ds_length = len(ds.index)
    #train test split
    train, test = train_test_split(ds, random_state=seed_value, test_size=0.20, shuffle=True)
    x_train = train.iloc[:,0:num_in]
    y_train = train.iloc[:,-num_out:]
    x_test = test.iloc[:,0:num_in]
    y_test = test.iloc[:,-num_out:]

    print("x_train shape: {}, x_test shape: {}".format(x_train.shape,x_test.shape))
    print("x_train type: {}, x_test type: {}".format(type(x_train),type(x_test)))
    print("y_train_shape: {}, y_test shape: {}".format(y_train.shape,y_test.shape))
    print("y_train type: {}, y_test type: {}".format(type(y_train),type(y_test)))
    print("y_train[0]: {}".format(y_train.iloc[0]))

    #feature scaling
    x_scaler = MinMaxScaler(feature_range=(0,1))
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

    #Calculate propensities
    N = ds_length
    L = num_out
    A = 0.55 
    B = 1.5
    propensity = []
    for i in range(L):
        Nl = np.count_nonzero(y_train[:,i])
        propensity.append(1.0 / (1.0 + tf.math.log(N-1.0)*tf.math.pow(1.0+B, A)*tf.math.exp(-A*tf.math.log(Nl + B))))
    propensity = np.array(propensity)

    """# 1. Identifying the optimiser"""

    #making 3 models that uses SGD,SGD_momentum, Adam with default BCE loss
    #plus 3 models that uses SGD,SGD_momentum, Adam with propensity loss
    model_names = ['sgd_bce', 'sgdmomentum_bce', 'adam_bce','sgd_propensity', 'sgdmomentum_propensity', 'adam_propensity']
    optimizers_ = {'sgd': SGD(lr=0.001, nesterov=True),
            'sgd_momentum': SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
            'adam': Adam(lr=0.001)} 
    models = []     
    for _, j in optimizers_.items():
        temp = build_model(j, num_in, num_out)
        models.append(temp)
    for _, j in optimizers_.items():
        temp = build_model(j, num_in, num_out, propensity)
        models.append(temp)

    #fitting respective models
    histories = []
    execution_time = []
    for (model, model_name) in zip(models, model_names):
        print("Fitting model: {}".format(model_name))
        start_time = time.time()
        histories.append(model.fit(x_scaled_train, y_train, batch_size=batch_size,epochs=epochs,shuffle=True, verbose=0,validation_data=(x_scaled_test, y_test)))
        stop_time = time.time()
        time_lapsed = stop_time-start_time
        execution_time.append(time_lapsed)
        print("Fitting done for model: {}, execution time: {:0.3f} seconds".format(model_name, time_lapsed))

    scores = []
    #calculate each model score
    for (model, model_name) in zip(models, model_names):
        score = calculate_scores(model, x_scaled_test, y_test)
        scores.append(score)
        print("model: {}, acc: {}".format(model_name, score[1]))

    y_pos = np.arange(len(models))
    hamm_loss_list = []
    acc_list = []
    for score in scores:
        hamm_loss_list.append(score[0])
        acc_list.append(score[1])

    #plot hamming lost comparison
    fig, ax = plt.subplots()    
    hamm_plot = ax.bar(y_pos, hamm_loss_list, x_label_width, color='r')
    ax.set_ylabel('Hamming loss')
    ax.set_title('Hamming loss comparison')
    ax.set_xticks(y_pos + x_label_width / 7)
    ax.set_xticklabels(model_names)
    for rect in hamm_plot:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%0.3f' %(height),
                ha='center', va='bottom')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}] Hamming loss comparison.png'.format(dataset)))
    plt.clf()

    #plot exact match score comparison
    fig, ax = plt.subplots()   
    acc_plot = ax.bar(y_pos, acc_list, x_label_width, color='b')
    ax.set_ylabel('Exact match')
    ax.set_title('Exact match comparison')
    ax.set_xticks(y_pos + x_label_width / 7)
    ax.set_xticklabels(model_names)
    for rect in acc_plot:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}] Exact match comparison.png'.format(dataset)))
    plt.clf()

    #plot execution time comparison
    fig, ax = plt.subplots()    
    exe_time_plot = ax.bar(y_pos, execution_time, x_label_width, color='g')
    ax.set_ylabel('Execution time')
    ax.set_title('Execution time comparison')
    ax.set_xticks(y_pos + x_label_width / 7)
    ax.set_xticklabels(model_names)
    for rect in exe_time_plot:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}] Execution time comparison.png'.format(dataset)))
    plt.clf()

    #plot training loss comparison
    epoch_range = range(1, epochs+1)
    for j in range(len(histories)):
        plt.plot(epoch_range, histories[j].history['loss'])
    plt.title('Training loss comparison')
    plt.ylabel('Training loss')
    plt.xlabel('Epoch')
    plt.legend(model_names, loc='upper right')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}] Training loss comparison.png'.format(dataset)))
    plt.clf()

    #plot validation loss comparison
    epoch_range = range(1, epochs+1)
    for j in range(len(histories)):
        plt.plot(epoch_range, histories[j].history['val_loss'])
    plt.title('Validation loss comparison')
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    plt.legend(model_names, loc='upper right')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}] Validation loss comparison.png'.format(dataset)))
    plt.clf()

    #plot training accuracy comparison
    for j in range(len(histories)):
        plt.plot(epoch_range, histories[j].history['accuracy'])
    plt.title('Training accuracy comparison')
    plt.ylabel('Training accuracy')
    plt.xlabel('Epoch')
    plt.legend(model_names, loc='lower right')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}] Training accuracy comparison.png'.format(dataset)))
    plt.clf()

    #plot validation accuracy comparison
    for j in range(len(histories)):
        plt.plot(epoch_range, histories[j].history['val_accuracy'])
    plt.title('Validation accuracy comparison')
    plt.ylabel('Validation accuracy')
    plt.xlabel('Epoch')
    plt.legend(model_names, loc='lower right')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}] Validation accuracy comparison.png'.format(dataset)))
    plt.clf()

    """# 2. Dimensionality reduction"""

    #re-establish the optimum optimiser
    optimum_optimiser = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    """## 2.1 Feature Selection_ Variance threshold"""
    #calculate threshold for variance threshold
    variance = np.var(x_scaled_train,axis=0)
    thres = np.percentile(variance,25)

    #apply threshold on features
    temp_x = np.concatenate((x_scaled_train, x_scaled_test), axis=0)
    print("temp_x shape: {}".format(temp_x.shape))
    features_selector = VarianceThreshold(thres)
    x_fs = features_selector.fit_transform(temp_x)
    print(x_fs.shape)
    x_train_fs = x_fs[:int(ds_length*0.8),:]
    x_test_fs = x_fs[int(ds_length*0.8):,:]
    print("x_train_fs shape: {}, x_test_fs shape: {}".format(x_train_fs.shape,x_test_fs.shape))
    reduced_num_in = np.size(x_train_fs,1)

    #training variance_threshold model
    var_thres_model = build_model(optimum_optimiser,reduced_num_in, num_out)
    start_time = time.time()
    var_thres_history = var_thres_model.fit(x_train_fs, y_train, batch_size=batch_size,epochs=epochs,shuffle=True, verbose=0,validation_data=(x_test_fs, y_test))
    stop_time = time.time()
    var_thres_exeTime = stop_time-start_time
    print("fitting done for var thres model (SGD momentum, BCE loss), execution time: {}".format(var_thres_exeTime))

    #calculate variance threshold scores
    var_thres_scores = calculate_scores(var_thres_model,x_test_fs,y_test)
    print("hamming loss: {}, acc score: {}, log loss: {}".format(var_thres_scores[0],var_thres_scores[1],var_thres_scores[2]))

    """## 2.2 Feature Extraction_PCA"""
    #set the PCA information retainment threshold
    pca = PCA(0.99)
    pca.fit(x_scaled_train)
    pca.n_components_

    #apply threshold on features
    train_x = pca.transform(x_scaled_train)
    test_x = pca.transform(x_scaled_test)
    print("train_x shape: {}, test_x: {}".format(train_x.shape,test_x.shape))

    #train PCA model
    pca_model = build_model(optimum_optimiser,pca.n_components_, num_out)
    start_time = time.time()
    pca_history = pca_model.fit(train_x, y_train, batch_size=batch_size,epochs=epochs,shuffle=True, verbose=0,validation_data=(test_x, y_test))
    stop_time = time.time()
    pca_exeTime = stop_time-start_time
    print("fitting done for pca model (SGD momentum, BCE loss), execution time: {}".format(pca_exeTime))

    #calculate PCA score
    pca_scores = calculate_scores(pca_model,test_x,y_test)
    print("hamming loss: {}, acc score: {}, log loss: {}".format(pca_scores[0],pca_scores[1],pca_scores[2]))

    """## 2.3 Plotting the results"""
    x_labels = ["Original","Var_thres","PCA"]
    exe_time_list = [execution_time[1], var_thres_exeTime, pca_exeTime]
    hamm_loss_list_2 = [hamm_loss_list[1], var_thres_scores[0], pca_scores[0]]
    acc_list_2 = [acc_list[1], var_thres_scores[1], pca_scores[1]]
    y_pos = np.arange(len(x_labels))
    
    #plot exe time comparison
    fig, ax = plt.subplots()  
    exe_time_plot_2 = ax.bar(y_pos, exe_time_list, x_label_width, color='g')
    ax.set_ylabel('Execution time')
    ax.set_title('Execution time comparison')
    ax.set_xticks(y_pos + x_label_width / 7)
    ax.set_xticklabels(x_labels)
    for rect in exe_time_plot_2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}][After PCA] Execution time comparison.png'.format(dataset)))
    plt.clf()

    #plot hamming loss comparison
    fig, ax = plt.subplots()    
    hamm_loss_plot_2 = ax.bar(y_pos, hamm_loss_list_2, x_label_width, color='g')
    ax.set_ylabel('Hamming loss ')
    ax.set_title('Hamming loss comparison')
    ax.set_xticks(y_pos + x_label_width / 7)
    ax.set_xticklabels(x_labels)
    for rect in hamm_loss_plot_2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}][After PCA] Hamming loss comparison.png'.format(dataset)))
    plt.clf()

    #plot exact match accuracy comparison
    fig, ax = plt.subplots()    
    acc_plot_2 = ax.bar(y_pos, acc_list_2, x_label_width, color='g')
    ax.set_ylabel('Exact match accuracy ')
    ax.set_title('Exact match accuracy comparison')
    ax.set_xticks(y_pos + x_label_width / 7)
    ax.set_xticklabels(x_labels)
    for rect in acc_plot_2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%0.3f' %(height),
                ha='center', va='bottom')
    plt.xticks(rotation = -45)
    plt.savefig(os.path.join('figures', '[{}][After PCA] Exact match accuracy comparison.png'.format(dataset)))
    plt.clf()