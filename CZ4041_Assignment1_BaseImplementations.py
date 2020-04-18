#!/usr/bin/env python
# coding: utf-8

# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.ensemble import RakelD, RakelO, MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.embedding import SKLearnEmbedder, EmbeddingClassifier
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import make_scorer
import skmultilearn.problem_transform as skpt
import pandas as pd
import numpy as np
import skmultilearn.adapt as skadapt
import sklearn.metrics as metrics
from sklearn import preprocessing

def FindBestMNBParams(classif, dataset_train_x, dataset_train_y):
    rangefloat = [round(x * 0.1, 1) for x in range(1, 11)]
    parameters = {
        'classifier': [MultinomialNB()],
        'classifier__alpha': rangefloat,
    }
    clf = GridSearchCV(classif, parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=3)
    clf.fit(dataset_train_x, dataset_train_y)
    print(clf.best_params_)
    #print(pd.DataFrame(clf.cv_results_))
    
    return clf.best_params_

def FindBestSVCParams(classif, dataset_train_x, dataset_train_y):
    parameters = {
            'classifier': [SVC()],
            'classifier__degree': [2,3,4],
            'classifier__kernel': ['linear','poly','rbf'],
            #'classifier__max_iter': [10000],
            #'classifier__loss': ['hinge','squared_hinge'],
    }
    
    clf = GridSearchCV(classif, parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=3)
    clf.fit(dataset_train_x, dataset_train_y)
    print(clf.best_params_)
    #print(pd.DataFrame(clf.cv_results_))
    
    return clf.best_params_

def BinaryRelevance(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title):
    #print(base_classif)
    classifier = skpt.BinaryRelevance(base_classif)
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy(title, predictions ,dataset_test_y)
    
def ClassifierChain(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title):
    classifier = skpt.ClassifierChain(base_classif)
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy(title, predictions ,dataset_test_y)

def LabelPowerset(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title):
    classifier = skpt.LabelPowerset(base_classif)
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy(title, predictions ,dataset_test_y)
 
def MLkNN(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_neighbours, smoothing_param):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    classifier = skadapt.MLkNN(k=num_neighbours,s=smoothing_param)
    get_ipython().run_line_magic('timeit', 'classifier.fit(x_train,y_train)')
    predictions = classifier.predict(x_test)
    
    text = "MLkNN w/ k=" + str(num_neighbours) + " s="+str(smoothing_param)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
    
def MLARAM(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_vigilance, num_threshold):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    #Threshold controls number of prototypes to participate; vigilance controls how large hyperbox is
    classifier = skadapt.MLARAM(threshold = num_threshold, vigilance = num_vigilance)
    get_ipython().run_line_magic('timeit', 'classifier.fit(x_train,y_train)')
    predictions = classifier.predict(x_test)
    
    text = "MLARAM w/ Threshold = " + str(num_threshold) + ", Vigilance = "+ str(num_vigilance)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
        
    
#Random Label Space Partitionining with Label Powerset
def RAkELd(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,base_clasif,num_labels):
    classifier = RakelD(
        base_classifier=base_clasif,
        labelset_size=num_labels
    )

    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("RAkELd", predictions ,dataset_test_y)
    
#random overlapping label space division with Label Powerset
def RAkELO(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,base_clasif,num_labels, num_models):
    classifier = RakelO(
        base_classifier=base_clasif,
        labelset_size=num_labels,
        model_count=num_models
    )

    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("RAkELO", predictions ,dataset_test_y)

def LabelSpacePartitioningClassifier(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = MajorityVotingClassifier(
        clusterer=FixedLabelSpaceClusterer(clusters = [[1,3,4], [0,2,5]]),
        classifier = skpt.ClassifierChain(classifier=SVC())
    )
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("Label Space Partition", predictions ,dataset_test_y)

def BRkNNa(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, num_neighbours):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    classifier = skadapt.BRkNNaClassifier(k=num_neighbours)
    get_ipython().run_line_magic('timeit', 'classifier.fit(x_train,y_train)')
    predictions = classifier.predict(x_test)
    
    text = "BRkNNa w/ k=" + str(num_neighbours)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
    
def BRkNNb(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, num_neighbours):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    classifier = skadapt.BRkNNbClassifier(k=num_neighbours)
    get_ipython().run_line_magic('timeit', 'classifier.fit(x_train,y_train)')
    predictions = classifier.predict(x_test)
    
    text = "BRkNNb w/ k=" + str(num_neighbours)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
    
def EmbeddingClassifierMethod(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = EmbeddingClassifier(
        SKLearnEmbedder(SpectralEmbedding(n_components=10)),
        RandomForestRegressor(n_estimators=10),
        skadapt.MLkNN(k=5)
    )
    get_ipython().run_line_magic('timeit', 'classifier.fit(lil_matrix(dataset_train_x).toarray(), lil_matrix(dataset_train_y).toarray())')
    predictions = classifier.predict(dataset_test_x)

    Metrics_Accuracy("Embedded Classifier", predictions ,dataset_test_y)
    
def Metrics_Accuracy(classifier,predictions,dataset_test_y):
    #results
    print("Results for ",classifier)
    # accuracy
    print("Accuracy = ",accuracy_score(dataset_test_y,predictions))
    # hamming loss
    print("Hamming loss = ",metrics.hamming_loss(dataset_test_y,predictions))
    # log loss
    #print(type(predictions)==np.ndarray)
    print("Log loss = ",metrics.log_loss(dataset_test_y,predictions.toarray() if type(predictions)!=np.ndarray else predictions))
    # Exact Match Score
    #exact_match_score = np.all(predictions.toarray() == dataset_test_y, axis=1).mean()
    #print('Exact match score (Whole row must match):', exact_match_score)
    
    print("")
    
def Util_ClassifierMethods(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y):
    BinaryRelevance(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y)
    ClassifierChain(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y)
    ClassifierChainCV(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y)
    LabelPowerset(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y)
    
#estimating best params using hamming loss for multi label problems
def FindBestK(classif, dataset_train_x, dataset_train_y):
    rangefloatv = [round(x * 0.1, 1) for x in range(5, 11)]
    
    parameters = {'k': range(1,20), 's': rangefloatv} 
    if type(classif) == type(skadapt.BRkNNaClassifier()) or type(classif) == type(skadapt.BRkNNbClassifier()):
        parameters = {'k': range(1,20)}

    clf = GridSearchCV(classif, parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=2)
    clf.fit(lil_matrix(dataset_train_x).toarray(), lil_matrix(dataset_train_y).toarray())
    print(clf.best_params_)
    return clf.best_params_

def FindBestVT(dataset_train_x, dataset_train_y):
    rangefloat = [round(x * 0.01, 2) for x in range(1, 11)]
    rangefloatv = [round(x * 0.1, 1) for x in range(5, 11)]
    parameters = {'threshold': rangefloat, 'vigilance': rangefloatv} #default thres = 0.02, vigi = 0.9

    clf = GridSearchCV(skadapt.MLARAM(), parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=2)
    clf.fit(lil_matrix(dataset_train_x).toarray(), lil_matrix(dataset_train_y).toarray())
    print(clf.best_params_)
    return clf.best_params_

def GridSearchCV_base(classif, dataset_train_x, dataset_train_y):
    rangefloat = [round(x * 0.1, 1) for x in range(1, 11)]
    parameters = [
        {
            'base_classifier': [GaussianNB()],
            #'labelset_size': 
        },
        {
            'base_classifier': [MultinomialNB()],
            'base_classifier__alpha': rangefloat, #for smoothing {Additive smoothing parameter NB}
        },
        {
            'base_classifier': [SVC()],
            'base_classifier__kernel': ['rbf','linear','sigmoid'],
        },
    ]
    
    classifier = GridSearchCV(RakelD(), parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=3)
    classifier.fit(dataset_train_x, dataset_train_y)
    return classifier.best_params_
    
def GridSearchCV_baseRakel(classif, dataset_train_x, dataset_train_y):
    #labelset_size denotes the desired size of partition
    range_labelset_size = list(range(1,11))
    rangefloat = [round(x * 0.1, 1) for x in range(1, 11)]
    parameters = [
        {
            'base_classifier': [GaussianNB()],
            'labelset_size': range_labelset_size,
        },
        {
            'base_classifier': [MultinomialNB()],
            'base_classifier__alpha': rangefloat, #for smoothing {Additive smoothing parameter NB}
            'labelset_size': range_labelset_size,
        },
        {
            'base_classifier': [SVC()],
            'base_classifier__kernel': ['rbf','linear','sigmoid'],
            'labelset_size': range_labelset_size,
        },
    ]
    print(type(classif) == type(RakelO()))
    if (type(classif) == type(RakelO())):
        end_range = dataset_train_y.shape[1]//2 if dataset_train_y.shape[1]//2 > (3+1) else dataset_train_y.shape[1]
        range_labelset_size = list(range(3, end_range))
        #starting_range = dataset_train_y.shape[1]//range_labelset_size[0]
        range_model_count = list(range(2*dataset_train_y.shape[1],2*dataset_train_y.shape[1]+1)) #[x*2 for x in range((starting_range), (starting_range+1))]#[x*2 for x in range(dataset_train_y.shape[1]//6, dataset_train_y.shape[1]//2)]
        print(dataset_train_y.shape[1])
        print(range_labelset_size)
        print(range_model_count)
        parameters = [
            {
                'base_classifier': [GaussianNB()],
                'labelset_size': range_labelset_size,
                'model_count': range_model_count,
            },
            {
                'base_classifier': [MultinomialNB()],
                'base_classifier__alpha': rangefloat, #for smoothing {Additive smoothing parameter NB}
                'labelset_size': range_labelset_size,
                'model_count': range_model_count,
            },
            {
                'base_classifier': [SVC()],
                'base_classifier__kernel': ['rbf','linear','sigmoid'],
                'labelset_size': range_labelset_size,
                'model_count': range_model_count,
            },
        ]
    
    classifier = GridSearchCV(classif, parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=3)
    classifier.fit(dataset_train_x, dataset_train_y)
    return classifier.best_params_


# In[69]:


#birds
print("Load Birds dataset")
birds1 = pd.read_csv(r"C:/Users/K/Desktop/Assignment1/birds-train.csv")
birds2 = pd.read_csv(r"C:/Users/K/Desktop/Assignment1/birds-test.csv")
birds = birds1.append(birds2)

#scale based on columns before split
mms = preprocessing.MinMaxScaler()

#print(birds.iloc[:,0:260])
birds.iloc[:,0:260] = mms.fit_transform(birds.iloc[:,0:260])

#print(birds.iloc[:,0:260])

#split dataset
dataset_train_bird, dataset_test_bird = train_test_split(birds,random_state=42, test_size=0.20, shuffle=True)

dataset_train_x_bird = dataset_train_bird.iloc[:,0:260]
dataset_train_y_bird = dataset_train_bird.iloc[:,-19:]

dataset_test_x_bird = dataset_test_bird.iloc[:,0:260]
dataset_test_y_bird = dataset_test_bird.iloc[:,-19:]

#emotions
print("Load Emotions dataset")
emotions = pd.read_csv(r"C:/Users/K/Desktop/Assignment1/emotions.csv")

#scale based on columns before split
mms = preprocessing.MinMaxScaler()
emotions.iloc[:,0:72] = mms.fit_transform(emotions.iloc[:,0:72])

#split dataset
dataset_train_emotions, dataset_test_emotions = train_test_split(emotions,random_state=42, test_size=0.20, shuffle=True)

dataset_train_x_emotions = dataset_train_emotions.iloc[:,0:72]
dataset_train_y_emotions = dataset_train_emotions.iloc[:,-6:]

dataset_test_x_emotions = dataset_test_emotions.iloc[:,0:72]
dataset_test_y_emotions = dataset_test_emotions.iloc[:,-6:]

#yeast
print("Load Yeast dataset")
yeast = pd.read_csv(r"C:/Users/K/Desktop/Assignment1/yeast.csv")

#scale based on columns before split
mms = preprocessing.MinMaxScaler()
yeast.iloc[:,0:103] = mms.fit_transform(yeast.iloc[:,0:103])

#split dataset
dataset_train_yeast, dataset_test_yeast = train_test_split(yeast,random_state=42, test_size=0.20, shuffle=True)

dataset_train_x_yeast = dataset_train_yeast.iloc[:,0:103]
dataset_train_y_yeast = dataset_train_yeast.iloc[:,-14:]

dataset_test_x_yeast = dataset_test_yeast.iloc[:,0:103]
dataset_test_y_yeast = dataset_test_yeast.iloc[:,-14:]


# In[70]:


#Binary Relevance
print("%Comparison Binary Relevance GaussianNB%")
base_classif = GaussianNB()
print("Bird dataset")
BinaryRelevance(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,base_classif, "GaussianNB")
print("Emotions dataset")
BinaryRelevance(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "GaussianNB")
print("Yeast dataset")
BinaryRelevance(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,base_classif, "GaussianNB")


# In[71]:


print("%Comparison Binary Relevance SVC%")
base_classif = SVC()
print("Bird dataset")
BinaryRelevance(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "SVC")
print("Emotions dataset")
BinaryRelevance(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "SVC")
print("Yeast dataset")
BinaryRelevance(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "SVC")


# In[72]:


print("%Comparison Binary Relevance SVC aft tuned%")
print("Bird dataset")
dict_res = FindBestSVCParams(skpt.BinaryRelevance(),dataset_train_x_bird, dataset_train_y_bird)
#base_classif = LinearSVC(max_iter=10000, loss = dict_res['classifier__loss'])
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
BinaryRelevance(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "SVC tuned")

print("Emotions dataset")
dict_res = FindBestSVCParams(skpt.BinaryRelevance(),dataset_train_x_emotions,dataset_train_y_emotions)
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
BinaryRelevance(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "SVC tuned")

print("Yeast dataset")
dict_res = FindBestSVCParams(skpt.BinaryRelevance(),dataset_train_x_yeast,dataset_train_y_yeast)
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
BinaryRelevance(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "SVC tuned")


# In[73]:


print("%Comparison Binary Relevance MNB%")
base_classif = MultinomialNB()
print("Bird dataset")
BinaryRelevance(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "MNB")
print("Emotions dataset")
BinaryRelevance(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "MNB")
print("Yeast dataset")
BinaryRelevance(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "MNB")


# In[74]:


print("%Comparison Binary Relevance MNB aft tuned%")
print("Bird dataset")
dict_res = FindBestMNBParams(skpt.BinaryRelevance(),dataset_train_x_bird, dataset_train_y_bird)
#base_classif = LinearSVC(max_iter=10000, loss = dict_res['classifier__loss'])
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
BinaryRelevance(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "MNB tuned")

print("Emotions dataset")
dict_res = FindBestMNBParams(skpt.BinaryRelevance(),dataset_train_x_emotions,dataset_train_y_emotions)
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
BinaryRelevance(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "MNB tuned")

print("Yeast dataset")
dict_res = FindBestMNBParams(skpt.BinaryRelevance(),dataset_train_x_yeast,dataset_train_y_yeast)
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
BinaryRelevance(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "MNB tuned")


# In[75]:


#Classifier Chain
print("CC")
base_classif = GaussianNB()
print("Bird dataset")
ClassifierChain(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,base_classif,"GaussianNB")
print("Emotions dataset")
ClassifierChain(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif,"GaussianNB")
print("Yeast dataset")
ClassifierChain(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif,"GaussianNB")


# In[76]:


print("%Comparison Classifier Chain SVC%")
base_classif = SVC()
print("Bird dataset")
ClassifierChain(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "SVC")
print("Emotions dataset")
ClassifierChain(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "SVC")
print("Yeast dataset")
ClassifierChain(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "SVC")


# In[77]:


print("%Comparison Classifier Chain SVC aft tuned%")
print("Bird dataset")
dict_res = FindBestSVCParams(skpt.ClassifierChain(),dataset_train_x_bird, dataset_train_y_bird)
#base_classif = LinearSVC(max_iter=10000, loss = dict_res['classifier__loss'])
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
ClassifierChain(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "SVC tuned")

print("Emotions dataset")
dict_res = FindBestSVCParams(skpt.ClassifierChain(),dataset_train_x_emotions,dataset_train_y_emotions)
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
ClassifierChain(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "SVC tuned")

print("Yeast dataset")
dict_res = FindBestSVCParams(skpt.ClassifierChain(),dataset_train_x_yeast,dataset_train_y_yeast)
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
ClassifierChain(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "SVC tuned")


# In[78]:


print("%Comparison Classifier Chain MNB%")
base_classif = MultinomialNB()
print("Bird dataset")
ClassifierChain(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "MNB")
print("Emotions dataset")
ClassifierChain(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "MNB")
print("Yeast dataset")
ClassifierChain(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "MNB")


# In[79]:


print("%Comparison Classifier Chain MNB aft tuned%")
print("Bird dataset")
dict_res = FindBestMNBParams(skpt.ClassifierChain(),dataset_train_x_bird, dataset_train_y_bird)
#base_classif = LinearSVC(max_iter=10000, loss = dict_res['classifier__loss'])
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
ClassifierChain(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "MNB tuned")

print("Emotions dataset")
dict_res = FindBestMNBParams(skpt.ClassifierChain(),dataset_train_x_emotions,dataset_train_y_emotions)
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
ClassifierChain(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "MNB tuned")

print("Yeast dataset")
dict_res = FindBestMNBParams(skpt.ClassifierChain(),dataset_train_x_yeast,dataset_train_y_yeast)
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
ClassifierChain(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "MNB tuned")


# In[80]:


#Label Powerset
print("Comparison LP GaussianNB")
base_classif = GaussianNB()
print("Bird dataset")
LabelPowerset(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,base_classif,"GaussianNB")
print("Emotions dataset")
LabelPowerset(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif,"GaussianNB")
print("Yeast dataset")
LabelPowerset(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif,"GaussianNB")


# In[81]:


print("%Comparison Label Powerset SVC%")
base_classif = SVC()
print("Bird dataset")
LabelPowerset(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "SVC")
print("Emotions dataset")
LabelPowerset(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "SVC")
print("Yeast dataset")
LabelPowerset(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "SVC")


# In[82]:


print("%Comparison Label Powerset SVC aft tuned%")
print("Bird dataset")
dict_res = FindBestSVCParams(skpt.LabelPowerset(),dataset_train_x_bird, dataset_train_y_bird)
#base_classif = LinearSVC(max_iter=10000, loss = dict_res['classifier__loss'])
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
LabelPowerset(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "SVC tuned")

print("Emotions dataset")
dict_res = FindBestSVCParams(skpt.LabelPowerset(),dataset_train_x_emotions,dataset_train_y_emotions)
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
LabelPowerset(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "SVC tuned")

print("Yeast dataset")
dict_res = FindBestSVCParams(skpt.LabelPowerset(),dataset_train_x_yeast,dataset_train_y_yeast)
base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
LabelPowerset(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "SVC tuned")


# In[83]:


print("%Comparison Label Powerset MNB%")
base_classif = MultinomialNB()
print("Bird dataset")
LabelPowerset(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "MNB")
print("Emotions dataset")
LabelPowerset(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "MNB")
print("Yeast dataset")
LabelPowerset(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "MNB")


# In[84]:


print("%Comparison Label Powerset MNB aft tuned%")
print("Bird dataset")
dict_res = FindBestMNBParams(skpt.LabelPowerset(),dataset_train_x_bird, dataset_train_y_bird)
#base_classif = LinearSVC(max_iter=10000, loss = dict_res['classifier__loss'])
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
LabelPowerset(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird, base_classif, "MNB tuned")

print("Emotions dataset")
dict_res = FindBestMNBParams(skpt.LabelPowerset(),dataset_train_x_emotions,dataset_train_y_emotions)
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
LabelPowerset(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions, base_classif, "MNB tuned")

print("Yeast dataset")
dict_res = FindBestMNBParams(skpt.LabelPowerset(),dataset_train_x_yeast,dataset_train_y_yeast)
base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
LabelPowerset(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast, base_classif, "MNB tuned")


# In[85]:


#to do loop to find 
dict_res = GridSearchCV_base(RakelD(),dataset_train_x_bird, dataset_train_y_bird)
print(dict_res)
RAkELd(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,dict_res['base_classifier'],3)
dict_res = GridSearchCV_base(RakelD(),dataset_train_x_emotions, dataset_train_y_emotions)
print(dict_res)
RAkELd(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,dict_res['base_classifier'],3)
dict_res = GridSearchCV_base(RakelD(),dataset_train_x_yeast, dataset_train_y_yeast)
print(dict_res)
RAkELd(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,dict_res['base_classifier'],3)


# In[86]:


#to do loop to find 
dict_res = GridSearchCV_baseRakel(RakelD(),dataset_train_x_bird, dataset_train_y_bird)
print(dict_res)
RAkELd(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,dict_res['base_classifier'],dict_res['labelset_size'])
dict_res = GridSearchCV_baseRakel(RakelD(),dataset_train_x_emotions, dataset_train_y_emotions)
print(dict_res)
RAkELd(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,dict_res['base_classifier'],dict_res['labelset_size'])
dict_res = GridSearchCV_baseRakel(RakelD(),dataset_train_x_yeast, dataset_train_y_yeast)
print(dict_res)
RAkELd(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,dict_res['base_classifier'],dict_res['labelset_size'])


# In[87]:


#to do loop to find 
dict_res = GridSearchCV_baseRakel(RakelO(),dataset_train_x_bird, dataset_train_y_bird)
#print(dict_res)
RAkELO(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,dict_res['base_classifier'],dict_res['labelset_size'],dict_res['model_count'])
dict_res = GridSearchCV_baseRakel(RakelO(),dataset_train_x_emotions, dataset_train_y_emotions)
#print(dict_res)
RAkELO(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,dict_res['base_classifier'],dict_res['labelset_size'],dict_res['model_count'])
dict_res = GridSearchCV_baseRakel(RakelO(),dataset_train_x_yeast, dataset_train_y_yeast)
#print(dict_res)
RAkELO(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,dict_res['base_classifier'],dict_res['labelset_size'],dict_res['model_count'])


# In[88]:


#Adapted Algorithms
#MLkNN with k =10 (default) smoothing_param = 1
k = 10
s = 1
print("MLkNN")
print("Bird dataset")
MLkNN(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,k,s)
print("Emotions dataset")
MLkNN(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,k,s)
print("Yeast dataset")
MLkNN(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,k,s)


# In[89]:


#Adapted Algorithms
#MLkNN with Find the best K
print("MLkNN")
print("Bird dataset")
dict_res = FindBestK(skadapt.MLkNN(),dataset_train_x_bird, dataset_train_y_bird)
MLkNN(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,dict_res['k'],dict_res['s'])
print("Emotions dataset")
dict_res= FindBestK(skadapt.MLkNN(), dataset_train_x_emotions,dataset_train_y_emotions)
MLkNN(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,dict_res['k'],dict_res['s'])
print("Yeast dataset")
dict_res= FindBestK(skadapt.MLkNN(), dataset_train_x_yeast,dataset_train_y_yeast)
MLkNN(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,dict_res['k'],dict_res['s'])


# In[90]:


#MLARAM
v = 0.95
t = 0.05
print("MLARAM")
print("Bird dataset")
MLARAM(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,v,t)
print("Emotions dataset")
MLARAM(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,v,t)
print("Yeast dataset")
MLARAM(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,v,t)


# In[91]:


#MLARAM with tuning
print("MLARAM")
print("Bird dataset")
dict_res = FindBestVT(dataset_train_x_bird, dataset_train_y_bird)
MLARAM(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,dict_res['vigilance'],dict_res['threshold'])
print("Emotions dataset")
dict_res = FindBestVT(dataset_train_x_emotions,dataset_train_y_emotions)
MLARAM(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,dict_res['vigilance'],dict_res['threshold'])
print("Yeast dataset")
dict_res = FindBestVT(dataset_train_x_yeast,dataset_train_y_yeast)
MLARAM(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,dict_res['vigilance'],dict_res['threshold'])


# In[92]:


#Adapted Algorithms
#BRkNNa with k = 10 (default)
k = 10
print("BRkNNa")
print("Bird dataset")
BRkNNa(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,k)
print("Emotions dataset")
BRkNNa(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,k)
print("Yeast dataset")
BRkNNa(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,k)


# In[93]:


#Adapted Algorithms
#BRkNNa with Find the best K
print("BRkNNa tuned")
print("Bird dataset")
dict_res = FindBestK(skadapt.BRkNNaClassifier(), dataset_train_x_bird, dataset_train_y_bird)
BRkNNa(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,dict_res['k'])
print("Emotions dataset")
dict_res= FindBestK(skadapt.BRkNNaClassifier(), dataset_train_x_emotions,dataset_train_y_emotions)
BRkNNa(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,dict_res['k'])
print("Yeast dataset")
dict_res= FindBestK(skadapt.BRkNNaClassifier(), dataset_train_x_yeast,dataset_train_y_yeast)
BRkNNa(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,dict_res['k'])


# In[94]:


#Adapted Algorithms
#BRkNNb with k = 10 (default)
k = 10
print("BRkNNb")
print("Bird dataset")
BRkNNb(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,k)
print("Emotions dataset")
BRkNNb(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,k)
print("Yeast dataset")
BRkNNb(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,k)


# In[95]:


#Adapted Algorithms
#BRkNNb with Find the best K
print("BRkNNb tuned")
print("Bird dataset")
dict_res = FindBestK(skadapt.BRkNNbClassifier(), dataset_train_x_bird, dataset_train_y_bird)
BRkNNb(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,dict_res['k'])
print("Emotions dataset")
dict_res= FindBestK(skadapt.BRkNNbClassifier(), dataset_train_x_emotions,dataset_train_y_emotions)
BRkNNb(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,dict_res['k'])
print("Yeast dataset")
dict_res= FindBestK(skadapt.BRkNNbClassifier(), dataset_train_x_yeast,dataset_train_y_yeast)
BRkNNb(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,dict_res['k'])


# In[96]:


#todo label relations exploration
print("Bird dataset")
LabelSpacePartitioningClassifier(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
LabelSpacePartitioningClassifier(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
LabelSpacePartitioningClassifier(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[97]:


#Embedded Classifier
print("Bird dataset")
EmbeddingClassifierMethod(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
EmbeddingClassifierMethod(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
EmbeddingClassifierMethod(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[98]:


rangefloat = [round(x * 0.01, 2) for x in range(1, 5)]
print(rangefloat)


# In[99]:


rangefloat = [round(x * 0.1, 1) for x in range(1, 11)]
print(rangefloat)


# In[100]:


xx = list(range(1,11))
print(xx)


# In[101]:


list(range(2,10))


# In[102]:


print ([x*2 for x in range(1,10)])


# In[103]:


[round(x * 0.1, 1) for x in range(1, 11)]


# In[ ]:




