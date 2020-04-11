#!/usr/bin/env python
# coding: utf-8

# In[117]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

def BinaryRelevance(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = skpt.BinaryRelevance(GaussianNB())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("Binary Relevance w/ GaussianNB", predictions ,dataset_test_y)
    
def BinaryRelevanceSVC(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):    
    classifier = skpt.BinaryRelevance(SVC())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("Binary Relevance w/ SVC", predictions ,dataset_test_y)
      
def BinaryRelevanceMNB(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):      
    classifier = skpt.BinaryRelevance(MultinomialNB())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("Binary Relevance w/ MNB", predictions ,dataset_test_y)    
    
def ClassifierChain(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = skpt.ClassifierChain(GaussianNB())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("CC w/ GaussianNB", predictions ,dataset_test_y)

def ClassifierChainSVC(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = skpt.ClassifierChain(SVC())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("CC w/ SVC", predictions ,dataset_test_y)

def ClassifierChainMNB(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):    
    classifier = skpt.ClassifierChain(MultinomialNB(alpha = 1.0))
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("CC w/ MNB", predictions ,dataset_test_y)
    
#     rangefloat = [round(x * 0.1, 1) for x in range(1, 10)] #degree of smoothing
#     parameters = {'classifier__alpha' : rangefloat}

#     clf = GridSearchCV(skpt.ClassifierChain(MultinomialNB()), parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=2)
#     #print(clf.get_params().keys())
#     clf.fit(dataset_train_x, dataset_train_y)
#     print(clf.cv_results_)
#     #return clf.best_params_
    
#     classifier = skpt.ClassifierChain(MultinomialNB(alpha = clf.best_params_['classifier__alpha']))
#     %timeit classifier.fit(dataset_train_x, dataset_train_y)
#     predictions = classifier.predict(dataset_test_x)
    
#     Metrics_Accuracy("CC w/ MNB tuning", predictions ,dataset_test_y)
   
def LabelPowerset(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = skpt.LabelPowerset(GaussianNB())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("LP w/ GaussianNB", predictions ,dataset_test_y)

def LabelPowersetSVC(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):    
    classifier = skpt.LabelPowerset(SVC())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("LP w/ SVC", predictions ,dataset_test_y)

def LabelPowersetMNB(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):    
    classifier = skpt.LabelPowerset(MultinomialNB())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("LP w/ MNB", predictions ,dataset_test_y)

#Choose best classifier between MNB and SVC - combined
def BinaryRelevanceCV(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    parameters = [
        {
            'classifier': [MultinomialNB()],
            'classifier__alpha': [0.7, 1.0], #for smoothing {Additive smoothing parameter NB}
        },
        {
            'classifier': [SVC()],
            'classifier__kernel': ['rbf','linear'],
        },
    ]

    classifier = GridSearchCV(skpt.BinaryRelevance(), parameters, scoring = 'accuracy')
    print(classifier)
    classifier.fit(dataset_train_x, dataset_train_y)
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("Binary Relevance w/ CV",predictions, dataset_test_y)
    
def ClassifierChainCV(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    parameters = [
        {
            'classifier': [MultinomialNB()],
            'classifier__alpha': [0.7, 1.0],
        },
        {
            'classifier': [SVC()],
            'classifier__kernel': ['rbf', 'linear'],
        },
    ]
    classifier = GridSearchCV(skpt.ClassifierChain(),parameters,scoring='accuracy',n_jobs=2)
    print(classifier)
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    Metrics_Accuracy("CC Cross Validate", predictions ,dataset_test_y)    
 
def MLkNN(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_neighbours, smoothing_param):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    classifier = skadapt.MLkNN(k=num_neighbours,s=smoothing_param)
    get_ipython().run_line_magic('timeit', 'classifier.fit(x_train,y_train)')
    predictions = classifier.predict(x_test)
    
    text = "MLkNN w/ k=" + str(num_neighbours)
    
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
def RAkELd(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_labels):
    classifier = RakelD(
        base_classifier=GaussianNB(),
        base_classifier_require_dense=[True, True],
        labelset_size=num_labels
    )

    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("RAkELd", predictions ,dataset_test_y)
    
#random overlapping label space division with Label Powerset
def RAkELO(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_labels):
    classifier = RakelO(
        base_classifier=GaussianNB(),
        base_classifier_require_dense=[True, True],
        labelset_size=dataset_train_y.shape[1],
        model_count=12
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
def FindBestK(dataset_train_x, dataset_train_y):
    rangefloatv = [round(x * 0.1, 1) for x in range(5, 10)]
    parameters = {'k': range(1,5), 's': rangefloatv}

    clf = GridSearchCV(skadapt.MLkNN(), parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=2)
    clf.fit(lil_matrix(dataset_train_x).toarray(), lil_matrix(dataset_train_y).toarray())
    print(clf.best_params_)
    return clf.best_params_

def FindBestVT(dataset_train_x, dataset_train_y):
    rangefloat = [round(x * 0.01, 2) for x in range(1, 10)]
    rangefloatv = [round(x * 0.1, 1) for x in range(5, 10)]
    parameters = {'threshold': rangefloat, 'vigilance': rangefloatv} #default thres = 0.02, vigi = 0.9

    clf = GridSearchCV(skadapt.MLARAM(), parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=2)
    clf.fit(lil_matrix(dataset_train_x).toarray(), lil_matrix(dataset_train_y).toarray())
    print(clf.best_params_)
    return clf.best_params_


# In[118]:


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


# In[119]:


#Binary Relevance
print("%Comparison Binary Relevance%")
print("Bird dataset")
BinaryRelevance(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
BinaryRelevance(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
BinaryRelevance(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)

print("%Comparison Binary Relevance SVC%")
print("Bird dataset")
BinaryRelevanceSVC(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
BinaryRelevanceSVC(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
BinaryRelevanceSVC(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)

print("%Comparison Binary Relevance MNB%")
print("Bird dataset")
BinaryRelevanceMNB(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
BinaryRelevanceMNB(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
BinaryRelevanceMNB(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)

# print("%Comparison Binary Relevance Grid Search CV%")
# print("Bird dataset")
# BinaryRelevanceCV(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
# print("Emotions dataset")
# BinaryRelevanceCV(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
# print("Yeast dataset")
# BinaryRelevanceCV(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[120]:


#Classifier Chain
print("Comparison CC")
print("Bird dataset")
ClassifierChain(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
ClassifierChain(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
ClassifierChain(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)

print("Comparison CC SVC")
print("Bird dataset")
ClassifierChainSVC(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
ClassifierChainSVC(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
ClassifierChainSVC(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)

print("Comparison CC MNB")
print("Bird dataset")
ClassifierChainMNB(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
ClassifierChainMNB(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
ClassifierChainMNB(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[121]:


#Label Powerset
print("Comparison LP")
print("Bird dataset")
LabelPowerset(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
LabelPowerset(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
LabelPowerset(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)

print("Comparison LP SVC")
print("Bird dataset")
LabelPowersetSVC(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
LabelPowersetSVC(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
LabelPowersetSVC(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)

print("Comparison LP MNB")
print("Bird dataset")
LabelPowersetMNB(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
LabelPowersetMNB(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
LabelPowersetMNB(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[ ]:





# In[122]:


#Test other methods
RAkELd(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,19)
RAkELd(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,6)
RAkELd(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,14)


# In[123]:


#Test other methods
RAkELO(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,19)
RAkELO(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,6)
RAkELO(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,14)


# In[124]:


#Adapted Algorithms
#MLkNN with k =3
print("MLkNN")
print("Bird dataset")
MLkNN(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,3,1)
print("Emotions dataset")
MLkNN(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,3,1)
print("Yeast dataset")
MLkNN(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,3,1)


# In[125]:


#Adapted Algorithms
#MLkNN with Find the best K
print("MLkNN")
print("Bird dataset")
dict_res = FindBestK(dataset_train_x_bird, dataset_train_y_bird)
MLkNN(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,dict_res['k'],dict_res['s'])
print("Emotions dataset")
dict_res= FindBestK(dataset_train_x_emotions,dataset_train_y_emotions)
MLkNN(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,dict_res['k'],dict_res['s'])
print("Yeast dataset")
dict_res= FindBestK(dataset_train_x_yeast,dataset_train_y_yeast)
MLkNN(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,dict_res['k'],dict_res['s'])


# In[126]:


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


# In[127]:


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


# In[128]:


#todo label relations exploration
print("Bird dataset")
LabelSpacePartitioningClassifier(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
LabelSpacePartitioningClassifier(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
LabelSpacePartitioningClassifier(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[129]:


#Embedded Classifier
print("Bird dataset")
EmbeddingClassifierMethod(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
EmbeddingClassifierMethod(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
EmbeddingClassifierMethod(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[130]:


rangefloat = [round(x * 0.01, 2) for x in range(1, 5)]
print(rangefloat)


# In[ ]:




