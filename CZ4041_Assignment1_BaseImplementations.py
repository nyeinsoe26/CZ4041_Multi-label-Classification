#!/usr/bin/env python
# coding: utf-8

# In[59]:


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
#from skmultilearn.embedding import SKLearnEmbedder, EmbeddingClassifier
from sklearn.manifold import SpectralEmbedding
import skmultilearn.problem_transform as skpt
import pandas as pd
import numpy as np
import skmultilearn.adapt as skadapt
import sklearn.metrics as metrics
from sklearn import preprocessing

"""" Brief Description of each base classifier
Base classifiers: MultinomialNB, C-Support Vector Classification (SVC), Logistic Regression, GaussianNB
MultinomialNB:



""""

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
    classifier = skpt.ClassifierChain(LogisticRegression(max_iter=120000))
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("CC w/ Logistic Regression (iter=120000)", predictions ,dataset_test_y)

def ClassifierChainSVC(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = skpt.ClassifierChain(SVC())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("CC w/ SVC", predictions ,dataset_test_y)

def ClassifierChainMNB(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = skpt.ClassifierChain(MultinomialNB())
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("CC w/ MNB", predictions ,dataset_test_y)
   
def LabelPowerset(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = skpt.LabelPowerset(LogisticRegression(max_iter=120000))
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("LP w/ Logistic Regression (iter=120000)", predictions ,dataset_test_y)

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
    classifier = GridSearchCV(skpt.ClassifierChain(LogisticRegression(max_iter=120000)),parameters,scoring='accuracy',n_jobs=2)
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("CC Cross Validate w/ Logistic Regression (iter=120000)", predictions ,dataset_test_y)    
 
def MLkNN(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_neighbours):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    classifier = skadapt.MLkNN(k=num_neighbours)
    get_ipython().run_line_magic('timeit', 'classifier.fit(x_train,y_train)')
    predictions = classifier.predict(x_test)
    
    text = "MLkNN w/ k=" + str(num_neighbours)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
    
def MLARAM(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_vigilance, num_threshold):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
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

def EmbeddingClassifier(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = EmbeddingClassifier(
        SKLearnEmbedder(SpectralEmbedding(n_components=10)),
        RandomForestRegressor(n_estimators=10),
        MLkNN(k=5)
    )
    get_ipython().run_line_magic('timeit', 'classifier.fit(dataset_train_x, dataset_train_y)')
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


# In[60]:


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


# In[48]:


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


# In[49]:


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


# In[50]:


#Label Powerset Chain
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


# In[51]:


#GridSearchCV
print("Comparison GridSearchCV for CC")
print("Bird dataset")
ClassifierChainCV(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
ClassifierChainCV(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
ClassifierChainCV(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[52]:


#Test other methods
RAkELd(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,19)
RAkELd(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,6)
RAkELd(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,14)


# In[53]:


#Test other methods
RAkELO(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,19)
RAkELO(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,6)
RAkELO(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,14)


# In[54]:


#Adapted Algorithms
#MLkNN
print("MLkNN")
print("Bird dataset")
MLkNN(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird,2)
print("Emotions dataset")
MLkNN(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions,2)
print("Yeast dataset")
MLkNN(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast,2)


# In[55]:


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


# In[58]:


#todo label relations exploration
LabelSpacePartitioningClassifier(dataset_train_x_bird, dataset_train_y_bird, dataset_test_x_bird, dataset_test_y_bird)
print("Emotions dataset")
LabelSpacePartitioningClassifier(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)
print("Yeast dataset")
LabelSpacePartitioningClassifier(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




