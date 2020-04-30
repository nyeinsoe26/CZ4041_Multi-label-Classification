#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.ensemble import RakelD, RakelO
from sklearn.metrics import make_scorer
import skmultilearn.problem_transform as skpt
import pandas as pd
import numpy as np
import skmultilearn.adapt as skadapt
import sklearn.metrics as metrics
from sklearn import preprocessing

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

def TwinMLSVM(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,nc_k, omega):
    classifier = skadapt.MLTSVM(c_k = nc_k, sor_omega = omega)
    classifier.fit(csr_matrix(dataset_train_x),csr_matrix(dataset_train_y))
    predictions = classifier.predict(csr_matrix(dataset_test_x))
    
    Metrics_Accuracy("MLTSVM", predictions ,dataset_test_y)


    
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

def FindCKParam(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    rangev = [2**i for i in range(-5, 3, 2)]
    #introduce back 0 default to rangev
    rangev = rangev+ [0]
    rangefloat = [round(x * 0.1, 1) for x in range(1, 11)]    
    #rangefloat2 = [1e-06, 1e-05]
    parameters = {'c_k': rangev, 'sor_omega': rangefloat} 
    
    clf = GridSearchCV(skadapt.MLTSVM(), parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=2, verbose = 10)
    clf.fit(csr_matrix(dataset_train_x),csr_matrix(dataset_train_y))
    print(clf.best_params_)
    return clf.best_params_
    
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
        end_range = 10 if dataset_train_y.shape[1]//2 > (3+1) else dataset_train_y.shape[1] #dataset_train_y.shape[1]//2 if dataset_train_y.shape[1]//2 > (3+1) else dataset_train_y.shape[1]
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
    print(classifier.best_params_)
    return classifier.best_params_

def Util_Title(title):
    print("====================================",title,"====================================")
    
def Util_ClassifierMethods(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y):
    #BR
    Util_Title("Binary Relevance")
    base_classif = GaussianNB()
    BinaryRelevance(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "GaussianNB")

    dict_res = FindBestSVCParams(skpt.BinaryRelevance(),dataset_train_x,dataset_train_y)
    base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
    BinaryRelevance(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "SVC tuned")

    dict_res = FindBestMNBParams(skpt.BinaryRelevance(),dataset_train_x,dataset_train_y)
    base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
    BinaryRelevance(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "MNB tuned")
    
    #CC
    Util_Title("Classifier Chain")
    base_classif = GaussianNB()
    ClassifierChain(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "GaussianNB")

    dict_res = FindBestSVCParams(skpt.ClassifierChain(),dataset_train_x,dataset_train_y)
    base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
    ClassifierChain(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "SVC tuned")

    dict_res = FindBestMNBParams(skpt.ClassifierChain(),dataset_train_x,dataset_train_y)
    base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
    ClassifierChain(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "MNB tuned")

    #LP
    Util_Title("Label Powerset")
    base_classif = GaussianNB()
    LabelPowerset(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "GaussianNB")

    dict_res = FindBestSVCParams(skpt.LabelPowerset(),dataset_train_x,dataset_train_y)
    base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
    LabelPowerset(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "SVC tuned")

    dict_res = FindBestMNBParams(skpt.LabelPowerset(),dataset_train_x,dataset_train_y)
    base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
    LabelPowerset(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y, base_classif, "MNB tuned")
    
    #MLkNN
    Util_Title("MLkNN")
    dict_res= FindBestK(skadapt.MLkNN(), dataset_train_x,dataset_train_y)
    MLkNN(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y,dict_res['k'],dict_res['s'])

    #MLARAM
    Util_Title("MLARAM")
    dict_res = FindBestVT(dataset_train_x,dataset_train_y)
    MLARAM(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y,dict_res['vigilance'],dict_res['threshold'])

    #BRkNNa
    Util_Title("BRkNNa")
    dict_res= FindBestK(skadapt.BRkNNaClassifier(), dataset_train_x,dataset_train_y)
    BRkNNa(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y,dict_res['k'])
    
    #BRkNNb
    Util_Title("BRkNNb")
    dict_res= FindBestK(skadapt.BRkNNbClassifier(), dataset_train_x,dataset_train_y)
    BRkNNb(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y,dict_res['k'])
    
    #RAkELD
    Util_Title("RAkELd")
    dict_res = GridSearchCV_baseRakel(RakelD(),dataset_train_x,dataset_train_y)
    RAkELd(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y,dict_res['base_classifier'],dict_res['labelset_size'])
    
    #RAkELo
    Util_Title("RAkELo")
    dict_res = GridSearchCV_baseRakel(RakelO(),dataset_train_x,dataset_train_y)
    RAkELO(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y,dict_res['base_classifier'],dict_res['labelset_size'],dict_res['model_count'])

    #MLTSVM
    Util_Title("MLTSVM")
    dict_res = FindCKParam(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y)
    TwinMLSVM(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y,dict_res['c_k'],dict_res['sor_omega'])

def Util_ClassifierMethodsBookmarks(train_x, y_train, test_x, y_test):    
    #Scale negatives for BR/ CC and LP for MultinomialNB
    x_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    train_x_scaledb = x_scaler.fit_transform(train_x)
    test_x_scaledb = x_scaler.fit_transform(test_x)

    #BR
    Util_Title("Binary Relevance")
    base_classif = GaussianNB()
    BinaryRelevance(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "GaussianNB")

    dict_res = FindBestSVCParams(skpt.BinaryRelevance(),train_x_scaledb, y_train)
    base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
    BinaryRelevance(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "SVC tuned")

    dict_res = FindBestMNBParams(skpt.BinaryRelevance(),train_x_scaledb,y_train)
    base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
    BinaryRelevance(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "MNB tuned")
    
    #CC
    Util_Title("Classifier Chain")
    base_classif = GaussianNB()
    ClassifierChain(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "GaussianNB")

    dict_res = FindBestSVCParams(skpt.ClassifierChain(),train_x_scaledb, y_train)
    base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
    ClassifierChain(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "SVC tuned")

    dict_res = FindBestMNBParams(skpt.ClassifierChain(),train_x_scaledb, y_train)
    base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
    ClassifierChain(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "MNB tuned")

    #LP
    Util_Title("Label Powerset")
    base_classif = GaussianNB()
    LabelPowerset(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "GaussianNB")

    dict_res = FindBestSVCParams(skpt.LabelPowerset(),train_x_scaledb, y_train)
    base_classif = SVC(kernel = dict_res['classifier__kernel'], degree = dict_res['classifier__degree'])
    LabelPowerset(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "SVC tuned")

    dict_res = FindBestMNBParams(skpt.LabelPowerset(),train_x_scaledb, y_train)
    base_classif = MultinomialNB(alpha = dict_res['classifier__alpha'])
    LabelPowerset(train_x_scaledb, y_train, test_x_scaledb, y_test, base_classif, "MNB tuned")
    
    #RAkELo
    Util_Title("RAkELo")
    lbs_size = 3
    mod_count = 4
    RAkELO(train_x, y_train,test_x, y_test, LinearSVC(max_iter=500,verbose=1),lbs_size,mod_count)

    #RAkELd
    lbs_size = 3
    RAkELd(train_x, y_train, test_x, y_test, LinearSVC(verbose =2), lbs_size)

    #MLkNN
    base_classif = skadapt.MLkNN()
    k = 10
    s = 1
    MLkNN(train_x,y_train,test_x, y_test,k,s)

    #MLARAM
    v = 0.95
    t = 0.05
    dict_res = FindBestVT(train_x, y_train)
    MLARAM(train_x,y_train,test_x, y_test,dict_res['vigilance'],dict_res['threshold'])

    #BRkNNa
    dict_res= FindBestK(skadapt.BRkNNaClassifier(), train_x, y_train)
    BRkNNa(train_x,y_train,test_x, y_test,dict_res['k'])

    #BRkNNb
    dict_res= FindBestK(skadapt.BRkNNbClassifier(), train_x, y_train)
    BRkNNb(train_x,y_train,test_x, y_test,dict_res['k'])

    #MLTSVM
    #Test for 0 
    #TwinMLSVM(train_x,y_train,test_x,y_test,0,1)
    #Test for 0.125
    TwinMLSVM(train_x,y_train,test_x,y_test,0.125,1)
    #Test for 0.25
    #TwinMLSVM(train_x,y_train,test_x,y_test,0.25,1)    
    
def LoadEmotionsDataset(path):
    #Emotions Dataset
    #emotions
    print("Load Emotions dataset")
    emotions = pd.read_csv(path)

    #scale based on columns before split
    mms = preprocessing.MinMaxScaler()
    emotions.iloc[:,0:72] = mms.fit_transform(emotions.iloc[:,0:72])

    #split dataset
    dataset_train_emotions, dataset_test_emotions = train_test_split(emotions,random_state=42, test_size=0.20, shuffle=True)

    dataset_train_x_emotions = dataset_train_emotions.iloc[:,0:72]
    dataset_train_y_emotions = dataset_train_emotions.iloc[:,-6:]

    dataset_test_x_emotions = dataset_test_emotions.iloc[:,0:72]
    dataset_test_y_emotions = dataset_test_emotions.iloc[:,-6:]
    
    return dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions

def LoadYeastDataset(path): 
    print("Load Yeast dataset")
    yeast = pd.read_csv(path)

    #scale based on columns before split
    mms = preprocessing.MinMaxScaler()
    yeast.iloc[:,0:103] = mms.fit_transform(yeast.iloc[:,0:103])

    #split dataset
    dataset_train_yeast, dataset_test_yeast = train_test_split(yeast,random_state=42, test_size=0.20, shuffle=True)

    dataset_train_x_yeast = dataset_train_yeast.iloc[:,0:103]
    dataset_train_y_yeast = dataset_train_yeast.iloc[:,-14:]

    dataset_test_x_yeast = dataset_test_yeast.iloc[:,0:103]
    dataset_test_y_yeast = dataset_test_yeast.iloc[:,-14:]
    
    return dataset_train_x_yeast, dataset_train_y_yeast, dataset_test_x_yeast, dataset_test_y_yeast

def LoadBookmarksDataset(path):
    #bookmarks 1/10
    print("Bookmarks dataset")
    book = pd.read_csv(path)
    book1, book2 = train_test_split(book, random_state=42, test_size=0.90, shuffle=True)
    print(book1.shape)
    print(book2.shape)
    
    bookmark_train, bookmark_test = train_test_split(book1, random_state=42, test_size=0.20, shuffle=True)
    x_train = bookmark_train.iloc[:,0:2150]
    y_train = bookmark_train.iloc[:,-208:]

    x_test = bookmark_test.iloc[:,0:2150]
    y_test = bookmark_test.iloc[:,-208:]
    x_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_scaled_train =  np.float32(x_scaler.fit_transform(x_train))
    x_scaled_test = np.float32(x_scaler.fit_transform(x_test))
    print("x_scaled_train shape: {}, x_scaled_test shape: {}".format(x_scaled_train.shape,x_scaled_test.shape))
    print("x_scaled_train type: {}, x_scaled_test type: {}".format(type(x_scaled_train),type(x_scaled_test)))

    from sklearn.decomposition import PCA
    pca = PCA(0.9)
    pca.fit(x_scaled_train)
    train_x = pca.transform(x_scaled_train)
    test_x = pca.transform(x_scaled_test)
    print("train_x shape: {}, test_x: {}".format(train_x.shape,test_x.shape))
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train shape: {}, y_test shape: {}".format(y_train.shape,y_test.shape))
    print("y_train type: {}, y_test type: {}".format(type(y_train),type(y_test)))
    print("y_train[0]: {}".format(y_train[0]))
    
    return train_x, y_train, test_x, y_test


# In[8]:


path = "C:/Users/K/Desktop/Assignment1/"
dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions = LoadEmotionsDataset(path+"emotions.csv") #r"C:/Users/K/Desktop/Assignment1/emotions.csv")

print("Emotions Dataset")
Util_ClassifierMethods(dataset_train_x_emotions,dataset_train_y_emotions,dataset_test_x_emotions,dataset_test_y_emotions)


# In[9]:


#yeast
path = "C:/Users/K/Desktop/Assignment1/"
dataset_train_x_yeast, dataset_train_y_yeast, dataset_test_x_yeast, dataset_test_y_yeast = LoadYeastDataset(path+"yeast.csv")

print("Yeast Dataset")
Util_ClassifierMethods(dataset_train_x_yeast,dataset_train_y_yeast,dataset_test_x_yeast,dataset_test_y_yeast)


# In[5]:


#bookmarks 1/10
path = "C:/Users/K/Desktop/Assignment1/"
train_x, y_train, test_x, y_test = LoadBookmarksDataset(path+"bookmarks.csv")
print(train_x.shape)
print(y_train.shape)
print(test_x.shape)
print(y_test.shape)
      
print("Bookmarks Dataset")
Util_ClassifierMethodsBookmarks(train_x, y_train, test_x, y_test)


# In[ ]:





# In[ ]:




