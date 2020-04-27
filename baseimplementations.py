from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.ensemble import RakelD, RakelO, MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.embedding import SKLearnEmbedder, EmbeddingClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import make_scorer
import skmultilearn.problem_transform as skpt
import pandas as pd
import numpy as np
import time
import os
import skmultilearn.adapt as skadapt
import sklearn.metrics as metrics
from sklearn import preprocessing

def BinaryRelevance(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title):
    #print(base_classif)
    classifier = skpt.BinaryRelevance(base_classif)
    start_time = time.time()
    classifier.fit(dataset_train_x, dataset_train_y)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy(title, predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))
    
def ClassifierChain(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title):
    classifier = skpt.ClassifierChain(base_classif)
    start_time = time.time()
    classifier.fit(dataset_train_x, dataset_train_y)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy(title, predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))

def LabelPowerset(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title):
    classifier = skpt.LabelPowerset(base_classif)
    start_time = time.time()
    classifier.fit(dataset_train_x, dataset_train_y)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy(title, predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))
 
def MLkNN(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_neighbours, smoothing_param):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    classifier = skadapt.MLkNN(k=num_neighbours,s=smoothing_param)

    start_time = time.time()
    classifier.fit(x_train,y_train)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(x_test)
    
    text = "MLkNN w/ k=" + str(num_neighbours) + " s="+str(smoothing_param)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))

def MLARAM(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_vigilance, num_threshold):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    #Threshold controls number of prototypes to participate; vigilance controls how large hyperbox is
    classifier = skadapt.MLARAM(threshold = num_threshold, vigilance = num_vigilance)
    start_time = time.time()
    classifier.fit(x_train,y_train)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(x_test)
    
    text = "MLARAM w/ Threshold = " + str(num_threshold) + ", Vigilance = "+ str(num_vigilance)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))
        
    
#Random Label Space Partitionining with Label Powerset
def RAkELd(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,base_clasif,num_labels):
    classifier = RakelD(
        base_classifier=base_clasif,
        labelset_size=num_labels
    )

    start_time = time.time()
    classifier.fit(dataset_train_x, dataset_train_y)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("RAkELd", predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))
    
#random overlapping label space division with Label Powerset
def RAkELO(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,base_clasif,num_labels, num_models):
    classifier = RakelO(
        base_classifier=base_clasif,
        labelset_size=num_labels,
        model_count=num_models
    )

    start_time = time.time()
    classifier.fit(dataset_train_x, dataset_train_y)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("RAkELO", predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))

def LabelSpacePartitioningClassifier(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = MajorityVotingClassifier(
        clusterer=FixedLabelSpaceClusterer(clusters = [[1,3,4], [0,2,5]]),
        classifier = skpt.ClassifierChain(classifier=SVC())
    )
    start_time = time.time()
    classifier.fit(dataset_train_x, dataset_train_y)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(dataset_test_x)
    
    Metrics_Accuracy("Label Space Partition", predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))

def BRkNNa(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, num_neighbours):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    classifier = skadapt.BRkNNaClassifier(k=num_neighbours)
    start_time = time.time()
    classifier.fit(x_train,y_train)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(x_test)
    
    text = "BRkNNa w/ k=" + str(num_neighbours)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))
    
def BRkNNb(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, num_neighbours):
    x_train = lil_matrix(dataset_train_x).toarray()
    y_train = lil_matrix(dataset_train_y).toarray()
    x_test = lil_matrix(dataset_test_x).toarray()
    
    classifier = skadapt.BRkNNbClassifier(k=num_neighbours)
    start_time = time.time()
    classifier.fit(x_train,y_train)
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(x_test)
    
    text = "BRkNNb w/ k=" + str(num_neighbours)
    
    Metrics_Accuracy(text, predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))
    
def EmbeddingClassifierMethod(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y):
    classifier = EmbeddingClassifier(
        SKLearnEmbedder(SpectralEmbedding(n_components=10)),
        RandomForestRegressor(n_estimators=10),
        skadapt.MLkNN(k=5)
    )
    start_time = time.time()
    classifier.fit(lil_matrix(dataset_train_x).toarray(), lil_matrix(dataset_train_y).toarray())
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(dataset_test_x)
    predictions = classifier.predict(dataset_test_x)

    Metrics_Accuracy("Embedded Classifier", predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))

def TwinMLSVM(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,nc_k, omega):
    classifier = skadapt.MLTSVM(c_k = nc_k, sor_omega = omega)
    start_time = time.time()
    classifier.fit(csr_matrix(dataset_train_x),csr_matrix(dataset_train_y))
    stop_time = time.time()
    time_lapsed = stop_time-start_time
    predictions = classifier.predict(csr_matrix(dataset_test_x))
    
    Metrics_Accuracy("MLTSVM", predictions ,dataset_test_y)
    print("Execution time: {}s".format(time_lapsed))


    
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
    
    clf = GridSearchCV(skadapt.MLTSVM(), parameters, scoring=make_scorer(metrics.hamming_loss,greater_is_better=False), n_jobs=4)
    clf.fit(csr_matrix(dataset_train_x),csr_matrix(dataset_train_y))
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

datasets = ['emotions', 'bookmarks', 'yeast']
numinputoutputs = [[72, 6], [2150, 208], [103, 14]]

for (dataset, [num_in, num_out]) in zip(datasets, numinputoutputs):
    print("Dataset name:", dataset)
    #import dataset
    ds = pd.read_csv(os.path.join("datasets", dataset+".csv"))
    ds_length = len(ds.index)
    #train test split
    train, test = train_test_split(ds, random_state=42, test_size=0.20, shuffle=True)
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

    Util_ClassifierMethods(x_scaled_train, y_train, x_scaled_test, y_test)