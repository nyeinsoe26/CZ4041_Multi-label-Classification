# CZ4041 group assignment
## Project overview
In this project, we have opted to implement the multi-label classification task. We will be using "emotions", "yeast" and "bookmark" datasets to assess the performance of 
the respective classifiers. The datasets are obtained from [https://sci2s.ugr.es/keel/multilabel.php].  
There are 5 python files submitted.  
	1. EDA_emotions.ipynb
	2. EDA.ipynb
	3. EDA.py
	4. CZ4041_Assignment1_BaseImplementations.py
	5. neuralnet.py

## Tools and envrionment
The codes can generally be run in any operating system, however, Ubuntu 18.04 or 17.04 is recommended. Additionally, the following libraries are required.  
	1. numpy, ver ≥ 1.18.1
	2. pandas, ver ≥ 1.0.1
	3. matplotlib, ver ≥ 3.2.1
	4. tqdm, ver ≥ 4.45.0
	5. sklearn (scikit learn), ver 0.22.2 
	6. skmultilearn (scikit-multilearn), ver 0.2.0
	7. tensorflow, ver ≥ 1.14.0
	8. keras, ver ≥ 2.3.1 
Packages can be installed by entering "pip install [package_name]" at the terminal without the enclosing square brackets. For examply, to install numpy package, 
the following could be entered at the terminal, "pip install numpy".


## Implementation
### Exploratory Data Analysis
In this project, the EDA performed was simply to determine the pre-processing techniques required to be performed before feeding the data to respective classifiers. 
In "EDA.py" file, within main() function, adjust the "perform_EDA = dataset[0]" to perform EDA on one of the datasets at any one time. The 2 lines of codes are as follows.

	dataset = ["emotions", "yeast","bookmarks"]
	perform_EDA = dataset[0]

Index 0 corresponds to "emotions dataset, index 1 corresponds to "yeast" dataset and index 2 corresponds to "bookmarks" dataset. Adjust the index accordingly to 
perform EDA on the desired dataset. Once selected, simply run the program in desired IDE. For better visualization, it is recommended to run the EDA.ipynb. 
For a more detailed explanation on how to extract information from respective boxplots and data distribution plots, refer to EDA_emotions.ipynb.

### Proposed model




### Base Implementation
Variable name used in the subsequent sections
	a. dataset_train_x - attributes
	b. dataset_train_y - labels
	c. dataset_test_x - attributes
	d. dataset_test_y - labels
	e. base_classif - base classifier for single label classifying
	f. title - string object of method for easy identification i.e. "Binary Relevance"
	g. num_neighbours - number of neighbours
	h. smoothing_param - smoothing parameter
	i. num_vigilance - vigilance threshold value
	j. num_threshold - threshold value
	k. num_labels - number of labels
	l. num_models - number of models 
	m. nc_k - c_k emphirical penalty value
	n. omega - sor_omega 

In this project, the following state-of-the-arts (SOTA) methods were implemented.
	1. Binary Relevance : BinaryRelevance(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title)
	2. Label Powerset : LabelPowerset(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title)
	3. Classifier Chain : ClassifierChain(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, base_classif, title)
	4. Random K-labelsets : RAkELd(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,base_clasif,num_labels) / RAkELO(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,base_clasif,num_labels, num_models)
	5. Binary Relevance k-nearest Neighbour : BRkNNa(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, num_neighbours) / BRkNNb(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, num_neighbours)
	6. Multilabel K-nearest Neighbour : MLkNN(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_neighbours, smoothing_param)
	7. Multilabel Adaptive Resonance Associative Map : MLARAM(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,num_vigilance, num_threshold)
	8. Multilabel Twin Support Vector Machines : TwinMLSVM(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,nc_k, omega)

For the base classifiers that will be used by methods of Problem Transformation (PT) domain, the following 3 classifier methods were used.
	1. Gaussian Naive Bayes
	2. Multinomial Naive Bayes
	3. C-Support Vector Classification/ Linear Support Vector Machine*

*Linear Support Vector Machine was used for Bookmarks dataset to reduce training time, instead of C-Support Vector Classification for Random K-labelsets methods.

The relevant GridSearchCV used for optimal parameter finding are as follow:
	1. GridSearchCV_baseRakel(classif, dataset_train_x, dataset_train_y) - for Rakel classifier methods
	2. FindBestSVCParams(skpt.BinaryRelevance(),dataset_train_x,dataset_train_y) - for SVC base classifier
	3. FindBestMNBParams(skpt.BinaryRelevance(),dataset_train_x,dataset_train_y) - for MultinomialNB base classifier
	4. FindBestVT(dataset_train_x,dataset_train_y) - for MLARAM classifier
	5. FindBestK(classif, dataset_train_x, dataset_train_y) - for BRkNN, MLkNN methods
	6. FindCKParam(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y) - for MLTSVM method
The return values for these GridSearchCV methods are a dictionary object which contains the classifier's instance details that has the lowest hamming loss score. To access the parameters i.e. k = dict_res['k']

For simplification, these methods and its relevant parameter tuning steps can be found in the python file CZ4041_Assignment1_BaseImplementations.py. To replicate results, the following method Util_ClassifierMethods method can be used. It is however not recommended to use with Bookmarks dataset, for Bookmarks dataset, it is recommended to extract the relevant methods and run it individually. Each SOTA method implemented will have its own individual method implementation to allow users to perform individual testing. Transformation of dataset to either sparse matrix or array will be performed by the method itself. Most of the SOTA methods requires input of dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y and relevant key parameters. 

Prior to calling the method, each dataset has its individual method to perform scaling, train-test data split. Util_ClassifierMethods method contains a prior GridSearchCV method to find optimal parameters which will subsequently be used as the input for each SOTA method implemented. 

## Experiment Settings
### Approach
As mentioned in Base Implmentation, the replication of results can be done through Util_ClassifierMethods method. The experiment settings approach used are as follow:
	1. 80-20 split of dataset for train/ test split.
	2. Scaling is done in the range of 0-1.
	3. Optimal parameter tuning through relevant GridSearchCV methods.
	4. Using found parameter as input to relevant classifier methods.

### Steps
	1. For Emotions dataset 
		1a. Specify the path to Emotions dataset
		1b. Run LoadEmotionsDataset with path as input. Output of x_train, y_train, x_test and y_train respectively
		1c. Run Util_ClassifierMethods with the output from LoadEmotionsDataset in the same sequence
	2. For Yeast dataset
		2a. Specify the path to Yeast dataset
		2b. Run LoadYeastDatasetwith with path as input. Output of x_train, y_train, x_test and y_train respectively
		2c. Run Util_ClassifierMethods with the output from LoadEmotionsDataset in the same sequence
	3. For Bookmarks dataset
		3a. Specify the path to Bookmarks dataset
		3b. Run LoadBookmarksDataset with path as input. Output of x_train, y_train, x_test and y_train respectively
		3c. Run Util_ClassifierMethodsBookmarks with the output from LoadBookmarksDataset in the same sequence*
	*As mentioned above, it is recommended to run each method individually to avoid long waiting times, you may just copy the section out and run it.