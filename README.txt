# CZ4041 group assignment
## 1. Project overview
In this project, we have opted to implement the multi-label classification task. We will be using "emotions", "yeast" and "bookmark" datasets to assess the performance of 
the respective classifiers. The datasets are obtained from [https://sci2s.ugr.es/keel/multilabel.php].  
There are 5 python files submitted.  
	1. EDA_emotions.ipynb
	2. EDA.ipynb
	3. EDA.py
	4. CZ4041_Assignment1_BaseImplementations.py
	5. neuralnet.py

## 2. Tools and envrionment
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


## 3. Implementation
### 3.1 Exploratory Data Analysis
In this project, the EDA performed was simply to determine the pre-processing techniques required to be performed before feeding the data to respective classifiers. 
In "EDA.py" file, within main() function, adjust the "perform_EDA = dataset[0]" to perform EDA on one of the datasets at any one time. The 2 lines of codes are as follows.

	dataset = ["emotions", "yeast","bookmarks"]
	perform_EDA = dataset[0]

Index 0 corresponds to "emotions dataset, index 1 corresponds to "yeast" dataset and index 2 corresponds to "bookmarks" dataset. Adjust the index accordingly to 
perform EDA on the desired dataset. Once selected, simply run the program in desired IDE. For better visualization, it is recommended to run the EDA.ipynb. 
For a more detailed explanation on how to extract information from respective boxplots and data distribution plots, refer to EDA_emotions.ipynb.


### 3.2 Metric used to measure performance
In this project, the following 3 metrics were are used to assess the performance of the respective classifiers.

	1. Hamming loss _ measures the fraction of labels that are incorrectly predicted, i.e., the fraction of the wrong labels to the total number of labels.
			_ the lower the loss, the better the performance of the model

	2. Accuracy	_ measures the exact match accuracy ratio
			_ eg. if the actual label vector is [0 1 1 0], the prediction must be [0 1 1 0] as well to consider 1 count of correct prediction
			_ the higher the accuracy, the better the performance of the model

	3. Log loss	_ quantifies the accuracy of a classifier by penalising false classifications
			_ the lower the loss, the better the performance of the model	
				 



### 3.3 Proposed model
In this project, the proposed method is a shallow neural network. The configuration is as follows.

	1. Input layer 		_ the number of neurons in input layer equals to the number of features in a dataset.
		       		_ eg. for "emotions" dataset which has 72 features, the number of neurons in input layer will be 72 with 1-to-1 correspondence
				_ "relu" activation function is used

	2. 3 hidden layers 	_ the number of hidden neurons are 50,100 and 50 respectively
				_ "relu" activation function is used
	
	3. output layer		_ number of neurons in output layer equals to number of labels in a datset
				_ eg. for "emotions" dataset which has 6 labels, the number of neurons in output layer will be 6 with 1-to-1 correspondence
				_ input vector will be assigned the corresponding labels of the activated neurons at the output layer
				_ "sigmoid" activation function is used
 
#### 3.3.1 Dimensionality reduction_PCA 
In this project, feature selection technique "Variance threshold" and feature extraction technique "Principal Component Analysis(PCA)" were implemented to assess 
which dimnensionality reduction technique is to be used. "Variance threshold" removes features that have lesser variance than the pre-set threshold while PCA uses
statistical analysis to derives a totally new dataset. This newly derived dataset will have number of features depending on how much information it was set to retain
from the original dataset. The configuration that was tested was "variance threshold of 25th percentile" against 99% information retainment for PCA.
In the end, PCA is selected as it performs significantly better. In line 333 of neuralnet.py, 
		"pca = PCA(0.99)" 
specify that PCA should derives a new dataset that retains 99% information of the original dataset. 
	- This parameter can be adjusted between 0-1 according to the user. 
	- If a user specifies smaller percentage of information retainment, the newly derived dataset will have less number of features but is less accurate.

#### 3.3.2 Optimiser
In this process, a series if tests were conducted to identify the optimiser that is best suited for our multi-label classification task. The 3 optimisers tested were

	1. Stochastic Gradient Descent (SGD)			- learning rate 0.001

	2. Stochastic Gradient Descent (SGD) with momentum	- learning rate 0.001
								- Momentum 0.9
								- Decay  1e^-6

	3. Adaptive Moment Estimation (Adam)			- learning rate 0.001
								- beta_1 0.9
								- beta_2 0.999 

In line 301 of "neuralnet.py", the code to initialise the SGD with momentum is as follows.
	optimum_optimiser = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
The values can be adjusted based on user preference and tests could be rerun, however, to achieve fair comparison across the 3 optimisers, it is recommended that the values 
of the common hyperparameters are kept the same. The values of the hyperparameters used in this project are as shown above. 3 different models are built using the respective optimisers
and their individual performance was assessed and the following results were obtained.
	
	1. SGD			- performs the worst out of the 3
	2. SGD with momentum	- performs consistently well on both training and testing dataset
	3. Adam			- performs the best on training dataset but poor on unseen test data

(refer to the visualization plots for further confirmation)  
Therefore, in this project, SGD with momentum is used for further tuning of other components.

#### 3.3.3 Loss function 
In this project, 2 loss functions were implemented.
	1. Binary Cross Entropy - Binary Cross Entropy is a simple log loss for each output label	
	2. Custom Propensity Loss - Our custom loss function consists of a few components, a hamming loss at the core, a binary selector function, a propensity multiplier, and a hinge filter. 
	
The hamming loss simply gives the squared difference of the predicted value and the true value of each label. This squared difference is then passed through a binary selector, which only returns 0 for true positive predictions. 
This is done as the number of labels which are 0 far outweigh the number of labels which are 1 for all of our chosen datasets. For any combination other than true positive, the full squared difference will be propagated.
After the selector, the loss goes through a propensity multiplier , which applies a different multiplier for each label, based on the number of times that label occurs, or the propensity for that label to occur.
Finally, a hinge filter is applied as an overall multiplier for all the labels in each data point. The hinge filter will allow the loss to propagate inversely proportionately to the number of correctly classified labels (both true positive and true negatives).
That is, if all labels are classified correctly, then the resulting loss will be 0, if 50% of the labels are classified correctly, the loss will be multiplied by 0.5.
		



### 3.4 Base Implementation
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
	2. Multinomial Naive Bayes**
	3. C-Support Vector Classification/ Linear Support Vector Machine*

*Linear Support Vector Machine was used for Bookmarks dataset to reduce training time, instead of C-Support Vector Classification for Random K-labelsets methods.
**Base classifier Multinomial Naive Bayes requires input to be non-negative ("x" train/test splits). Do take note to scale or check for min before running large datasets to avoid warnings. Example can be shown in Util_ClassifierMethodsBookmarks()

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


## 4. Experiment Settings
### 4.1 Approach
=======
## Experiment Settings
### Approach
As mentioned in Base Implmentation, the replication of results can be done through Util_ClassifierMethods method. The experiment settings approach used are as follow:
	1. 80-20 split of dataset for train/ test split.
	2. Scaling is done in the range of 0-1.
	3. Optimal parameter tuning through relevant GridSearchCV methods.
	4. Using found parameter as input to relevant classifier methods.


### 4.2 Steps
=======
### Steps
	1. For Emotions dataset 
		1a. Specify the path to Emotions dataset
		1b. Run LoadEmotionsDataset with path as input. Output of x_train, y_train, x_test and y_train respectively
		1c. Run Util_ClassifierMethods with the output from LoadEmotionsDataset in the same sequence
	2. For Yeast dataset
		2a. Specify the path to Yeast dataset
		2b. Run LoadYeastDatasetwith with path as input. Output of x_train, y_train, x_test and y_train respectively
		2c. Run Util_ClassifierMethods with the output from LoadEmotionsDataset in the same sequence
	3. For Bookmarks dataset**
		3a. Specify the path to Bookmarks dataset
		3b. Run LoadBookmarksDataset with path as input. Output of x_train, y_train, x_test and y_train respectively
		3c. Run Util_ClassifierMethodsBookmarks with the output from LoadBookmarksDataset in the same sequence*
	*As mentioned above, it is recommended to run each method individually to avoid long waiting times, you may just copy the section out and run it.
	**For base implementation methods other than BR, CC and LP. To reduce complexity and training time needed, only 10% of the sample is used. 