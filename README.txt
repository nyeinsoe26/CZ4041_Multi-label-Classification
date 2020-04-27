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
