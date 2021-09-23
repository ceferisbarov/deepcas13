# DeepCas13 #

A deep learning model to predict the CRISPR-Cas13d sgRNA on-target activity with high accuracy from sgRNA sequences and RNA secondary structures.

DeepCas13 uses convolutional recurrent neural network (CRNN) to extract spatial-temporal features for both sequence and secondary structure of a specific sgRNA and then predicts the efficiency by a fully connected neural network (FCNN).

## How to run this source code ##

### Install dependencies ###

1. Create a conda environment

```
	conda create --name deepcas13 python=3.6
```
	
2. Activate the environment

```
	conda activate deepcas13
```

3. Install the dependencies

```
    conda install pandas numpy=1.19.1 seaborn scikit-learn 
	
	conda install -c conda-forge jupyterlab
	
    conda install -c bioconda viennarna
	
	conda install -c conda-forge tensorflow=2.2
	
	
```

### Run the demo ###

Here, we provie a jupyter notebook that show how to run our model. We prepared the train and test files in the data folder and you just need to run the notebook file.

In this notebook file, you can train the model and predict Deep Score for the test dataset. You can also evalute the preformance by ploting ROC curve.

1. Activate the conda environment

```
	conda activate deepcas13
```

2. Start Jupyter Notebook

```
	jupyter notebook
```

3. Run the demo step by step

### About the output ###

The output of our model is a score, named Deep Score. 

Deep Score, which ranges from 0 to 1, is used to indicate the on-target efficiency of a specific sgRNA. The higher the Deep Score, the more likely sgRNA is to be effective.

### Authors ###

* Xiaolong Cheng: xcheng@childrensnational.org