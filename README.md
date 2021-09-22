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


* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact