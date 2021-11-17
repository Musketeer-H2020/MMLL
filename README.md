# MMLL: Musketeer Machine Learning Library

## Installing 

`pip install git+https://github.com/Musketeer-H2020/MMLL.git`

## Dependencies

* `six==1.15.0`
* `transitions==0.6.9`
* `pyparsing==2.3`
* `pygraphviz==1.5`
* `numpy==1.19.2`
* `sklearn==0.0`
* `scikit-learn`
* `matplotlib`
* `tensorflow==2.4.1`
* `phe==1.4.0`
* `dill==0.3.2`
* `tqdm==4.61.0`
* `pympler==0.8`
* `torchvision==0.8.1`
* `pillow==7.2.0`
* `skl2onnx==1.8.0`
* `sklearn2pmml==0.71.1`
* `tf2onnx==1.8.5`

## Content

The library supports the following Privacy Operation Modes (POMs) and models:

### POM1:

* Kmeans
* Neural networks
* Support Vector Machine
* Federated Budget Support Vector Machine
* Distributed Support Vector Machine

### POM2:

* Kmeans
* Neural networks
* Support Vector Machine
* Federated Budget Support Vector Machine

### POM3:

* Kmeans
* Neural networks
* Support Vector Machine
* Federated Budget Support Vector Machine

### POM4: 

* Linear Regression
* Logistic Classifier
* Multiclass Logistic Classifier
* Clustering Kmeans
* Kernel Regression
* Budget Distributed Support Vector Machine

### POM5: 

* Linear Regression
* Logistic Classifier
* Multiclass Logistic Classifier
* Clustering Kmeans
* Kernel Regression
* Budget Distributed Support Vector Machine
* Multiclass Budget Distributed Support Vector Machine

## POM6: 

* Ridge Regression
* Logistic Classifier
* Multiclass Logistic Classifier
* Clustering Kmeans
* Kernel Regression
* Budget Distributed Support Vector Machine
* Multiclass Budget Distributed Support Vector Machine

## Usage 

Please, visit the following git repository that contains a collection of demos that illustrate the usage of this library:

[MMLL-demo](https://github.com/Musketeer-H2020/MMLL-demo)


## Installation using Anaconda (Windows and Linux)

0. Requisites:
  - Conda: https://www.anaconda.com
  - Git client: https://git-scm.com/
  
1. Create conda environment:
```
conda create -n mmll python=3.7 pip
```
2. Activate environment:
```
conda activate mmll
```
3. Install dependencies:
```
pip install git+https://github.com/Musketeer-H2020/MMLL.git
```

## Installation using venv in Linux

Alternatively, you can use Python venv built-in module to create a working environment.

1. Install Python 3.7:
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
sudo apt-get install python3.7-venv -y
```
2. Update pip:
```
python3.7 -m pip install --upgrade pip
```
3. Create virtual environment in your home folder:
```
cd ~
python3.7 -m venv mmll
```
Please note: "mmll" is the environment name. You can use whatever name you prefer.

4. Activate environment and install auxiliary libraries:
```
source ~/mmll/bin/activate
sudo apt-get install python3.7-dev -y
```
6. Install project library:
```
pip install git+https://github.com/Musketeer-H2020/MMLL.git
```



## Acknowledgement 

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 824988. https://musketeer.eu/

![](./EU.png)
