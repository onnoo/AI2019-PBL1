# Manual

## Requirement

Training/test files are not included in this repository.
Following files must be included in current directory, which you can get in [this link](http://yann.lecun.com/exdb/mnist/).

* train-images.idx3-ubyte
* train-labels.idx1-ubyte
* t10k-images.idx3-ubyte
* t10k-labels.idx1-ubyte



## Module Description

### main.py

* Run this for watching result
* Modify this for changing data size and parameter tuning

### svm_model.py

* Encapsulated SVM model for training, prediction, and evaluation
* Doesn't have to be modified

### loader.py

* Return actually trainable/testable dataset from reader.py
* Scale by standardization(axis=1)
* Doesn't have to be modified

### reader.py

* Return raw data from file
* File names are hard-coded(Can be modified)