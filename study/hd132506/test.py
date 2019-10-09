import reader
import loader
import svm_model
import numpy as np

"""
Unit test each module 
"""

# loader returns standardized training/test dataset
size = 10
imgs, labels = loader.load(data_size=size)
assert len(imgs) == size and len(labels) == size, \
    "loader can't load correct size of data"

for i in range(size):
    img, label = imgs[i], labels[i]
    assert len(img) == 28*28, "Wrong image shape, {} != {}".format(len(img), 28*28)
    assert abs(np.mean(img)) < 0.01 and abs(np.std(img)-1) < 0.01, \
        "Image isn't standardized"
    assert isinstance(label, np.int8), "Type of label isn't np.int8"
    assert label >= 0 and label <= 9, "label isn't between 0 and 9" 
print('loader test success')

