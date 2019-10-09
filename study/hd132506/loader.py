import reader
import numpy as np

"""
Retun actually trainable/testable dataset using reader function
dta_size: Number of training data. Maximum: 10,000
"""
def load(dataset="training", data_size=10000):
    if dataset == "training":
        data_size = min(data_size, 10000) 
    elif dataset == "testing":
        data_size = min(data_size, 60000)
    else:
        print("dataset parameter must be \"training\" or \"testing\"")
        return

    ztransform = lambda arr : (arr - np.mean(arr))/np.std(arr)
    flatten_imgs = lambda dset : \
        np.array([ztransform(data[1].reshape(-1)) for data in dset], dtype='f')
    flatten_labels = lambda dset : np.array([data[0] for data in dset])

    dset = list(reader.read(dataset))[:data_size]

    return (flatten_imgs(dset), flatten_labels(dset))