from chainer import datasets
from chainer.dataset import concat_examples
import numpy as np

# https://github.com/rizaudo/chainer-Fashion-MNIST
from fashionmnist import get_fmnist

class Data(object):
    
    def __init__(self, data_name):
        # get data from chainer
        # images are normalized to [0.0, 1.0]
        if data_name == 'mnist':
            train_tuple, test_tuple = datasets.get_mnist(ndim=3)
        elif data_name == 'fmnist':
            train_tuple, test_tuple = get_fmnist(withlabel=True, ndim=3, scale=1.0)
        elif data_name == 'cifar10':
            train_tuple, test_tuple = datasets.get_cifar10()
        else:
            raise ValueError('Invalid data')
            
        self.data_name = data_name
        
        # preprocess
        # convert to array
        train_image, train_label = concat_examples(train_tuple)
        test_image, test_label = concat_examples(test_tuple)
        
        # set images to [-0.5, 0.5] 
        self.train_image = np.array(train_image, dtype=np.float32) - 0.5
        self.train_label = np.array(train_label, dtype=np.int32)
        self.test_image = np.array(test_image, dtype=np.float32) - 0.5
        self.test_label = np.array(test_label, dtype=np.int32)
        
        # re-convert to TupleDataset
        self.train_tuple = datasets.TupleDataset(self.train_image, self.train_label)
        self.test_tuple = datasets.TupleDataset(self.test_image, self.test_label)
