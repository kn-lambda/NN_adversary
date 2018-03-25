"""
This script trains the following classifer models.

1. base model
   normally trained model
2. teacher model
   used as a teacher model when distillation is performed
3. distilled model
   (trined by distilation of the above teacher model)
   used as a defense for adversarial images
   the original idea is in the paper:
   "Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks"
   by Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami, 2015

Data sets are assumed to be the followings.

1. MNIST
2. FASION MNIST
3. CIFAR10
"""

import chainer
from chainer import training, datasets, optimizers, iterators, reporter, serializers
from chainer.training import extensions
from chainer import cuda
from chainer.dataset import concat_examples

import os
import sys
import numpy as np

from modules.data import Data
from modules.classifer import layer_params, ClassiferNN


# use GPU if possible
uses_device = 0

if uses_device >= 0:
    chainer.cuda.get_device_from_id(uses_device).use()
    chainer.cuda.check_cuda_available()
    import cupy as xp
else:
    xp = np


########################################################################################
## functions ###########################################################################

def train_model(save_name, model, loss_func, train_data, test_data=None, n_epochs=50, batch_size=128):
    """ This function train input model.
        
    Args:
        
        save_name (str)           : used in file names when loging and saving 
        model (ClassiferNN)       : model to be trained
        loss_func (function)      : loss function which back propagated
        train_data (TupleDataset) : (image, true label) for training
        test_data  (TupleDataset) : (image, true label) for test
                                     if None, test is ignored
        n_epochs (int)            : number of epochs when training
        batch_size (int)          : batch data size when training
        
    Returns:
        trained model (ClassiferNN)
        (Trained model is also saved.) 
    """
        
    # set data to iterators
    train_iter = iterators.SerialIterator(train_data, batch_size, shuffle=True)
    
    if test_data is not None:
        test_iter = iterators.SerialIterator(test_data, batch_size, repeat=False)
    
    
    # set optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model) 
   
    # set updater
    updater = training.StandardUpdater(train_iter, optimizer, device=uses_device, loss_func=loss_func)
    
    # set trainer
    trainer = training.Trainer(updater, (n_epochs, 'epoch'), out='training_report')
    
    # set extensions
    log_file = '{0}.log'.format(save_name)
    trainer.extend(extensions.LogReport(log_name=log_file))
    trainer.extend(extensions.ProgressBar())
    
    if test_data is not None:
        trainer.extend(extensions.Evaluator(test_iter, model, device=uses_device, eval_func=loss_func))
    
    # run training
    trainer.run()
    
    # save trained model
    save_dir = 'trained_models'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    save_file = '{0}.hdf5'.format(save_name)
    save_path = os.path.join(save_dir, save_file)
    serializers.save_hdf5(save_path, model)
    
    return model



def get_soft_label(model, image):
    """This function calculates soft labelings of input image data.
    
    Args:
        model (ClassiferNN)   : model which calculates the soft labelings
        image (numpy ndarray) : image data which is fed into the model
        
    Return:
        TupleDataset (image, soft label) located in GPU if possible
    """
    
    xp_image = xp.array(image)
    iterator = iterators.SerialIterator(xp_image, batch_size=400, shuffle=False, repeat=False)
 
    soft_label = xp.empty((0, model.n_classes), dtype=xp.float32)
    y = []
    
    # loops until iterator gets end
    while iterator.epoch == 0:
        x = iterator.next()
        y.append(model.predict_proba(x).data)
        
    soft_label = xp.concatenate(y, axis=0)
    tuple_data = datasets.TupleDataset(xp_image, soft_label) 
    
    return tuple_data



def estimete_accuracy(model, tuple_data):
    """This function estimates acuracy of the model on the given dataset.
    
    Args:
        model (ClassiferNN)       : model tested
        tuple_data (TupleDataset) : (image, true_label) fed into the model
        
    Return:
        accuracy (numpy.float) : the ratio of the number of datum which correctly classified
    """
    
    image, true_label = concat_examples(tuple_data)
    image = xp.array(image, dtype=xp.float32)
    
    iterator = iterators.SerialIterator(image, batch_size=400, shuffle=False, repeat=False)
 
    soft_label = xp.empty((0, 10), dtype=xp.float32)
    y = []
    
    # loops until iterator gets end
    while iterator.epoch == 0:
        x = iterator.next()
        y.append(model.predict_proba(x).data)
        
    soft_label = xp.concatenate(y, axis=0)
    hard_label = xp.argmax(soft_label, axis=1)
    
    true_label = cuda.to_cpu(true_label).flatten().astype(np.int32)
    hard_label = cuda.to_cpu(hard_label).flatten().astype(np.int32)
    
    n_all = len(true_label)
    n_correct = np.sum(true_label == hard_label)
    
    ratio = n_correct / n_all
    
    return ratio



########################################################################################
## main ################################################################################

if __name__ == '__main__':

    #set data neme
    args = sys.argv
    
    if len(args) == 2:
        data_name = args[1]
    else:
        print('enter data name which you use ---- mnist or cifar10')
        print('python <this script> <data name>')
        quit()
   
    print("------------------------")
    print("{}".format(data_name))
    print("------------------------")

 
    # load data
    data = Data(data_name)
    
    
    #####################################################################
    ## make three models base, teacher and distilled
    #####################################################################
    
    ######################################################
    ## 1. base model which is for merely comparison
    ##    Temperature = 1

    print("")
    print("-----------------------------")
    print("making base model ...")
    print("-----------------------------")
    print("")

    model = ClassiferNN(layer_params[data_name], T=1.0)
    if uses_device >= 0:
        model.to_gpu()
    
    save_name = '{0}_base'.format(data_name)
    base_model = train_model(save_name, model, model.hard_cross_entropy_loss, data.train_tuple, data.test_tuple)

    print("Done.")    
    
    ######################################################
    ## 2. teacher model
    ##    Temperature = 100

    print("")
    print("-----------------------------")
    print("making teacher model ...")
    print("-----------------------------")
    print("")

    model = ClassiferNN(layer_params[data_name], T=100.0)
    if uses_device >= 0:
        model.to_gpu()
    
    save_name = '{0}_teacher'.format(data_name)
    teacher_model = train_model(save_name, model, model.hard_cross_entropy_loss, data.train_tuple, data.test_tuple)
    
    print("Done.")    
    
    ######################################################
    ## 3. distilled model
    ##    Temperature = 100 in learning process
    
    print("")
    print("-----------------------------")
    print("making distilled model ...")
    print("-----------------------------")
    print("")
    
    # prepare soft labeling used in distillation learning
    distilled_data = get_soft_label(model, data.train_image)
    # train model
    model = ClassiferNN(layer_params[data_name], T=100.0)
    if uses_device >= 0:
        model.to_gpu() 
    
    save_name = '{0}_distilled'.format(data_name)
    distilled_model = train_model(save_name, model, model.soft_cross_entropy_loss, distilled_data)
    
    # reduce Temperature = 1 in inference
    distilled_model.T = 1.0
    
    print("Done.") 
   
 
    #####################################################################
    ## estimate accuracy of the above models
    #####################################################################

    # not in train mode
    chainer.config.train = False
    
    # for base-line model
    bea_train = estimete_accuracy(base_model, data.train_tuple)
    bea_test = estimete_accuracy(base_model, data.test_tuple)
    
    # for teacher model
    tea_train = estimete_accuracy(teacher_model, data.train_tuple)
    tea_test = estimete_accuracy(teacher_model, data.test_tuple)
    
    # for distilled model
    dis_train = estimete_accuracy(distilled_model, data.train_tuple)
    dis_test = estimete_accuracy(distilled_model, data.test_tuple)
    
    # print accuracy
    
    print("")
    print('=================================')
    print('Accuracy for {0}'.format(data_name))
    print('=================================')
    print('model     : train, test')
    print('---------------------------------')
    print('base      : {0:.4f}, {1:.4f}'.format(bea_train, bea_test))
    print('teacher   : {0:.4f}, {1:.4f}'.format(tea_train, tea_test))
    print('distilled : {0:.4f}, {1:.4f}'.format(dis_train, dis_test))
    print('=================================')

