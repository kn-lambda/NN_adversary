import numpy as np
from chainer import cuda
from chainer import iterators



########################################################
# functions ############################################

def get_predictions(model, images, batch_size=400):
    """
    This function calculates most probable and least likely labels. 
    
    Args:
        model (classifer.ClassiferNN)  :  classifer which predict labels
        images (numpy or cupy ndarray) :  input images
        batch_size (int)               :  number of images once fed into the model
   
   Returns:
        predicted probabilities, most probable labels, most probable probabilities, 
                                 least likely labels, least likely pribabilities
                                 (numpy array or array)
    """

    # chech if the model is in cpu or gpu
    gen = model.params()
    xp = cuda.get_array_module(next(gen)) # numpy or cupy

    xp_images = xp.array(images)
    iterator = iterators.SerialIterator(xp_images, batch_size=batch_size, shuffle=False, repeat=False)
 
    pred_labels = xp.empty((0, model.n_classes), dtype=xp.float32)
    pred_labels = xp.empty((0, model.n_classes), dtype=xp.float32)
    
    y = []
    
    # loops until iterator gets end
    while iterator.epoch == 0:
        x = iterator.next()
        y.append(model.predict_proba(x).data)
        
    probs = xp.concatenate(y, axis=0)
    probs = cuda.to_cpu(probs)
    
    predicted_labels = np.argmax(probs, axis=1)
    predicted_probs = np.max(probs, axis=1)
    
    least_likely_labels = np.argmin(probs, axis=1)
    least_likely_probs = np.min(probs, axis=1) 
    
    return probs, predicted_labels, predicted_probs, least_likely_labels, least_likely_probs
    
