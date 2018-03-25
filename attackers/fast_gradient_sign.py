import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators
from chainer import cuda
from chainer import Variable

from PIL import Image
import os
import sys
import gc
from datetime import datetime

import numpy as np


#########################################################################
# generator of adversarial images #######################################

class FastGradientSign(object):
    """
    This object generates adversarial images. 
    
    Used Algorithm is 'Fast Gradient Sign' including iterative method:
    
    "Explaining and Harnessing Adversarial Examples" 
     by Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy, 2014
     
    "Adversarial examples in the physical world"
     by Alexey Kurakin, Ian Goodfellow, Samy Bengio, 2017
    
    """
    
    def __init__(self, model, images, targets, batch_size=200, num_iterations = 10000,
                 learning_rate = 0.005, clip_eps = 0.01, early_abort = True):
        """
        Args:
            model (ClassiferNN)     : adversarial images are genarated with respect to this model
            images (numpy ndarray)  : original input image
            targets (numpy ndarray) : target labels toward which adverarial images are generated
            batch size (int)        : batch size
            num_iterations(int)     : number of iterations when gradient decent optimization
            learning_rate(int)      : larning rate called epsilon or alpha in the papers
            clip_eps(float)         : upper bound on pixel wise absolute value of the adversarial perturbation
            early_abort(bool)       : if improvement stops, braeak iteration
        """
        
        if images.shape[0] != targets.shape[0]:
            raise Exception('number of input images and labels does not match')
        
        # check if the model is in cpu or gpu
        gen = model.params()
        self.xp = cuda.get_array_module(next(gen)) # numpy or cupy

        self.model = model
        self.org_images = images
        self.targets = targets 
        self.batch_size = batch_size
        
        self.num_iterations = num_iterations
        self.lr = learning_rate
        self.clip_eps = clip_eps
        self.early_abort = early_abort
        
        # storages for the results
        # predicted labels and probabilities of the generated adversarial images
        self.adv_images = np.empty((0,) + self.org_images[0].shape, dtype=np.float32)
        self.adv_labels = np.empty((0,) + self.targets[0].shape, dtype=np.int32)
        self.adv_probs = np.empty((0,) + self.targets[0].shape, dtype=np.float32)
        self.adv_l2_squared = np.empty((0,) + self.targets[0].shape, dtype=np.float32)
        
        # storage for predicted labels and probabilities of the original images
        self.org_labels = np.empty((0,) + self.targets[0].shape, dtype=np.int32)
        self.org_probs = np.empty((0,) + self.targets[0].shape, dtype=np.float32)
        
        # iterators for batch
        self.img_iter = iterators.SerialIterator(self.org_images, batch_size, shuffle=False, repeat=False)
        self.targ_iter = iterators.SerialIterator(self.targets, batch_size, shuffle=False, repeat=False)
        
   
    ## loss functions ######################
    
    def loss(self):
        """
        ordinary cross entropy
        """
        return self.model.hard_cross_entropy_loss(self.batch_adv, self.batch_targ)
    
    
    def l2_loss(self):
        diff = self.batch_adv - self.batch_org
        return F.batch_l2_norm_squared(diff)
    
    
    ## main ################################
    
    def run(self):
        """
        Run the full process to generate adversarial images.
        The result is set to the following attributes.
        =====================================================================================================
        adv_images (numpy float ndarray)   : generated adversarial images
        adv_labels (numpy int array)       : predicted labels of the adversarial images
        adv_probs (numpy float array)      : probabilties of the predicted labels
        adv_l2_squared (numpy float array) : l2 squared distance between original and adversarial image
        -----------------------------------------------------------------------------------------------------
        org_images (numpy float ndarray)   : original input images
        org_labels (numpy int array)       : predicted labels of the original images
        org_probs (numpy float array)      : probabilties of the predicted labels o
        =====================================================================================================
        If adversarial image is not generated, label = -1 and image is filled with '0'.
        """

        n_data = self.org_images.shape[0]
        n_batch = int(np.ceil(n_data / self.batch_size))
        
        # loop for batch
        loop_cnt_batch = 1
        while self.img_iter.epoch == 0 and self.targ_iter.epoch == 0:
            
            self._prepare_batch()
            self._do_iteration()
            self._store_results_of_this_batch()
            gc.collect()
                
            print('batch: {0}/{1}  ----  finished at {2}'.format(
                    loop_cnt_batch, n_batch, 
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
                  )
            sys.stdout.flush()
            loop_cnt_batch += 1
            
        # aggregate results and print
        n_failure = np.sum(self.targets != self.adv_labels)
        ratio_failure = 1.0 * n_failure / n_data
        n_success = n_data - n_failure
        ratio_success = 1.0 * n_success / n_data
        mean_l2sq = np.mean(self.adv_l2_squared[self.adv_labels != -1])
        
        print("")
        print("============================================")
        print("number of input data : {0}".format(n_data))
        print("number of success    : {0} ({1:.2f} %)".format(n_success, 100*ratio_success))
        print("number of failure    : {0} ({1:.2f} %)".format(n_failure, 100*ratio_failure))
        print("--------------------------------------------")
        print("mean squared loss    : {0:.4f}".format(mean_l2sq))
        print("============================================")

        return n_data, n_success, ratio_success, n_failure, ratio_failure, mean_l2sq
            
            
    def _prepare_batch(self):
        """
        Set the next batch data, and initialize several states.
        """
        xp = self.xp
        if self.img_iter.epoch != 0 or self.targ_iter.epoch != 0:
            return
        
        # get the next batch data
        batch_org = self.img_iter.next() 
        batch_targ = self.targ_iter.next()
        # to gpu if possible
        self.batch_org = xp.array(batch_org, dtype=xp.float32)
        self.batch_targ = xp.array(batch_targ, dtype=xp.int32)
        # reset adversarial image
        self.batch_adv = Variable(self.batch_org.copy())
        
        # reset storages for the succeeded adversaries in this batch
        self.batch_success_adv = xp.zeros_like(self.batch_org, dtype=xp.float32)
        self.batch_success_l2sq = xp.ones_like(self.batch_targ, dtype=xp.float32) * 1e10
        self.batch_success_labs = xp.ones_like(self.batch_targ, dtype=xp.int32) * (-1)
        self.batch_success_probs = xp.ones_like(self.batch_targ, dtype=xp.float32) * (-1)
        
        # retain predicted labels and probabilities of the orginal images
        # just for convenience, not affect on the later steps
        probs = self.model.predict_proba(self.batch_org).data
        labs = xp.argmax(probs, axis=1).astype(xp.int32)
        probs = xp.max(probs, axis=1)
        self.org_labels = np.append(self.org_labels, cuda.to_cpu(labs), axis=0)
        self.org_probs = np.append(self.org_probs, cuda.to_cpu(probs), axis=0)
        
    
    def _update_adversary(self):
        """
        Back propagate losses and update adversarial images. 
        """
        xp = self.xp
        loss = self.loss()
        # set initial gradient
        loss.grad = xp.ones_like(loss, dtype=xp.float32)
        loss.backward()
        # update adversary
        perturb = xp.clip(self.lr * xp.sign(self.batch_adv.grad), -self.clip_eps, self.clip_eps)
        new_adv = xp.clip(self.batch_adv.data - perturb, -0.5, 0.5)
        self.batch_adv = Variable(new_adv)
 

    def _do_iteration(self):
        """
        Iteratively update the adversarial images.
        """
        xp = self.xp
        prev_loss = xp.full_like(self.batch_targ, 1e20, dtype=xp.float32)
  
        for cnt_iter in range(self.num_iterations):
            # update
            self._update_adversary()
            
            # if imporovement of all adversaries seems to stop, abort this search
            if self.num_iterations > 10 and self.early_abort and (cnt_iter + 1)%(self.num_iterations//10) == 0:
                cur_loss = self.loss().data
                should_abort = (cur_loss > prev_loss * 0.9999)
                if xp.sum(xp.logical_not(should_abort)) == 0:
                    break
                prev_loss = cur_loss
                
            # print progress
            if self.num_iterations > 10 and (cnt_iter + 1)%(self.num_iterations//10) == 0:
                print('iterations: {0}/{1}  ----  finished at {2}'.format(
                    cnt_iter + 1, self.num_iterations, 
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
                sys.stdout.flush()


    def _store_results_of_this_batch(self):
        """
        Before cotinue to the next batch, retain results of this batch.
        """
        xp = self.xp
        # get losses and predicted labels
        l2sq = self.l2_loss().data # l2 squared loss
        probs = self.model.predict_proba(self.batch_adv).data
        labs = xp.argmax(probs, axis=1).astype(xp.int32)
        probs = xp.max(probs, axis=1) 
        
        # check if succeeded ? 
        is_success = (labs == self.batch_targ) # is the model fooled ?
        # gather succeeded images
        self.batch_success_adv[is_success] = self.batch_adv.data[is_success].copy()
        self.batch_success_labs[is_success] = labs[is_success].copy()
        self.batch_success_probs[is_success] = probs[is_success].copy()
        self.batch_success_l2sq[is_success] = l2sq[is_success].copy()
        # retain results 
        self.adv_images = np.append(self.adv_images, cuda.to_cpu(self.batch_success_adv), axis=0)
        self.adv_labels = np.append(self.adv_labels, cuda.to_cpu(self.batch_success_labs), axis=0)
        self.adv_probs = np.append(self.adv_probs, cuda.to_cpu(self.batch_success_probs), axis=0)
        self.adv_l2_squared = np.append(self.adv_l2_squared, cuda.to_cpu(self.batch_success_l2sq), axis=0)

    
    ## utils ###############################
    
    def save_adv(self, save_dir):
        """
        Save generated adversarial images as image files.
        Also, save their predicted labels and probabilities, l2_error and 'c'.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # adjust and rescale images to [0, 255]
        images_pil = ((self.adv_images + 0.5) * 255).astype(np.uint8)
        # exchange axis as PIL Image
        images_pil = images_pil.transpose(0, 2, 3, 1)
        # RGB or Monotone
        n_chanels = images_pil.shape[3]
        img_shape = (images_pil.shape[1], images_pil.shape[2])
        
        # header of detail file
        details = ['\t'.join(['ID', 'predicted_label', 'probability', 'l2_squared'])]
        
        for i, (arr, lab, prob, l2sq) in enumerate(zip(images_pil, self.adv_labels, self.adv_probs, 
                                                      self.adv_l2_squared)):
            if n_chanels == 1:
                img = Image.fromarray(arr.reshape(img_shape), mode='L')
            elif n_chanels == 3:
                img = Image.fromarray(arr, mode='RGB')
            else:
                raise Exception("number of chanels must be 1 or 3")
            
            details.append('{0}\t{1}\t{2:.4f}\t{3}'.format(i, lab, prob, l2sq))
            
            # save image
            img_file = os.path.join(save_dir, 'adv_{0}.png'.format(i))
            img.save(img_file)
            
        # save details
        details = '\n'.join(details)
        detail_file = os.path.join(save_dir, 'details.tsv')
        with open(detail_file, 'w', encoding='utf8') as f:
            f.writelines(details)
        
    
    def save_org(self, save_dir):
        """
        Save original images as image files.
        Also, save their predicted labels and probabilities.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # adjust and rescale images to [0, 255]
        images_pil = ((self.org_images + 0.5) * 255).astype(np.uint8)
        # exchange axis as PIL Image
        images_pil = images_pil.transpose(0, 2, 3, 1)
        # RGB or Monotone
        n_chanels = images_pil.shape[3]
        img_shape = (images_pil.shape[1], images_pil.shape[2])
        
        # header of detail file
        details = ['\t'.join(['ID', 'predicted_label', 'probability'])]
        
        for i, (arr, lab, prob) in enumerate(zip(images_pil, self.org_labels, self.org_probs)):
            if n_chanels == 1:
                img = Image.fromarray(arr.reshape(img_shape), mode='L')
            elif n_chanels == 3:
                img = Image.fromarray(arr, mode='RGB')
            else:
                raise Exception("number of chanels must be 1 or 3")
            
            details.append('{0}\t{1}\t{2:.4f}'.format(i, lab, prob))
            
            # save image
            img_file = os.path.join(save_dir, 'org_{0}.png'.format(i))
            img.save(img_file)
            
        # save details
        details = '\n'.join(details)
        detail_file = os.path.join(save_dir, 'details.tsv')
        with open(detail_file, 'w', encoding='utf8') as f:
            f.writelines(details)
