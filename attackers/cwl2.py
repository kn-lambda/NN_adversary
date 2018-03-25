import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, iterators
from chainer import cuda
from chainer import Parameter
from chainer import initializers

from PIL import Image
import os
import sys
import gc
from datetime import datetime

import numpy as np


#########################################################################
# perturbation itself ###################################################

class Perturbation(chainer.Link):
    """
    This object represets adversarial perturbation.
    
    Attributes:
        delta (chainer.Parameter) : adversarial perturbation which will be trained
    """
    
    def __init__(self, shape):
        super(Perturbation, self).__init__()
        init = initializers.Normal(scale=0.05, dtype=np.float32)
       
        with self.init_scope():
            self.delta = chainer.Parameter(init, shape)
            
    def __call__(self):
        return self.delta
    
    @property
    def data(self):
        return self.delta.data


#########################################################################
# generator of adversarial images #######################################

class CarliniWagnerL2(object):
    """
    This object generates adversarial images. 
    
    Used Algorithm is 'L2 Attack' in the paper:
    "Towards Evaluating the Robustness of Neural Networks" 
     by Nicholas Carlini and David Wagner, at IEEE Symposium on Security & Privacy, 2017
    
    """
    
    def __init__(self, model, images, targets, batch_size=200, confidence = 0.0,
                 initial_c = 1e-3, num_c_search = 9, num_iterations = 10000,
                 learning_rate = 1e-2, early_abort = True):
        """
        Args:
            model (ClassiferNN)     : adversarial images are genarated with respect to this model
            images (numpy ndarray)  : original input image
            targets (numpy ndarray) : target labels toward which adverarial images are generated
            batch size (int)        : batch size
            confidence (float)      : how strong adversarial images get high probabilities
            initial_c (float)       : initial value of 'c' whici is the relative importance of two losses
            num_c_search(int)       : number of times to search best 'c'
            num_iterations(int)     : number of iterations when gradient decent optimization
            learning_rate(int)      : optimizers.Adam's alpha
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
        self.confidence = confidence
        
        self.initial_c = initial_c
        self.num_c_search = num_c_search
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.early_abort = early_abort
        
        # storages for the results
        # predicted labels and probabilities of the generated adversarial images
        self.adv_images = np.empty((0,) + self.org_images[0].shape, dtype=np.float32)
        self.adv_labels = np.empty((0,) + self.targets[0].shape, dtype=np.int32)
        self.adv_probs = np.empty((0,) + self.targets[0].shape, dtype=np.float32)
        self.adv_l2_squared = np.empty((0,) + self.targets[0].shape, dtype=np.float32)
        self.adv_c = np.empty((0,) + self.targets[0].shape, dtype=np.float32)
        
        # storage for predicted labels and probabilities of the original images
        self.org_labels = np.empty((0,) + self.targets[0].shape, dtype=np.int32)
        self.org_probs = np.empty((0,) + self.targets[0].shape, dtype=np.float32)
        
        # iterators for batch
        self.img_iter = iterators.SerialIterator(self.org_images, batch_size, shuffle=False, repeat=False)
        self.targ_iter = iterators.SerialIterator(self.targets, batch_size, shuffle=False, repeat=False)
        
   
    ## loss functions ######################
    
    def total_loss(self):
        """
        Carlini Wagner L2 loss, which is composed of two loss.
        The relative impotance of these two losses is refered as 'c'.
        """
        return self.l2_loss() + self.c * self.confidence_loss()
    
    
    def _one_hot(self, labels):
        xp = self.xp
        labels_flat = labels.flatten()
        return xp.eye(self.model.n_classes, dtype=xp.float32)[labels_flat]
        
        
    def l2_loss(self):
        diff = self.batch_adv - self.batch_org
        return F.batch_l2_norm_squared(diff)
    
    
    def confidence_loss(self):
        xp = self.xp
        out = self.model(self.batch_adv) # output of the model before softmax
        one_hot_targ = self._one_hot(self.batch_targ)
        inf = 1e20
        # calculate differnce of outputs between 'max + confidence' and 'target' 
        out_max = F.max(out + one_hot_targ * (-inf), axis=1)
        out_targ = F.sum(out * one_hot_targ, axis=1)
        conf_arr = xp.ones_like(out_max, dtype=xp.float32) * self.confidence
        out_diff = out_max + conf_arr - out_targ
        # clip the result to [0, inf]
        return F.clip(out_diff, 0.0, inf)

    
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
        adv_c (numpy float array)          : selected value of 'c' where each adversarial image is generated 
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
            
            # loop for searching c
            loop_cnt_c = 1
            for _ in range(self.num_c_search):
                self._reset_states_for_c()
                self._check_this_c()
                self._prepare_next_c()
                
                print('batch: {0}/{1}, c-search: {2}/{3}  ----  finished at {4}'.format(
                    loop_cnt_batch, n_batch, loop_cnt_c, self.num_c_search, 
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
                     )
                sys.stdout.flush()
                
                loop_cnt_c += 1
                gc.collect()
                
            self._store_results_of_this_batch()
            
            loop_cnt_batch += 1
            
        # aggregate results and print
        n_failure = np.sum(self.targets != self.adv_labels)
        ratio_failure = 1.0 * n_failure / n_data
        n_success = n_data - n_failure
        ratio_success = 1.0 * n_success / n_data
        mean_l2sq = np.mean(self.adv_l2_squared[self.adv_labels != -1])
        
        print("")
        print("============================================")
        print("confidence : {0}".format(self.confidence))
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
        Set the next batch data, and initialize several states including 'c'.
        """
        xp = self.xp
        if self.img_iter.epoch != 0 or self.targ_iter.epoch != 0:
            return
        
        # get the next batch data
        batch_org = self.img_iter.next() 
        batch_targ = self.targ_iter.next()
        # to gpu
        self.batch_org = xp.array(batch_org, dtype=xp.float32)
        self.batch_targ = xp.array(batch_targ, dtype=xp.int32)
        # convert input image into arctanh apace
        self.batch_org_arctanh = xp.arctanh(2.0 * self.batch_org)
        
        # initialize c
        self.c = xp.ones_like(self.batch_targ, dtype=xp.float32) * self.initial_c
        # reset lower and upper bound of c, among which we search the best results
        self.lower_bound = xp.zeros_like(self.batch_targ, dtype=xp.float32)
        self.upper_bound = xp.ones_like(self.batch_targ, dtype=xp.float32) * 1e10    
        
        # reset storages for the best adversaries in this batch
        self.batch_best_adv = xp.zeros_like(self.batch_org, dtype=xp.float32)
        self.batch_best_l2sq = xp.ones_like(self.batch_targ, dtype=xp.float32) * 1e10
        self.batch_best_labs = xp.ones_like(self.batch_targ, dtype=xp.int32) * (-1)
        self.batch_best_probs = xp.ones_like(self.batch_targ, dtype=xp.float32) * (-1)
        self.batch_best_c = xp.ones_like(self.batch_targ, dtype=xp.float32) * (-1)
        
        # retain predicted labels and probabilities of the orginal images
        # just for convenience, not affect on the later steps
        probs = self.model.predict_proba(self.batch_org).data
        labs = xp.argmax(probs, axis=1).astype(xp.int32)
        probs = xp.max(probs, axis=1)
        self.org_labels = np.append(self.org_labels, cuda.to_cpu(labs), axis=0)
        self.org_probs = np.append(self.org_probs, cuda.to_cpu(probs), axis=0)      
        
    
    def _reset_states_for_c(self):
        """
        Now, new 'c' is set. 
        Before checking this 'c' can generate adversarial images, 
        reset previous adversarial images and storages. 
        """
        xp = self.xp
        # reset perturbation
        self.perturb = Perturbation(self.batch_org.shape)
        if xp is not np:
            self.perturb = self.perturb.to_gpu() 
        # reset adversarial image
        self.batch_adv = F.tanh(self.batch_org_arctanh + self.perturb()) / 2.0
        # reset optimizer
        self.optimizer = optimizers.Adam(alpha=self.learning_rate) # use Adam
        self.optimizer.setup(self.perturb)       
        # reset storages
        ## for the best (smallest perturbation) adversary with this c
        self.c_best_l2sq = xp.ones_like(self.batch_targ, dtype=xp.float32) * 1e10 # l2-squared
        ## for 'success or failure' in generating adversaries with this c
        self.c_is_success = xp.full_like(self.batch_targ, False)
        self.c_not_success = None
    
    
    def _update_adversary(self):
        """
        Back propagate losses and update adversarial images. 
        """
        xp = self.xp
        loss = self.total_loss()
        # set initial gradient
        loss.grad = xp.ones_like(loss, dtype=xp.float32)
        self.perturb.cleargrads()
        loss.backward()
        # update perturbation
        self.optimizer.update()
        # update adversary
        self.batch_adv = F.tanh(self.batch_org_arctanh + self.perturb()) / 2.0
        
    
    def _check_this_c(self):
        """
        With 'c' fixed, iteratively update the adversarial images.
        """
        xp = self.xp
        prev_loss = xp.full_like(self.batch_targ, 1e20, dtype=xp.float32)
  
        for cnt_iter in range(self.num_iterations):
            # update
            self._update_adversary()
            # get losses and predicted labels
            l2sq = self.l2_loss().data # l2 squared loss
            confl = self.confidence_loss().data # confidence loss
            probs = self.model.predict_proba(self.batch_adv).data
            labs = xp.argmax(probs, axis=1).astype(xp.int32)
            probs = xp.max(probs, axis=1)
            
            # check improvement, and update the best adversary if exists
            ## among this c
            is_improved = (l2sq < self.c_best_l2sq) # is smaller perturbation ?
            is_fooled = (confl <= 0.0) # is the model fooled ?
            is_best = xp.logical_and(is_improved, is_fooled)
            
            self.c_best_l2sq[is_best] = l2sq[is_best].copy()
            
            self.c_is_success = xp.logical_or(self.c_is_success, is_best)
            self.c_not_success = xp.logical_not(self.c_is_success)
            
            ## among this batch
            is_improved = (l2sq < self.batch_best_l2sq)
            is_best = xp.logical_and(is_improved, is_fooled)
            
            self.batch_best_adv[is_best] = self.batch_adv.data[is_best].copy()
            self.batch_best_l2sq[is_best] = l2sq[is_best].copy()
            self.batch_best_labs[is_best] = labs[is_best].copy()
            self.batch_best_probs[is_best] = probs[is_best].copy()
            self.batch_best_c[is_best] = self.c[is_best].copy()

            # if imporovement of all adversaries seems to stop, abort this search
            if self.early_abort and (cnt_iter + 1)%(self.num_iterations//10) == 0:
                cur_loss = self.total_loss().data
                should_abort = (cur_loss > prev_loss * 0.9999)
                if xp.sum(xp.logical_not(should_abort)) == 0:
                    break
                prev_loss = cur_loss
            
            # print progress
            #if (cnt_iter + 1)%(self.num_iterations//10) == 0:
            #    print('iteratins: {0}/{1}  ----  finished at {2}'.format(
            #        cnt_iter + 1, self.num_iterations, 
            #        datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
                
            
    def _prepare_next_c(self):
        """
        Update 'c' depending on whether adversarial images have been sucessfully generated.
        The next value of 'c' is determined by binary search.
        """
        xp = self.xp
        tmp_c = self.c.reshape(1, -1) 
        
        # if succeeded, c is decreased
        tmp_upper_bound = self.upper_bound.reshape(1, -1)
        new_upper_bound = xp.min(xp.concatenate([tmp_upper_bound, tmp_c], axis=0), axis=0)
        new_c = (self.lower_bound + new_upper_bound) / 2.0
        self.upper_bound[self.c_is_success] = new_upper_bound[self.c_is_success].copy()
        ## if upper bound is near limit, skip update
        not_near_limit = (self.upper_bound < 1e9)
        can_update = xp.logical_and(self.c_is_success, not_near_limit)
        self.c[can_update] = new_c[can_update].copy()
        
        # if failed, c is increased
        tmp_lower_bound = self.lower_bound.reshape(1, -1)
        new_lower_bound = xp.max(xp.concatenate([tmp_lower_bound, tmp_c], axis=0), axis=0)        
        new_c = (new_lower_bound + self.upper_bound) / 2.0
        self.lower_bound[self.c_not_success] = new_lower_bound[self.c_not_success].copy()
        ## if upper bound is near limit, multiplied by 10
        is_near_limit = xp.logical_not(not_near_limit)
        can_update = xp.logical_and(self.c_not_success, is_near_limit)
        self.c[can_update] = self.c[can_update].copy() * 10.0
        ## otherwise perform binary search
        can_update = xp.logical_and(self.c_not_success, not_near_limit)
        self.c[can_update] = new_c[can_update].copy()
        
    
    def _store_results_of_this_batch(self):
        """
        Before cotinue to the next batch, retain results of this batch.
        """
        self.adv_images = np.append(self.adv_images, cuda.to_cpu(self.batch_best_adv), axis=0)
        self.adv_labels = np.append(self.adv_labels, cuda.to_cpu(self.batch_best_labs), axis=0)
        self.adv_probs = np.append(self.adv_probs, cuda.to_cpu(self.batch_best_probs), axis=0)
        self.adv_l2_squared = np.append(self.adv_l2_squared, cuda.to_cpu(self.batch_best_l2sq), axis=0)
        self.adv_c = np.append(self.adv_c, cuda.to_cpu(self.batch_best_c), axis=0)

    
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
        details = ['\t'.join(['ID', 'predicted_label', 'probability', 'l2_squared', 'c'])]
        
        for i, (arr, lab, prob, l2sq, c) in enumerate(zip(images_pil, self.adv_labels, self.adv_probs, 
                                                      self.adv_l2_squared, self.adv_c)):
            if n_chanels == 1:
                img = Image.fromarray(arr.reshape(img_shape), mode='L')
            elif n_chanels == 3:
                img = Image.fromarray(arr, mode='RGB')
            else:
                raise Exception("number of chanels must be 1 or 3")
            
            details.append('{0}\t{1}\t{2:.4f}\t{3}\t{4}'.format(i, lab, prob, l2sq, c))
            
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
