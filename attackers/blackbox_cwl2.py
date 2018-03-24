import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from PIL import Image
import os
import sys
import gc
from datetime import datetime

import numpy as np


#########################################################################
# optimizers ############################################################

# Adam

class PixelAdam(object):
    """pixel wise Adam optimizer
    
    Simplified implementation of the paper:
    
    "ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models" 
    by Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, Cho-Jui Hsieh
    
    The following points are omitted:
    1. attack-space dimensional reduction
    2. hierarchical attack
    3. sampling from important pixcels
      -> not efficient for large images such as ImageNet 
    4. 'tanh' as clipping to [-0.5, 0.5]  
    """
    
    def __init__(self, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999):
        # Adam's lerning parameters
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        
        
    def setup(self, image, target, num_sampling = 128):
        """
        Set up one original image and its target label.
        
        Args:
            image (numpy or cupy ndarray) : array of one image
            target (int)                  : adversary's target label
            num_sampling (int)            : number of pixel points where gradients are calculated at one update
        """
        # check if the image is in cpu or gpu
        self.xp = cuda.get_array_module(image) # numpy or cupy
        xp = self.xp
        self.num_sampling = num_sampling
        self.target = target
        self.org_image = xp.array(image.copy())
        # for convenience add batch-dimension
        if len(self.org_image.shape) == 3:
            self.org_image = self.org_image.reshape((1,) + self.org_image.shape)
        self.image_shape = self.org_image.shape
        # initialize adversarial image
        self.adv_image = self.org_image.copy()
        self.num_pixels = image.size
        self.init_state()
        # the following bool arrays are used in update(self)
        self.odd_ind = xp.array([True if i % 2 != 0 else False for i in range(self.num_sampling * 2 + 1)])
        self.even_ind = xp.logical_not(self.odd_ind)
        self.even_ind[0] = False
    
    
    def init_state(self):
        """
        Initialize Adam's internal states.
        """
        xp = self.xp
        # Adam's states for each pixel
        self.m = xp.zeros(self.num_pixels, dtype = np.float32)
        self.v = xp.zeros(self.num_pixels, dtype = np.float32)
        # counter of how many times each pixel has been updated
        self.t = xp.zeros(self.num_pixels, dtype = np.int32)
        
        
    def update(self, loss_func):
        """
        Perform update one step.
        
        Args:
            loss_func (function) : calculate the loss from the original/adversarial image and the target label
        """
        xp = self.xp
        # variations from the current adversarial image
        ## 0th : original image, odd-index : positive variation, even-index : negative variation
        var_images = xp.repeat(self.adv_image, self.num_sampling * 2 + 1, axis = 0)
        ## randomly select pixels which are varied
        pix_ind = xp.random.choice(self.num_pixels, self.num_sampling, replace=False)
        for i in range(self.num_sampling):
            var_images[2 * i + 1].reshape(-1)[pix_ind[i]] += 0.0001
            var_images[2 * i + 2].reshape(-1)[pix_ind[i]] -= 0.0001
        
        # approximate the gradient by directly calculate the differences
        var_loss = loss_func(self.org_image, var_images, self.target).data
        grad = (var_loss[self.odd_ind] - var_loss[self.even_ind]) / 0.0002
        
        # update Adam's internal state
        self.m[pix_ind] = self.beta1 * self.m[pix_ind] + (1 - self.beta1) * grad
        self.v[pix_ind] = self.beta2 * self.v[pix_ind] + (1 - self.beta2) * grad * grad
        self.t[pix_ind] += 1
        
        # update the adversarial image
        adv_image = self.adv_image.reshape(-1)[pix_ind].copy()
        t = self.t[pix_ind]
        m = self.m[pix_ind]
        v = self.v[pix_ind]
        beta_coef = xp.sqrt(1 - xp.power(self.beta2, t)) / (1 - xp.power(self.beta1, t)) 
        adv_image = adv_image - self.lr * beta_coef * m / (xp.sqrt(v) + 1e-8)
        adv_image = xp.clip(adv_image, -0.5, 0.5)
        self.adv_image.reshape(-1)[pix_ind] = adv_image 
        
        return self.adv_image


# Newton method

class PixelNewton(object):
    """pixel wise saddle-free Newton method optimizer
    
    Simplified implementation of the paper:
    
    "ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models" 
    by Pin-Yu Chen*, Huan Zhang*, Yash Sharma, Jinfeng Yi, Cho-Jui Hsieh
    
    The following points are omitted:
    1. attack-space dimensional reduction
    2. hierarchical attack
    3. sampling from important pixcels
      -> not efficient for large images such as ImageNet 
    4. 'tanh' as clipping to [-0.5, 0.5]
    
    "saddle-free Newton method" is proposed in the paper:
    
    "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"
    by Yann Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, Yoshua Bengio
    """
    
    def __init__(self, learning_rate=0.01):
        # learning parameters
        self.lr = learning_rate
        
        
    def setup(self, image, target, num_sampling = 128):
        """
        Set up one original image and its target label.
        
        Args:
            image (numpy or cupy ndarray) : array of one image
            target (int)                  : adversary's target label
            num_sampling (int)            : number of pixel points where gradients are calculated at one update
        """
        # check if the image is in cpu or gpu
        self.xp = cuda.get_array_module(image) # numpy or cupy
        xp = self.xp
        self.num_sampling = num_sampling
        self.target = target
        self.org_image = xp.array(image.copy())
        # for convenience add batch-dimension
        if len(self.org_image.shape) == 3:
            self.org_image = self.org_image.reshape((1,) + self.org_image.shape)
        self.image_shape = self.org_image.shape
        # initialize adversarial image
        self.adv_image = self.org_image.copy()
        self.num_pixels = image.size
        # the following bool arrays are used in update(self)
        self.odd_ind = xp.array([True if i % 2 != 0 else False for i in range(self.num_sampling * 2 + 1)])
        self.even_ind = xp.logical_not(self.odd_ind)
        self.even_ind[0] = False
        

    def init_state(self):
        pass
    
    
    def update(self, loss_func):
        """
        Perform update one step.
        
        Args:
            loss_func (function) : calculate the loss from the original/adversarial image and the target label
        """
        xp = self.xp
        # variations from the current adversarial image
        ## 0th : original image, odd-index : positive variation, even-index : negative variation
        var_images = xp.repeat(self.adv_image, self.num_sampling * 2 + 1, axis = 0)
        ## randomly select pixels which are varied
        pix_ind = xp.random.choice(self.num_pixels, self.num_sampling, replace=False)
        for i in range(self.num_sampling):
            var_images[2 * i + 1].reshape(-1)[pix_ind[i]] += 0.0001
            var_images[2 * i + 2].reshape(-1)[pix_ind[i]] -= 0.0001
        
        # approximate the gradient and hessian by directly calculate the differences
        var_loss = loss_func(self.org_image, var_images, self.target).data
        grad = (var_loss[self.odd_ind] - var_loss[self.even_ind]) / 0.0002
        hess = (var_loss[self.odd_ind] - 2 * var_loss[0] + var_loss[self.even_ind]) / (0.0001 * 0.0001)
        hess = xp.abs(hess) # saddle-free method
        hess[hess < 0.01] = 0.01 # care about division by too small value
        
        # update the adversarial image
        adv_image = self.adv_image.reshape(-1)[pix_ind].copy()
        adv_image = adv_image - self.lr * grad / hess
        adv_image = xp.clip(adv_image, -0.5, 0.5)
        self.adv_image.reshape(-1)[pix_ind] = adv_image 
        
        return self.adv_image


#########################################################################
# generator of adversarial images #######################################

class BlackBoxCWL2(object):
    """
    This object generates adversarial images. 
    
    Used Algorithm is 'Black-Box L2 Attack' in the paper:
    
    "ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models" 
    by Pin-Yu Chen*, Huan Zhang*, Yash Sharma, Jinfeng Yi, Cho-Jui Hsieh
    """
    
    def __init__(self, model, images, targets, num_sampling = 128, confidence = 0.0,
                 initial_c = 0.01, num_c_search = 9, num_iterations = 10000,
                 learning_rate = 0.01, early_abort = True):
        """
        Args:
            model (ClassiferNN)     : adversarial images are genarated with respect to this model
            images (numpy ndarray)  : original input image
            targets (numpy ndarray) : target labels toward which adverarial images are generated
            num_sampling (int)      : number of sampling points when coordinate-wise optimization
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
        self.num_sampling = num_sampling
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
   

    ## loss functions ######################
    
    def total_loss(self, org_img_arr, adv_img_arr, targ_arr):
        """
        Carlini Wagner L2 loss, which is composed of two losses.
        The relative impotance of these two losses is refered as 'c'.
        """
        return self.l2_loss(adv_img_arr, org_img_arr) + self.c * self.confidence_loss(adv_img_arr, targ_arr)
    
    
    def _one_hot(self, labels):
        xp = self.xp
        labels_flat = labels.flatten()
        return xp.eye(self.model.n_classes, dtype=xp.float32)[labels_flat]
        
        
    def l2_loss(self, adv_img_arr, org_img_arr):
        diff = adv_img_arr - org_img_arr
        return F.batch_l2_norm_squared(diff)
    
    
    def confidence_loss(self, adv_img_arr, targ_arr):
        xp = self.xp
        out = self.model(adv_img_arr) # output of the model before softmax
        one_hot_targ = self._one_hot(targ_arr)
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
        
        # loop for input images, one image at a time
        for i_data in range(n_data):
            self._prepare_image(i_data)
            
            # loop for searching c
            loop_cnt_c = 1
            for _ in range(self.num_c_search):
                self._reset_states_for_c()
                self._check_this_c()
                self._prepare_next_c()
                
                print('images: {0}/{1}, c-search: {2}/{3}  ----  finished at {4}'.format(
                    i_data + 1, n_data, loop_cnt_c, self.num_c_search, 
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
                     )
                sys.stdout.flush()
                
                loop_cnt_c += 1
                gc.collect()
                
            self._store_results_of_current_image()
            
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
            
            
    def _prepare_image(self, i_data):
        """
        Set the next data (image, target), and initialize several states including 'c'.
        """
        xp = self.xp
        # get the next image and target, and set currently focusing image/target
        cur_org = self.org_images[i_data] 
        cur_targ = self.targets[i_data]
        # put to gpu if possible, set the batch-dimension to 1 for convenience 
        self.cur_org = xp.array(cur_org, dtype=xp.float32).reshape((1,) + cur_org.shape)
        self.cur_targ = xp.array([cur_targ], dtype=xp.int32)
        
        # initialize c
        self.c = xp.ones_like(self.cur_targ, dtype=xp.float32) * self.initial_c
        # reset lower and upper bound of c, among which we search the best results
        self.lower_bound = xp.zeros_like(self.cur_targ, dtype=xp.float32)
        self.upper_bound = xp.ones_like(self.cur_targ, dtype=xp.float32) * 1e10 
        
        # reset storages for the best adversary of the current image
        self.cur_best_adv = xp.zeros_like(self.cur_org, dtype=xp.float32)
        self.cur_best_l2sq = xp.ones_like(self.cur_targ, dtype=xp.float32) * 1e10
        self.cur_best_lab = xp.ones_like(self.cur_targ, dtype=xp.int32) * (-1)
        self.cur_best_prob = xp.ones_like(self.cur_targ, dtype=xp.float32) * (-1)
        self.cur_best_c = xp.ones_like(self.cur_targ, dtype=xp.float32) * (-1)
        
        # retain predicted labels and probabilities of the orginal images
        # just for convenience, not affect on the later steps
        probs = self.model.predict_proba(self.cur_org).data
        lab = xp.argmax(probs, axis=1).astype(xp.int32)
        prob = xp.max(probs, axis=1)
        self.org_labels = np.append(self.org_labels, cuda.to_cpu(lab), axis=0)
        self.org_probs = np.append(self.org_probs, cuda.to_cpu(prob), axis=0) 
        
    
    def _reset_states_for_c(self):
        """
        Now, new 'c' is set. 
        Before checking this 'c' can generate adversarial images, 
        reset previous adversarial images and storages. 
        """
        xp = self.xp
        # reset optimizer 
        ## also, adversarial image is initialized and retained in the optimizer
        self.optimizer = PixelAdam(learning_rate=self.learning_rate)
        #self.optimizer = PixelNewton(learning_rate=self.learning_rate)
        self.optimizer.setup(self.cur_org, self.cur_targ)
        
        # reset storages
        ## for the best (smallest perturbation) adversary with this c
        self.c_best_l2sq = xp.ones_like(self.cur_targ, dtype=xp.float32) * 1e10 # l2-squared
        ## for 'success or failure' in generating adversaries with this c
        self.c_is_success = xp.full_like(self.cur_targ, False)
        self.c_not_success = None
    
    
    def _update_adversary(self):
        """
        Directly calculate difference of losses, and update adversarial images. 
        """
        # update adversary
        self.cur_adv = self.optimizer.update(self.total_loss)
        
    
    def _check_this_c(self):
        """
        With 'c' fixed, iteratively update the adversarial image.
        """
        xp = self.xp
        prev_loss = xp.ones_like(self.cur_targ, dtype=xp.float32) * 1e20
  
        for cnt_iter in range(self.num_iterations):
            # update
            self._update_adversary()
            # get losses and predicted labels
            l2sq = self.l2_loss(self.cur_adv, self.cur_org).data # l2 squared loss
            confl = self.confidence_loss(self.cur_adv, self.cur_targ).data # confidence loss
            probs = self.model.predict_proba(self.cur_adv).data
            lab = xp.argmax(probs, axis=1).astype(xp.int32)
            prob = xp.max(probs, axis=1)
            
            
            # check improvement, and update the best adversary if exists
            ## among this c
            is_improved = (l2sq < self.c_best_l2sq) # is smaller perturbation ?
            is_fooled = (confl <= 0.0) # is the model fooled ?
            is_best = xp.logical_and(is_improved, is_fooled)
            
            self.c_best_l2sq[is_best] = l2sq[is_best].copy()
            
            self.c_is_success = xp.logical_or(self.c_is_success, is_best)
            self.c_not_success = xp.logical_not(self.c_is_success)
            
            ## among current image
            is_improved = (l2sq < self.cur_best_l2sq)
            is_best = xp.logical_and(is_improved, is_fooled)
            
            self.cur_best_adv[is_best] = self.cur_adv[is_best].copy()
            self.cur_best_l2sq[is_best] = l2sq[is_best].copy()
            self.cur_best_lab[is_best] = lab[is_best].copy()
            self.cur_best_prob[is_best] = prob[is_best].copy()
            self.cur_best_c[is_best] = self.c[is_best].copy()
            

            # if imporovement of the adversary seems to stop, abort this search
            if self.early_abort and (cnt_iter + 1)%(self.num_iterations//10) == 0:
                cur_loss = self.total_loss(self.cur_org, self.cur_adv, self.cur_targ).data
                should_abort = (cur_loss > prev_loss * 0.9999)
                if xp.sum(xp.logical_not(should_abort)) == 0:
                    break
                prev_loss = cur_loss
            
            # when the confidence_loss reaches 0, reset Adam's state
            if confl <= 0.0:
                self.optimizer.init_state()

            # print progress
            #cnt_iter += 1
            #if cnt_iter%(self.num_iterations//10) == 0:
            #    print('        iteratins: {0}/{1}  ---- finished at {2}'.format(
            #        cnt_iter, self.num_iterations, 
            #        datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
                
            
    def _prepare_next_c(self):
        """
        Update 'c' depending on whether adversarial image has been sucessfully generated.
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
        
    
    def _store_results_of_current_image(self):
        """
        Before cotinue to the next image, retain results of the current image.
        """
        self.adv_images = np.append(self.adv_images, cuda.to_cpu(self.cur_best_adv), axis=0)
        self.adv_labels = np.append(self.adv_labels, cuda.to_cpu(self.cur_best_lab), axis=0)
        self.adv_probs = np.append(self.adv_probs, cuda.to_cpu(self.cur_best_prob), axis=0)
        self.adv_l2_squared = np.append(self.adv_l2_squared, cuda.to_cpu(self.cur_best_l2sq), axis=0)
        self.adv_c = np.append(self.adv_c, cuda.to_cpu(self.cur_best_c), axis=0)

    
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
