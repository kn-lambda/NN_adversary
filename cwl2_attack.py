"""
Carlini & Wagner's L2 attack
"""

import chainer
from chainer import iterators, serializers
from chainer import cuda

import os
import sys
import shutil
from datetime import datetime

import numpy as np

# use GPU if possible
uses_device = 0

if uses_device >= 0:
    chainer.cuda.get_device_from_id(uses_device).use()
    chainer.cuda.check_cuda_available()

# import several tools
from modules.data import Data
from modules.classifer import layer_params, ClassiferNN
from modules.utils import get_predictions
from attackers.cwl2 import CarliniWagnerL2


###########################################################
# main ####################################################

#set data-neme and model-type
args = sys.argv

if len(args) == 3:
    data_name = args[1]
    model_type = args[2]
else:
    print('enter data name and model type')
    print('python <this script> <data name> <model type>')
    quit()

print("")
print("STARTED at {0}".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
print("==================================")
print("data       : {0}".format(data_name))
print("model type : {0}".format(model_type))
print("==================================")


# load data
data = Data(data_name)
data_images = data.test_image[:1000]
data_labels = data.test_label[:1000]

# load trained model
model = ClassiferNN(layer_params[data_name], T=1.0)
trained_model_dir = 'trained_models'
trained_model_file = '{0}_{1}.hdf5'.format(data_name, model_type)
trained_model = os.path.join(trained_model_dir, trained_model_file)
serializers.load_hdf5(trained_model, model)
model.T = 1.0

# model to gpu
if uses_device >= 0:
    model.to_gpu()
    
# model cannot be trained 
for param in model.params():
    param._requires_grad = False
    
# make train-mode false
chainer.config.train = False


# reporting success rate
success_report = []
header = '\t'.join(['org_label', 'targ_label', 'n_data', 
                    'n_success', 'ratio_success', 'n_failure', 'ratio_failure', 
                    'mean_l2_squared'])
success_report.append(header)


# generate adversarial images

# prepare labels
all_labels = np.unique(data_labels)    

# loops for all label pairs
for org_label in all_labels:
    # pick up data to use
    is_use = (data_labels == org_label)
    uses_images = data_images[is_use]

    # confined to data which our classifer can correctly classify
    _, pred_label, _, _, _ = get_predictions(model, uses_images)
    is_correct = (pred_label == org_label)
    uses_images = uses_images[is_correct]

    # iterate for target labels
    org_saved = False # check if original images are already saved
    for targ_label in all_labels:
        # skip original = target
        if targ_label == org_label:
            continue
        # set target labels
        uses_targets = np.ones((uses_images.shape[0],), dtype=np.int32) * targ_label
        
        print("")
        print("")
        print("")
        print("generating adversaries ...")
        print("----------------------------")
        print("original label : {0}".format(org_label))
        print("target label   : {0}".format(targ_label))
        print("----------------------------")

        att = CarliniWagnerL2(model, uses_images, uses_targets, num_iterations=1000, confidence=0, batch_size=128)
        # run
        n_data, n_success, ratio_success, n_failure, ratio_failure, mean_l2sq = att.run()

        # reporting aggregated result
        success_report.append('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}'.format(
                               org_label, targ_label,
                               n_data, n_success, ratio_success, n_failure, ratio_failure,
                               mean_l2sq))

        # saave images
        img_dir = 'images/cwl2/{0}/{1}/org={2}'.format(data_name, model_type, org_label)
        # for adversaries
        adv_dir = os.path.join(img_dir, 'adv={}'.format(targ_label))
        att.save_adv(adv_dir)
        # for original, save only once
        if org_saved is False:
            org_dir = os.path.join(img_dir, 'org')
            att.save_org(org_dir)
            org_saved = True
            

# save report
report_dir = 'success_rate'
os.makedirs(report_dir, exist_ok=True)
report_file = 'cwl2__{0}_{1}.tsv'.format(data_name, model_type)
report_path = os.path.join(report_dir, report_file)

contents = '\n'.join(success_report)
with open(report_path, 'w', encoding='utf8') as f:
        f.writelines(contents)


print("----------------------------")
print('Done.')


