from GSD.utils.cnn_visualizations.src.gradcam import *
from GSD.utils.cnn_visualizations.src.misc_functions import *
import os
import numpy as np
import pickle as pkl
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from hsic import HSICLoss
from networks import HSICClassifier
from GSD.train.datasets import create_dataloaders
from GSD.train.train_utils import get_device, AnnealingRestartScheduler
from torch.autograd import Variable


# experiment parameters
lambda_hsic = 0
feature_opt = 'Concat'  # {'None', 'Concat', 'HSIC', 'HSIC+Concat'}
engineered_features = True
learned_features = True
feature_subset = 'ALL'
use_comp = False
n_training_fields = 1

exp_name = f"eng_{engineered_features}_learned_{learned_features}_lambda{lambda_hsic}_{n_training_fields if learned_features else ''}"
print(exp_name)
model_path = '/home/roironen/GSD/GSD/saved_models/DONE_GSD1A_eng_featTrue_learned_featTrue_lambda0_8/model.pkl'

# training parameters
update_lambda = True

cuda_id = 0
disorders = ['GSD1A']#['APBD']#['GSD1A']

num_classes = len(disorders) + 1
assert num_classes>1

dict_features = {'feature_opt':feature_opt,
                 'feature_subset':feature_subset,
                 'engineered_features':engineered_features,
                 'learned_features':learned_features,
                 'disorders': disorders,
                 'use_comp': use_comp}

file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', 'DONE_'+disorders[0]+'_'+exp_name)


device = get_device(cuda_id)
batch_size=1
train_loader, val_loader, test_loader = create_dataloaders(batch_size, dict_features,n_training_fields, num_classes)


model = HSICClassifier(num_classes=num_classes, feature_len=train_loader.dataset.feature_len,
                       dict_features=dict_features, gap_norm_opt='batch_norm').to(device)

def vis_valid_or_test(mode, perf_dict=None):

    loader = val_loader if mode=='valid' else test_loader
    for batch_idx, (data, target, eng_features, label) in enumerate(tqdm(loader)):
        assert learned_features
        data, target, eng_features = data.squeeze(1).to(device), target.to(device), eng_features.squeeze(1).to(device)
        im_as_var = Variable(data, requires_grad=True)

        target_class = target
        # Grad cam
        grad_cam = GradCam(model, target_layer=6)
        # Generate cam mask
        cam = grad_cam.generate_cam(im_as_var, eng_features, None)
        show_class_activation_images(data[0,np.array([0,1,3])],cam)


if __name__ == "__main__":
    perf_dict = None
    print(f'Started training main task, {exp_name}')

    model.load_state_dict(torch.load(model_path))
    vis_valid_or_test(mode='test')
    print(f'{exp_name} finished training')
