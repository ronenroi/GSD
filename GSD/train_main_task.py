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

# experiment parameters
lambda_hsic = 0
feature_opt = 'Concat'  # {'None', 'Concat', 'HSIC', 'HSIC+Concat'}
engineered_features = True
learned_features = True
feature_subset = 'ALL'
dict_features = {'feature_opt':feature_opt,
                 'feature_subset':feature_subset,
                 'engineered_features':engineered_features,
                 'learned_features':learned_features}
exp_name = f"eng_feat{engineered_features}_learned_feat{learned_features}_lambda{lambda_hsic}_{feature_subset}"
n_training_fields = 1
# training parameters
update_lambda = True
lr = 5e-4
num_epochs = 20
batch_size = 1 # int(16/n_training_fields)
cuda_id = 0

file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', exp_name)
if not os.path.exists(file_dir):
    os.mkdir(file_dir)

device = get_device(cuda_id)

train_loader, val_loader, test_loader = create_dataloaders(batch_size, dict_features,n_training_fields)

model = HSICClassifier(num_classes=2, feature_len=train_loader.dataset.feature_len,
                       dict_features=dict_features, gap_norm_opt='batch_norm').to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)
classification_criterion = nn.CrossEntropyLoss()
independence_criterion = HSICLoss(feature_opt, lambda_hsic, model.activation_size, device, decay_factor=0.7,
                                  external_feature_std=3)
# lr_scheduler = AnnealingRestartScheduler(lr_min=lr/100, lr_max=lr, steps_per_epoch=len(train_loader),
#                                          lr_max_decay=0.6, epochs_per_cycle=num_epochs, cycle_length_factor=1.5)
lambda_vec = lambda_hsic * np.hstack([np.linspace(0, 1, num_epochs//2), np.ones(num_epochs-num_epochs//2)])


def train(epoch, lambda_hsic):
    model.train()
    train_loss = 0
    cum_classification_loss = 0
    cum_hsic_loss = 0
    correct = 0
    for batch_idx, (data, target, eng_features) in enumerate(tqdm(train_loader)):
        if data.shape[0] > 0:
            data, target, eng_features = data.to(device), target.to(device), eng_features.squeeze(1).to(device)
        else:
            target, eng_features = target.to(device), eng_features.squeeze(0).to(
                device)
        optimizer.zero_grad()
        # for g in optimizer.param_groups:
        #     g['lr'] = lr_scheduler.lr
        logits, _, gap = model(data, eng_features)

        classification_loss = classification_criterion(logits, target)
        hsic_loss = independence_criterion.calc_loss(gap, eng_features)

        loss = classification_loss + hsic_loss

        loss.backward()
        cum_classification_loss += classification_loss.item()
        cum_hsic_loss += hsic_loss.item()
        train_loss += loss.item()

        _, predicted = torch.max(logits.data, 1)

        if torch.cuda.is_available():
            correct += (predicted.detach().cpu().numpy() == target.detach().cpu().numpy()).sum()
        else:
            correct += (predicted == target).sum()

        optimizer.step()

        # lr_scheduler.on_batch_end_update()

        if update_lambda:
            lambda_hsic = lambda_vec[epoch-1]

    epoch_accuracy = 100 * float(correct) / train_loader.dataset.__len__()
    print(f'Training Accuracy: {epoch_accuracy}')
    print(f'====> Epoch: {epoch} Losses: total={train_loss / len(train_loader.dataset):.4f}'
          f', classification={cum_classification_loss / len(train_loader.dataset):.4f},'
          f' hsic={cum_hsic_loss / len(train_loader.dataset):.4f}')

    return lambda_hsic


def valid_or_test(mode, perf_dict=None):
    if perf_dict is None:
        perf_dict = {'accuracy': [], 'f1': {'naf': [], 'af': []}}
    # model.eval()
    tot_loss = 0
    correct = 0
    pred_list = []
    label_list = []
    with torch.no_grad():
        loader = val_loader if mode=='valid' else test_loader
        for batch_idx, (data, target, eng_features) in enumerate(tqdm(loader)):
            if data.shape[0] > 0:
                data, target, eng_features = data.to(device), target.to(device), eng_features.squeeze(1).to(device)
            else:
                target, eng_features = target.to(device), eng_features.squeeze(0).to(device)
            logits, _, gap = model(data, eng_features)

            hsic_loss = independence_criterion.calc_loss(gap, eng_features)

            loss = classification_criterion(logits, target) + hsic_loss

            tot_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == target.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == target).sum()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(target.detach().cpu().numpy())

    preds = np.concatenate(pred_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    f1_total = f1_score(labels, preds, labels=[0, 1], average=None)

    tot_loss /= len(loader.dataset)
    epoch_accuracy = 100 * float(correct) / loader.dataset.__len__()



    if mode == 'valid' and (len(perf_dict['accuracy']) == 0 or epoch_accuracy > np.max(perf_dict['accuracy'])):
        torch.save(model.state_dict(), os.path.join(file_dir, f'{exp_name}_params.pkl'))
        print(['Saved @  ' + str(epoch_accuracy) + '%'])

    perf_dict['accuracy'].append(epoch_accuracy)
    perf_dict['f1']['naf'].append(f1_total[0])
    perf_dict['f1']['af'].append(f1_total[1])

    print(f'====> {mode} set loss: {tot_loss:.5f}')
    print(f'{mode} accuracy: {epoch_accuracy:.4f}')
    print(f'{mode} F1: {f1_total[0]}, {f1_total[1]}')

    if mode == 'test':
        with open(os.path.join(file_dir, f'{exp_name}_test_perf.pkl'), 'wb') as handle:
            pkl.dump(perf_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return perf_dict


if __name__ == "__main__":
    perf_dict = None
    print(f'Started training main task, {exp_name}')
    for epoch in range(num_epochs):
        lambda_hsic = train(epoch, lambda_hsic)
        perf_dict = valid_or_test(mode='valid', perf_dict=perf_dict)
        # lr_scheduler.on_epoch_end_update(epoch=epoch)
    model.load_state_dict(torch.load(os.path.join(file_dir, f'{exp_name}_params.pkl')))
    valid_or_test(mode='valid')
    valid_or_test(mode='test')
    print(f'{exp_name} finished training')
