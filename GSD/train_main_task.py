import os

import matplotlib.pyplot as plt
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
if True:
    lambda_hsic = 0.05
    feature_opt = 'HSIC+Concat'  # {'None', 'Concat', 'HSIC', 'HSIC+Concat'}
    engineered_features = True
    learned_features = True
    feature_subset = 'ALL'
    use_comp = False
    n_training_fields = 8
    grad_accumulation_steps = 8
    classifier='mlp2'
    network_name='cnn2'
else:
    lambda_hsic=0
    feature_opt='Concat'
    engineered_features = True
    learned_features = False
    feature_subset = 'ALL'
    use_comp = False
    n_training_fields = 1
    grad_accumulation_steps = 1
    classifier = 'mlp2'
    network_name = 'none'

disorders = ['GSD1A']#['APBD']#['GSD1A']

exp_name = f"{''.join(disorders)}_eng_{engineered_features}_learned_{learned_features}_{feature_opt}_classifier_{classifier}_{network_name}_lambda{lambda_hsic}{'_'+str(n_training_fields) if learned_features else ''}"
print(exp_name)
# training parameters
update_lambda = True
lr = 1e-4
num_epochs = 40
batch_size = 1 # int(16/n_training_fields)
cuda_id = 0

num_classes = len(disorders) + 1
assert num_classes>1

dict_features = {'feature_opt':feature_opt,
                 'feature_subset':feature_subset,
                 'engineered_features':engineered_features,
                 'learned_features':learned_features,
                 'disorders': disorders,
                 'use_comp': use_comp,
                 'network_name': network_name,
                 'n_training_fields': n_training_fields,
                 'classifier': classifier}

file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', exp_name)
if not os.path.exists(file_dir):
    os.mkdir(file_dir)

device = get_device(cuda_id)

train_loader, val_loader, test_loader = create_dataloaders(batch_size, dict_features,n_training_fields, num_classes)

model = HSICClassifier(num_classes=num_classes, feature_len=train_loader.dataset.feature_len,
                       dict_features=dict_features, gap_norm_opt='batch_norm').to(device)
# no_decay = list()
# decay = list()
# all = list()
# for m in model.model.features.modules():
#       if isinstance(m, (nn.Conv2d)):
#         decay.append(m.weight)
#         if m.bias is not None:
#             no_decay.append(m.bias)
#       elif hasattr(m, 'weight'):
#         no_decay.append(m.weight)
#       elif hasattr(m, 'bias'):
#         no_decay.append(m.bias)
#       else:
#           all.append(m)
#
# no_decay.append(model.fc_layer.weight)
# no_decay.append( model.model.classifier.weight)
# no_decay.append( model.model.classifier.bias)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.AdamW([{'params': no_decay, 'weight_decay': 0}, {'params': decay, 'weight_decay': 0}], lr=lr)
# optimizer = optim.AdamW(, lr=lr, betas=(0.95, 0.99), eps=1e-08, weight_decay=5)
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
    false_negative = 0
    all_GSD_patients = 0

    dataset_eng_features = torch.tensor(train_loader.dataset.eng_features_val, device=device, dtype=torch.float)
    gaps = []
    for batch_idx, (data, target, eng_features, label) in enumerate(tqdm(train_loader)):
        if len(data.shape)==2:
            continue
        if learned_features:
            data, target, eng_features = data.to(device), target.to(device), eng_features.squeeze(1).to(device)
        else:
            target, eng_features = target.to(device), eng_features.squeeze(1).to(device)

        # optimizer.zero_grad()
        # for g in optimizer.param_groups:
        #     g['lr'] = lr_scheduler.lr
        try:
            logits, _, gap = model(data, eng_features)
            gaps.append(gap)
            classification_loss = classification_criterion(logits, target)
            # hsic_loss = independence_criterion.calc_loss(gap, eng_features)

            loss = classification_loss #+ hsic_loss
            train_loss += loss.item()

            # Normalize the Gradients

            loss = loss / grad_accumulation_steps
            loss.backward(retain_graph='HSIC' in feature_opt)

            if ((batch_idx + 1) % grad_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                # Update Optimizer
                if learned_features and 'HSIC' in feature_opt:
                    hsic_loss = independence_criterion.calc_loss(torch.vstack(gaps), dataset_eng_features)
                    cum_hsic_loss += hsic_loss.item()

                    hsic_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                gaps = []

                # loss.backward()
        except:
            print('Error in current iteration')
            optimizer.zero_grad()
            gaps = []

        cum_classification_loss += classification_loss.item()

        _, predicted = torch.max(logits.data, 1)

        # if torch.cuda.is_available():
        predicted = predicted.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        correct += (predicted == target).sum()
        GSD_element = (np.array(label) != 'HCNONE')*1.0
        # print(HC_element)

        false_negative += (GSD_element * (predicted != target)).sum()
        all_GSD_patients += (GSD_element==1).sum()
        # else:
        #     correct += (predicted == target).sum()

        # optimizer.step()

        # lr_scheduler.on_batch_end_update()

        if update_lambda:
            lambda_hsic = lambda_vec[epoch-1]

    epoch_accuracy = 100 * float(correct) / train_loader.dataset.__len__()
    print(f'GSD patients {all_GSD_patients}')
    epoch_false_negative = 100 * float(false_negative) / all_GSD_patients
    print(f'Training Accuracy: {epoch_accuracy}, False Negative: {epoch_false_negative}')
    print(f'====> Epoch: {epoch} Losses: total={train_loss / len(train_loader.dataset):.4f}'
          f', classification={cum_classification_loss / len(train_loader.dataset):.4f},'
          f' hsic={cum_hsic_loss / len(train_loader.dataset):.4f}')


    return lambda_hsic, (train_loss / len(train_loader.dataset))


def valid_or_test(mode, perf_dict=None):
    if perf_dict is None:
        perf_dict = {'accuracy': [], 'f1': {'naf': [], 'af': []}}
    # model.eval()
    tot_loss = 0
    correct = 0
    false_negative = 0
    all_GSD_patients = 0
    pred_list = []
    label_list = []
    with torch.no_grad():
        loader = val_loader if mode=='valid' else test_loader
        for batch_idx, (data, target, eng_features, label) in enumerate(tqdm(loader)):
            try:
                if learned_features:
                    data, target, eng_features = data.to(device), target.to(device), eng_features.squeeze(1).to(device)
                else:
                    target, eng_features = target.to(device), eng_features.squeeze(1).to(device)
                logits, _, gap = model(data, eng_features)
            except:
                print()
            hsic_loss = independence_criterion.calc_loss(gap, eng_features)

            loss = classification_criterion(logits, target) + hsic_loss

            tot_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)

            # if torch.cuda.is_available():
            predicted = predicted.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            correct += (predicted == target).sum()
            GSD_element = (np.array(label) != 'HCNONE') * 1.0
            # print(HC_element)

            false_negative += (GSD_element * (predicted != target)).sum()
            all_GSD_patients += (GSD_element == 1).sum()
            # else:
            #     correct += (predicted == target).sum()

            pred_list.append(predicted)
            label_list.append(target)

    preds = np.concatenate(pred_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    f1_total = f1_score(labels, preds, labels=[0, 1], average=None)

    tot_loss /= len(loader.dataset)
    epoch_accuracy = 100 * float(correct) / loader.dataset.__len__()
    epoch_false_negative = 100 * float(false_negative) / all_GSD_patients

    if mode == 'valid' and (len(perf_dict['accuracy']) == 0 or epoch_accuracy > np.max(perf_dict['accuracy'])):
        torch.save(model.state_dict(), os.path.join(file_dir, f'{exp_name}_params.pkl'))
        print(['Saved @  ' + str(epoch_accuracy) + '%'])


    perf_dict['accuracy'].append(epoch_accuracy)
    perf_dict['f1']['naf'].append(f1_total[0])
    perf_dict['f1']['af'].append(f1_total[1])

    print(f'GSD patients {all_GSD_patients}')

    print(f'====> {mode} set loss: {tot_loss:.5f}')
    print(f'{mode} accuracy: {epoch_accuracy:.4f}, False Negative: {epoch_false_negative:.4f}')
    print(f'{mode} F1: {f1_total[0]}, {f1_total[1]}')

    if mode == 'test':
        with open(os.path.join(file_dir, f'{exp_name}_test_perf.pkl'), 'wb') as handle:
            pkl.dump(perf_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return perf_dict, tot_loss


if __name__ == "__main__":
    perf_dict = None
    train_losses = []
    val_losses = []
    print(f'Started training main task, {exp_name}')
    for epoch in range(num_epochs):
        lambda_hsic, loss = train(epoch, lambda_hsic)
        train_losses.append(loss)
        perf_dict, val_loss = valid_or_test(mode='valid', perf_dict=perf_dict)
        val_losses.append(val_loss)

        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.legend(['train','val'])
        plt.yscale('log')
        plt.savefig(os.path.join(file_dir, 'plot.png'))
        plt.close()
    model.load_state_dict(torch.load(os.path.join(file_dir, f'{exp_name}_params.pkl')))
    # valid_or_test(mode='valid')
    valid_or_test(mode='test')
    print(f'{exp_name} finished training')
