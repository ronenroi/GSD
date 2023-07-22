import json
import os, glob
import numpy as np
import pandas as pd
import pickle as pkl
# import wfdb
import torch
# from wfdb import processing
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image

import ECG.feature_utils as futil


# def single_rr(sig):
#     xqrs = wfdb.processing.XQRS(sig=sig.ravel(), fs=300)
#     xqrs.detect()
#     rr = wfdb.processing.calc_rr(xqrs.qrs_inds, fs=300, min_rr=xqrs.rr_min, max_rr=xqrs.rr_max)
#     return rr
DATA_PATH = '/media/roironen/8AAE21F5AE21DB09/Data/GSD'

class GSDDataset(Dataset):

    def __init__(self, experiments, plates, feature_subset='all', feature_opt=None, oversample=False, is_baseline=False, naf=False,
                 idxs=None):

        self.feature_subset = feature_subset
        self.feature_opt = feature_opt
        self.feature_subset = feature_subset
        self.is_baseline = is_baseline
        self.data_dir = os.path.join(DATA_PATH)
        self.dataset_path = os.path.join(self.data_dir)
        self.idxs = idxs
        self.eng_features_names = ['Local Outlier Factor 10', 'Local Outlier Factor 8',	'Local Outlier Factor 30', 'calc_area1',
                                   'calc_area2', 'calc_area3', 'calc_area4', 'calc_text1', 'calc_intensity',
                                   'nuc_area1',	'nuc_area2', 'nuc_area3', 'nuc_area4', 'nuc_text1', 'nuc_intensity',
                                   'lyso_area2', 'lyso_area1', 'lyso_area3', 'lyso_text2', 'lyso_text1',
                                   'lyso_intensity1', 'tmre_area2', 'tmre_area1', 'tmre_area3', 'tmre_intensity1',
                                   'tmre_text1', 'tmre_text2']
        # These options are called only from plot_utils.py, so need to go 1 folder up
        # if (oversample == 'af') or (oversample == 'normal'):
            # assert 0  # need to check if this still works..
            # main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
            # dataset_path = os.path.join(main_dir, self.dataset_path)
            # data_dir = os.path.join(main_dir, data_dir)
        self.df = pd.read_csv(os.path.join(self.data_dir, 'df_clean_final_roi_proj.csv'))

        group_roi = self.df['group_roi'].to_list()
        row = []
        col = []
        pla = []
        exp = []
        mask_df = []
        for group in group_roi:
            p = (int(group.split('p')[-1]))
            e = int(np.floor((p-1) / 6)) + 1
            exp.append(e)
            plate_new = p - 6 * (e - 1)
            pla.append(plate_new)
            row.append(int(group.split('r')[-1].split('c')[0]))
            col.append(int(group.split('c')[-1].split('p')[0]))
            mask_df.append(np.any((e == experiments) * (plate_new == plates)))
            # new_col.append(c.split('NONE')[-1].split('p')[0]+f'p{plate_new}e{e}')

        self.df['experiment'] = exp
        self.df['plate_num'] = pla
        self.df['plate_row'] = row
        self.df['plate_col'] = col

        self.df = self.df.loc[mask_df]
        self.df = self.df.reset_index()
        self.load_labels()

        # if naf:
        #     # unify normal and other classes to label=0, AF is label=1
        #     self.labels.loc[self.labels.target == 2, 'target'] = 0
        #
        # if oversample == 'af':
        #     self.labels = self.labels[self.labels.target == 1]
        #     oversample = 'none'
        # elif oversample == 'normal':
        #     self.labels = self.labels[self.labels.target == 0]
        #     oversample = 'none'

        # self.labels = self.labels.reset_index()
        # self.labels.drop(columns=['index'], inplace=True)

        # self.waveform_path = os.path.join(self.dataset_path, 'waveforms')
        #
        # self.load_waveforms()

        self.feature_len = 0
        if oversample:
            self.sampler = self.balance_classes()
        else:
            self.sampler = None

        self.load_features()
        self.dataset_size = self.classes.shape[0]
        assert self.dataset_size == self.classes.shape[0] and self.dataset_size == self.df.shape[0] and \
               self.dataset_size == self.eng_features_val.shape[0]


    def get_field_images(self, index):
        # name = self.image_name[index][len(self.labels[self.classes[index]]):]
        # row = name.split('r')[-1].split('c')[0]
        # row = row if len(row) > 1 else '0' + row
        # col = name.split('c')[-1].split('p')[0]
        # col = col if len(col) > 1 else '0' + col
        # plate = str(int(name.split('p')[-1]))
        # plate = plate if len(plate) > 1 else '0' + plate
        try:
            exp = self.df['experiment'][index].astype(str)
            plate = self.df['plate_num'][index].astype(str)
            row = self.df['plate_row'][index].astype(str)
            row = row if len(row) > 1 else '0' + row
            col = self.df['plate_col'][index].astype(str)
            col = col if len(col) > 1 else '0' + col
            file_name = f'r{row}c{col}f*p01-ch{1}*'
            max_n_fields = len(glob.glob(os.path.join(self.dataset_path, f'exp{exp}', f'e{exp}p{plate}*', 'Images', file_name)))

            if max_n_fields==28:
                fields_ind = ['10', '11', '12', '13', '18', '17',  '01', '16', '19', '20', '21', '22', '28', '27', '26', '25']
            elif max_n_fields==32:
                fields_ind = ['09', '10', '11', '12', '17', '16',  '01', '15', '18', '19', '20', '21', '27', '26', '25', '24']
            elif max_n_fields==38:
                fields_ind = ['11', '12', '13', '14', '19', '18',  '01', '17', '22', '23', '24', '25', '31', '30', '29', '28']
            else:
                NotImplementedError()

            field_images = []


            for f in fields_ind:
                # channel_image = np.full(shape=(4,1024,1360),fill_value=np.nan,dtype=np.float32)
                channel_image = []
                for c in range(1, 5):

                    file_name = f'r{row}c{col}f{f}*p01-ch{c}*'
                    channel_images = glob.glob(os.path.join(self.dataset_path,f'exp{exp}',f'e{exp}p{plate}*','Images', file_name))[-1]

                    # for channel_image_path in channel_images:
                    ampl = int(channel_images.split('max_val')[-1].split('.jpeg')[0])
                    channel_image.append(np.array(Image.open(channel_images)).astype('float32') * ampl / 255)
                field_images.append(np.stack(channel_image))
            field_images = np.stack(field_images)#.swapaxes(0, 1)
        except:
            print()
        return field_images
    def __getitem__(self, index):
        target = self.classes[index]
        eng_features = self.eng_features_val[index, :]
        eng_features = eng_features.reshape(1, eng_features.shape[0]).astype('float32')

        field_images = self.get_field_images(index)


        # feature = np.zeros(1, dtype='float32')
        # feature_rep = np.zeros(1, dtype='float32')
        #
        # signal_name = self.labels.iloc[index]['signal']
        # real_feature = self.real_features.loc[signal_name.split(sep='_')[0]].values
        #
        # if self.feature_len > 0:
        #     feature = self.features.loc[signal_name.split(sep='_')[0]].values.astype('float32')
        #     feature_rep = feature  # relic from a time we used representations learned from another network

        return field_images, target, eng_features

    def __len__(self):
        return self.dataset_size

    def label2class(self, labels):
        self.labels = list(set(labels))
        classes = np.zeros(len(labels),dtype=int)-1
        for i in range(len(self.labels)):
            classes[np.array(self.labels[i])==labels] = i

        assert np.all(classes>=0)
        return classes

    def load_labels(self):
        # df = df.reset_index()
        self.classes = self.label2class(self.df['group'].tolist())


        # # self.labels = pd[.DataFrame.from_dict(data=json_dict, orient='index')]
        # # self.labels = self.labels.reset_index()
        # self.labels = self.labels.rename(columns={'index': 'signal', 0: 'target'})
        # # exclude noisy
        # self.labels = self.labels[self.labels.target != 3]
        # self.labels = self.labels.reset_index()
        # self.labels.drop(columns=['index'], inplace=True)

        if self.idxs is not None:
            self.labels = self.labels.iloc[self.idxs]


    def load_features(self):
        self.eng_features_val = self.df[self.eng_features_names].to_numpy()
        self.feature_len = self.eng_features_val.shape[1]
        assert (self.feature_len) == 27
        # self.image_name = self.df['group_roi'].to_numpy()

        # if ('HSIC' in self.feature_opt.lower()) or ('concat' in self.feature_opt.lower()):
        #
        #     if 'rr' in self.feature_subset:
        #         feature_names = futil.rr_feature_names
        #     elif 'all' in self.feature_subset:
        #         feature_names = futil.all_feature_names
        #     elif 'p_wave' in self.feature_subset:
        #         feature_names = futil.p_wave_feature_names
        #
        #     self.features = df[feature_names]
        #     self.feature_len = len(list(self.features))

            # if self.is_baseline:
            #     self.feature_len = 0

    def balance_classes(self):
        total = len(self.labels)
        self.labels['weights'] = 1

        counts = self.labels.target.value_counts()
        for i in range(2):
            total2 = counts[0] + counts[1]
            self.labels.loc[self.labels.target == i, 'weights'] = total2 / counts[i]
        self.labels.loc[self.labels.target == 2, 'weights'] = 0.0001  # just a small probability
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(self.labels.weights), replacement=True,
                                        num_samples=total)
        return sampler

    # def generate_all_rrs(self):
    #     max_len = 0
    #     rr_dict = {}
    #     for idx in range(self.dataset_size):
    #         signal_name = self.labels.iloc[idx]['signal']
    #         rr_dict[signal_name] = single_rr(self.waveforms[idx, :])
    #         if max_len < rr_dict[signal_name].shape[0]:
    #             max_len = rr_dict[signal_name].shape[0]
    #     for k, v in rr_dict.items():
    #         len_padding = max_len - v.shape[0]
    #         padding = np.ones(len_padding) * 0
    #         new_rr = np.append(v, padding)
    #         rr_dict[k] = new_rr
    #
    #     rr_df = pd.DataFrame.from_dict(rr_dict, orient='index')
    #     rr_df.to_csv(os.path.join(os.getcwd(), 'rr.csv'))
    #     return


def create_dataloaders(batch_size, feature_subset, feature_opt, naf):
    train_exp = val_exp =[1,2,3,4]
    n_train_plates = 6 * len(val_exp)
    train_val_plates = np.random.permutation(n_train_plates)+1
    train_val_plate_exp = np.array(train_exp).take(np.floor(train_val_plates/7).astype(int))
    train_val_plates = np.mod(train_val_plates-1,6)+1

    N = np.ceil(n_train_plates*0.8).astype(int)
    train_plate = train_val_plates[:N]
    val_plate = train_val_plates[N:]

    train_plate_exp = train_val_plate_exp[:N]
    val_plate_exp = train_val_plate_exp[N:]

    test_exp = np.array([5]*6)
    test_plate = np.arange(1,7)
    train_dataset = GSDDataset(experiments=train_plate_exp, plates=train_plate,feature_subset=feature_subset, feature_opt=feature_opt,
                               oversample=False, naf=naf)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                               sampler=train_dataset.sampler)

    val_dataset = GSDDataset(experiments=val_plate_exp, plates=val_plate, feature_subset=feature_subset, feature_opt=feature_opt,
                             oversample=False, naf=naf)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                             sampler=val_dataset.sampler)

    test_dataset = GSDDataset(experiments=test_exp, plates=test_plate, feature_subset=feature_subset, feature_opt=feature_opt,
                              oversample=False, naf=naf)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              sampler=test_dataset.sampler)

    return train_loader, val_loader, test_loader


def create_kfoldcv_loaders(batch_size, feature_subset, feature_opt, naf):
    num_folds = 5
    kfoldcv_testloaders = []
    test_dataset = GSDDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                              oversample=False, naf=naf)

    np.random.seed(24)
    idxs = (np.random.multinomial(1, 0.2 * np.ones(5).ravel(), size=len(test_dataset)) == 1).argmax(1).astype(int)

    for i_fold in range(num_folds):
        idx_test = idxs == i_fold
        idx_val = idxs == (i_fold + 1) % 5
        idx_train = ((idxs == (i_fold + 2) % 5) |
                     (idxs == (i_fold + 3) % 5) |
                     (idxs == (i_fold + 4) % 5))

        train_dataset = GSDDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                                   oversample=False, naf=naf, idxs=idx_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                                   sampler=train_dataset.sampler)
        val_dataset = GSDDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                                 oversample=False, naf=naf, idxs=idx_val)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                 sampler=val_dataset.sampler)
        test_dataset = GSDDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                                  oversample=False, naf=naf, idxs=idx_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  sampler=test_dataset.sampler)
        kfoldcv_testloaders.append((train_loader, val_loader, test_loader))
    return kfoldcv_testloaders
