import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision.models as models


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size, conv_filts, res_filts,
                 skip_filts, dilation_rate, res=True, skip=True):
        super(ResidualBlock, self).__init__()
        self.res = res
        self.skip = skip

        padding = int((kernel_size + ((kernel_size-1)*(dilation_rate-1)) - 1)/2)

        self.conv_filt = nn.Sequential(
            nn.Conv1d(in_channels=conv_filts, out_channels=res_filts, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation_rate, bias=False),
            nn.Tanh())

        self.conv_gate = nn.Sequential(
            nn.Conv1d(in_channels=conv_filts, out_channels=res_filts, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation_rate, bias=False), nn.Sigmoid())

        self.conv_res = nn.Sequential(
            nn.Conv1d(in_channels=res_filts, out_channels=conv_filts, kernel_size=1, stride=1,
                      padding=0, dilation=dilation_rate, bias=False))

        self.conv_skip = nn.Sequential(
            nn.Conv1d(in_channels=res_filts, out_channels=skip_filts, kernel_size=1, stride=1,
                      padding=0, dilation=dilation_rate, bias=False))

    def forward(self, x):
        outputs = dict()
        activation = self.conv_filt(x) * self.conv_gate(x)
        if self.res:
            outputs['res'] = self.conv_res(activation)
            outputs['res'] = outputs['res'] + x

        if self.skip:
            outputs['skip'] = self.conv_skip(activation)

        return outputs


class HSICClassifier(nn.Module):
    def __init__(self, num_classes,  dict_features, feature_len=0, gap_norm_opt='batch_norm'):
        super(HSICClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_opt = dict_features['feature_opt']
        self.feature_len = feature_len
        self.gap_norm_opt = gap_norm_opt
        self.classifier = dict_features['classifier']
        model_out_dim = 32
        if dict_features['learned_features']:
            if True:
                self.model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1,)
                # for name, module in self.model._modules.items():
                #     if 'bn' in name:
                #         self.model._modules[name] = nn.LayerNorm(self.model._modules[name].weight.shape)
                conv1_old = self.model.conv1
                conv1_new = nn.Conv2d(4,conv1_old.out_channels,
                                      kernel_size=conv1_old.kernel_size,stride=conv1_old.stride,
                                      padding=conv1_old.padding,dilation=conv1_old.dilation,
                                      bias=conv1_old.bias
                                      )
                with torch.no_grad():
                    new_weights = torch.tensor(torch.hstack([conv1_old.weight.data,conv1_old.weight.data.mean(1).unsqueeze(1)]))
                    conv1_new.weight = nn.Parameter(new_weights)
                    self.model.conv1 = conv1_new
                self.activation_size = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=model_out_dim, bias=True)
                # nn.Identity()
            # self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

            # padding = kernel_size//2
            #
            # self.conv1 = nn.Sequential(
            #     nn.Conv1d(in_channels=in_channels, out_channels=conv_filts, kernel_size=kernel_size, stride=1,
            #               padding=padding, bias=False, dilation=1),
            #     nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=padding),
            # )
            # self.res_block2 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                 skip_filts=skip_filts, dilation_rate=2, res=True, skip=True)
            #
            # self.res_block3 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                 skip_filts=skip_filts, dilation_rate=4, res=True, skip=True)
            #
            # self.res_block4 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                 skip_filts=skip_filts, dilation_rate=8, res=True, skip=True)
            #
            # self.res_block5 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                 skip_filts=skip_filts, dilation_rate=16, res=True, skip=True)
            #
            # self.res_block6 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                 skip_filts=skip_filts, dilation_rate=32, res=True, skip=True)
            #
            # self.res_block7 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                 skip_filts=skip_filts, dilation_rate=64, res=True, skip=True)
            #
            # self.res_block8 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                 skip_filts=skip_filts, dilation_rate=128, res=True, skip=True)
            #
            # self.res_block9 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                 skip_filts=skip_filts, dilation_rate=256, res=True, skip=True)
            #
            # self.res_block10 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
            #                                  skip_filts=skip_filts, dilation_rate=512, res=False, skip=True)
            #
            # self.tail = nn.Sequential(
            #     nn.ReLU(),
            #
            #     nn.Dropout(p=0.3),
            #     nn.Conv1d(in_channels=skip_filts, out_channels=256, kernel_size=kernel_size, stride=1,
            #               padding=padding, bias=False, dilation=1),
            #     nn.ReLU(),
            #     nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=padding),
            #     nn.Dropout(p=0.3),
            #     nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=1,
            #               padding=padding, bias=False, dilation=1),
            #     nn.ReLU(),
            #     nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=padding),
            #     nn.Dropout(p=0.3)
            # )
            #
                if gap_norm_opt == 'batch_norm':
                    self.batch_norm = nn.BatchNorm1d(num_features=self.activation_size)

            elif False:
                self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,)
                conv1_old = self.model.features[0][0]
                conv1_new = nn.Conv2d(4, conv1_old.out_channels,
                                      kernel_size=conv1_old.kernel_size, stride=conv1_old.stride,
                                      padding=conv1_old.padding, dilation=conv1_old.dilation,
                                      bias=conv1_old.bias
                                      )
                with torch.no_grad():
                    new_weights = torch.tensor(
                        torch.hstack([conv1_old.weight.data, conv1_old.weight.data.mean(1).unsqueeze(1)]))
                    conv1_new.weight = nn.Parameter(new_weights)
                    self.model.features[0][0] = conv1_new
                self.activation_size = self.model.classifier[0].in_features
                self.model.classifier = nn.Linear(in_features=self.activation_size, out_features=model_out_dim,
                                          bias=True)

            else:
                self.model = Net()
                self.activation_size = 32


        else:
            self.activation_size = 0
            self.model = None
        ext_rep_size = self.feature_len
        # fc_layer_in_size = self.activation_size + ext_rep_size*int('concat' in self.feature_opt.lower())
        fc_layer_in_size = model_out_dim + ext_rep_size*int('concat' in self.feature_opt.lower())
        if self.classifier=='linear':
            self.fc_layer = nn.Linear(in_features=fc_layer_in_size, out_features=self.num_classes, bias=False)
        # self.fc_layer = nn.Linear(in_features=fc_layer_in_size, out_features=32, bias=True)
        if self.classifier == 'mlp2':
            self.fc_layer = MLP2Layer(in_size=fc_layer_in_size, hidden_size1=64, hidden_size2=64, out_size=self.num_classes)


        self.relu = nn.ReLU(inplace=True)

        self.gradients = None

    def forward(self, x, features=None, get_cam=False):
        # skips = list()
        if self.model is not None:
            b,n = x.shape[:2]
            x = x.view(-1,*x.shape[2:]) #(B,N_field,c,w,h) -> (B*N_field,c,w,h)
            output = self.model(x).reshape(b,n,-1) #(B*N_field,d) -> (B,N_field,d)
            gap = output

            output = torch.mean(output, dim=1)
            # gap = torch.mean(output, dim=1)

            # if self.gap_norm_opt == 'batch_norm' and gap.shape[0]>1:
            #     gap = self.batch_norm(gap)

            if ('concat' in self.feature_opt.lower()) and (self.feature_len > 0):
                output = torch.cat([output, features], dim=-1)

            logits = self.fc_layer(output)

            # weight_fc = list(self.fc_layer.parameters())[0][:, :output.shape[-1]]
            # weight_fc_tile = weight_fc.repeat(output.shape[0], 1, 1)
            #
            # cam = torch.bmm(weight_fc_tile, output.unsqueeze(2)) if get_cam else None
            cam=None
            # gap = gap[:, :self.activation_size]
        else:
            logits = self.fc_layer(features)
            cam=None
            gap=None


        return logits, cam, gap


class MLP1Layer(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(MLP1Layer, self).__init__()
        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(in_features=in_size, out_features=hidden_size, bias=True)
        self.fc_layer2 = nn.Linear(in_features=hidden_size, out_features=out_size, bias=True)

    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        return x


class MLP2Layer(nn.Module):
    def __init__(self, in_size, hidden_size1, hidden_size2, out_size):
        super(MLP2Layer, self).__init__()
        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(in_features=in_size, out_features=hidden_size1, bias=True)
        self.fc_layer2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2, bias=True)
        self.fc_layer3 = nn.Linear(in_features=hidden_size2, out_features=out_size, bias=True)

    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        x = self.relu(x)
        x = self.fc_layer3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 7, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()