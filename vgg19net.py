# -*- coding: utf-8 -*-
import gc

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

device = torch.device('cuda')


class VGGNET():
    def __init__(self, pretained=False):
        self.vggnet = vgg.vgg19(pretrained=pretained)

    def getnet(self):
        return self.vggnet
    #


class VGGNET_v2(torch.nn.Module):

    def __init__(self, pretained=True, num_classes=1000):
        """
        这里将原来的vggnet进行了一次改造，将原来的单输出变为4输出，可以获取
        conv3_1,conv4_1,conv5_1的特征图。
        :param pretrained:
        :param num_classes:
        """
        super(VGGNET_v2, self).__init__()
        self.net = None
        self.layers_before = self.features_A_before()
        self.layers_A = self.features_A()
        self.layers_A_next = self.features_A_next()
        self.layers_B = self.features_B()
        self.layers_B_next = self.features_B_next()
        self.layers_C = self.features_C()
        self.layers_C_next = self.features_C_next()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.layers_before(x)
        x3 = self.layers_A(x)
        x = self.layers_A_next(x3)
        x4 = self.layers_B(x)
        x = self.layers_B_next(x4)
        x5 = self.layers_C(x)
        x = self.layers_C_next(x5)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x3, x4, x5, x

    def get_net(self):
        self.vnet = VGGNET_v2()
        # net.load_state_dict()
        load_dict = model_zoo.load_url(model_urls['vgg19'])
        aims_dict = self.vnet.state_dict()
        
        for i,j in zip(aims_dict, load_dict):
            aims_dict[i] = load_dict[j]
        self.vnet.load_state_dict(aims_dict)
        
        return self.vnet

    def clear_storage(self):
        try:
            del self.net
            gc.collect()
            print('clear net finish')
        except:
            print('fail, storage is in using')

    def features_A_before(self):
        cfg = [64, 64, 'M', 128, 128, 'M']
        return self.make_layers(cfg)

    def features_A(self):
        cfg = [256]
        return self.make_layers(cfg, in_channels=128)
    def features_A_next(self):
        cfg = [256, 256, 256, 'M']
        return self.make_layers(cfg, in_channels=256)

    def features_B(self):
        cfg = [512]
        return self.make_layers(cfg, in_channels=256)

    def features_B_next(self):
        cfg = [512, 512, 512, 'M']
        return self.make_layers(cfg, in_channels=512)

    def features_C(self):
        cfg = [512]
        return self.make_layers(cfg, in_channels=512)

    def features_C_next(self):
        cfg = [512, 512, 512, 'M']
        return self.make_layers(cfg, in_channels=512)

    def make_layers(self, cfg, batch_norm=False, in_channels=3):
        layers = []
        in_channels = in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


