import os
import torch
import vgg19net
import torch.nn as nn
import sys
from torchvision.transforms import transforms
import numpy as np
sys.path.append('.')
from utils import load_discrete_lfw_attributes,reduce_img_size,load_images

_VGG_MEAN = [103.939, 116.779, 123.68]

class vgg19(nn.Module):

    WIDTH = 224
    HEIGHT = 224
    CHANNELS = 3

    model = {}
    model_save_path = None
    model_save_freq = 0

    learning_rate = 0.05

    _inpurRGB = None
    _inpurBGR = None
    _inputNormalizedBGR = None

    _preds = None
    'the predictions tensor, shape of [?,1000]'

    _loss = None
    _optimizer = None
    _train_labels = None
    
    net = None

    device = torch.device('cuda')
    transform_train = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])
    
    
    transform_test = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
         ])
    transform_PIL = transforms.Compose(
        [
            transforms.ToPILImage()
        ])


    def __init__(self,
                 model=None,
                 model_save_path = None,
                 model_save_freq=0,
                 inputs_palceholder=True,
                 data_dir=None,
                 random_start=True,
                 start_image_path=0
                 ):
        """
        this is the training model used for get out3,out4,out5
        :param model: the model either for back-propagation or forward-propagation.
        :param model_save_path: The model path for training process.
        :param model_save_freq: Save the model (in training process) every N
        :param inputs_palceholder:
        :param data_dir:
        :param random_start:
        :param start_image_path:
        """
        '''
        labels = [None,1000]
        inputRGB = [None,3,224,224]
        
        tensorflow 中可以使用占位符号，pytorch不需要使用占位
        '''
        super(vgg19, self).__init__()

        self.model = model
        self.model_save_path = model_save_path
        self.model_save_freq = model_save_freq

        if inputs_palceholder:
            self._inpurRGB = torch.zeros((1, 3, 224, 224), device=self.device)

        else:
            if random_start:
                rand_img = torch.randint(0, 255, [1, 3, 224, 224], device=self.device)
                self._inpurRGB = rand_img
            else:
                path = start_image_path

                start_img = reduce_img_size(load_images(*[path]))[0]
                self._inpurRGB = np.reshape(start_img, [-1, 3, 224, 224])
        self.net = vgg19net.VGGNET_v2().get_net()

        r, g, b = (self._inpurRGB[0, i] for i in range(3))
        self._inputBGR = np.dstack([r, g, b]).reshape(-1, 3, 224, 224)

    def forward(self):
        return None

    @property
    def inputRGB(self):
        return self._inpurRGB

    @property
    def inputBGR(self):
        return self._inpurBGR

    @property
    def preds(self):
        return self._preds

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def loss(self):
        return self.loss

    @property
    def optimizer(self):
        return self._optimizer