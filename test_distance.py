# -*- coding: utf-8 -*-

import vgg19net
import torch
import numpy as np
import PIL.Image as Image 
vnet = vgg19net.VGGNET_v2().get_net()
start_img = Image.open('start_img.png').convert('RGB')
start_img = np.array(start_img).reshape(-1,3,224,224)
start_tensor = torch.tensor(start_img,requires_grad=True,dtype=torch.float)
outs1,outs2,outs3,_ = vnet(start_tensor)

test_img = Image.open('./out/outpour54000.jpg').convert('RGB')
test_img = np.array(test_img).reshape(-1,3,224,224)
test_tensor = torch.tensor(test_img,requires_grad=True,dtype=torch.float)
oute1,oute2,oute3,_ = vnet(test_tensor)