import argparse
import vgg19net
from dfi import DFI
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import *

parser = argparse.ArgumentParser('this is the DFI parser')
parser.add_argument('--data-dir', '-d', default='./data', type=str, help='Path to data directory containing the images')
parser.add_argument('--pretrained', '-pr', default=False, action='store_true', help='using vgg19 pretrained net parameters')
parser.add_argument('--layers', default='conv3_1,conv4_1,conv5_1', type=str, help='Comma separated list of layers')
parser.add_argument('--feature', '-f', default='Sunglasses', type=str, help='Name of the feature')
parser.add_argument('--person-index','-p', default=0, type=int, help='Index of the start image')
parser.add_argument('--person-image', default='./data/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg', type=str, help='Start image path')
# parser.add_argument('--person-image', default='./data/lfw-deepfunneled/Jose_Sarney/Jose_Sarney_0001.jpg', type=str, help='Start image path')
parser.add_argument('--list-features', '-l', default=False, action='store_true', help='List all available features')
parser.add_argument('--optimizer', '-o', default=False, type=str, help='Optimizer type')
parser.add_argument('--lr', '-lr', default=0.05, type=float, help='learning rate of dfi')
parser.add_argument('--steps', '-s', type=int, default=50000, help='Number of steps')
parser.add_argument('--eps','-e', type=str, help='Epsilon interval in log10')
parser.add_argument('--tk', default=False, action='store_true', help='Use TkInter')
parser.add_argument('--K', '-k', default=10, type=int, help='Number of nearest neighbors')

parser.add_argument('--diff', default=6, type=int, help='Name of the feature')
parser.add_argument('--alpha', '-a', type=float, default=1.2, help='Alpha_param, using for W weight')
parser.add_argument('--beta', '-b', help='Beta param', type=float, default=2)


parser.add_argument('--lamb', type=float, default=0.0001, help='Lambda param, using for tv_loss')
parser.add_argument('--rebuild-cache', '-rc', help='Rebuild the cache', default=False,
                    action='store_true')
parser.add_argument('--random-start', '-rs', help='Set versbose', action='store_true',
                    default=False)
parser.add_argument('--invert', '-i', help='Invert deep feature difference(No beard -> beard)',
                    default=False, )
parser.add_argument('--output', '-out', type=str, help='output directory',
                    default='out')
parser.add_argument('--discrete-knn', default=False, action='store_true',
                    help='为knn启用离散化功能呢个，Enable feature discretization of knn')
parser.add_argument('--epoch', '-ep', default=1000, type=int, help='')
parser.add_argument('--use-cuda', '-cuda', action='store_true', default=True, help='use gpu uprate')
parser.add_argument('--save-dir', '-sd', default = './out/', type=str, help='save the output images from the torch network')
parser.add_argument('--save-step', '-ss', default=1000, type=int, help='the save step of the image')
FLAGS = parser.parse_args()
FLAGS.feature = FLAGS.feature.replace('_',' ')
FLAGS.layers = [l+'/Relu:0' for l in FLAGS.layers.split(',')]
print(FLAGS.feature)
# test = plt.imread(FLAGS.person_image)
# test = test[:224, :224, :]
#
# test = torch.tensor(test, device=torch.device('cuda'))
# test = test.float()
# test.rqures_grad = True
# out1, out2, out3, _ = vggnet(test.view(1, 3, 224, 224))

dfi = DFI(FLAGS)
dfi.run()
print('finish')


