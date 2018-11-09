from time import strftime, gmtime

import matplotlib as mpl
# DFI_V2

mpl.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from PIL import Image

import torch
from torch import optim
from torch import nn
from sklearn.neighbors import KNeighborsClassifier
from utils import load_discrete_lfw_attributes,load_lfw_attributes,get_person_idx_by_path,reduce_img_size,load_images
from vgg19net import VGGNET_v2
import os.path
import vgg19
from tensorboardX import SummaryWriter
writer = SummaryWriter()
import numpy as np
import tqdm as tqdm

class DFI:
    _conv_layer_tensors = None

    def __init__(self, flags):
        self._nn = vgg19.vgg19()
        self.use_transfer = True
        self.vnet = VGGNET_v2().get_net().cuda().train()
        for i in self.vnet.parameters():
            i.requires_grad =False
        self.FLAGS = flags
        self.pretrained = self.FLAGS.pretrained
        self._conv_layer_tensors = ['conv3_1,conv4_1,conv5_1']
        self._loss_log = []

        self.device = torch.device('cuda') if self.FLAGS.use_cuda else torch.device('cpu')
        
    def run(self):
        """
        start DFI
        :return: None
        """
        print('starting DFI')

        phi_z = self._get_phi_z_const()
        reversed_mapped_z = self._reverse_map_z(phi_z)
        # self._save_output(reversed_mapped_z)

    def _reverse_map_z(self, phi_z):
        vgg = vgg19.vgg19(model=None,
                          inputs_palceholder=False,
                          data_dir=self.FLAGS.data_dir,
                          random_start=self.FLAGS.random_start,
                          start_image_path=self.FLAGS.person_image
                          )
        
        self._nn = vgg
        return self._optimize_z_torch(phi_z)

    def _optimize_z_torch(self, phi_z):
        phi_z_const_tensor = phi_z
        inputs_image = Image.open('start_img.png').convert('RGB')
        inputs_image.save('./out/start_img.png')
        inputs_numpy = self._nn.transform_test(inputs_image).detach().numpy().reshape(1, 3, 224, 224)
#        inputs_numpy = torch.rand(1, 3, 224, 224).numpy() 
#        Image.fromarray(inputs_numpy.transpose(2,0,1).astype('float')).save('./out/input_z.png')
        plt.imsave('./out/input_z.jpg', inputs_numpy.reshape(3, 224, 224).transpose(1,2,0).astype('float'))
        
        input_z = torch.tensor(inputs_numpy, requires_grad = True, device=self.device, dtype = torch.float)
#        plt.imsave('{}z_input.png'.format(self.FLAGS.save_dir), (inputs_numpy).reshape(224, 224, 3))
        optimizer = optim.Adam([input_z], lr=self.FLAGS.lr)
        writer.add_graph(self.vnet, (input_z, ))
        diffloss_fn = Myloss()
        tvloss_fn = TVLoss()
        luloss_fn = IU_loss()
        it = tqdm.tqdm(range(self.FLAGS.steps + 1))
        
        for i in it:
            # 这一段将inputs作为梯度下降的参数，然后通过对比神经网络中conv31。conv41。conv51的不同之处来反向传播梯度

            optimizer.zero_grad()
            z_out = self._z_prime(input_z)
            phi_z_const_tensor = torch.tensor(phi_z_const_tensor, dtype=torch.float, device=self.device)
            diff_loss = self.FLAGS.diff * diffloss_fn(self.FLAGS, input_z, z_out, phi_z_const_tensor)
            tv_loss = self.FLAGS.lamb * tvloss_fn(input_z)*0.2
            lu_loss =4 * luloss_fn(input_z)
            
            loss = diff_loss + tv_loss + lu_loss
            writer.add_scalars('data/loss_group',{'total_loss':loss,
                                                  'diff_loss':diff_loss,
                                                  'tv_loss':tv_loss,
                                                  'lu_loss':lu_loss,
                                                  },i)

#            print("-------the lr is {}, the loss is {:.9f}, the step is {} ------".format(self.FLAGS.lr, loss, i))
            loss.backward()
            optimizer.step()

            if i % self.FLAGS.save_step == 0:
                save_dir = self.FLAGS.save_dir
                image = input_z.cpu().detach().squeeze()
                image = self._re_normalize(image)
                image = self._nn.transform_PIL(image)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                writer.add_image('data/output_image',np.array(image).transpose(2,0,1),i)
                image.save('{}/outpour{}.jpg'.format(self.FLAGS.save_dir, i), 'jpeg')

        writer.export_scalars_to_json('./test.json')
        writer.close()


    def _z_prime(self, img_tensor):
        vnet = self.vnet
        out3z, out4z, out5z, _ = vnet(img_tensor)
        out3n, out4n, out5n = out3z.cpu().detach().view(-1).numpy(),out4z.cpu().detach().view(-1).numpy(),  out5z.cpu().detach().view(-1).numpy()

        phi_img = np.array([])
        phi_img = np.append(phi_img, out3n.reshape(-1))
        phi_img = np.append(phi_img, out4n.reshape(-1))
        phi_img = np.append(phi_img, out5n.reshape(-1))

        phi_img_tensor = torch.cat([out3z.view(-1), out4z.view(-1), out5z.view(-1)])
        square = phi_img_tensor * phi_img_tensor
        reduce_sum = torch.sum(square)
        sqrt = torch.sqrt(reduce_sum)
        return phi_img_tensor/sqrt

    def _phi(self, imgs):
        print('get phi!')
        """
        将图像转换到深度空间上去
        :param imgs:input images
        :return:deep feature transformed images
        """
        if not isinstance(imgs, list):
            input_images = [imgs]
        else:
            input_images = imgs

        inputs = []
        if self.use_transfer:
            for im in input_images:
                im = Image.fromarray(im.astype('uint8')).convert('RGB')
                inputs.append(self._nn.transform_train(im).numpy())
            input_images = np.array(inputs)

        input_images = torch.tensor(input_images, device=torch.device('cuda'), dtype=torch.float, requires_grad=True).view(-1, 3, 224, 224)

        out3t, out4t, out5t, _ = self.vnet(input_images)
        out3, out4, out5 = out3t.cpu().detach().numpy(), out4t.cpu().detach().numpy(), out5t.cpu().detach().numpy()
        res = []
        for img_idx in range(len(input_images)):
            phi_img = np.array([])

            phi_img = np.append(phi_img, out3[img_idx].reshape(-1))
            phi_img = np.append(phi_img, out4[img_idx].reshape(-1))
            phi_img = np.append(phi_img, out5[img_idx].reshape(-1))

            res.append(phi_img)

        if not isinstance(imgs, list):
            return res[0] / np.linalg.norm(res[0])
        else:
            return [x / np.linalg.norm(x) for x in res]


    def _get_phi_z_const(self):
        if self.FLAGS.discrete_knn:
            atts = load_discrete_lfw_attributes(self.FLAGS.data_dir)
        else:
            atts = load_lfw_attributes(self.FLAGS.data_dir)
        imgs_path = atts['path'].values

        if self.FLAGS.person_image:
            start_img_path = self.FLAGS.person_image
        else:
            start_img_path = imgs_path[self.FLAGS.person_index]

        person_index = get_person_idx_by_path(atts, start_img_path)
        start_img = reduce_img_size(load_images(*[start_img_path]))[0]
        plt.imsave(fname='start_img.png', arr=start_img)

        pos_paths, neg_paths = self._get_sets(atts, self.FLAGS.feature, person_index)

        pos_imgs = reduce_img_size(load_images(*pos_paths))
        neg_imgs = reduce_img_size(load_images(*neg_paths))

        # 此处运行网络获得 @w
        pos_deep_features = self._phi(pos_imgs)
        neg_deep_features = self._phi(neg_imgs)

        w = np.mean(pos_deep_features, axis=0) - np.mean(neg_deep_features, axis=0)
        w /= np.linalg.norm(w)

        inv = -1 if self.FLAGS.invert else 1
        # 此处运行网络获得 @x
        phi = self._phi(start_img)
        print('w_sum is :', np.sum(phi))
        phi_z = phi + self.FLAGS.alpha * w * inv
#        phi_z = phi
        print('w_sum is :', np.sum(phi_z))
        np.save('chache.ch', phi_z)
        return phi_z




    def _get_sets(self, atts, feat, person_index):
        person = atts.loc[person_index]

        del person['person']
        del person['path']

        atts = atts.drop(person_index)

        pos_set = atts.loc[atts[feat] > 0]
        neg_set = atts.loc[atts[feat] < 0]

        pos_paths = self._get_k_neighbors(pos_set, person)
        print(pos_paths)
        neg_paths = self._get_k_neighbors(neg_set, person)
        print(pos_paths)

        return pos_paths.as_matrix(), neg_paths.as_matrix()

    def _get_k_neighbors(self, subset, person):
        """
        这里将获取与初始图片相似的钱k个相关图片，subset.shape[0]代表原始subset的长度
        :param subset:带有所有图像参数的dataframe
        :param person:
        :return:
        """
        del subset['person']
        paths = subset['path']
        del subset['path']

        knn = KNeighborsClassifier(n_jobs=-1)

        dummy_target = [0 for x in range(subset.shape[0])]
        knn.fit(X=subset.as_matrix(), y=dummy_target)

        knn_indices = knn.kneighbors(person.as_matrix(), n_neighbors=self.FLAGS.K, return_distance=False)[0]
        print('knn indices is ', knn_indices)
        neighbor_paths = paths.iloc[knn_indices]

        return neighbor_paths

    def features(self):
        atts = load_lfw_attributes(self.FLAGS.data_dir)
        del atts['path']
        del atts['person']
        return atts.columns.values

    def _re_normalize(self,normalized_image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for t, m, s in zip(range(3), mean, std):
            normalized_image[t] = normalized_image[t] * s + m
        return normalized_image

class Myloss (nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()


    def forward(self, flags, z_prime, phi_z_prime, phi_z_const_tensor):

        subtract = phi_z_prime - phi_z_const_tensor
        square = (subtract)** 2
        reduce_sum = torch.sum(square)
        diff_loss = 0.5 * reduce_sum

        loss = diff_loss 
        return loss
        

class IU_loss(nn.Module):
    def __init__(self):
        super(IU_loss,self).__init__()
    
    def forward(self, z_prime):
        shape = torch.Tensor([224, 224, 3])
        shape = shape.float().cuda()
        loss_lower = 0.05 * -1 * torch.sum(z_prime - torch.abs(z_prime)) / torch.prod(shape)
        sub = (z_prime - 1.0)
        loss_upper = torch.sum((sub + torch.abs(sub)) / 2.0) / torch.prod(shape)
        loss = loss_lower + loss_upper
        return loss


class TVLoss(nn.Module):
  # 设置网络的loss函数
  def __init__(self, eps=1e-08, beta=2):
    super(TVLoss, self).__init__()
    self.eps = eps
    self.beta = beta

  def forward(self, input):
    x_diff = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
    y_diff = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]

    sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, self.eps, 10000000)
    return torch.norm(sq_diff, self.beta / 2.0) ** (self.beta / 2.0)

