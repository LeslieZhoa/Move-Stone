import ntpath
# import onnx
# from onnx_tf.backend import prepare
import os
import sys
import warnings

import numpy as np
import tqdm
from torch import nn

import torch
from torch.autograd import Variable

import os
from collections import OrderedDict
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
import torch


def arg_parser(args):
    parser = argparse.ArgumentParser()

        
    parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')  
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')    
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--netG', type=str, default='douyin_a_taiyi', help='import net')  
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')

    return parser.parse_args(args)

opt = arg_parser(sys.argv[1:])

def get_transform():
    transform_list = []
    

    transform_list += [transforms.ToTensor()]

    
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class ExportNet(nn.Module):
    def __init__(self, opt):
        super(ExportNet, self).__init__()
        # self.net = Net()
        if opt.netG.startswith('douyin_'):
            sub = opt.netG[7:]
            m = __import__('Pix2Pix.douyin_generator_{}'.format(sub), fromlist=True)
            norm_layer = get_norm_layer(norm_type=opt.norm)
            self.model = m.DouyinGenerator(opt.input_nc, opt.output_nc, opt.ngf, norm_layer=norm_layer, dropout_rate=0.0, n_blocks=9)       

    def forward(self, img):
        # preprocess
        img = img.permute(0, 3, 1, 2) / 255.
        img = img * 2 - 1
        # infer
        pred1,pred2 = self.model(img)
        # pred = self.net(img)
        # postprocess
        pred1 = (pred1 + 1) / 2
        pred1 = pred1.permute(0,2,3,1) * 255.
        return pred1.to(dtype=torch.uint8),pred2

def export_model(opt):
    # the input image shape of your network
    input_shape = (3, 256, 256)
    export_net = ExportNet(opt).model.netG.cpu()
    export_net.eval()
    remove_all_spectral_norm(export_net)
    input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2], requires_grad=True).to(dtype=torch.float32)
    output1,output2 = export_net(input)
    torch.onnx.export(export_net,
                      input,
                      os.path.join('./checkpoint/', 'color.onnx'),
                      export_params=True,
                      opset_version=9,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output1',"output2"],
                      dynamic_axes={'input':{0:'batch_size'},
                                    'output1':{0:'batch_size'},
                                    'output2':{0:'batch_size'}})


def onnx2pb(onnx_input_path, pb_output_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model, strict=False)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model
    pass


def remove_all_spectral_norm(item):
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass

        for child in item.children():
            remove_all_spectral_norm(child)

    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)

    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)

if __name__ == '__main__':
    # opt = TestOptions().parse()
    # print(' '.join(sys.argv))
    export_model(opt)
    # onnx2pb('model.onnx','model.pb')

