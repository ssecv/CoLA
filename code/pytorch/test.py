import time
import torch
import torch.nn.functional as F
import sys
import numpy as np
import os, argparse
import cv2
from Net import BaselineControlNet
from data import test_dataset
import numpy as np
from torchvision import utils,transforms
from PIL import Image
import cv2
from thop import profile
from torchstat import stat
import clip
import options
import torchvision.transforms as transforms
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='PATH_TO_TEST',help='test dataset path')
opt = parser.parse_args()
print('检查dropout 概率')
print('检查切换的epoch')
dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = BaselineControlNet()
model.load_state_dict(torch.load('PATH_TO_MODEL'))
model.cuda()
model.eval()

clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")


test_datasets =['VT821','VT1000','VT5000']

for dataset in test_datasets:
    save_path = 'PATH_TO_SAVE' + dataset +'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'  
    gt_root = dataset_path + dataset + '/GT/'
    t_root=dataset_path +dataset +'/T/' 
    test_loader = test_dataset(image_root, gt_root,t_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,t, name, image_for_post, image_clip = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        t = t.cuda()
        res,_,_,_,_= model(image,t,[image_clip],epoch = opt.change_epoch+1)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')