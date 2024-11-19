import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from Net import BaselineControlNet
from data import test_dataset
import numpy as np
import cv2
import clip
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
# 指定模式
parser.add_argument('--modality', type=str, default='Modality-Complete', help='select modality of [Modality-Complete,Modality-Only-RGB,Modality-Only-T]')
# save_path
parser.add_argument('--test_path',type=str,default='RGBT_Dataset/',help='test dataset path')
# checkpoint path
parser.add_argument('--checkpoint', type=str, default='checkpoint/rgb-t.pth', help='model checkpoint path')
opt = parser.parse_args()
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
model.load_state_dict(torch.load(opt.checkpoint), strict=False)
model.cuda()
model.eval()

clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")


test_datasets =['VT821','VT1000','VT5000']
if opt.modality == 'Modality-Complete':
    Path_Modality_1  = 'RGB/'
    Path_Modality_2  = 'T/'
elif opt.modality == 'Modality-Only-RGB':
    Path_Modality_1  = 'RGB/'
    Path_Modality_2  = 'black/'
elif opt.modality == 'Modality-Only-T':
    Path_Modality_1  = 'black/'
    Path_Modality_2  = 'T/'


for dataset in test_datasets:
    save_path = '/mnt/data4T/ZCL-RGB-T/our/result/'+ opt.modality + '/' + dataset +'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/' + Path_Modality_1  
    gt_root = dataset_path + dataset + '/GT/' 
    t_root=dataset_path +dataset +'/' + Path_Modality_2 
    test_loader = test_dataset(image_root, gt_root,t_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,t, name, image_for_post, image_clip = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        t = t.cuda()
        res,_,_,_,_= model(image,t,[image_clip])
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')