import torch
import random
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Net import Baseline,BaselineControlNet
from data import get_loader,test_dataset,SalObjDataset
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
import pytorch_iou
import os
import options
from PIL import Image


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#build the model
model = BaselineControlNet()
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ',opt.load)


model.cuda()
params = model.parameters()
for param in model.clip_alpha.clip_model.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,params), lr=opt.lr)

#set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
t_root=opt.t_root
save_path=opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
image_paths = []
image_paths = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
image_paths = sorted(image_paths)
train_loader = get_loader(image_root, gt_root,t_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Baseline-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))

bce_loss = torch.nn.BCELoss(size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

#set loss function
def bce_iou_loss(pred,target):

    bce_out = bce_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss = bce_out + iou_out

    return loss

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate, decay_epoch):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

# fixed random seed
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_torch()

step=0
writer = SummaryWriter(save_path+'summary')
best_mae=1
best_epoch=0

#train function
def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    
    try:
        for i, (images, gts, ts, image_clips) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            ts=ts.cuda()
            #冻结clip
            if epoch  == opt.change_epoch+1:
                for param in model.clip_alpha.clip_model.parameters():
                    param.requires_grad = False
                for param in model.baseline.parameters():
                    param.requires_grad = False
                    param.detach()
                model.baseline.eval()
                for param in model.decoder.parameters():
                    param.requires_grad = False
                    param.detach()
                model.decoder.eval()
                for param in model.clip_alpha.parameters():
                    param.requires_grad = False
                    param.detach()
                params = model.parameters()
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,params), lr=opt.lr)
            s1,u4,u3,u2,u1 = model(images, ts, image_clips, epoch)

            loss1 = bce_iou_loss(s1, gts)
            loss2 = bce_iou_loss(u4, gts)
            loss3 = bce_iou_loss(u3, gts)
            loss4 = bce_iou_loss(u2, gts)
            loss5 = bce_iou_loss(u1, gts)

            loss=loss1+loss2+loss3+loss4+loss5
            loss.backward(retain_graph=True)

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 100 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}'.
                             format(epoch, opt.epoch, i, total_step, loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res=s1[0].clone()
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s1', torch.tensor(res), step,dataformats='HW')
               
            
        loss_all/=epoch_step
        logging.basicConfig(filename='PATH_TO_LOG', level=logging.INFO)
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 20 == 0:
            torch.save(model.state_dict(), save_path+'TNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'TNet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        if epoch <= opt.change_epoch:
            cur_lr=adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        else:
            cur_lr=adjust_lr(optimizer, opt.lr, epoch-opt.change_epoch, opt.decay_rate, decay_epoch=25)
        print('clip lr:',cur_lr)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch,save_path)
        if epoch == opt.change_epoch:
            model.baseline_ctrl.load_state_dict(model.baseline.state_dict())