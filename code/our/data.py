import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import cv2
import torch

#several data augumentation strategies
def cv_random_flip(img, label,depth):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth
def randomCrop(image, label,depth):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region)
def randomRotation(image,label,depth):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
    return image,label,depth

def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def gauss_peper(img, ksize, sigma):
    n = np.random.randint(10)
    if n == 1:
        k_list = list(ksize)
        kw = (k_list[0] * 2) + 1
        kh = (k_list[1] * 2) + 1
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        resultImg = cv2.GaussianBlur(img, (kw, kh), sigma)
        resultImg= Image.fromarray(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB))
        img=resultImg
        img = np.array(img)
        noiseNum = int(0.75 * img.shape[0] * img.shape[1])
        for i in range(noiseNum):
            randX = random.randint(0, img.shape[0] - 1)
            randY = random.randint(0, img.shape[1] - 1)
            if random.randint(0, 1) == 0:
                img[randX, randY] = 0
            else:
                img[randX, randY] = 255
        img=Image.fromarray(img)

    return img


# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,t_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.ts=[t_root + f for f in os.listdir(t_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.ts=sorted(self.ts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.ts_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        t=self.binary_loader(self.ts[index])
        image,gt,t =cv_random_flip(image,gt,t)
        image,gt,t=randomCrop(image, gt,t)
        image,gt,t=randomRotation(image, gt,t)
        image=colorEnhance(image)
        
        image=gauss_peper(image, (3, 3), 0)
        image_clip = image
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        t=self.ts_transform(t)

        image=image.permute(1, 2, 0)
        gt=gt.permute(1, 2, 0)
        t=t.permute(1,2,0)
         
        
        return image, gt, t, image_clip
    
    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        ts=[]
        for img_path, gt_path,t_path in zip(self.images, self.gts, self.ts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            t= Image.open(t_path)
            if img.size == gt.size and gt.size==t.size:
                images.append(img_path)
                gts.append(gt_path)
                ts.append(t_path)
        self.images = images
        self.gts = gts
        self.ts=ts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, t):
        assert img.size == gt.size and gt.size==t.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),t.resize((w, h), Image.NEAREST)
        else:
            return img, gt, t

    def collate(self, batch):
        # size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        size = 384
        image, mask, edge, image_clip = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            
            image[i] = cv2.resize(np.asarray(image[i]), dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(np.asarray(mask[i]), dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i] = cv2.resize(np.asarray(edge[i]), dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            # image_clip[i] = cv2.resize(np.asarray(image_clip[i]), dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        return image, mask, edge, image_clip

    def __len__(self):
        return self.size



#dataloader for training
def get_loader(image_root, gt_root,t_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, t_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  collate_fn=dataset.collate,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  )
    return data_loader

#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root,t_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.ts=[t_root + f for f in os.listdir(t_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.ts=sorted(self.ts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.ts_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image_clip = image
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        t=self.binary_loader(self.ts[self.index])
        t=self.ts_transform(t).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt,t, name,np.array(image_for_post), image_clip

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size