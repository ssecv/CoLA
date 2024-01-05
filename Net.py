from statistics import mode
import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50, ResNet101
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import options
import clip
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
import clip
import torchvision.transforms as transforms
from options import opt
from PIL import Image

black_path = opt.black_path

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        x = self.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,align_corners=True)
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
    def forward(self, ftr):
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True)
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True)
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1)
        att_map = F.sigmoid(self.conv(ftr_cat))
        return att_map

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class ChannelConv(nn.Module):
    def __init__(self, in_1, in_2):
        super(ChannelConv, self).__init__()
        self.conv1 = convblock(in_1, 128, 3, 1, 1)
        self.conv_out = convblock(128, in_2, 3, 1, 1)

    def forward(self, pre,cur):
        cur_size = cur.size()[2:]
        pre = self.conv1(F.interpolate(pre, cur_size, mode='bilinear', align_corners=True))
        fus = pre
        return self.conv_out(fus)

class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class FinalOut(nn.Module):
    def __init__(self):
        super(FinalOut, self).__init__()
        self.ca =CA(128)
        self.score = nn.Conv2d(128, 1, 1, 1, 0)
    def forward(self,f1,f2,xsize):
        f1 = torch.cat((f1,f2),1)
        f1 = self.ca(f1)
        score = F.interpolate(self.score(f1), xsize, mode='bilinear', align_corners=True)
        return score

class SaliencyNet(nn.Module):
    def __init__(self):
        super(SaliencyNet, self).__init__()
        self.c4=nn.Conv2d(2048*2,2048,kernel_size=1)
        self.c3=nn.Conv2d(1024*2, 1024, kernel_size=1)
        self.c2=nn.Conv2d(512*2,512,kernel_size=1)
        self.c1 = nn.Conv2d(256*2, 256, kernel_size=1)
        self.c = nn.Conv2d(64*2, 64, kernel_size=1)
        self.Convs = [self.c4, self.c3, self.c2, self.c1]
        self.spa = SpatialAttention()
        self.ca4 = CA(2048*2)
        self.ca3 = CA(2048)
        self.ca2 = CA(1024)
        self.ca1 = CA(512)
        self.ca = CA(128)
        self.CAs = [self.ca4, self.ca3, self.ca2, self.ca1, self.ca]

        self.ChannelConv4 = ChannelConv(2048,1024)
        self.ChannelConv3= ChannelConv(1024,512)
        self.ChannelConv2= ChannelConv(512,256)
        self.ChannelConv1 = ChannelConv(256, 64)
        self.Chan_conv = [self.ChannelConv4, self.ChannelConv3, self.ChannelConv2, self.ChannelConv1]

        self.score4 = nn.Conv2d(1024, 1, 1, 1, 0)
        self.score3 = nn.Conv2d(512, 1, 1, 1, 0)
        self.score2 = nn.Conv2d(256, 1, 1, 1, 0)
        self.score1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score = nn.Conv2d(128, 1, 1, 1, 0)
        self.output_convs = [self.score4, self.score3, self.score2, self.score1, self.score]

    def forward(self, x, x_layers, x_t_layers, alpha):
        width = x.size(-1)
        xsize = x.size()[2:]
        scale = [32, 16, 8, 4, 4]
        results = []
        alpha_scaled = repeat(alpha, 'b n  -> b n h w', h=int(width/32), w=int(width/32))
        x_t_scaeld = torch.mul(x_t_layers[4], 1 - alpha_scaled)
        sa = self.spa(x_t_scaeld)
        temp = x_layers[4] + x_layers[4].mul(sa)
        result = temp.mul(alpha_scaled) + x_t_layers[4].mul(1-alpha_scaled)
        result = torch.cat((result, temp), 1)
        result = self.CAs[0](result)
        result = self.Convs[0](result)
        result = self.Chan_conv[0](result, x_layers[3])
        results.append(result)
        for i in range(1, 4):
            alpha_scaled = repeat(alpha, 'b n  -> b n h w', h=int(width/scale[i]), w=int(width/scale[i]))
            x_t_scaeld = torch.mul(x_t_layers[4-i], 1 - alpha_scaled)
            sa = self.spa(x_t_scaeld)
            temp = results[-1] + results[-1].mul(sa)
            result = x_layers[4-i].mul(alpha_scaled) + x_t_layers[4-i].mul(1-alpha_scaled)
            result = torch.cat((temp, result), 1)
            result = self.CAs[i](result)
            result = self.Convs[i](result)
            result = self.Chan_conv[i](result, x_layers[3-i])
            results.append(result)
        alpha_scaled = repeat(alpha, 'b n  -> b n h w', h=int(width/scale[i]), w=int(width/scale[i]))
        x_t_scaeld = torch.mul(x_t_layers[0], 1 - alpha_scaled)
        sa = self.spa(x_t_scaeld)
        temp = results[-1] + results[-1].mul(sa)
        result = x_layers[0].mul(alpha_scaled) + x_t_layers[0].mul(1-alpha_scaled)
        result = torch.cat((result, temp), 1)
        result = self.CAs[4](result)
        results.append(result)    
        for i in range(len(results)):
            results[i] = F.interpolate(self.output_convs[i](results[i]), xsize, mode='bilinear', align_corners=True)


        # return result,u4,u3,u2,u1
        return results[4], results[0], results[1], results[2], results[3]

def modality_drop(x_rgb, x_depth):
    if opt.mode == 'train':
        modality_combination = [[1, 0], [0, 1], [1, 1]]
    else:
        modality_combination = [[1, 1], [1, 1], [1, 1]]
    index_list = [x for x in range(3)]

    p = []
    prob = np.array(( 1 / 3 , 1 / 3, 1 / 3))
    for i in range(x_rgb.shape[0]):
        index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
        p.append(modality_combination[index])
    p = np.array(p)
    p = torch.from_numpy(p)
    p = torch.unsqueeze(p, 2)
    p = torch.unsqueeze(p, 3)
    p = torch.unsqueeze(p, 4)
    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_depth = x_depth * p[:, 1]

    return x_rgb,x_depth,p

#baseline
class Baseline(nn.Module):
    def __init__(self, channel=32):
        super(Baseline, self).__init__()

        self.resnet = ResNet50('rgb')
        self.resnet_t = ResNet50('rgbt')
        self.cc = nn.Conv2d(2048, 1, kernel_size=1)

        self.cc1 = nn.ConvTranspose2d(1, 1, kernel_size=16, padding=4, stride=8)
        self.cc2 = nn.ConvTranspose2d(1, 1, kernel_size=16, padding=4, stride=8)
        self.cc3 = nn.ConvTranspose2d(1, 1, kernel_size=8, padding=2, stride=4)
        self.cc4 = nn.ConvTranspose2d(1, 1, kernel_size=4, padding=1, stride=2)
        self.cc_layers = [self.cc1, self.cc2, self.cc3, self.cc4]
        self.sigmoid = nn.Sigmoid()

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_t, alpha):
        aaa = x.size(-1)

        x_layers = self.process_through_resnet(x, self.resnet)
        resnet_t_layers = [self.resnet_t.layer1, self.resnet_t.layer2, 
                           self.resnet_t.layer3, self.resnet_t.layer4]
        x_t = self.process_through_resnet(x_t, self.resnet_t)[0]
        
        last_x_feature = self.cc(x_layers[-1])
        x_t_layers = [x_t]
        scaled = [4, 4, 8, 16, 32]
        for i in range(5):
            alpha_scaled = self.scale_alpha(alpha, aaa, scaled[i])
            if i == 4:
                attention_map = self.sigmoid(last_x_feature).mul(alpha_scaled)
            else:
                attention_map = self.create_attention_map(self.cc_layers[i], last_x_feature, alpha_scaled)
            if i == 0:
                x_t_modified = self.apply_attention(x_t_layers[-1], attention_map)
            else:
                x_t_modified = self.apply_attention(resnet_t_layers[i-1](x_t_layers[-1]), attention_map)
            x_t_layers.append(x_t_modified)            

        return x_layers, x_t_layers[-5:], last_x_feature

    def process_through_resnet(self, x, model):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x1 = model.layer1(x)
        x2 = model.layer2(x1)
        x3 = model.layer3(x2)
        x4 = model.layer4(x3)

        x_layers = [x, x1, x2, x3, x4]
        return x_layers

    def scale_alpha(self, alpha, aaa, layer_index):
        return repeat(alpha, 'b n -> b n h w', h=int(aaa/layer_index), w=int(aaa/layer_index))

    def create_attention_map(self, cc_layer, ttt, alpha_scaled):
        attention_map = cc_layer(ttt)
        attention_map = self.sigmoid(attention_map)
        return attention_map.mul(alpha_scaled)

    def apply_attention(self, x_t, attention_map):
        return x_t + x_t.mul(attention_map)
    
    def initialize_weights(self):
        res101 = models.resnet101(pretrained=True)
        pretrained_dict = res101.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_t.state_dict().items():
            if k=='conv1.weight':
                all_params[k]=torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_t.state_dict().keys())
        self.resnet_t.load_state_dict(all_params)


class Decoder(nn.Module):
    def __init__(self,channel=32):
        super(Decoder, self).__init__()
        self.s_net = SaliencyNet()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_layers, x_t_layers, last_x_feature, alpha):
        #Decoder
        result_r,u4,u3,u2,u1 = self.s_net(x, x_layers, x_t_layers, alpha)
        result_r=self.sigmoid(result_r)
        u4=self.sigmoid(u4)
        u3 = self.sigmoid(u3)
        u2 = self.sigmoid(u2)
        u1 = self.sigmoid(u1)
        return result_r,u4,u3,u2,u1

class zeroConv(nn.Module):
    def __init__(self):
        super(zeroConv, self).__init__()
        self.ori_ctrl_zero = zero_module(nn.Conv2d(3, 3, kernel_size=1))
        self.x_ctrl_zero = zero_module(nn.Conv2d(64, 64, kernel_size=1))
        self.x1_ctrl_zero = zero_module(nn.Conv2d(256, 256, kernel_size=1))
        self.x2_ctrl_zero = zero_module(nn.Conv2d(512, 512, kernel_size=1))
        self.x3_ctrl_zero = zero_module(nn.Conv2d(1024, 1024, kernel_size=1))
        self.x4_ctrl_zero = zero_module(nn.Conv2d(2048, 2048, kernel_size=1))
        self.x_t4_ctrl_zero = zero_module(nn.Conv2d(2048, 2048, kernel_size=1))
        self.x_t_ctrl_zero = zero_module(nn.Conv2d(64, 64, kernel_size=1))
        self.x_t1_ctrl_zero = zero_module(nn.Conv2d(256, 256, kernel_size=1))
        self.x_t2_ctrl_zero = zero_module(nn.Conv2d(512, 512, kernel_size=1))
        self.x_t3_ctrl_zero = zero_module(nn.Conv2d(1024, 1024, kernel_size=1))
        self.last_ctrl_zero = zero_module(nn.Conv2d(1, 1, kernel_size=1))

    def forward(self, x_layers_ctrl, x_t_layers_ctrl, last_x_feature_ctrl):
        x_layers_ctrl[0] = self.x_ctrl_zero(x_layers_ctrl[0])
        x_layers_ctrl[1] = self.x1_ctrl_zero(x_layers_ctrl[1])
        x_layers_ctrl[2] = self.x2_ctrl_zero(x_layers_ctrl[2])
        x_layers_ctrl[3] = self.x3_ctrl_zero(x_layers_ctrl[3])
        x_layers_ctrl[4] = self.x4_ctrl_zero(x_layers_ctrl[4])
        x_t_layers_ctrl[0] = self.x_t_ctrl_zero(x_t_layers_ctrl[0])
        x_t_layers_ctrl[1] = self.x_t1_ctrl_zero(x_t_layers_ctrl[1])
        x_t_layers_ctrl[2] = self.x_t2_ctrl_zero(x_t_layers_ctrl[2])
        x_t_layers_ctrl[3] = self.x_t3_ctrl_zero(x_t_layers_ctrl[3])
        x_t_layers_ctrl[4] = self.x_t4_ctrl_zero(x_t_layers_ctrl[4])
        last_x_feature_ctrl = self.last_ctrl_zero(last_x_feature_ctrl)
        return x_layers_ctrl, x_t_layers_ctrl, last_x_feature_ctrl

class CLIPAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.text = clip.tokenize(["A photo of high quality", "A photo of low quality"]).cuda()
        self.text_features = self.clip_model.encode_text(self.text)
        self.text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True).float()
        self.text_learner = nn.parameter.Parameter(torch.zeros([2, 512]))
        self.logit_scale = nn.parameter.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x, batch_size,p):
        text_feature = self.text_features + self.text_learner
        i = 0
        clip_score = torch.zeros([batch_size, 1]).cuda()
        for clip_img in x:
            if p[i,0] == 0:
                image_clip = self.clip_preprocess(Image.open(black_path)).unsqueeze(0).to('cuda')
            else:
                image_clip = self.clip_preprocess(clip_img).unsqueeze(0).to('cuda')
            image_features = self.clip_model.encode_image(image_clip)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features.float() @ text_feature.t()
            probs_learner = logits_per_image.softmax(dim=-1)
            probs_learner_clone = probs_learner.clone()
            probs_learner_clone[0][0] = (probs_learner[0][0] * 0.2 + 0.4)
            clip_score[i] = probs_learner_clone[0][0]
            i = i + 1
        clip_score = clip_score.view(-1,1)
        return clip_score

class BaselineControlNet(nn.Module):
    def __init__(self):
        super(BaselineControlNet, self).__init__()
        self.baseline = Baseline()
        self.baseline_ctrl = Baseline()
        self.zeroConv = zeroConv()
        self.decoder = Decoder()
        self.clip_alpha = CLIPAlpha()

    def forward(self,x,x_t,image_clips,epoch):
        if epoch > opt.change_epoch:
            x,x_t,p = modality_drop(x, x_t)
        else :
            p = torch.full((x.size(0), 2, 1, 1, 1), 1).to('cuda:0')
        x_copy = x
        x_t_copy = x_t
        alpha = self.clip_alpha(image_clips, x.shape[0],p)
        x_layers, x_t_layers, last_x_feature = self.baseline(x, x_t, alpha)
        if epoch > opt.change_epoch:
            x_layers_ctrl, x_t_layers_ctrl, last_x_feature_ctrl = self.baseline_ctrl(x_copy, x_t_copy, alpha)
            x_layers_ctrl, x_t_layers_ctrl, last_x_feature_ctrl = self.zeroConv(x_layers_ctrl, x_t_layers_ctrl, last_x_feature_ctrl)
            for i in range(len(x_layers)):
                x_layers[i] = x_layers[i] + x_layers_ctrl[i]
            for i in range(len(x_t_layers)):
                x_t_layers[i] = x_t_layers[i] + x_t_layers_ctrl[i]
            last_x_feature = last_x_feature + last_x_feature_ctrl
        result_r,u4,u3,u2,u1 = self.decoder(x, x_layers, x_t_layers, last_x_feature, alpha)
        return result_r,u4,u3,u2,u1

