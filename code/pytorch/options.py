import argparse
parser = argparse.ArgumentParser()
# parameters for the train
parser.add_argument('--epoch', type=int, default=101, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
# parameters for the train strategy
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=45, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
# epoch to change stage
parser.add_argument('--change_epoch', type=int, default=200, help='change epoch')

parser.add_argument('--pre_model', type=str, default='ResNet-50', help='pretrained model')
# black image path for CLIP when modality missing
parser.add_argument('--black_path', type=str, default='/VT821/black/1.jpg', help='the black_path rgb images root')
# dataset path
parser.add_argument('--rgb_root', type=str, default='/VT5000Train/RGB/', help='the training rgb images root')
parser.add_argument('--t_root', type=str, default='/VT5000Train/T/', help='the training t images root')
parser.add_argument('--gt_root', type=str, default='/VT5000Train/GT/', help='the training gt images root')
parser.add_argument('--save_path', type=str, default='./result/', help='the path to save models and logs')
opt = parser.parse_args()