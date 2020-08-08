import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import numpy as np
from math import ceil
from functools import reduce
from torch.nn.utils import spectral_norm

def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)


def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def conv_bank_cal(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            #nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            #nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
            #nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        )
    def forward(self, x):
        return x + self.main(x)

#卷积填充https://blog.csdn.net/qq_26369907/article/details/88366147
#d = (d - kennel_size + 2 * padding) / stride + 1
class VideoEncoder(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(VideoEncoder, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
            #layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.BatchNorm2d(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        layers.append(nn.BatchNorm2d(curr_dim))
        # Up-sampling layers.
        # for i in range(2):
        #     layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim // 2
#之前缩减了2倍  64-->16

        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=2, padding=1))
        layers.append(nn.Conv2d(curr_dim, 128, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1))
        layers.append(nn.MaxPool2d((2,2)))
        #x = torch.flatten(t, start_dim=1)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        out=self.main(x)# out torch.Size([4, 64, 3, 4])
        # dim=out.shape[0]*out.shape[1]*out.shape[2]
        out = torch.flatten(out, start_dim=1,end_dim=-1)
        # out.unsqueeze_(0)
        # linear = torch.nn.Linear(dim, 64)  # 20,30是指维度
        # out = linear(out)
        return out

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(1, 4, kernel_size=k) for k in range(1, 8 + 1, 1)])
        #bank_channels = c_bank * (bank_size // bank_scale) + c_in
        self.conv2 = nn.Conv1d(in_channels=33, out_channels=16, kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        #out = pad_layer(out, conv2)  # [16,256]
        # 最大池化
        self.pool = nn.MaxPool1d(4)
        self.act=nn.LeakyReLU()
        self.batch_norm16 = nn.BatchNorm1d(16)
        self.batch_norm8 = nn.BatchNorm1d(8)

    def forward(self, input):
        out = conv_bank_cal(input, self.conv_bank, act=nn.LeakyReLU())  #[batch,33,512]
        out = pad_layer(out, self.conv2)  # [16,256]
        out = self.batch_norm16(out)
        out = self.act(out)
        # 最大池化
        out = self.pool(out)  # [16,64]
        out = pad_layer(out, self.conv3)
        out = self.batch_norm8(out)
        out = self.act(out)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        return out

class CrossModel(nn.Module):
    def __init__(self, config):
        super(CrossModel, self).__init__()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()

    def forward(self, audio_data,video_data):
        audio_emb = self.audio_encoder(audio_data)
        video_emb = self.video_encoder(video_data)
        return audio_emb, video_emb

# Find total parameters and trainable parameters
# model=SpeakerEncoder()
# total_params = sum(p.numel() for p in model.parameters())
# print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(
#     p.numel() for p in model.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')