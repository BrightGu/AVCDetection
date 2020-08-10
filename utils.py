import torch 
import numpy as np
from tensorboardX import SummaryWriter
import pickle
import editdistance
import torch.nn as nn
import torch.nn.init as init
#import pyworld
def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)

def load_data_zero_audio(path,key):
    with open(path,'rb') as f:
        data=pickle.load(f)
        data_list=data[key]
    data_zeroaudio_list=[]
    for item in data_list:
        zero_audio_feature=np.zeros(np.array(item[2]).shape).tolist()
        data_zeroaudio_list.append([item[0],item[1],zero_audio_feature])
    return data_zeroaudio_list

def load_data(path,key):
    with open(path,'rb') as f:
        data=pickle.load(f)
        data_list=data[key]
    return data_list


class Logger(object):
    def __init__(self, logdir=r'D:\document\pycharmproject\mouth_voice\output\logs'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, tag, dictionary, step):
        self.writer.add_scalars(tag, dictionary, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

    def audio_summary(self, tag, value, step, sr):
        self.writer.add_audio(tag, value, step, sample_rate=sr)



