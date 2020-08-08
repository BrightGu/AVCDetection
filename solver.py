import pickle
import torch
import numpy as np
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model.crossmodel import CrossModel
from model.utils import *
from functools import reduce
from collections import defaultdict
from evaluate.count_metrics import calculate_metrics
import random

class Solver(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # args store other information
        self.args = args
        print(self.args)

        # logger to use tensorboard
        self.logger = Logger(self.args.logdir)

        # get dataloader
        self.data=load_data(self.args.train_set_path,self.args.data_key)
        self.test_data=load_data(self.args.test_set_path,self.args.test_data_key)
        #self.margin=self.config['loss']['margin']
        # init the model with config
        self.build_model()

        self.save_config()

        if self.args.load_model:
            self.load_model_num = self.args.load_model_num
            self.load_model(self.args.load_model_num)
        else:
            self.load_model_num=0

    def build_model(self):
        # create model, discriminator, optimizers
        self.model = cc(CrossModel(self.config))
        print(self.model)
        #optimizer = self.config['optimizer']
        self.opt = torch.optim.Adam(self.model.parameters(),
                lr=self.args.lr, betas=(self.args.beta1, self.args.beta2),
                amsgrad=self.args.amsgrad, weight_decay=self.args.weight_decay)
        print(self.opt)
        return

    def load_model(self,iteration):
        #path=self.args.store_model_path + '/model_' + f'{iteration}'
        path=os.path.join(self.args.store_model_path,'model_' + f'{iteration}')
        print(f'Load model from {path}')
        self.model.load_state_dict(torch.load(f'{path}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{path}.opt'))
        return

    def save_config(self):
        with open(f'{self.args.store_model_path}.config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def save_model(self, iteration):
        # save model and discriminator and their optimizer
        #path = self.args.store_model_path + '/model_' + f'{iteration}'
        path = os.path.join(self.args.store_model_path, 'model_' + f'{iteration}')
        torch.save(self.model.state_dict(), f'{path}.ckpt')
        torch.save(self.opt.state_dict(), f'{path}.opt')

    def cal_distance(self,video_embed, audio_embed):
        batch_size = video_embed.shape[0]
        d1 = torch.pow((video_embed - audio_embed), 2)
        d2 = torch.sum(d1, [0, 1])
        #avg
        d3 = d2 / batch_size
        return d3

    def contrastive_margin_loss(self,video_embed, audio_embed, label, margin):
        batch_size = video_embed.shape[0]
        #print("batch_size", batch_size)
        d1 = torch.pow((video_embed - audio_embed), 2)
        d2 = torch.sum(d1, [0, 1])
        loss1 = d2 / batch_size / 2

        d3 = torch.sqrt(torch.sum(d1, 1))
        x=torch.zeros(d3.shape)
        y=margin - d3
        d4 = torch.max(y.cuda(), x.cuda())
        d5 = torch.sum(torch.pow(d4, 2), 0)
        loss2 = d5 / batch_size / 2
        loss = label * loss1 + (1 - label) * loss2
        return loss

    def ae_step(self,item,margin):
        video_data = cc(torch.Tensor(item[1]))
        video_data.transpose(1, 2)
        video_data.unsqueeze_(1)
        audio_data = cc(torch.Tensor(item[2]))
        audio_data.unsqueeze_(1)  # (B, 1, 512)
        if 'real' in item[0]:
            label = 1
        elif 'fake' in item[0]:
            label = 0
        else:
            assert 1==0

        audio_emb,video_emb = self.model(audio_data,video_data)
        self.opt.zero_grad()
        loss = self.contrastive_margin_loss(video_emb, audio_emb, label, margin)
        print("label:"+item[0]+"  loss:"+str(loss))
        loss.backward(torch.ones_like(loss))
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                max_norm=self.args.grad_norm)
        self.opt.step()
        #use for auc
        distance = self.cal_distance(video_emb, audio_emb)

        meta = {'loss': loss.item(),
                'grad_norm': grad_norm,
                'distance': distance.item()}
        return meta

    def test_evaluation(self,item):
        video_data = cc(torch.Tensor(item[1]))
        video_data.transpose(1, 2)
        video_data.unsqueeze_(1)
        audio_data = cc(torch.Tensor(item[2]))
        audio_data.unsqueeze_(1)  # (B, 1, 512)
        audio_emb,video_emb = self.model(audio_data,video_data)
        #use for auc
        distance = self.cal_distance(video_emb, audio_emb)
        return distance.item()

    def train(self, n_iterations):
        start_iterations = self.load_model_num
        for iteration in range(start_iterations,n_iterations):
            #self.data.shuffle()
            random.shuffle(self.data)
            real_clip_distance_map=defaultdict(lambda: [])
            fake_clip_distance_map=defaultdict(lambda: [])
            meta={}
            for item in self.data:
                label=item[0]
                metagu = self.ae_step(item,self.args.margin)
                meta=metagu
                if 'real' in label:
                    real_clip_distance_map[label].append(meta['distance'])
                elif 'fake' in label:
                    fake_clip_distance_map[label].append(meta['distance'])
                else:
                    assert 1==0
            #calculate auc
            if iteration % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                #计算AUC
                clip_auc=calculate_metrics(real_clip_distance_map,fake_clip_distance_map,'train',iteration,self.args.auc_path)
                line='iteration:'+str(iteration)+" "+"clip_auc:"+str(clip_auc)
                print(line)
                self.save_model(iteration=iteration)
                self.logger.scalars_summary(f'{self.args.tag}/ae_train', meta, iteration)
                loss = meta['loss']
                print(f'AE:[{iteration + 1}/{n_iterations}], loss={loss:.2f}')

                # test_set evaluation
                test_real_clip_distance_map = defaultdict(lambda: [])
                test_fake_clip_distance_map = defaultdict(lambda: [])
                for item in self.test_data:
                    label = item[0]
                    distance = self.test_evaluation(item)
                    if 'real' in label:
                        test_real_clip_distance_map[label].append(distance)
                    elif 'fake' in label:
                        test_fake_clip_distance_map[label].append(distance)
                    else:
                        assert 1 == 0
                test_clip_auc = calculate_metrics(test_real_clip_distance_map, test_fake_clip_distance_map, 'test',
                                              iteration, self.args.auc_path)
                line = 'test_iteration:' + str(iteration) + " " + "test_clip_auc:" + str(test_clip_auc)
                print(line)

    def infer(self):
         # test_set evaluation
        test_real_clip_distance_map = defaultdict(lambda: [])
        test_fake_clip_distance_map = defaultdict(lambda: [])
        for item in self.test_data:
            label = item[0]
            distance = self.test_evaluation(item)
            if 'real' in label:
                test_real_clip_distance_map[label].append(distance)
            elif 'fake' in label:
                test_fake_clip_distance_map[label].append(distance)
            else:
                assert 1 == 0
        test_clip_auc = calculate_metrics(test_real_clip_distance_map, test_fake_clip_distance_map, 'test',
                                      0, self.args.auc_path)
        line = 'test_iteration:' + str(0) + " " + "test_clip_auc:" + str(test_clip_auc)
        print(line)


