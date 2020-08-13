from  os.path import join,basename
import torch.nn as nn
import yaml
from crossmodel import CrossModel
from utils import *
from collections import defaultdict
from count_metrics import calculate_metrics
import random

class Solver(object):
    def __init__(self, config):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # file path
        self.train_set_path = self.config['train_set_path']
        self.test_set_path = self.config['test_set_path']
        self.store_model_dir = self.config['store_model_dir']
        self.evaluate_path = self.config['evaluate_path']
        # train set
        self.save_steps = self.config['train']['save_steps']
        self.loaded_model_num = self.config['train']['loaded_model_num']
        self.loaded_model = self.config['train']['loaded_model']
        self.iters = self.config['train']['iters']
        # logger to use tensorboard
        self.logger = Logger(self.config['logger']['logger_dir'])
        self.tag = self.config['logger']['tag']
        # loss
        self.margin = self.config['loss']['margin']
        # optimizer
        self.lr = self.config['optimizer']['lr']
        self.beta1 = self.config['optimizer']['beta1']
        self.beta2 = self.config['optimizer']['beta2']
        self.amsgrad = self.config['optimizer']['amsgrad']
        self.weight_decay = self.config['optimizer']['weight_decay']
        self.grad_norm = self.config['optimizer']['grad_norm']

        # get dataloader
        # key of high/low set
        train_data_key=basename(self.train_set_path[:-4]) # train_high or train_low
        self.data=load_data(self.train_set_path,train_data_key)
        test_data_key = basename(self.test_set_path[:-4])  # test_high or test_low
        self.test_data=load_data(self.test_set_path,test_data_key)

        # init the model with config
        self.build_model()
        self.save_config()
        # load model
        if self.loaded_model:
            self.load_model(self.loaded_model_num)
        else:
            self.loaded_model_num=0

    def build_model(self):
        # create model, optimizers
        self.model = cc(CrossModel(self.config))
        print(self.model)
        #optimizer = self.config['optimizer']
        self.opt = torch.optim.Adam(self.model.parameters(),
                lr=self.lr, betas=(self.beta1, self.beta2),
                amsgrad=self.amsgrad, weight_decay=self.weight_decay)
        print(self.opt)
        return

    def load_model(self,iteration):
        path=join(self.store_model_dir,'model_' + f'{iteration}')
        print(f'Load model from {path}')
        self.model.load_state_dict(torch.load(f'{path}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{path}.opt'))
        return

    def save_config(self):
        #with open(f'{self.store_model_dir}.config.yaml', 'w') as f:
        with open(join(self.store_model_dir,'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        return

    def save_model(self, iteration):
        # save model and optimizer
        path = join(self.store_model_dir, 'model_' + f'{iteration}')
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
        #d4 = torch.max(y.cuda(), x.cuda())
        d4 = torch.max(cc(y), cc(x))
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
        #print("label:"+item[0]+"  loss:"+str(loss))
        loss.backward(torch.ones_like(loss))
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                max_norm=self.grad_norm)
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


    def train(self):
        start_iterations = self.loaded_model_num
        for iteration in range(start_iterations,self.iters):
            random.shuffle(self.data)
            real_clip_distance_map=defaultdict(lambda: [])
            fake_clip_distance_map=defaultdict(lambda: [])
            meta={}
            for item in self.data[:15]:
                label=item[0]
                meta = self.ae_step(item,self.margin)
                if 'real' in label:
                    real_clip_distance_map[label].append(meta['distance'])
                elif 'fake' in label:
                    fake_clip_distance_map[label].append(meta['distance'])
                else:
                    assert 1==0
            # calculate metrics
            if iteration % self.save_steps == 0 or iteration + 1 == self.iters:
                clip_auc=calculate_metrics(real_clip_distance_map,fake_clip_distance_map,'train',iteration,self.evaluate_path)
                line = 'iteration:'+str(iteration)+' '+'clip_auc:'+str(clip_auc)
                print(line)
                self.save_model(iteration=iteration)
                self.logger.scalars_summary(f'{self.tag}/ae_train', meta, iteration)
                loss = meta['loss']
                print(f'AE:[{iteration + 1}/{self.iters}], loss={loss:.2f}')

                # test_set evaluation
                test_real_clip_distance_map = defaultdict(lambda: [])
                test_fake_clip_distance_map = defaultdict(lambda: [])
                #for item in self.test_data[:len(self.test_data)//10]:
                random.shuffle(self.test_data)
                for item in self.test_data[:18]:
                    label = item[0]
                    distance = self.test_evaluation(item)
                    if 'real' in label:
                        test_real_clip_distance_map[label].append(distance)
                    elif 'fake' in label:
                        test_fake_clip_distance_map[label].append(distance)
                    else:
                        assert 1 == 0
                test_clip_auc = calculate_metrics(test_real_clip_distance_map, test_fake_clip_distance_map, 'test',
                                              iteration, self.evaluate_path)
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
                                      0, self.evaluate_path)
        line = 'test_iteration:' + str(0) + " " + "test_clip_auc:" + str(test_clip_auc)
        print(line)


