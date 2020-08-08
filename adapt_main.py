from argparse import ArgumentParser, Namespace
import torch
from solver import Solver
import yaml
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d',
            default='/storage/feature/LibriTTS/sr_24000_mel_norm')
    parser.add_argument('-train_set', default='train')

    parser.add_argument('-logdir', default='log/')
    #add
    parser.add_argument('-load_model_num', default='0')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('-store_model_path', default='/storage/model/adaptive_vc/model')
    parser.add_argument('-load_model_path', default='/storage/model/adaptive_vc/model')
    parser.add_argument('-summary_steps', default=100, type=int)
    parser.add_argument('-save_steps', default=5000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=0, type=int)

    args = parser.parse_args()

    args.data_dir= r'D:\gyw\document\pycharmproject\mouth_voice\output'

    #args.test_set=r'D:\document\paper\personpaper\audio-visual_consistance\data\test\test_high.pkl'
    #args.log_dir= r'D:\gyw\document\pycharmproject\mouth_voice\output\logs'

    # args.store_model_path= '/root/PycharmProjects/vcc/dataset/model'
    # args.load_model_path= '/root/PycharmProjects/vcc/dataset/model'


    args.config=r'D:\gyw\document\pycharmproject\mouth_voice\model\config.yaml'

    args.train_set_path = r'D:\gyw\document\vac\data\train\train_high.pkl'
    args.data_key='train_high'
    args.test_set_path = r'D:\gyw\document\vac\data\test\test_high.pkl'
    args.test_data_key = 'test_high'

    args.tag = 'guzi2'

    args.load_model=True
    args.load_model_num=5

    args.save_steps = 1
    # args.summary_steps=1000
    args.iters=1000
    # args.load_model_num=460045
    #args.store_model_path = r'D:\gyw\document\pycharmproject\mouth_voice\output\model'
    args.store_model_path = r'D:\gyw\document\pycharmproject\mouth_voice\model\model\pu_high'
    args.auc_path = r'D:\gyw\document\pycharmproject\mouth_voice\output\result_data.txt'
    args.margin = 700
    args.lr = 0.0035
    args.beta1 = 0.9
    args.beta2 = 0.999
    args.amsgrad = True
    args.weight_decay = 0.0003
    args.grad_norm = 20
    # load config file
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    solver = Solver(config=config, args=args)

    if args.iters > 0:
        #solver.train(n_iterations=args.iters)
        solver.infer()