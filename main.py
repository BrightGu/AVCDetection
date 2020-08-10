from argparse import ArgumentParser, Namespace
import torch
from solver import Solver
import yaml


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    args = parser.parse_args()

    # load config file
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    solver = Solver(config=config)

    if args.iters > 0:
        solver.train()
        #solver.infer()