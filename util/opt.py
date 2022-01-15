import argparse
import os
import random
import time

import numpy as np
import torch


def get_parser():
    parser = argparse.ArgumentParser(description='arguments of program')
    parser.add_argument('--data_directory', type=str, default='./dataset', help='original data directory')
    parser.add_argument('--checkpoints_directory', type=str, default='./checkpoint', help='checkpoints directory')

    #ilsvrc2012 SIGGRAPH
    parser.add_argument('--dataset_name', type=str, default='ilsvrc2012', help='the name of data set')
    parser.add_argument('--model', type=str, default='SIGRES', help='the name of model')

    parser.add_argument('--stage', type=str, default='train', help='train/test')
    parser.add_argument('--logging_directory',
                        type=str, default='./log', help='the directory the log would be saved to')
    parser.add_argument('--random_state', type=int, default=2022, help='the random seed')

    # arguments for runner
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda:0 / cpu')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='number of epochs')
    parser.add_argument('--epoch', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--test_epoch', type=int, default=2,
                        help='test with a period of some epoch (-1 means no test)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization in optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size while training/ validating')

    # args for model
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=176, help='then crop to this size')
    parser.add_argument('--ab_norm', type=float, default=256., help='colorization normalization factor')
    parser.add_argument('--ab_max', type=float, default=0., help='maximimum ab value')
    parser.add_argument('--ab_quant', type=float, default=10., help='quantization factor')

    args, unknown = parser.parse_known_args()
    setattr(args, 'A', 2 * args.ab_max / args.ab_quant + 1)
    setattr(args, 'B', 2 * args.ab_max / args.ab_quant + 1)

    setattr(args, 'logging_file_name', os.path.join(
        args.logging_directory,
        '{}_{}_{}.txt'.format(args.model,
                              args.dataset_name,
                              time.strftime('%Y.%m.%d', time.localtime()))
    ))
    setattr(args, 'result_file_name', os.path.join(
        args.logging_directory,
        '{}_{}_{}.pkl'.format(args.model,
                              args.dataset_name,
                              time.strftime('%Y.%m.%d', time.localtime()))
    ))
    setattr(args, 'result_pics_path', os.path.join(
        args.logging_directory,
        '{}_{}'.format(args.model,
                              args.dataset_name)
    ))
    setattr(args, 'dataset_path', os.path.join(
        args.data_directory, args.dataset_name
    ))
    prefix = args.checkpoints_directory + '/' + '{}_{}'.format(args.model, args.dataset_name)
    setattr(args, 'checkpoints_prefix',
            prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    if not os.path.exists(args.data_directory):
        os.makedirs(args.data_directory)
    if not os.path.exists(args.checkpoints_directory):
        os.makedirs(args.checkpoints_directory)
    if not os.path.exists(args.logging_directory):
        os.makedirs(args.logging_directory)
    if not os.path.exists(args.result_pics_path):
        os.makedirs(args.result_pics_path)

    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True

    return args
