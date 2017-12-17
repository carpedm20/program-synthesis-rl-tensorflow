#-*- coding: utf-8 -*-
import argparse

from utils import str2bool

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--kernel_dims', type=eval, default='[]', help='')
net_arg.add_argument('--stride_size', type=eval, default='[]', help='')
net_arg.add_argument('--channel_dims', type=eval, default='[]', help='')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='data')
data_arg.add_argument('--data_ext', type=str, default='npz')
data_arg.add_argument('--world_height', type=int, default=8)
data_arg.add_argument('--world_width', type=int, default=8)
data_arg.add_argument('--max_marker_in_cell', type=int, default=1)

# Train
train_arg = add_argument_group('Train')
train_arg.add_argument('--base_dir', type=str, default='logs')
train_arg.add_argument('--model_path', type=str, default=None,
                       help='default is {config.base_dir}/{config.tag}_{timestring}')
train_arg.add_argument('--pretrain_path', type=str, default=None)
train_arg.add_argument('--epoch', type=int, default=100)
train_arg.add_argument('--lr', type=float, default=0.001)
train_arg.add_argument('--seed', type=int, default=123)
train_arg.add_argument('--use_rl', type=str2bool, default=False)
train_arg.add_argument('--batch_size', type=int, default=64)

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--world', type=str, default=None)

# ETC
etc_arg = add_argument_group('ETC')
etc_arg.add_argument('--train', type=str2bool, default=True,
                     help='whether run under train or test mode')
etc_arg.add_argument('--tag', type=str, default='karel')
etc_arg.add_argument('--log_level', type=str, default='info')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
