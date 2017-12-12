#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

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
data_arg.add_argument('--grid_height', type=int, default=8)
data_arg.add_argument('--grid_width', type=int, default=8)
data_arg.add_argument('--max_marker_in_cell', type=int, default=1)

# Train
train_arg = add_argument_group('Train')
train_arg.add_argument('--lr', type=float, default=0.001)
train_arg.add_argument('--seed', type=int, default=123)
train_arg.add_argument('--use_rl', type=str2bool, default=False)
train_arg.add_argument('--pretrain_path', type=str, default=None)

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--test', type=str2bool, default=False)
test_arg.add_argument('--map', type=str, default=None)

# ETC
etc_arg = add_argument_group('ETC')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
