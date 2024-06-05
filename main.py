import os
import sys
import torch
import torch, random
import numpy as np
from tools import train_net
from utils import parser


def main():

    args = parser.get_args()
    parser.setup(args)    
    print(args)
    init_seed(args.seed+rank)
    train_net(args, rank, world_size)

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    main()