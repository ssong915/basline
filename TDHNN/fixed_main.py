from our_utils import setup_seed, arg_parse, visualization
from load_data import load_data
from our_data_load import load_our_data, split_data
from fixed_train import train_dhl,train_gcn,train_gat
import json
import torch
from networks import HGNN_classifier, GCN, GAT
import torch.nn.functional as F
import time

chosse_trainer = {
    'dhl':train_dhl
    }

args = arg_parse()
torch.cuda.set_device(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

setup_seed(args.seed)

args.min_num_edges = args.k_e
args.in_dim = 128
args_list = []

final_roc, final_ap = chosse_trainer[args.model](device, args)
