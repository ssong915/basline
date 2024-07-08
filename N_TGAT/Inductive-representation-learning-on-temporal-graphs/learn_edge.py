"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler
from itertools import groupby
from utils import Decoder
from process import get_data

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='email-Eu')
parser.add_argument('--bs', type=int, default=32, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=1, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument("--time_split_type", default='sec', type=str, help='snapshot split type')
parser.add_argument("--freq_size", default=2628000, type=int, help='batch size')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
#NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim


MODEL_SAVE_PATH = f'/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/saved_models/seed_{args.seed}/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{args.n_degree}.pth'
get_checkpoint_path = lambda epoch: f'/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/saved_checkpoints/seed_{args.seed}/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}-{args.n_degree}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


def eval_one_epoch(hint, tgan, sampler, src, dst, ts):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []

    group_lengths = [len(list(group)) for _, group in groupby(ts)]
    test_hyperedges = np.unique(ts).shape[0]

    tbatch_sources = []
    tbatch_destinations = []
    tbatch_timestamps = []

    start = 0
    for length in group_lengths:
        end = start + length
        tbatch_sources.append(src[start:end])
        tbatch_destinations.append(dst[start:end])
        tbatch_timestamps.append(ts[start:end])
        start = end

    with torch.no_grad():
        tgan = tgan.eval()

        TEST_BATCH_SIZE = 32
        num_test_instance = test_hyperedges
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]


            src_l_cut, dst_l_cut = tbatch_sources[s_idx:e_idx], \
                                                    tbatch_destinations[s_idx:e_idx]
            ts_l_cut = tbatch_timestamps[s_idx:e_idx]

            src_l_cut = np.concatenate(src_l_cut).tolist()
            dst_l_cut = np.concatenate(dst_l_cut).tolist()
            ts_l_cut = np.concatenate(ts_l_cut).tolist()
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            num_hyperedges = np.unique(ts_l_cut).shape[0]

            pos_prob, neg_prob = tgan.hyper_contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, device, NUM_NEIGHBORS)
            
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(num_hyperedges), np.zeros(num_hyperedges)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return val_acc, val_ap, val_f1, val_auc

### Load data and train val test split
g_df = pd.read_csv('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}.csv'.format(DATA))
e_feat = np.load('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}.npy'.format(DATA))
n_feat = np.load('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}_node.npy'.format(DATA))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

r_seed = args.seed
set_random_seeds(r_seed)

node_features, edge_features, full_data, train_data, val_data, test_data, max_node_idx = get_data(DATA)

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)

train_src_l = train_data.sources
train_dst_l = train_data.destinations
train_e_idx_l = train_data.edge_idxs
train_ts_l = train_data.timestamps

adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

src_l = full_data.sources
dst_l = full_data.destinations
e_idx_l = full_data.edge_idxs
ts_l = full_data.timestamps
# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)

### Model initialize
device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

hyperedge_unit = [list(group) for _, group in groupby(train_ts_l)]

timestamps = train_ts_l
freq_sec = args.freq_size 
split_criterion = timestamps // freq_sec
groups = np.unique(split_criterion)
groups = np.sort(groups)
num_batch = groups.shape[0]

train_hyperedges = np.unique(train_data.timestamps).shape[0]
group_lengths = [len(list(group)) for _, group in groupby(train_data.timestamps)]

num_instance = train_hyperedges
BATCH_SIZE = args.bs
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(len(hyperedge_unit)))
logger.info('num of batches per epoch: {}'.format(num_batch))
#idx_list = np.arange(num_instance)
#np.random.shuffle(idx_list) 

early_stopper = EarlyStopMonitor()

batch_sources = []
batch_destinations = []
batch_edge_idx = []
batch_timestamps = []

start = 0
for length in group_lengths:
    end = start + length
    batch_sources.append(train_src_l[start:end])
    batch_destinations.append(train_dst_l[start:end])
    batch_edge_idx.append(train_e_idx_l[start:end])
    batch_timestamps.append(train_ts_l[start:end])
    start = end

for epoch in tqdm(range(NUM_EPOCH)):
    # Training 
    # training use only training graph
    tgan.ngh_finder = train_ngh_finder
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    #np.random.shuffle(idx_list)
    logger.info('start {} epoch'.format(epoch))
    for k in tqdm(range(num_batch)):
        # percent = 100 * k / num_batch
        # if k % int(0.2 * num_batch) == 0:
        #     logger.info('progress: {0:10.4f}'.format(percent))
        start_idx = k * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)


        sources_batch, destinations_batch = batch_sources[start_idx:end_idx], \
                                            batch_destinations[start_idx:end_idx]
        edge_idxs_batch = batch_edge_idx[start_idx: end_idx]
        timestamps_batch = batch_timestamps[start_idx:end_idx]

        sources_batch = np.concatenate(sources_batch).tolist()
        destinations_batch = np.concatenate(destinations_batch).tolist()
        edge_idxs_batch = np.concatenate(edge_idxs_batch).tolist()
        timestamps_batch = np.concatenate(timestamps_batch).tolist()

        size = len(sources_batch)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

        # def is_sorted(lst):
        #     return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))
        
        # sorted_time = sorted(ts_l_cut)
        
        # if list(sorted_time) == list(ts_l_cut):
        #   print("리스트는 정렬되어 있습니다.")
        # else:
        #   print("리스트는 정렬되어 있지 않습니다.") 

        num_hyperedges = np.unique(timestamps_batch).shape[0]
        with torch.no_grad():
            pos_label = torch.ones(num_hyperedges, dtype=torch.float, device=device)
            neg_label = torch.zeros(num_hyperedges, dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        tgan = tgan.train()
        pos_prob, neg_prob = tgan.hyper_contrast(sources_batch, destinations_batch, dst_l_fake, timestamps_batch, device, NUM_NEIGHBORS)
        pos_prob = pos_prob.squeeze(1)
        neg_prob = neg_prob.squeeze(1)
    
        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)
        
        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            tgan = tgan.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(num_hyperedges), np.zeros(num_hyperedges)])
            #acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            # f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))

    # validation phase use all information
    tgan.ngh_finder = full_ngh_finder
    val_src_l = val_data.sources
    val_dst_l = val_data.destinations
    val_ts_l = val_data.timestamps

    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_src_l, 
    val_dst_l, val_ts_l)

        
    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

    if early_stopper.early_stop_check(val_ap):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        tgan.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        tgan.eval()
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch))


# testing phase use all information
tgan.ngh_finder = full_ngh_finder
test_src_l = test_data.sources
test_dst_l = test_data.destinations
test_ts_l = test_data.timestamps
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, test_dst_l, test_ts_l)

logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(np.mean(test_acc), np.mean(test_auc), np.mean(test_ap)))

logger.info('Saving TGAN model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('TGAN models saved')