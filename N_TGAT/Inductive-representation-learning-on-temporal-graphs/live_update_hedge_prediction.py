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
from process import get_data, get_snapshot_data, get_new_data

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='threads-math-sx')
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
parser.add_argument('--uniform', type=str, default="true", help='take uniform sampling from temporal neighbors')
parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument("--time_split_type", default='sec', type=str, help='snapshot split type')
parser.add_argument("--freq_size", default=2628000000, type=int, help='batch size')
#parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')
#parser.add_argument('--randomize_features', action='store_true',help='Whether to randomize node features')

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

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label):
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


TOTAL_DATA, n_feat, e_feat, max_node_idx = get_snapshot_data(args, DATA)
### Load data and train val test split
time_length = len(TOTAL_DATA)
auc_roc_list = []
ap_list = []

full_data, train_data, val_data, test_data = get_new_data(TOTAL_DATA[0], future=1)
    ### Initialize the data structure for graph and edge sampling
    # build the graph for fast query
    # graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(max_node_idx + 1)]
for src, dst, eidx, ts in zip(train_data.sources, train_data.destinations, train_data.edge_idxs, train_data.timestamps):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)
### Model initialize
device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)


for curr_time in tqdm(range(len(TOTAL_DATA))):
    
    logger.info('start {} time'.format(curr_time))

    if curr_time > 0 :
        tgan.load_state_dict(torch.load(best_model_path))

    if (curr_time+1) > (time_length-1) :
        break
    '''
    n_full_data, n_train_data, n_val_data, n_test_data, n_new_node_val_data, \
        n_new_node_test_data = get_data(TOTAL_DATA[curr_time], future=0,
                                      different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)
        
    ### Extract data for training, validation and testing
    full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data = get_data(TOTAL_DATA[curr_time+1], future=1,
                                    different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)
    '''

    n_full_data, n_train_data, n_val_data, n_test_data = get_new_data(TOTAL_DATA[:curr_time+1], future=0)
        
    ### Extract data for training, validation and testing
    full_data, train_data, val_data, test_data = get_new_data(TOTAL_DATA[curr_time+1], future=1)
    ### Initialize the data structure for graph and edge sampling
    # build the graph for fast query
    # graph only contains the training data (with 10% nodes removal)
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for src, dst, eidx, ts in zip(n_full_data.sources, n_full_data.destinations, n_full_data.edge_idxs, n_full_data.timestamps):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_node_idx + 1)]
    for src, dst, eidx, ts in zip(n_full_data.sources, n_full_data.destinations, n_full_data.edge_idxs, n_full_data.timestamps):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations)
    #nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations)
    #nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations)
    ### Model initialize
    train_hyperedges = np.unique(train_data.timestamps).shape[0]
    group_lengths = [len(list(group)) for _, group in groupby(train_data.timestamps)]

    num_instance = train_hyperedges
    BATCH_SIZE = args.bs
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    #idx_list = np.arange(num_instance)
    #np.random.shuffle(idx_list) 

    early_stopper = EarlyStopMonitor()

    batch_sources = []
    batch_destinations = []
    batch_labels = []
    batch_timestamps = []

    start = 0
    for length in group_lengths:
        end = start + length
        batch_sources.append(train_data.sources[start:end])
        batch_destinations.append(train_data.destinations[start:end])
        batch_labels.append(train_data.labels[start:end])
        batch_timestamps.append(train_data.timestamps[start:end])
        start = end

    for epoch in (range(NUM_EPOCH)):
        # Training 
        # training use only training graph
        tgan.ngh_finder = train_ngh_finder
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        #np.random.shuffle(idx_list)
        #logger.info('start {} epoch'.format(epoch))
        for k in (range(num_batch)):
            # percent = 100 * k / num_batch
            # if k % int(0.2 * num_batch) == 0:
            #     logger.info('progress: {0:10.4f}'.format(percent))
            
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance, s_idx + BATCH_SIZE)

            src_l_cut, dst_l_cut = batch_sources[s_idx:e_idx], batch_destinations[s_idx:e_idx]
            label_l_cut = batch_labels[s_idx:e_idx]
            ts_l_cut = batch_timestamps[s_idx:e_idx]

            src_l_cut = np.concatenate(src_l_cut).tolist()
            dst_l_cut = np.concatenate(dst_l_cut).tolist()
            label_l_cut = np.concatenate(label_l_cut).tolist()
            ts_l_cut = np.concatenate(ts_l_cut).tolist()

            src_l_cut, dst_l_cut = train_data.sources[s_idx:e_idx], train_data.destinations[s_idx:e_idx]
            ts_l_cut = train_data.timestamps[s_idx:e_idx]
            label_l_cut = train_data.labels[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            # def is_sorted(lst):
            #     return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))
            
            # sorted_time = sorted(ts_l_cut)
            
            # if list(sorted_time) == list(ts_l_cut):
            #   print("리스트는 정렬되어 있습니다.")
            # else:
            #   print("리스트는 정렬되어 있지 않습니다.") 

            num_hyperedges = np.unique(ts_l_cut).shape[0]

            with torch.no_grad():
                pos_label = torch.ones(num_hyperedges, dtype=torch.float, device=device)
                neg_label = torch.zeros(num_hyperedges, dtype=torch.float, device=device)
            
            optimizer.zero_grad()
            tgan = tgan.train()
            pos_prob, neg_prob = tgan.hyper_contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, device, NUM_NEIGHBORS)
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
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_data.sources, 
        val_data.destinations, val_data.timestamps, val_data.labels)

        #nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgan, val_rand_sampler, new_node_val_data.sources, 
        #new_node_val_data.destinations, new_node_val_data.timestamps, new_node_val_data.labels)
            
        #logger.info('epoch: {}:'.format(epoch))
        #logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        #logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
        #logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
        # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

        if early_stopper.early_stop_check(val_ap):
            #logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            #logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgan.load_state_dict(torch.load(best_model_path))
            #logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgan.eval()
            break
        else:
            torch.save(tgan.state_dict(), get_checkpoint_path(epoch))


    # testing phase use all information
    tgan.ngh_finder = full_ngh_finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, 
                                                          test_data.sources, test_data.destinations, test_data.timestamps, test_data.labels)

    #nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, 
    #                                                                  new_node_test_data.sources, new_node_test_data.destinations, new_node_test_data.timestamps, new_node_test_data.labels)

    logger.info('Test statistics: Old nodes -- auc: {}, ap: {}'.format(np.mean(test_auc), np.mean(test_ap)))
    #logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

    #auc_roc_list.append(nn_test_auc)
    #ap_list.append(nn_test_ap)

    auc_roc_list.append(np.mean(test_auc))
    ap_list.append(np.mean(test_ap))

    logger.info('Saving TGAN model')
    torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGAN models saved')

logger.info(f'============================================ Test End =================================================\n')
final_auc = sum(auc_roc_list)/len(auc_roc_list)
final_ap = sum(ap_list)/len(ap_list)
logger.info('Total avg Test statistics: Old nodes -- auc: {}, ap: {}'.format(final_auc, final_ap))