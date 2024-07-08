#!/usr/bin/env python

import argparse
import os
import torch
import copy
import time
import random
from config import get_config
from datasets import source_select
from torch import nn
import torch.optim as optim
from models import model_select
import sklearn
from sklearn import neighbors
import numpy as np
import pandas as pd
import networkx as nx

from utils.construct_hypergraph import _edge_dict_to_H, _generate_G_from_H
from utils.layer_utils import reindex_snapshot
import our_utils
from tqdm import tqdm
from models.decoder import *

#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

torch.autograd.set_detect_anomaly(True)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model, all_node_feat, train_node_feat, val_node_feat, origin_train_node, origin_val_node, train_unique_node, val_unique_node,
          train_neighbor_dict, val_neighbor_dict, train_G, val_G,
          decoder, next_pos_edges, next_neg_edges, n_train_idx, n_val_idx, optimizer, scheduler, device, num_epochs=25, print_freq=500):

    since = time.time()
    state_dict_updates = 0          # number of epochs that updates state_dict
    model = model.cuda()

    best_model = copy.deepcopy(model.state_dict())

    best_perform = 0.0
    running_loss = 0.0

    train_all_v_feat = all_node_feat.clone()
    train_fts = train_node_feat.clone()
    train_update_fts = train_node_feat.clone()
    val_fts = val_node_feat.clone()
    val_update_fts = val_node_feat.clone()
    val_all_v_feat = all_node_feat.clone()

    origin_train_node = np.array(origin_train_node)
    origin_val_node  = np.array(origin_val_node)

    criterion = nn.BCELoss()

    for epoch in tqdm(range(num_epochs)):
        epo = epoch
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                decoder.train()
            else:
                model.eval()  # Set model to evaluate mode
                decoder.eval()

                preds = []
                labels = []

            if phase == 'train':
                updated_n_feat = model(ids=train_unique_node, feats=train_fts, edge_dict=train_neighbor_dict, G=train_G, ite=epo, phase=phase)
                '''idx = torch.tensor(idx[:, None]).to(device)
                fts.scatter_(0, idx, updated_n_feat)

                s_index = torch.tensor(snapshot_node_index[:, None]).to(device)
                train_all_v_feat.scatter_(0, s_index, fts) '''   
                train_update_fts[train_unique_node] = updated_n_feat
                train_all_v_feat[origin_train_node] = train_update_fts
                train_pos_edges = [next_pos_edges[idx] for idx in n_train_idx]
                train_neg_edges = [next_neg_edges[idx] for idx in n_train_idx]

                # all_v_feat은 학습된(model 통과+update) node feature 반영된 상태 
                train_all_v_feat = train_all_v_feat.clone().detach()

                pos_preds = decoder(train_all_v_feat.to(device), train_pos_edges, 'Maxmin')
                pos_preds = pos_preds.squeeze()
    
                neg_preds = decoder(train_all_v_feat.to(device), train_neg_edges, 'Maxmin')
                neg_preds = neg_preds.squeeze()
            
            elif phase == 'val':

                updated_n_feat = model(ids=val_unique_node, feats=val_fts, edge_dict=val_neighbor_dict, G=val_G, ite=epo, phase=phase)
                '''idx = torch.tensor(idx[:, None]).to(device)
                val_fts.scatter_(0, idx, updated_n_feat)

                s_index = torch.tensor(snapshot_node_index[:, None]).to(device)
                val_all_v_feat.scatter_(0, s_index, val_fts)'''
                val_update_fts[val_unique_node] = updated_n_feat
                val_all_v_feat[origin_val_node] = val_update_fts  
                val_pos_edges = [next_pos_edges[idx] for idx in n_val_idx]
                val_neg_edges = [next_neg_edges[idx] for idx in n_val_idx]

                # all_v_feat은 학습된(모델 통과) node feature 반영된 상태 
                pos_preds = decoder(val_all_v_feat, val_pos_edges, 'Maxmin')
                pos_preds = pos_preds.squeeze()
    
                neg_preds = decoder(val_all_v_feat, val_neg_edges, 'Maxmin')
                neg_preds = neg_preds.squeeze()
                
            pos_labels = torch.ones_like(pos_preds)
            neg_labels = torch.zeros_like(neg_preds)

                # 3. Compute training loss and update parameters
            if phase == 'train':
                
                real_loss = criterion(pos_preds, pos_labels)
                fake_loss = criterion(neg_preds, neg_labels)
            
                train_loss = real_loss + fake_loss 
                train_loss = train_loss/ 2
               
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            elif phase == 'val':
                preds = ( pos_preds.tolist() + neg_preds.tolist() )
                labels = ( pos_labels.tolist() + neg_labels.tolist() )

                val_roc, val_ap  = our_utils.measure(labels, preds) 
                epoch_perform = val_roc

                if epoch_perform > best_perform:
                    best_perfom = epoch_perform
                    best_model = copy.deepcopy(model.state_dict())
                    best_v_feat = train_all_v_feat
                    best_epoch = epoch 
                    state_dict_updates += 1


    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'\nState dict updates {state_dict_updates}')

    return (best_model, best_epoch), best_v_feat


def test(model, updated_model, all_node_feat, test_node_feat, origin_test_node, test_unique_node,
         test_neighbor_dict, test_G, 
         decoder, next_pos_edges, sns_neg_hedge, mns_neg_hedge, cns_neg_hedge, n_test_idx, device, test_time = 1):
    """
    gcn-style whole graph test
    :param model_best:
    :param fts:
    :param lbls:
    :param idx_test:
    :param edge_dict:
    :param G: G for input HGNN layer
    :param device:
    :param test_time: test for several times and vote
    :return:
    """
    best_model, epo = updated_model
    model = model.cuda()
    model.load_state_dict(best_model)
    model.eval()

    running_corrects = 0.0
    test_fts = test_node_feat.clone()
    test_update_fts = test_node_feat.clone()
    test_all_v_feat = all_node_feat.clone()

    preds = []
    labels = []

    average_avg_roc =[]
    average_avg_ap = []
    phase = 'test'

    for _ in range(test_time):

        with torch.no_grad():

            updated_n_feat = model(ids=test_unique_node, feats=test_fts, edge_dict=test_neighbor_dict, G=test_G, ite=epo, phase=phase)
            test_update_fts[test_unique_node] = updated_n_feat
            test_all_v_feat[origin_test_node] = test_fts

            test_all_v_feat = test_all_v_feat.clone().detach()

            test_pos_edges = [next_pos_edges[idx] for idx in n_test_idx]
            pos_preds = decoder(test_all_v_feat, test_pos_edges, 'Maxmin')
            pos_preds = pos_preds.squeeze()
            pos_labels = torch.ones_like(pos_preds)


            sns_neg_edges = [sns_neg_hedge[idx] for idx in n_test_idx]
            sns_neg_preds = decoder(test_all_v_feat, sns_neg_edges, 'Maxmin')
            sns_neg_preds = sns_neg_preds.squeeze()
            neg_labels = torch.zeros_like(sns_neg_preds)
            preds = ( pos_preds.tolist() + sns_neg_preds.tolist() )
            labels = ( pos_labels.tolist() + neg_labels.tolist() )

            sns_roc, sns_ap = our_utils.measure(labels, preds) 

            mns_neg_edges = [mns_neg_hedge[idx] for idx in n_test_idx]
            mns_neg_preds = decoder(test_all_v_feat, mns_neg_edges, 'Maxmin')
            mns_neg_preds = mns_neg_preds.squeeze()
            neg_labels = torch.zeros_like(mns_neg_preds)
            preds = ( pos_preds.tolist() + mns_neg_preds.tolist() )
            labels = ( pos_labels.tolist() + neg_labels.tolist() )

            mns_roc, mns_ap = our_utils.measure(labels, preds) 

            cns_neg_edges = [cns_neg_hedge[idx] for idx in n_test_idx]
            cns_neg_preds = decoder(test_all_v_feat, cns_neg_edges, 'Maxmin')
            cns_neg_preds = cns_neg_preds.squeeze()
            neg_labels = torch.zeros_like(cns_neg_preds)
            preds = ( pos_preds.tolist() + cns_neg_preds.tolist() )
            labels = ( pos_labels.tolist() + neg_labels.tolist() )

            cns_roc, cns_ap = our_utils.measure(labels, preds) 

            d = len(next_pos_edges) // 3
            mixed_neg_hedge = sns_neg_hedge[0:d]+cns_neg_hedge[0:d]+mns_neg_hedge[0:d]
            mix_neg_preds = decoder(test_all_v_feat, mixed_neg_hedge, 'Maxmin')
            mix_neg_preds = mix_neg_preds.squeeze()
            neg_labels = torch.zeros_like(mix_neg_preds)
            preds = ( pos_preds.tolist() + mix_neg_preds.tolist() )
            labels = ( pos_labels.tolist() + neg_labels.tolist() )

            mix_roc, mix_ap = our_utils.measure(labels, preds) 

            roc_average = (sns_roc+mns_roc+cns_roc+mix_roc)/4
            ap_average = (sns_ap+mns_ap+cns_ap+mix_ap)/4        

    print('*' * 20)
    print(f'Test AUROC: {roc_average} Test AP: {ap_average} @Epoch-{epo}')
    print('*' * 20)

    return roc_average, ap_average


def graph_from_hypergraph(H_list, G):
    # 각 hyperedge를 clique로 확장하여 그래프에 추가
    for hedge in H_list:
        clique_edges = [(node_i, node_j) for node_i in hedge for node_j in hedge if node_i != node_j]
        G.add_edges_from(clique_edges)

    # 그래프의 노드와 인접 행렬 출력
    nodes = list(G.nodes())
    adj_matrix = nx.to_numpy_array(G)
    adjacency_matrix = np.array(adj_matrix)

    return adjacency_matrix


def train_test_model(cfg, args):

    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_data, num_node = our_utils.load_snapshot(args.data)
    unique_counts_per_snapshot = [len(set(sum(snapshot, []))) for snapshot in snapshot_data]
    #min_clusters = min(unique_counts_per_snapshot)
    #min_clusters = int(min(unique_counts_per_snapshot)/2)
    min_clusters = 5
    total_hedge_set = [list(hedge) for t_hedge in snapshot_data for hedge in t_hedge]
    total_hedge_set = [node for sublist in total_hedge_set for node in sublist]
    unique_node = np.unique(total_hedge_set)

    total_time = len(snapshot_data)

    feature_dim = 128
    node_features = torch.ones((len(unique_node), feature_dim))
    all_node_feat = node_features.type(torch.float32)
    all_node_feat = all_node_feat.to(device)

    all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = our_utils.get_all_neg_samples(args)
    
    n_category = 5
    model = model_select(cfg['model'])\
        (dim_feat=all_node_feat.size(1), n_categories=n_category, k_structured=cfg['k_structured'], k_nearest=cfg['k_nearest'],
        k_cluster=cfg['k_cluster'], wu_knn=cfg['wu_knn'], wu_kmeans=cfg['wu_kmeans'], wu_struct=cfg['wu_struct'],
        clusters= min_clusters, adjacent_centers=cfg['adjacent_centers'], n_layers=cfg['n_layers'], layer_spec=cfg['layer_spec'],
        dropout_rate=cfg['drop_out'], has_bias=cfg['has_bias'])

    #initialize model
    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            nn.init.xavier_uniform_(state_dict[key])
        elif 'bias' in key:
            state_dict[key] = state_dict[key].zero_()
    
    cls_layers = [128, 128, 8, 1]
    decoder = Decoder(cls_layers)
    decoder = decoder.to(device)

    auc_roc_list = []
    ap_list = []

    optimizer = optim.Adam(list(model.parameters())+list(decoder.parameters()), lr=cfg['lr'],weight_decay=cfg['weight_decay'], eps=1e-20)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])

    for t in tqdm(range(total_time-1)): 

        next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = our_utils.get_next_samples(snapshot_data[t+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)

        time_hyperedge = snapshot_data[t]
        total_size = len(time_hyperedge)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.2)
        test_size = total_size - (train_size + val_size)

        train_idx = np.arange(0, train_size)
        val_idx = np.arange(train_size, train_size + val_size)
        test_idx = np.arange(train_size + val_size, total_size)

        train_hypergraph = [time_hyperedge[i] for i in train_idx]
        val_hypergraph = [time_hyperedge[i] for i in val_idx]
        test_hypergraph = [time_hyperedge[i] for i in test_idx]

        reindexed_train_hypergraph, origin_train_node = reindex_snapshot(train_hypergraph)
        reindexed_val_hypergraph, origin_val_node = reindex_snapshot(val_hypergraph)
        reindexed_test_hypergraph, origin_test_node = reindex_snapshot(test_hypergraph)

        train_hedge_set = [node for sublist in reindexed_train_hypergraph for node in sublist]
        train_unique_node = np.unique(train_hedge_set) 
        train_node_feat = all_node_feat[origin_train_node].to(device)

        val_hedge_set = [node for sublist in reindexed_val_hypergraph for node in sublist]
        val_unique_node = np.unique(val_hedge_set) 
        val_node_feat = all_node_feat[origin_val_node].to(device)

        test_hedge_set = [node for sublist in reindexed_test_hypergraph for node in sublist]
        test_unique_node = np.unique(test_hedge_set) 
        test_node_feat = all_node_feat[origin_test_node].to(device)

        if args.neg_mode == 'sns':
            next_neg_edges = sns_neg_hedge
        elif args.neg_mode == 'mns':
            next_neg_edges = mns_neg_hedge
        elif args.neg_mode == 'cns':
            next_neg_edges = cns_neg_hedge
        elif args.neg_mode == 'mix':
            d = len(next_pos_edges) // 3
            next_neg_edges = sns_neg_hedge[0:d]+cns_neg_hedge[0:d]+mns_neg_hedge[0:d]
        
        train_G = nx.Graph()
        val_G = nx.Graph()
        test_G = nx.Graph()

        for node in train_unique_node:
            train_G.add_node(node)
        train_adj_matrix = graph_from_hypergraph(reindexed_train_hypergraph, train_G)
        train_neighbor_dict = [list(np.nonzero(row)[0]) for row in train_adj_matrix]

        for node in val_unique_node:
            val_G.add_node(node)
        val_adj_matrix = graph_from_hypergraph(reindexed_val_hypergraph, val_G)
        val_neighbor_dict = [list(np.nonzero(row)[0]) for row in val_adj_matrix]

        for node in test_unique_node:
            test_G.add_node(node)
        test_adj_matrix = graph_from_hypergraph(reindexed_test_hypergraph, test_G)
        test_neighbor_dict = [list(np.nonzero(row)[0]) for row in test_adj_matrix]
    
        n_total_size = len(next_pos_edges)
        n_train_size = int(n_total_size * 0.1)
        n_val_size = int(n_total_size * 0.3)
        n_test_size = n_total_size - (n_train_size + n_val_size)

        np.random.shuffle(next_pos_edges)

        n_train_idx = np.arange(0, n_train_size)
        n_val_idx = np.arange(n_train_size, n_train_size + n_val_size)
        n_test_idx = np.arange(n_train_size + n_val_size, n_total_size)

        # transductive learning mode
        updated_model, update_v_feat = train(model, all_node_feat, train_node_feat, val_node_feat, origin_train_node, origin_val_node, train_unique_node, val_unique_node,
                                             train_neighbor_dict, val_neighbor_dict, train_G, val_G,
                                             decoder, next_pos_edges, next_neg_edges, n_train_idx, n_val_idx, optimizer, schedular, device,cfg['max_epoch'], cfg['print_freq'])

        if test_idx is not None:
            print('**** Model of lowest val loss ****')
            test_roc, test_ap = test(model, updated_model, all_node_feat, test_node_feat, origin_test_node, test_unique_node,
                                     test_neighbor_dict, test_G, 
                                     decoder, next_pos_edges, sns_neg_hedge, mns_neg_hedge, cns_neg_hedge, n_test_idx, device, 1) 

        all_node_feat = update_v_feat
        all_node_feat = all_node_feat.clone().detach()
         
        auc_roc_list.append(test_roc)
        ap_list.append(test_ap)

    final_roc = sum(auc_roc_list)/len(auc_roc_list)
    final_ap = sum(ap_list)/len(ap_list)

    print('[ FINAL ]')
    print('AUROC\t AP\t ')
    print(f'{final_roc:.4f}\t{final_ap:.4f}')

if __name__ == '__main__':
    seed_num = 1000

    setup_seed(seed_num) 
    print('Using random seed: ', seed_num)

    cfg = get_config('config/config.yaml')
    args = our_utils.parse_args()
    cfg['model'] = args.model_version

    train_test_model(cfg, args)
