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


def train(model, all_v_feat, train_dataset, val_dataset,  
decoder, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, optimizer, scheduler, device, num_epochs=25, print_freq=500):

    state_dict_updates = 0          # number of epochs that updates state_dict
    model = model.cuda()

    train_all_v_feat = all_v_feat.clone()
    train_update_feat = all_v_feat.clone()
    val_all_v_feat = all_v_feat.clone()
    val_update_feat = all_v_feat.clone()

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

            if phase == 'train':
                total_time = len(train_dataset)
                
                for time in range(total_time-1):

                    next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = our_utils.get_next_samples(train_dataset[time+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)

                    if args.neg_mode == 'sns':
                        next_neg_edges = sns_neg_hedge
                    elif args.neg_mode == 'mns':
                        next_neg_edges = mns_neg_hedge
                    elif args.neg_mode == 'cns':
                        next_neg_edges = cns_neg_hedge
                    elif args.neg_mode == 'mix':
                        d = len(train_dataset[time+1]) // 3
                        next_neg_edges = sns_neg_hedge[0:d]+cns_neg_hedge[0:d]+mns_neg_hedge[0:d]
                    
                    reindexed_train_hypergraph, origin_train_node = reindex_snapshot(train_dataset[time])
                    time_hedge_set = [node for sublist in reindexed_train_hypergraph for node in sublist]
                    time_unique_node = np.unique(time_hedge_set) 
                    time_node_feat = train_all_v_feat[origin_train_node].to(device)
                    update_node_feat = time_node_feat.clone()
                   
                    time_G = nx.Graph()
                    for node in time_unique_node:
                        time_G.add_node(node)
                    time_adj_matrix = graph_from_hypergraph(reindexed_train_hypergraph, time_G)
                    time_neighbor_dict = [list(np.nonzero(row)[0]) for row in time_adj_matrix]


                    updated_n_feat = model(ids=time_unique_node, feats=time_node_feat, edge_dict=time_neighbor_dict, G=time_G, ite=epo, phase=phase)
                    update_node_feat[time_unique_node] = updated_n_feat

                    train_update_feat[origin_train_node] = time_node_feat

                    train_pos_edges = next_pos_edges
                    train_neg_edges = next_neg_edges
                    # all_v_feat은 학습된(model 통과+update) node feature 반영된 상태 
                    train_update_feat = train_update_feat.clone().detach()

                    pos_preds = decoder(train_update_feat.to(device), train_pos_edges, 'Maxmin')
                    pos_preds = pos_preds.squeeze()
        
                    neg_preds = decoder(train_update_feat.to(device), train_neg_edges, 'Maxmin')
                    neg_preds = neg_preds.squeeze()

                    pos_labels = torch.ones_like(pos_preds)
                    neg_labels = torch.zeros_like(neg_preds)

                    # 3. Compute training loss and update parameters
                    real_loss = criterion(pos_preds, pos_labels)
                    fake_loss = criterion(neg_preds, neg_labels)
                    
                    train_loss = real_loss + fake_loss 
                    train_loss = train_loss/ 2

                    # backward + optimize only if in training phase
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            elif phase == 'val':

                total_time = len(val_dataset)

                preds = []
                labels = []

                best_perform = 0.0
                
                for time in range(total_time-1):

                    next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = our_utils.get_next_samples(val_dataset[time+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)

                    if args.neg_mode == 'sns':
                        next_neg_edges = sns_neg_hedge
                    elif args.neg_mode == 'mns':
                        next_neg_edges = mns_neg_hedge
                    elif args.neg_mode == 'cns':
                        next_neg_edges = cns_neg_hedge
                    elif args.neg_mode == 'mix':
                        d = len(val_dataset[time+1]) // 3
                        next_neg_edges = sns_neg_hedge[0:d]+cns_neg_hedge[0:d]+mns_neg_hedge[0:d]

                    reindexed_val_hypergraph, origin_val_node = reindex_snapshot(val_dataset[time])
                    val_time_hedge_set = [node for sublist in reindexed_val_hypergraph for node in sublist]
                    val_time_unique_node = np.unique(val_time_hedge_set) 
                    val_node_feat = val_all_v_feat[origin_val_node].to(device)
                    updatae_val_node_feat = val_node_feat.clone() 
                   
                    val_G = nx.Graph()
                    for node in val_time_unique_node:
                        val_G.add_node(node)
                    val_adj_matrix = graph_from_hypergraph(reindexed_val_hypergraph, val_G)
                    val_neighbor_dict = [list(np.nonzero(row)[0]) for row in val_adj_matrix]

                    updated_n_feat = model(ids=val_time_unique_node, feats=val_node_feat, edge_dict=val_neighbor_dict, G=val_G, ite=epo, phase=phase)
                    updatae_val_node_feat[val_time_unique_node] = updated_n_feat
                    val_update_feat[origin_val_node] = updatae_val_node_feat
                    val_pos_edges = next_pos_edges
                    val_neg_edges = next_neg_edges

                    # all_v_feat은 학습된(모델 통과) node feature 반영된 상태 
                    pos_preds = decoder(val_update_feat, val_pos_edges, 'Maxmin')
                    pos_preds = pos_preds.squeeze()
        
                    neg_preds = decoder(val_update_feat, val_neg_edges, 'Maxmin')
                    neg_preds = neg_preds.squeeze()

                    pos_labels = torch.ones_like(pos_preds)
                    neg_labels = torch.zeros_like(neg_preds)

                    preds += ( pos_preds.tolist() + neg_preds.tolist() )
                    labels += ( pos_labels.tolist() + neg_labels.tolist() )

                val_roc, val_ap  = our_utils.measure(labels, preds) 

                epoch_perform = val_roc

                if epoch_perform > best_perform:
                    best_perfom = epoch_perform

                    best_model = copy.deepcopy(model.state_dict())
                    best_epoch = epoch 
                    state_dict_updates += 1

    print(f'\nState dict updates {state_dict_updates}')

    return (best_model, best_epoch)


def test(model, updated_model, all_v_feat, test_dataset, decoder, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, device, test_time = 1):

    best_model, epo = updated_model
    model = model.cuda()
    model.load_state_dict(best_model)
    model.eval()

    running_corrects = 0.0
    test_all_v_feat = all_v_feat.clone()
    test_update_feat = all_v_feat.clone()

    auroc = []
    ap = []

    total_time = len(test_dataset)
    phase = 'test'

    for time in range(total_time-1):

        with torch.no_grad():

            reindexed_test_hypergraph, origin_test_node = reindex_snapshot(test_dataset[time])
            test_hedge_set = [node for sublist in reindexed_test_hypergraph for node in sublist]
            test_unique_node = np.unique(test_hedge_set) 
            test_node_feat = test_all_v_feat[origin_test_node].to(device)
            update_node_feat = test_node_feat.clone()
            
            test_G = nx.Graph()
            for node in test_unique_node:
                test_G.add_node(node)
            test_adj_matrix = graph_from_hypergraph(reindexed_test_hypergraph, test_G)
            test_neighbor_dict = [list(np.nonzero(row)[0]) for row in test_adj_matrix]

            next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = our_utils.get_next_samples(test_dataset[time+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)

            updated_n_feat = model(ids=test_unique_node, feats=test_node_feat, edge_dict=test_neighbor_dict, G=test_G, ite=epo, phase=phase)
            update_node_feat[test_unique_node] = updated_n_feat
            test_update_feat[origin_test_node] = update_node_feat

            test_update_feat = test_update_feat.clone().detach()

            test_pos_edges = next_pos_edges
            pos_preds = decoder(test_update_feat, test_pos_edges, 'Maxmin')
            pos_preds = pos_preds.squeeze()
            pos_labels = torch.ones_like(pos_preds)

            sns_neg_preds = decoder(test_update_feat, sns_neg_hedge, 'Maxmin')
            sns_neg_preds = sns_neg_preds.squeeze()
            neg_labels = torch.zeros_like(sns_neg_preds)
            preds = ( pos_preds.tolist() + sns_neg_preds.tolist() )
            labels = ( pos_labels.tolist() + neg_labels.tolist() )

            sns_roc, sns_ap = our_utils.measure(labels, preds) 

            mns_neg_preds = decoder(test_update_feat, mns_neg_hedge, 'Maxmin')
            mns_neg_preds = mns_neg_preds.squeeze()
            neg_labels = torch.zeros_like(mns_neg_preds)
            preds = ( pos_preds.tolist() + mns_neg_preds.tolist() )
            labels = ( pos_labels.tolist() + neg_labels.tolist() )

            mns_roc, mns_ap = our_utils.measure(labels, preds) 

            cns_neg_preds = decoder(test_update_feat, cns_neg_hedge, 'Maxmin')
            cns_neg_preds = cns_neg_preds.squeeze()
            neg_labels = torch.zeros_like(cns_neg_preds)
            preds = ( pos_preds.tolist() + cns_neg_preds.tolist() )
            labels = ( pos_labels.tolist() + neg_labels.tolist() )

            cns_roc, cns_ap = our_utils.measure(labels, preds) 

            d = len(next_pos_edges) // 3
            mixed_neg_hedge = sns_neg_hedge[0:d]+cns_neg_hedge[0:d]+mns_neg_hedge[0:d]
            mix_neg_preds = decoder(test_update_feat, mixed_neg_hedge, 'Maxmin')
            mix_neg_preds = mix_neg_preds.squeeze()
            neg_labels = torch.zeros_like(mix_neg_preds)
            preds = ( pos_preds.tolist() + mix_neg_preds.tolist() )
            labels = ( pos_labels.tolist() + neg_labels.tolist() )

            mix_roc, mix_ap = our_utils.measure(labels, preds) 

            roc_average = (sns_roc+mns_roc+cns_roc+mix_roc)/4
            ap_average = (sns_ap+mns_ap+cns_ap+mix_ap)/4     

            auroc.append(roc_average)
            ap.append(ap_average)
    
    test_roc = sum(auroc)/len(auroc)
    test_ap = sum(ap)/len(ap)

    print('*' * 20)
    print(f'Test AUROC: {test_roc} Test AP: {test_ap} @Epoch-{epo}')
    print('*' * 20)

    return test_roc, test_ap


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
    train_dataset, val_dataset, test_dataset, all_dataset = our_utils.split_data(args.data)

    total_hedge_set = [list(hedge) for hedge in all_dataset]
    total_node_set = [node for sublist in total_hedge_set for node in sublist]

    unique_node = np.unique(total_node_set)
    #min_clusters = int(unique_node.shape[0]/2)
    min_clusters = 5

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

    # transductive learning mode
    updated_model = train(model, all_node_feat, train_dataset, val_dataset,
        decoder, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, optimizer, schedular, device,cfg['max_epoch'], cfg['print_freq'])

    
    print('**** Model of lowest val loss ****')
    test_roc, test_ap = test(model, updated_model, all_node_feat, test_dataset,
    decoder, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, device, cfg['test_time']) 

if __name__ == '__main__':
    seed_num = 1000

    setup_seed(seed_num) 
    print('Using random seed: ', seed_num)

    cfg = get_config('config/config.yaml')
    args = our_utils.parse_args()
    cfg['model'] = args.model_version

    train_test_model(cfg, args)
