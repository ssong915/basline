import torch
import numpy as np
from collections import defaultdict
import dgl
from batch import HEBatchGenerator
import pandas as pd
import ast
from sampler import *

def gen_DGLGraph(args, ground):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    he = []
    hv = []
    for i, edge in enumerate(ground):
        for v in edge :
            he.append(i)
            hv.append(v)
    data_dict = {
        ('node', 'in', 'edge'): (hv, he),        
        ('edge', 'con', 'node'): (he, hv)
    }
    g = dgl.heterograph(data_dict)
    return g.to(device)

def gen_data(args, dataset_name):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
        
    r_data = pd.read_csv('/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/{}/new_hyper_{}.csv'.format(dataset_name, dataset_name))
    hypergraph_data = r_data['node_set'].apply(ast.literal_eval)
    
    args.input_dim = 400 #임의로 dimension 지정
    ne = r_data['h_edge_idx'].max() + 1
    nv = max(max(row) for row in list(hypergraph_data)) + 1
    
    args.ne = ne    
    args.nv = nv
        
    node_feat = torch.rand(nv, args.input_dim)
    args.v_feat = node_feat.to(device)
    args.e_feat = torch.ones(args.ne, args.dim_edge).to(device)
    
    # if isinstance(node_feat, np.ndarray):
    #     args.v = torch.from_numpy(node_feat.astype(np.float32)).to(device)
    # else:
    #     args.v = torch.from_numpy(np.array(node_feat.astype(np.float32).todense())).to(device)
        
    # args.incidence = torch.zeros(ne, nv).to(device)
    # for edge_idx, node_set in enumerate(hypergraph_data):
    #     for node_idx in node_set:
    #         args.incidence[edge_idx, node_idx] = 1
        
    # HNHN terms
    args.v_weight = torch.zeros(nv).to(device)
    args.e_weight = torch.zeros(ne).to(device)
    node2sum = defaultdict(list)
    edge2sum = defaultdict(list)
    e_reg_weight = torch.zeros(args.ne) 
    v_reg_weight = torch.zeros(args.nv) 
    for edge_idx, node_set in enumerate(hypergraph_data):
        for node_idx in node_set:
            e_wt = args.e_weight[edge_idx]
            e_reg_wt = e_wt**args.alpha_e 
            e_reg_weight[edge_idx] = e_reg_wt
            node2sum[node_idx].append(e_reg_wt) 
            
            v_wt = args.v_weight[node_idx]
            v_reg_wt = v_wt**args.alpha_v
            v_reg_weight[node_idx] = v_reg_wt
            edge2sum[edge_idx].append(v_reg_wt)      
        
    v_reg_sum = torch.zeros(nv) 
    e_reg_sum = torch.zeros(ne) 
    for node_idx, wt_l in node2sum.items():
        v_reg_sum[node_idx] = sum(wt_l)
    for edge_idx, wt_l in edge2sum.items():
        e_reg_sum[edge_idx] = sum(wt_l)

    e_reg_sum[e_reg_sum==0] = 1
    v_reg_sum[v_reg_sum==0] = 1
    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
    args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)

    return args

    
def load_train(train_pos, bs, device):
    # train_pos = data_dict["train_only_pos"] + data_dict["ground_train"]
    train_pos_label = [1 for i in range(len(train_pos))]
    train_batchloader = HEBatchGenerator(train_pos, train_pos_label, bs, device, test_generator=False)    
    return train_batchloader

def load_val(val, bs, device, label):
    if label=="pos":
        # val = data_dict["train_only_pos"] + data_dict["ground_train"]
        val = val 
        val_label = [1 for i in range(len(val))]
    else:
        if label == 'mns' :
            mns = MNSSampler(len(val))
            t_mns = mns(set(tuple(x) for x in val))
            t_mns = list(t_mns)
            neg_hedges = [list(edge) for edge in t_mns]        
            
        elif label == 'sns':
            sns = SNSSampler(len(val))
            t_sns = sns(set(tuple(x) for x in val))
            t_sns = list(t_sns)
            neg_hedges = [list(edge) for edge in t_sns]    
            
        elif label == 'cns':
            cns = CNSSampler(len(val))
            t_cns = cns(set(tuple(x) for x in val))    
            t_cns = list(t_cns)    
            neg_hedges = [list(edge) for edge in t_cns]            
        val = neg_hedges
        # val = data_dict[f"valid_{label}"]
        val_label = [0 for i in range(len(val))]
    val_batchloader = HEBatchGenerator(val, val_label, bs, device, test_generator=True)    
    return val_batchloader


def load_test(test, bs, device, label):
    if label=="pos":
        # val = data_dict["train_only_pos"] + data_dict["ground_train"]
        test = test 
        test_label = [1 for i in range(len(test))]
    else:
        if label == 'mns' :
            mns = MNSSampler(len(test))
            t_mns = mns(set(tuple(x) for x in test))
            t_mns = list(t_mns)
            neg_hedges = [list(edge) for edge in t_mns]        
            
        elif label == 'sns':
            sns = SNSSampler(len(test))
            t_sns = sns(set(tuple(x) for x in test))
            t_sns = list(t_sns)
            neg_hedges = [list(edge) for edge in t_sns]    
            
        elif label == 'cns':
            cns = CNSSampler(len(test))
            t_cns = cns(set(tuple(x) for x in test))    
            t_cns = list(t_cns)    
            neg_hedges = [list(edge) for edge in t_cns]            
        test = neg_hedges
        # val = data_dict[f"valid_{label}"]
        test_label = [0 for i in range(len(test))]
    test_batchloader = HEBatchGenerator(test, test_label, bs, device, test_generator=True)    
    return test_batchloader