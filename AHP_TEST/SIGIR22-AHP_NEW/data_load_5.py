import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import dgl
import preprocess


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H) # ( nv * ne )
    # DV = np.sum(H, axis=1) + 1e-5
    # DE = np.sum(H, axis=0) + 1e-5

    # invDE = np.mat(np.diag(np.power(DE, -1)))
    # DV2 = np.mat(np.diag(np.power(DV, -1)))
    # H = np.mat(H)
    HT = H.T

    # G = DV2 * H * invDE * HT * DV2
    G = H @ HT
    
    return G
    
    
def gen_DGLGraph(snapshot_data):
    """
        snapshot_data(edge별 node set) 기반 
        DGLGraph generate
    """
    
    he = []
    hv = []
    for i, edge in enumerate(snapshot_data):
        for v in edge:
            he.append(i)
            hv.append(v)
    data_dict = {
        ('node', 'in', 'edge'): (hv, he),        
        ('edge', 'con', 'node'): (he, hv)
    }
    
    g = dgl.heterograph(data_dict)
        
    return g

    
def gen_init_data(args, num_node):
    """
        snapshot 별로 변함 없는 data들을 미리 선언합니다.
        - node 수
        - node feature
        - node feature dim
    """
    args.device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'                 

    args.nv = num_node # node 수 (node의 최대 index)
    args.input_dim = args.dim_vertex # node feature dim
    
    args.v_feat = torch.rand(args.nv, args.input_dim)  
    
    return args

def find_min_max(matrix):
    flattened_matrix = np.array(matrix).flatten()
    min_val = np.min(flattened_matrix)
    max_val = np.max(flattened_matrix)
    return min_val, max_val

def gen_data(args, snapshot_data, snapshot_time):
    data_dict = {}  # 데이터를 저장할 딕셔너리 생성

    nv = args.nv
    
    # # Time encoder terms
    # data_dict['start_time'] = torch.FloatTensor([min(snapshot_time)])
    # data_dict['end_time'] = torch.FloatTensor([max(snapshot_time)])
    # data_dict['time_interval'] = torch.cat((data_dict['start_time'], data_dict['end_time']), dim=0)
    
    # Hyperedge terms
    ne = len(snapshot_data)  # snapshot내 edge 수
    args.ne = ne
    data_dict['e_feat'] = torch.rand(ne, args.dim_edge)

    if args.model == 'gcn':
        # GCN terms
        incidence = torch.zeros(ne, nv)
        for edge_idx, node_set in enumerate(snapshot_data):
            for node_idx in node_set:
                incidence[edge_idx, node_idx] = 1
        data_dict['h_incidence'] = incidence.T
        data_dict['GCN_G'] = generate_G_from_H(data_dict['h_incidence'])
        
    elif args.model == 'hnhn':
        # 1. structural proximity
        incidence = torch.zeros(nv, ne)
        for edge_idx, node_set in enumerate(snapshot_data):
            for node_idx in node_set:
                incidence[node_idx, edge_idx] += 1
        data_dict['h_incidence'] = incidence.T # ne, nv
        struct_edge_G = generate_G_from_H(data_dict['h_incidence']) # ne, ne        
        np.fill_diagonal(struct_edge_G, 0)
        
        # 2. temporal proximity
        temp_edge_G = torch.zeros(ne, ne)
        non_zero_indices = np.nonzero(struct_edge_G)
        
        rows, cols = non_zero_indices        
        for row, col in zip(rows, cols):
            edge_time_1 = snapshot_time[row]
            edge_time_2 = snapshot_time[col]
            difference_time = abs(edge_time_1 - edge_time_2)
            if difference_time != 0:
                temp_edge_G[row,col] = 1/ difference_time        
        
        # # 정규화
        # min_struct, max_struct = find_min_max(temp_edge_G)         
        # temp_edge_G = (temp_edge_G - min_struct) / (max_struct - min_struct)
        
        data_dict['temp_edge_G'] = temp_edge_G  # ne, ne            
        data_dict['struct_edge_G'] = torch.tensor(struct_edge_G)  # ne, ne
                
        # HNHN terms
        data_dict['v_weight'] = torch.zeros(nv, 1)
        data_dict['e_weight'] = torch.zeros(ne, 1)
        node2sum = defaultdict(list)
        edge2sum = defaultdict(list)
        e_reg_weight = torch.zeros(ne)
        v_reg_weight = torch.zeros(nv)

        for edge_idx, node_set in enumerate(snapshot_data):
            for node_idx in node_set:
                e_wt = data_dict['e_weight'][edge_idx]
                e_reg_wt = e_wt ** args.alpha_e
                e_reg_weight[edge_idx] = e_reg_wt
                node2sum[node_idx].append(e_reg_wt)

                v_wt = data_dict['v_weight'][node_idx]
                v_reg_wt = v_wt ** args.alpha_v
                v_reg_weight[node_idx] = v_reg_wt
                edge2sum[edge_idx].append(v_reg_wt)

        v_reg_sum = torch.zeros(nv)
        e_reg_sum = torch.zeros(ne)

        for node_idx, wt_l in node2sum.items():
            v_reg_sum[node_idx] = sum(wt_l)
        for edge_idx, wt_l in edge2sum.items():
            e_reg_sum[edge_idx] = sum(wt_l)

        e_reg_sum[e_reg_sum == 0] = 1
        v_reg_sum[v_reg_sum == 0] = 1
        data_dict['e_reg_weight'] = torch.Tensor(e_reg_weight).unsqueeze(-1)
        data_dict['v_reg_sum'] = torch.Tensor(v_reg_sum).unsqueeze(-1)
        data_dict['v_reg_weight'] = torch.Tensor(v_reg_weight).unsqueeze(-1)
        data_dict['e_reg_sum'] = torch.Tensor(e_reg_sum).unsqueeze(-1)
    return data_dict

def load_snapshot(args, DATA):
     
    # 1. get feature and edge index        
    r_data = pd.read_csv('/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/{}/new_hyper_{}.csv'.format(DATA, DATA))
    
    # 2. make snapshot  
    data_info = preprocess.get_datainfo(r_data)
    all_hyperedges, timestamps, num_node = preprocess.get_full_data(args, data_info)
    snapshot_data, snapshot_time = preprocess.split_in_snapshot(args,all_hyperedges,timestamps)
       
    return snapshot_data, snapshot_time, num_node

def load_fulldata(args, DATA):
     
    # 1. get feature and edge index        
    r_data = pd.read_csv('/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/{}/new_hyper_{}.csv'.format(DATA, DATA))
    
    # 2. make snapshot  
    data_info = preprocess.get_datainfo(r_data)
    all_hyperedges, timestamps, num_node = preprocess.get_full_data(args, data_info)
    
    return all_hyperedges, timestamps, num_node