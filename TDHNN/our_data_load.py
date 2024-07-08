import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull
import scipy.io as scio
import numpy as np
import torchvision
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pickle
import itertools
import random
import pandas as pd

def parse_string_to_list(string):
    # 문자열에서 숫자만 추출하여 리스트로 변환
    numbers = np.fromstring(string[1:-1], dtype=int, sep=', ')
    return numbers.tolist()

def get_datainfo(r_dataset):
    #-------- DATA 불러와서 node_set, time(HIT 참고하여 timestamp 단위 변환), hyperedge_idx 저장 ---------# 
    
    r_dataset['node_set'] = r_dataset['node_set'].apply(parse_string_to_list)
    r_dataset = r_dataset.sort_values(by="time")
    r_dataset.reset_index(inplace=True) 

    return r_dataset
def get_snapshot(dataset, time_window_factor, time_start_factor):

    time = dataset['time']
    ts_start = (time.max() - time.min()) * time_start_factor + time.min()
    ts_end = time.max() - (time.max() - time.min()) * time_window_factor
    filter_data = dataset[(dataset['time'] >= ts_start) & (dataset['time']<=ts_end)]

    all_hyperedges = filter_data['node_set']
    timestamps = filter_data['time']
    
    time_hyperedges = list()
    
    freq_sec = 10000
    split_criterion = timestamps // freq_sec
    groups = np.unique(split_criterion)
    groups = np.sort(groups)
    for t in groups:
        period_members = (split_criterion == t) # t시점에 있는 아이 
        edge_data = all_hyperedges[period_members]
        time_hyperedges.append(edge_data)
        print(len(edge_data))

    return time_hyperedges

def graph_from_hypergraph(H_list):
    """
    calculate H from edge_list
    :param edge_dict: edge_list[i] = adjacent indices of index i
    :return: H, (n_nodes, n_nodes) numpy ndarray
    """
    # clique 확장
    tmp_edge_index = []
    for hedge in H_list:
        tmp_edge_index.extend(list(itertools.permutations(hedge,2)))
    
    edge_s = [ x[0] for x in tmp_edge_index]
    edge_e = [ x[1] for x in tmp_edge_index]

    edge_index = torch.LongTensor([edge_s,edge_e])

    return edge_index

def load_data2(args):
    """
    parses the dataset
    """
    dataset = args.dataset
    splits = args.splits
    device = torch.device(args.device)
    path = osp.abspath(__file__)         #当前文件绝对路径
    d_path = osp.dirname(path)           #当前文件所在目录
    # f_path = osp.dirname(d_path)         #当前文件所在目录的父目录
    f_path = osp.join(d_path, ('data2'))
    
    d_path_dict = {
        'ca_cora':osp.join(osp.join(f_path, ('coauthorship')),'cora'),
        'ca_dblp':osp.join(osp.join(f_path, ('coauthorship')),'dblp'),
        'cc_cora':osp.join(osp.join(f_path, ('cocitation')),'cora'),
        'cc_citeseer':osp.join(osp.join(f_path, ('cocitation')),'citeseer'),
        'ca_pubmed':osp.join(osp.join(f_path, ('cocitation')),'pubmed')
    }

    pickle_file = osp.join(d_path_dict[dataset], "splits", str(splits) + ".pickle")

    with open(osp.join(d_path_dict[dataset], 'features.pickle'), 'rb') as handle:
        features = pickle.load(handle).todense()

    with open(osp.join(d_path_dict[dataset], 'labels.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    with open(pickle_file, 'rb') as H: 
        Splits = pickle.load(H)
        train, test = Splits['train'], Splits['test']

    with open(osp.join(d_path_dict[dataset], 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

    tmp_edge_index = []
    for key in hypergraph.keys():
        ms = hypergraph[key]
        tmp_edge_index.extend(list(itertools.permutations(ms,2)))
    
    edge_s = [ x[0] for x in tmp_edge_index]
    edge_e = [ x[1] for x in tmp_edge_index]

    edge_index = torch.LongTensor([edge_s,edge_e])

    features = torch.Tensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    data = {
        'fts':features,
        'edge_index':edge_index,
        'lbls':labels,
        'train_idx':train,
        'test_idx':test
    }

    return data

def load_ft(args):
    if args.dataset == '40':
        data_dir = './data/ModelNet40_mvcnn_gvcnn.mat'
    elif args.dataset == 'NTU':
        data_dir = './data/NTU2012_mvcnn_gvcnn.mat'

    device = torch.device(args.device)
    feature_name = args.fts

    data = scio.loadmat(data_dir)
    lbls = data['Y'].astype(np.long) # shape (12311 x 1)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        fts = data['X'][0].item().astype(np.float32) # shape (12311 x 4096)
        fts = torch.Tensor(fts).to(device)
    elif feature_name == 'GVCNN':
        fts = data['X'][1].item().astype(np.float32)
        fts = torch.Tensor(fts).to(device)
    else:
        fts1 = data['X'][0].item().astype(np.float32)
        fts2 = data['X'][1].item().astype(np.float32)
        fts1 = torch.Tensor(fts1).to(device)
        fts2 = torch.Tensor(fts2).to(device)

        fts = torch.cat((fts1,fts2),dim=-1)

    if args.split_ratio < 0:
        train_idx = np.where(idx == 1)[0]
        test_idx = np.where(idx == 0)[0]
    else:
        nums = lbls.shape[0] # 전체 data 수 (node 수)
        num_train = int(nums * args.split_ratio) # 전체 data에 대한 train data 비율 (split_ratio = 0.8)
        idx_list = [i for i in range(nums)] # 전체 data의 index 저장 (len : 12311)

        train_idx = random.sample(idx_list, num_train) # 전체 index 기준으로 train data 개수만큼 index random sampling
        test_idx = [i for i in idx_list if i not in train_idx] # train data에 포함되지 않은 index는 test set으로 저장

    # train_idx = np.where(idx == 1)[0]
    # test_idx = np.where(idx == 0)[0]

    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    train_idx = torch.Tensor(train_idx).long().to(device) # shape (9848)
    test_idx = torch.Tensor(test_idx).long().to(device) # shape (2463)

    data = {
        'fts':fts,
        'lbls':lbls,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data
def get_full_data(dataset):

    time = dataset['time']
    ts_start = time.min()
    ts_end = time.max() 
    filter_data = dataset[(dataset['time'] >= ts_start) & (dataset['time']<=ts_end)]

    max_node_idx = max(max(row) for row in list(filter_data['node_set']))
    num_node = max_node_idx + 1
    
    all_hyperedges = filter_data['node_set']
    timestamps = filter_data['time']
    
    return all_hyperedges, timestamps, num_node
def split_in_snapshot(all_hyperedges, timestamps):

    snapshot_data = list()
    
    freq_sec = 2628000
    split_criterion = timestamps // freq_sec
    groups = np.unique(split_criterion)
    groups = np.sort(groups)
    merge_edge_data = []
   
    for t in groups:
        period_members = (split_criterion == t) # snapshot 내 time index들
        edge_data = list(all_hyperedges[period_members]) # snapshot 내 hyperedge들

        t_hedge_set = [node for sublist in edge_data for node in sublist]
        unique_node = np.unique(t_hedge_set)
        
        if len(unique_node) < 30 : 
            # [ 조건 미충족 ] : snapshot 내 edge 수가 부족하거나 / 중복된 edge가 많다면
            # 다음 snapshot과 merge 
            merge_edge_data = merge_edge_data + edge_data

            merge_hedge_set = [node for sublist in merge_edge_data for node in sublist]
            merge_unique_node = np.unique(merge_hedge_set)
            
            if len(merge_unique_node) >= 30:
                # merge된 data가 이제 조건을 충족한다면 append
                snapshot_data.append(merge_edge_data)
                merge_edge_data = []
        else :
            # [ 조건 충족 ]
            if len(merge_edge_data) != 0 :
                # 이전 snapshot이 [ 조건 미충족 ] 이었다면 merge
                edge_data = merge_edge_data + edge_data
            snapshot_data.append(edge_data)
            merge_edge_data = []

    return snapshot_data

def load_our_data(args, device):

    DATA = args.data
    r_data = pd.read_csv('/home/dake/workspace/DHGNN/hypergraph/{}/new_hyper_{}.csv'.format(DATA, DATA))

    data_info = get_datainfo(r_data)
    all_hyperedges, timestamps, num_node = get_full_data(data_info)
    snapshot_data = split_in_snapshot(all_hyperedges, timestamps)

    feature_dim = 128
    node_features = torch.ones((num_node, feature_dim))
    fts = node_features.type(torch.float32)
    fts = torch.Tensor(fts).to(device)

    return snapshot_data, fts

def load_snapshot_data(t_snapshot_data, fts, device):

    current_hgraph = t_snapshot_data
    all_node_set = [node for sublist in current_hgraph for node in sublist]
    all_node_in_snap = np.unique(all_node_set)

    total_size = len(current_hgraph) # 전체 hyperedge 수 
    train_size = int(total_size * 0.8) # 전체 data에 대한 train data 비율 (split_ratio = 0.8)
    train_idx = np.arange(0, train_size)
    test_idx = np.arange(train_size, total_size)

    train_hypergraph = [current_hgraph[i] for i in train_idx]
    test_hypergraph = [current_hgraph[i] for i in test_idx]

    train_edge_idx = graph_from_hypergraph(train_hypergraph)
    train_edge_idx = torch.Tensor(train_edge_idx).long().to(device)

    test_edge_idx = graph_from_hypergraph(test_hypergraph)
    test_edge_idx = torch.Tensor(test_edge_idx).long().to(device)

    train_node_set = [node for sublist in train_hypergraph for node in sublist]
    train_unique_node = np.unique(train_node_set) 
    train_node_fts = fts[train_unique_node]

    test_node_set = [node for sublist in test_hypergraph for node in sublist]
    test_unique_node = np.unique(test_node_set) 
    test_node_fts = fts[test_unique_node]

    train_data = {
        'fts':train_node_fts,
        'edge_index':train_edge_idx
    }

    test_data = {
        'fts':test_node_fts,
        'edge_index':test_edge_idx
    }

    return train_data, test_data, train_unique_node, test_unique_node, all_node_in_snap


def loading_data(args):
    if args.dataset in ['our']:
        return load_our_data(args)
   


def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def split_data(DATA):
    rf_data = torch.load('/home/dake/workspace/DHGNN/hypergraph/{}/feature_hyper-{}.pt'.format(DATA, DATA))
    r_data = pd.read_csv('/home/dake/workspace/DHGNN/hypergraph/{}/new_hyper_{}.csv'.format(DATA, DATA))
    
    # 2. make snapshot  
    data_info = get_datainfo(r_data)
    all_hyperedges, timestamps, num_node = get_full_data(data_info)
    
    total_size = len(all_hyperedges)
    train_size = int(total_size * 0.8)
    test_size = total_size - (train_size)

    train_idx = np.arange(0, train_size)
    test_idx = np.arange(train_size, total_size)

    train_hedge_data = all_hyperedges[train_idx]
    train_time_data = timestamps[train_idx]
    test_hedge_data = all_hyperedges[test_idx]
    test_time_data = timestamps[test_idx]

    train_dataset = split_in_snapshot(train_hedge_data, train_time_data)
    test_dataset = split_in_snapshot(test_hedge_data, test_time_data)

    return train_dataset, test_dataset, all_hyperedges

def load_data(t_snapshot_data, fts, device):

    current_hgraph = t_snapshot_data

    train_edge_idx = graph_from_hypergraph(current_hgraph)
    train_edge_idx = torch.Tensor(train_edge_idx).long().to(device)

    train_node_set = [node for sublist in current_hgraph for node in sublist]
    train_unique_node = np.unique(train_node_set) 
    train_node_fts = fts[train_unique_node]

    train_data = {
        'fts':train_node_fts,
        'edge_index':train_edge_idx
    }

    return train_data, train_unique_node