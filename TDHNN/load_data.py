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

def load_cite(args):
    dname = args.dataset
    device = torch.device(args.device)
    path = osp.abspath(__file__)         #当前文件绝对路径
    d_path = osp.dirname(path)           #当前文件所在目录
    # f_path = osp.dirname(d_path)         #当前文件所在目录的父目录
    f_path = osp.join(d_path, ('data'))

    dataset = Planetoid(f_path,dname)      #dataset

    tmp = dataset[0].to(device)
    fts = tmp.x
    lbls = tmp.y

    if args.split_ratio < 0:
        train_idx = tmp.train_mask
        test_idx = tmp.test_mask
    else:
        nums = lbls.shape[0]
        num_train = int(nums * args.split_ratio)
        idx_list = [i for i in range(nums)]

        train_idx = random.sample(idx_list, num_train)
        test_idx = [i for i in idx_list if i not in train_idx]

        train_idx = torch.tensor(train_idx)
        test_idx = torch.tensor(test_idx)

    data = {
        'fts':fts,
        'edge_index':tmp.edge_index,
        'lbls':lbls,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat
def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features

def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


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

    H = None
    fts = None
    if feature_name == 'MVCNN':
        mvcnn_ft = data['X'][0].item().astype(np.float32) # shape (12311 x 4096)
        fts = feature_concat(fts, mvcnn_ft)
        tmp = construct_H_with_KNN(fts, K_neigs=10,
                                        split_diff_scale=False,
                                        is_probH=True, m_prob=1)
        H = hyperedge_concat(H, tmp)
        fts = torch.Tensor(mvcnn_ft).to(device)

    elif feature_name == 'GVCNN':
        gvcnn_ft = data['X'][1].item().astype(np.float32)
        fts = feature_concat(fts, gvcnn_ft)
        tmp = construct_H_with_KNN(fts, K_neigs=10,
                                        split_diff_scale=False,
                                        is_probH=True, m_prob=1)
        H = hyperedge_concat(H, tmp)
        fts = torch.Tensor(gvcnn_ft).to(device)

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
    
    edge_idx = []
    

    data = {
        'fts':fts,
        'lbls':lbls,
        'edge_idx': edge_idx,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data

def load_data(args):
    if args.dataset in ['40','NTU']:
        return load_ft(args)
    elif args.dataset in ['Cora','Citeseer','PubMed']:
        return load_cite(args)
    elif args.dataset in ['MINIST']:
        return load_minist(args)
    elif args.dataset in ['cora']:
        return load_citation_data()

def load_minist(args):
    device = torch.device(args.device)
    dataset = torchvision.datasets.MNIST(root='./data',transform=lambda x:list(x.getdata()),download=True)
    features = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]
    features = torch.Tensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    train_idx = [i for i in range(50000)]
    test_idx = [i for i in range(50000,60000)]

    data = {
        'fts':features,
        'lbls':labels,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data


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


def load_citation_data():
    """
    Copied from gcn
    citeseer/cora/pubmed with gcn split
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    cfg = {
        'citation_root':'./citation_data',
        'activate_dataset':'cora',
        'add_self_loop': True
    }


    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(cfg['citation_root'], cfg['activate_dataset'], names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(cfg['citation_root'], cfg['activate_dataset']))
    test_idx_range = np.sort(test_idx_reorder)

    if cfg['activate_dataset'] == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.todense()

    G = nx.from_dict_of_lists(graph)
    # print("=====> ", G)
    # edge_list = G.adjacency_list()
    adjacency = G.adjacency()
    edge_list = []
    for item in adjacency:
        # print(list(item[1].keys()))
        edge_list.append(list(item[1].keys()))

    degree = [0] * len(edge_list)
    if cfg['add_self_loop']:
        for i in range(len(edge_list)):
            edge_list[i].append(i)
            degree[i] = len(edge_list[i])
    max_deg = max(degree)
    mean_deg = sum(degree) / len(degree)
    print(f'max degree: {max_deg}, mean degree:{mean_deg}')

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]     # one-hot labels
    n_sample = labels.shape[0]
    n_category = labels.shape[1]
    lbls = np.zeros((n_sample,))
    if cfg['activate_dataset'] == 'citeseer':
        n_category += 1                                         # one-hot labels all zero: new category
        for i in range(n_sample):
            try:
                lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels
            except ValueError:                              # labels[i] all zeros
                lbls[i] = n_category + 1                        # new category
    else:
        for i in range(n_sample):
            lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    

    features = torch.Tensor(features)
    lbls = torch.LongTensor(lbls)

    data = {
        'fts':features,
        'lbls':lbls,
        'train_idx':idx_val,
        'test_idx':idx_test
    }

    return data

    # return features, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list