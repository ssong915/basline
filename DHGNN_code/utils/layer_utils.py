import torch
from torch import nn
import pandas as pd

def nan_to_num(tensor, nan=0.0, posinf=None, neginf=None):
    """
    Replace NaN/Inf with specified numerical values.
    """
    if posinf is None:
        posinf = tensor.max()
    if neginf is None:
        neginf = tensor.min()
    return torch.where(torch.isnan(tensor), torch.tensor(nan, device=tensor.device), 
                       torch.where(tensor == float('inf'), torch.tensor(posinf, device=tensor.device),
                       torch.where(tensor == float('-inf'), torch.tensor(neginf, device=tensor.device), tensor)))

def cos_dis(X):
        """
        cosine distance
        :param X: (N, d)
        :return: (N, N)
        """
#-----------수정사항 (우리 data에는 feature가 0으로 되어있기 때문에 normalization 수정) -----------#
        #X = nn.functional.normalize(X, eps=1e-10)
        #XT = torch.nan_to_num(X.transpose(0, 1), nan=1e-10, posinf=1e-10, neginf=1e-10)

        X = nn.functional.normalize(X)
        XT = nan_to_num(X.transpose(0, 1), nan=1e-10, posinf=1e-10, neginf=1e-10)
        return torch.matmul(X, XT)


def sample_ids(ids, k):
    """
    sample `k` indexes from ids, must sample the centroid node itself
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    #print("idx sampled from:", df)
    #print("k checking :", k)
    sampled_ids = df.sample(k - 1, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    sampled_ids.append(ids[-1])  # must sample the centroid node itself
    return sampled_ids


def sample_ids_v2(ids, k):
    """
    purely sample `k` indexes from ids
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sample_indices = df.sample(k, replace=True).values
    sampled_ids = sample_indices.flatten().tolist()
    return sampled_ids

def reindex_snapshot(snapshot_edges):
    org_node_index = []
    reindex_snapshot_edges = [[0 for _ in row] for row in snapshot_edges]
    for i, edge in enumerate(snapshot_edges):
        for j, node in enumerate(edge):
            if node not in org_node_index:
                org_node_index.append(node)
            new_idx = org_node_index.index(node)
            reindex_snapshot_edges[i][j] = new_idx
    
    return reindex_snapshot_edges, org_node_index