import numpy as np
import argparse
import sys
import pandas as pd
import torch
import time
import random
import math

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
# data set 이름 설정하는 argument # 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='data sources to use tags-ask-ubuntu, tags-math-sx',
                        choices=['tags-ask-ubuntu', 'tags-math-sx', 'threads-ask-ubuntu', 'threads-math-sx', 'email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school'],
                        default='threads-ask-ubuntu')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument("--gpu", type=int, default=1, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument('--window',type=str, help='time window split method', choices=['fixed', 'time_freq'], default='fixed')
    parser.add_argument('--ns-algo', type=str, default='MNS', help='Negative Sampling algorithm. Available algorithms are UNS, SNS, MNS, CNS.',
                        choices= ['MNS','SNS','CNS','UNS'] )
    parser.add_argument('--ns-ratio', type=int, default=5, help='Ratio of negative hyperedges to positive hyperedges for sampling.')
    parser.add_argument("--freq_size", default=2628000, type=int, help='batch size')
    parser.add_argument('--model_version', default='DHGNN_v1', help='DHGNN model version, acceptable: DHGNN_v1, DHGNN_v2')
    parser.add_argument("--batch_size", default=32, type=int, help='batch size')
    parser.add_argument("--neg_mode", type=str, default='sns', help='negative sampling: mns, sns, cns')
    parser.add_argument("--epochs", default=50, type=int, help='number of epochs')

    args = parser.parse_args()
    
    return args

# string 형태로 저장되어있는 node set -> int형으로 변환 #
def parse_string_to_list(string):
    # 문자열에서 숫자만 추출하여 리스트로 변환
    numbers = np.fromstring(string[1:-1], dtype=int, sep=', ')
    return numbers.tolist()

def reading_data(file_name):

    file_addr = './our_data/' + file_name + '/' + file_name

    fin_nverts = open(file_addr + '-nverts.txt', 'r')
    fin_simplices = open(file_addr + '-simplices.txt', 'r')
    fin_times = open(file_addr + '-times.txt', 'r')

    nverts = []
    simplices = []
    times = []
    node2idx = {}
    # text 파일 읽어와서 list 형태로 저장
    # nverts = 각 hyperedge의 size 
    for i in fin_nverts:
        nverts.append(int(i))
    count = 1
    # simplices = hyperedge를 구성하고 있는 list
    for i in fin_simplices:
        simplices.append(int(i))

    last_time = -1
    idx = -1
    for i in fin_times:
        idx += 1
        if int(i) >= last_time:
            last_time = int(i)
        else:
            pass
        times.append(int(i))

    # HIT timestamp preprocessing 참고
    times = np.array(times)
    _time = 1
    while _time < max(times):
        _time = _time * 10
    times = times * 1.0 / (_time * 1e-7)
    times = np.array(times)

    y = np.argsort(times) # 오름차순 정리
    print(y) ### 34946은 왜 맨 앞에 있는 거지...?

    nvertsList = np.array(nverts)
    print("max: ",max(nvertsList), "min: ", min(nvertsList))
    print("Average Size: ", np.mean(nvertsList[nvertsList>1]), "std size: ", np.std(nvertsList[nvertsList>1]))
    print("Total size", np.sum(nvertsList), "total hyperedges",len(nvertsList), "total nodes hyperedges > 1",np.sum(nvertsList[nvertsList>1]))

    simplices_i = 0
    edge_idx = 0
    node_list_total = []
    except_one_total = []

    final_data = {}
    final_data['h_edge_idx']=[]
    final_data['node_set']=[]
    final_data['time']=[]
    final_feature = {}
    existing_unique_node = []

    for _, nverts_i in enumerate(nverts):

        node_list_total.append(simplices[simplices_i: simplices_i + nverts_i])

        if nverts_i == 1: # there may be 1 simplex, which means doesn't have edge with other nodes, so remove them
            simplices_i += 1
            continue

        except_one_total.append(simplices[simplices_i: simplices_i + nverts_i])

        for i in simplices[simplices_i: simplices_i + nverts_i]:
            if not(i in node2idx):
                node2idx[i] = count
                count += 1
        
        simplices_i += nverts_i

    existing_node = [item for sublist in except_one_total for item in sublist]
    existing_unique_node = np.unique(existing_node)
    node_dict = {value: index for index, value in enumerate(existing_unique_node)}

    simplex_idx = -1
    edge_num = 0
    one_node_edge = 0

    for idx_y in y:
        node_list = node_list_total[idx_y]
        reindexing_node_list = [node_dict[value] for value in node_list]
        time_stamp = times[idx_y]
        if len(node_list) == 1:
            one_node_edge +=1
            continue
        simplex_idx += 1
        edge_num +=1
        final_data['h_edge_idx'].append(simplex_idx)
        final_data['node_set'].append(reindexing_node_list)
        final_data['time'].append(time_stamp)

    fin_times.close()
    fin_simplices.close()
    fin_nverts.close()


    df = pd.DataFrame.from_dict(data= final_data, orient='columns')
    m_df = df[df['node_set'].apply(lambda x: len(x) != 1)]
    are_equal = df.equals(m_df)

    print("Are the DataFrames equal?", are_equal)

    # CSV 파일로 저장
    m_df.to_csv('./hypergraph/'+ file_name +'/hyper_'+ file_name +'.csv', index=False)

    print("total nodes ", len(existing_unique_node))

    node_random_feat = np.zeros((len(existing_unique_node), 172))
    final_feature['node_feature'] = torch.from_numpy(node_random_feat).cpu()

    #------hyperedge feature를 쓰는 경우에 사용 ------#
    
    #edge_random_feat = np.zeros((simplex_idx, 172))
    #final_feature['edge_feature'] = torch.from_numpy(edge_random_feat).cpu()

    torch.save(final_feature, './hypergraph/'+file_name+'/feature_hyper-'+file_name+'.pt')
    print("----------pt file save " + file_name +" ----------")

def get_datainfo(r_dataset):
    #-------- DATA 불러와서 node_set, time(HIT 참고하여 timestamp 단위 변환), hyperedge_idx 저장 ---------# 
    
    r_dataset['node_set'] = r_dataset['node_set'].apply(parse_string_to_list)
    r_dataset = r_dataset.sort_values(by="time")
    r_dataset.reset_index(inplace=True) 

    return r_dataset

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
        
def get_dataloaders(dataset, batch_size, device, mode):

    if mode == 'test':
        test_hyperedge = list(dataset)
        test_dataloader = BatchDataloader(test_hyperedge, batch_size, device, is_Train=False)
        return test_dataloader
    
    else:
        time_hyperedge = list(dataset)
        total_size = len(time_hyperedge)
        idcs = np.arange(len(total_size)).tolist()

        valid_index = random.sample(idcs, int(total_size * 0.3))
        valid_hyperedge = [time_hyperedge[i] for i in valid_index]

        train_index = list(set(idcs)-set(valid_index))
        train_hyperedge = [time_hyperedge[i] for i in train_index]

        train_dataloader = BatchDataloader(train_hyperedge, batch_size, device, is_Train=True)
        valid_dataloader = BatchDataloader(valid_hyperedge, batch_size, device, is_Train=False)

        return train_dataloader, valid_dataloader
    
def write_to_file(fname, data):
    with open(fname, 'w') as f:
        for hedge in data:
            f.write(', '.join(str(node) for node in hedge))
            f.write('\n')
    return

def get_dataloaders(dataset, batch_size, device):
    
    time_hyperedge = list(dataset)
    total_size = len(time_hyperedge)
    idcs = np.arange(len(time_hyperedge)).tolist()
    
    test_size = int(math.ceil(total_size * 0.1))
    valid_size = int(math.ceil(total_size * 0.2))

    test_index = random.sample(idcs, test_size)
    test_hyperedge = [time_hyperedge[i] for i in test_index]
    
    valid_index = random.sample(list(set(idcs)-set(test_index)), valid_size)
    valid_hyperedge = [time_hyperedge[i] for i in valid_index]
    
    train_index = list(set(idcs)-set(valid_index)-set(test_index))
    train_hyperedge = [time_hyperedge[i] for i in train_index]
    
    train_dataloader = BatchDataloader(train_hyperedge, batch_size, device, is_Train=True)
    valid_dataloader = BatchDataloader(valid_hyperedge, batch_size, device, is_Train=False)
    test_dataloader = BatchDataloader(test_hyperedge, batch_size, device, is_Train=False)
    
    return train_dataloader, valid_dataloader, test_dataloader

class BatchDataloader(object):
    def __init__(self, hyperedges, batch_size, device, is_Train=False):
        self.hyperedges = hyperedges
        self.batch_size = batch_size
        self.device = device
        self.is_Train = is_Train
        self.idx = 0
        self.time_end = False

        if is_Train:
            self.shuffle()

    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
       
    
    def __iter__(self):
        self.idx = 0
        return self

    def next(self):
        return self._next_batch()

    def _next_batch(self):
        nidx = self.idx + self.batch_size # next cursor position

        if nidx >= len(self.hyperedges): # end of each epoch
            next_hyperedges = self.hyperedges[self.idx:]
            self.idx = 0
            self.time_end = True
            if self.is_Train:
                self.shuffle() # data shuffling at every epoch

        else:
            next_hyperedges = self.hyperedges[self.idx:self.idx + self.batch_size]
            self.idx = nidx % len(self.hyperedges)

        objects = next_hyperedges[:]

        return objects, self.time_end

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


def load_snapshot(DATA):
     
    # 1. get feature and edge index        
    rf_data = torch.load('/home/dake/workspace/DHGNN/hypergraph/{}/feature_hyper-{}.pt'.format(DATA, DATA))
    r_data = pd.read_csv('/home/dake/workspace/DHGNN/hypergraph/{}/new_hyper_{}.csv'.format(DATA, DATA))
    
    # 2. make snapshot  
    data_info = get_datainfo(r_data)
    all_hyperedges, timestamps, num_node = get_full_data(data_info)
    snapshot_data = split_in_snapshot(all_hyperedges, timestamps)
       
    return snapshot_data, num_node

def get_all_neg_samples(args):
    DATA = args.data
    feat_folder = f'/home/dake/workspace/DHGNN/hypergraph/{DATA}/'
    sns_neg_hedge = torch.load(feat_folder + f'sns_{DATA}.pt')
    cns_neg_hedge = torch.load(feat_folder + f'cns_{DATA}.pt')
    mns_neg_hedge = torch.load(feat_folder + f'mns_{DATA}.pt')
    
    # all_snapshot_edges = [element for row in snapshot_data for element in row]
    # sns_neg_hedge = neg_generator('sns', all_snapshot_edges, len(all_snapshot_edges))
    # cns_neg_hedge = neg_generator('cns', all_snapshot_edges, len(all_snapshot_edges))
    # mns_neg_hedge = neg_generator('mns', all_snapshot_edges, len(all_snapshot_edges))
    
    return sns_neg_hedge, cns_neg_hedge, mns_neg_hedge

def get_next_samples(next_snapshot,all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge):    
    
    pos_hedge = next_snapshot
    pos_size = len(pos_hedge)
    sns_neg_hedge = random.sample(all_sns_neg_hedge, pos_size)
    cns_neg_hedge = random.sample(all_cns_neg_hedge, pos_size)
    mns_neg_hedge = random.sample(all_mns_neg_hedge, pos_size)
    
    return pos_hedge, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge
 
def measure(label, pred):
    auc_roc = roc_auc_score(np.array(label), np.array(pred))
    ap = average_precision_score(np.array(label), np.array(pred))
                    
    return round(auc_roc,3), round(ap,3)

def split_data(DATA):
    rf_data = torch.load('/home/dake/workspace/DHGNN/hypergraph/{}/feature_hyper-{}.pt'.format(DATA, DATA))
    r_data = pd.read_csv('/home/dake/workspace/DHGNN/hypergraph/{}/new_hyper_{}.csv'.format(DATA, DATA))
    
    # 2. make snapshot  
    data_info = get_datainfo(r_data)
    all_hyperedges, timestamps, num_node = get_full_data(data_info)
    
    total_size = len(all_hyperedges)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.2)
    test_size = total_size - (train_size + val_size)

    train_idx = np.arange(0, train_size)
    val_idx = np.arange(train_size, train_size + val_size)
    test_idx = np.arange(train_size + val_size, total_size)

    train_hedge_data = all_hyperedges[train_idx]
    train_time_data = timestamps[train_idx]
    
    val_hedge_data = all_hyperedges[val_idx]
    val_time_data = timestamps[val_idx]

    test_hedge_data = all_hyperedges[test_idx]
    test_time_data = timestamps[test_idx]

    train_dataset = split_in_snapshot(train_hedge_data, train_time_data)
    val_dataset = split_in_snapshot(val_hedge_data, val_time_data)
    test_dataset = split_in_snapshot(test_hedge_data, test_time_data)

    return train_dataset, val_dataset, test_dataset, all_hyperedges




