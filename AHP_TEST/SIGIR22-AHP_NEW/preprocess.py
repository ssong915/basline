import math
import numpy as np
import argparse
import sys
import pandas as pd
import torch
import time
import random
import utils
from sampler import *
import itertools
# data set 이름 설정하는 argument # 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help='data sources to use tags-ask-ubuntu, tags-math-sx',
                        choices=['tags-ask-ubuntu', 'tags-math-sx', 'threads-ask-ubuntu', 'threads-math-sx', 'email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school'],
                        default='email-Enron')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--gpu', type=int, default=1, help='idx for the gpu to use')
    parser.add_argument('--window',type=str, help='time window split method', choices=['fixed', 'time_freq'], default='fixed')
    try : 
        args = parser.parse_args()
    except :
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


# string 형태로 저장되어있는 node set -> int형으로 변환 #
def parse_string_to_list(string):
    # 문자열에서 숫자만 추출하여 리스트로 변환
    numbers = np.fromstring(string[1:-1], dtype=int, sep=', ')
    return numbers.tolist()


def reading_data(file_name):

    file_addr = './data/' + file_name + '/' + file_name

    fin_nverts = open(file_addr + '-nverts.txt', 'r')
    fin_simplices = open(file_addr + '-simplices.txt', 'r')
    fin_times = open(file_addr + '-times.txt', 'r')

    nverts = []
    simplices = []
    times = []
    node2idx = {}
    for i in fin_nverts:
        nverts.append(int(i))
    count = 1
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
    # print(y) ### 34946은 왜 맨 앞에 있는 거지...?

    # nvertsList = np.array(nverts)
    # print("max: ",max(nvertsList), "min: ", min(nvertsList))
    # print("Average Size: ", np.mean(nvertsList[nvertsList>1]), "std size: ", np.std(nvertsList[nvertsList>1]))
    # print("Total size", np.sum(nvertsList), "total hyperedges",len(nvertsList), "total nodes hyperedges > 1",np.sum(nvertsList[nvertsList>1]))

    simplices_i = 0
    edge_idx = 0
    node_list_total = []

    final_data = {}
    final_data['h_edge_idx']=[]
    final_data['node_set']=[]
    final_data['time']=[]
    final_feature = {}

    for _, nverts_i in enumerate(nverts):
        node_list_total.append(simplices[simplices_i: simplices_i + nverts_i])

        if nverts_i == 1: # there may be 1 simplex, which means doesn't have edge with other nodes, so remove them
            simplices_i += 1
            continue
        
        for i in simplices[simplices_i: simplices_i + nverts_i]:
            if not(i in node2idx):
                node2idx[i] = count
                count += 1
        
        simplices_i += nverts_i

    simplex_idx = -1
    edge_num = 0
    one_node_edge = 0
    for idx_y in y:
        node_list = node_list_total[idx_y]
        time_stamp = times[idx_y]
        if len(node_list) == 1:
            one_node_edge +=1
            continue
        simplex_idx += 1
        edge_num +=1
        final_data['h_edge_idx'].append(simplex_idx)
        final_data['node_set'].append(node_list)
        final_data['time'].append(time_stamp)

    fin_times.close()
    fin_simplices.close()
    fin_nverts.close()

    df = pd.DataFrame.from_dict(data= final_data, orient='columns')
    m_df = df[df['node_set'].apply(lambda x: len(x) != 1)]

    # CSV 파일로 저장
    m_df.to_csv('./data/' + file_name + '/' + file_name+'hyper_'+ file_name +'.csv', index=False)

    print("total nodes ", len(node2idx))

    node_random_feat = np.zeros((len(node2idx), 172))
    final_feature['node_feature'] = torch.from_numpy(node_random_feat).cpu()

    #------hyperedge feature를 쓰는 경우에 사용 ------#
    edge_random_feat = np.zeros((edge_num, 172))
    final_feature['edge_feature'] = torch.from_numpy(edge_random_feat).cpu()

    torch.save(final_feature, './data/' + file_name + '/feature_hyper-'+file_name+'.pt')
    print("----------pt file save " + file_name +" ----------")
    

def get_datainfo(r_dataset):
    #-------- DATA 불러와서 node_set, time(HIT 참고하여 timestamp 단위 변환), hyperedge_idx 저장 ---------# 
    r_dataset['node_set'] = r_dataset['node_set'].apply(parse_string_to_list)
    # r_dataset = r_dataset.sort_values(by="time")
    r_dataset.reset_index(inplace=True) 
    # r_dataset = reindex_node_index(r_dataset)
        
    return r_dataset


def reindex_node_index(all_data):

    org_node_index = []

    for idx, row in all_data.iterrows():
        node_set = row['node_set']
        # node_set = [int(node.strip()) for node in node_set_str.strip('[]').split(',')]
        new_node_set = []
        for n_idx in node_set:
            if n_idx not in org_node_index:
                org_node_index.append(n_idx)
            new_idx = org_node_index.index(n_idx)
            new_node_set.append(new_idx)
        all_data.at[idx, 'node_set'] = new_node_set
    
    return all_data

def split_in_snapshot(args,all_hyperedges,timestamps):
    snapshot_data = list()
    snapshot_time = list()
    
    if args.time_split_type == 'sec':
        freq_sec = args.freq_size 
        split_criterion = timestamps // freq_sec
        groups = np.unique(split_criterion)
        groups = np.sort(groups)
        merge_edge_data = []
        merge_time_data = []
        for t in groups:
            period_members = (split_criterion == t) # snapshot 내 time index들
            edge_data = list(all_hyperedges[period_members]) # snapshot 내 hyperedge들
            time_data  = list(timestamps[period_members]) # snapshot 내 timestamp들
            
            unique_set = set()
            unique_list = []

            for item in edge_data:
                tuple_item = tuple(item)
                if tuple_item not in unique_set:
                    unique_set.add(tuple_item)
                    unique_list.append(item)
            
            if len(edge_data) < 4 or len(unique_list) < 2: 
                # [ 조건 미충족 ] : snapshot 내 edge 수가 부족하거나 / 중복된 edge가 많다면
                # 다음 snapshot과 merge 
                merge_edge_data = merge_edge_data + edge_data
                merge_time_data = merge_time_data + time_data
                
                if len(merge_edge_data) >= 4:
                    # merge된 data가 이제 조건을 충족한다면 append
                    snapshot_data.append(merge_edge_data)
                    merge_time_data = list(map(int,merge_time_data))
                    snapshot_time.append(merge_time_data)
                    merge_time_data = []
                    merge_edge_data = []
            else :
                # [ 조건 충족 ]
                if len(merge_edge_data) != 0 :
                    # 이전 snapshot이 [ 조건 미충족 ] 이었다면 merge
                    edge_data = merge_edge_data + edge_data
                    time_data = merge_time_data + time_data
                snapshot_data.append(edge_data)
                time_data = list(map(int,time_data))
                snapshot_time.append(time_data)
                merge_edge_data = []
                merge_time_data = []
        lengths = []
        for hyperedge in snapshot_data:
            lengths.append(len(hyperedge))
            
        average_length = sum(lengths) / len(lengths)
        print("average hyperedge num : ", average_length)
                    
    elif args.time_split_type == 'num':
        snapshot_size = args.batch_size
        for i in range(0, len(all_hyperedges),snapshot_size):
            if i+snapshot_size < len(all_hyperedges):
                edge_data = all_hyperedges[i:i+snapshot_size] 
                time_data  = timestamps[i:i+snapshot_size] 
            else:
                edge_data = all_hyperedges[i:]   
                time_data  = timestamps[i:]
            snapshot_data.append(edge_data) 
            time_data = list(map(int,time_data))
            snapshot_time.append(time_data)
            i += snapshot_size
        print("hyperedge num : ",snapshot_size)
            
    print('snapshot 수 : ',len(snapshot_data))

    return snapshot_data, snapshot_time

    
def get_full_data(args, dataset):

    time = dataset['time']
    ts_start = time.min()
    ts_end = time.max() 
    filter_data = dataset[(dataset['time'] >= ts_start) & (dataset['time']<=ts_end)]

    max_node_idx = max(max(row) for row in list(filter_data['node_set']))
    num_node = max_node_idx + 1
    
    all_hyperedges = filter_data['node_set']
    timestamps = filter_data['time']
    
    return all_hyperedges, timestamps, num_node   
    
    

def neg_generator(mode, HE, pred_num):
    
    if mode == 'mns' :
        mns = MNSSampler(pred_num)
        t_mns = mns(set(tuple(x) for x in HE))
        t_mns = list(t_mns)
        neg_hedges = [list(edge) for edge in t_mns]        
        
    elif mode == 'sns':
        sns = SNSSampler(pred_num)
        t_sns = sns(set(tuple(x) for x in HE))
        t_sns = list(t_sns)
        neg_hedges = [list(edge) for edge in t_sns]    
        
    elif mode == 'cns'or mode == 'none':
        cns = CNSSampler(pred_num)
        t_cns = cns(set(tuple(x) for x in HE))    
        t_cns = list(t_cns)    
        neg_hedges = [list(edge) for edge in t_cns]
        
    
    return neg_hedges

def get_all_neg_samples(args):
    DATA = args.dataset_name 
    feat_folder = f'/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/{DATA}/'
    sns_neg_hedge = torch.load(feat_folder + f'sns_{args.dataset_name}.pt')
    cns_neg_hedge = torch.load(feat_folder + f'cns_{args.dataset_name}.pt')
    mns_neg_hedge = torch.load(feat_folder + f'mns_{args.dataset_name}.pt')
    
    # all_snapshot_edges = [element for row in snapshot_data for element in row]
    # sns_neg_hedge = neg_generator('sns', all_snapshot_edges, len(all_snapshot_edges))
    # cns_neg_hedge = neg_generator('cns', all_snapshot_edges, len(all_snapshot_edges))
    # mns_neg_hedge = neg_generator('mns', all_snapshot_edges, len(all_snapshot_edges))
    
    return sns_neg_hedge, cns_neg_hedge, mns_neg_hedge
    
    
def make_and_get_samples(next_snapshot, all_snapshot):    
    
    random.shuffle(next_snapshot)
    
    all_snapshot_edges = [element for row in all_snapshot for element in row]
    random.shuffle(all_snapshot_edges)

    pos_hedge = next_snapshot
    sns_neg_hedge = neg_generator('sns', pos_hedge, len(pos_hedge))
    cns_neg_hedge = neg_generator('cns', pos_hedge, len(pos_hedge))
    mns_neg_hedge = neg_generator('mns', all_snapshot_edges, len(pos_hedge))
    
    return pos_hedge, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge
 
def get_next_samples(next_snapshot,all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge):    
    
    pos_hedge = next_snapshot
    pos_size = len(pos_hedge)
    sns_neg_hedge = random.sample(all_sns_neg_hedge, pos_size)
    cns_neg_hedge = random.sample(all_cns_neg_hedge, pos_size)
    mns_neg_hedge = random.sample(all_mns_neg_hedge, pos_size)
    
    return pos_hedge, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge
 

def get_dataloaders(dataset, time_info, batch_size, device):
    
    time_hyperedge = list(dataset)
    snapshot_time = list(time_info)
    total_size = len(time_hyperedge)
    idcs = np.arange(len(time_hyperedge)).tolist()
    
    test_size = int(math.ceil(total_size * 0.1))
    valid_size = int(math.ceil(total_size * 0.2))

    test_index = random.sample(idcs, test_size)
    test_hyperedge = [time_hyperedge[i] for i in test_index]
    test_time = [snapshot_time[i] for i in test_index]
    
    valid_index = random.sample(set(idcs)-set(test_index), valid_size)
    valid_hyperedge = [time_hyperedge[i] for i in valid_index]
    valid_time = [snapshot_time[i] for i in valid_index]
    
    train_index = list(set(idcs)-set(valid_index)-set(test_index))
    train_hyperedge = [time_hyperedge[i] for i in train_index]
    train_time = [snapshot_time[i] for i in train_index]
    
    
    train_dataloader = BatchDataloader(train_hyperedge, train_time, batch_size, device, is_Train=True)
    valid_dataloader = BatchDataloader(valid_hyperedge, valid_time, batch_size, device, is_Train=False)
    test_dataloader = BatchDataloader(test_hyperedge, test_time, batch_size, device, is_Train=False)
    
    return train_dataloader, valid_dataloader, test_dataloader
    
    
class BatchDataloader(object):
    def __init__(self, hyperedges, timestamps, batch_size, device, is_Train=False):
        self.hyperedges = hyperedges
        self.timestamps = timestamps
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
        self.timestamps = [self.timestamps[i] for i in idcs]
    
    def __iter__(self):
        self.idx = 0
        return self

    def next(self):
        return self._next_batch()

    def _next_batch(self):
        nidx = self.idx + self.batch_size # next cursor position

        if nidx >= len(self.hyperedges): # end of each epoch
            next_hyperedges = self.hyperedges[self.idx:]
            next_timestamps = self.timestamps[self.idx:]
            self.idx = 0
            self.time_end = True
            if self.is_Train:
                self.shuffle() # data shuffling at every epoch

        else:
            next_hyperedges = self.hyperedges[self.idx:self.idx + self.batch_size]
            next_timestamps = self.timestamps[self.idx:self.idx + self.batch_size]
            self.idx = nidx % len(self.hyperedges)
        

        objects = next_hyperedges[:]
        timestamps = next_timestamps[:]

        return objects, timestamps, self.time_end