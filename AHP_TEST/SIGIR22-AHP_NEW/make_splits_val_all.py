
'''
Processing data
'''
import torch
import numpy as np
from collections import defaultdict
import utils
import pickle
from sampler import *
import random
import pandas as pd
import ast
from tqdm import tqdm

def remove_duplicates(arr):
    # 이중 배열을 튜플로 변환하여 집합으로 중복 제거
    unique_set = {tuple(sub_arr) for sub_arr in arr}
    # 튜플을 다시 리스트로 변환하여 반환
    unique_list = [list(sub_arr) for sub_arr in unique_set]
    return unique_list

def split_dataset(dataset):
    data_path = None
    # try:
    #     data_path = f'/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/{dataset}.pt'             
    # except:
    #     raise Exception('dataset {} not supported!'.format(dataset))
    r_data = pd.read_csv('/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/{}/new_hyper_{}.csv'.format(dataset, dataset))
    r_HE = r_data['node_set'].apply(ast.literal_eval).tolist()
    HE = remove_duplicates(r_HE)
    random.shuffle(HE)
    
    # 데이터셋의 길이
    total_length = len(r_HE)
    
    input_HE = []
    for e in HE:
        input_HE.append(frozenset(e))
            
    train_sns, train_cns = neg_generator(input_HE, total_length) # 
        

    torch.save({'sns_sample': train_sns, 'cns_sample': train_cns, 'HE': HE},
            f'/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/splits/{dataset}_neg.pt')
      
def neg_generator(HE, pred_num):
    mns = MNSSampler(pred_num)
    sns = SNSSampler(pred_num)
    cns = CNSSampler(pred_num)
    
    # t_mns = mns(set(HE))
    t_sns = sns(set(HE))
    t_cns = cns(set(HE))
    
    # t_mns = list(t_mns)
    t_sns = list(t_sns)
    t_cns = list(t_cns)
    
    # t_mns = [list(edge) for edge in t_mns]
    t_sns = [list(edge) for edge in t_sns]
    t_cns = [list(edge) for edge in t_cns]
    
    return t_sns, t_cns
    

if __name__ == '__main__':
    args = utils.parse_args()
    args.dataset_name = 'email-Eu'
    split_dataset(args.dataset_name)