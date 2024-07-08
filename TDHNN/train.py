import torch
from our_data_load import load_our_data, load_snapshot_data
from our_utils import *
from Decoder import *
from networks import HGNN_classifier,GCN,GAT
import torch.nn.functional as F
import random
import numpy as np
import time
import datetime
from tqdm import tqdm
import copy

from our_utils import *
from Decoder import *

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def contrast_loss(H_raw,mask,labels):               #拥有相同lbl的结点应该属于相同的边
    lbl_num = labels.max().item()+1
    total_loss = 0
    for h in H_raw:
        for lbl in range(lbl_num):
            lbl_mask = labels == lbl
            src = h[mask][lbl_mask]
            target_idx = [i for i in range(src.shape[0])]
            random.shuffle(target_idx)
            target = src[target_idx]
            loss = F.mse_loss(src,target)
            total_loss = total_loss + loss
    return total_loss

def contrast_loss2(H, x):                         #相同边的结点特征应该相似
    total_loss = 0
    feartures = x
    for h in H:
        cc = h.ceil().abs()
        for i in range(cc.shape[1]):               #h 是 n*n 维
            col_mask = cc[:,i] == 1
            src = feartures[col_mask]             #属于同一条边的结点
            target_idx = [i for i in range(src.shape[0])]
            random.shuffle(target_idx)
            target = src[target_idx]                #随机调换idx
            loss = F.mse_loss(src,target) + 1e-8           #同一条边的结点特征应该相似
            if loss > 1e-8:
                total_loss = total_loss + loss
    return total_loss

def laplacian_rank(H_list, device):
    # L = I - Dv^(-1/2) W De^(-1) H^(T) Dv^(-1/2)
    rank_list = []
    for tmpH in H_list:
        H = tmpH.clone()

        ## 删除空边
        n_edge = H.shape[1]
        tmp_sum = H.sum(dim=0)
        index = []
        for i in range(n_edge):
            if tmp_sum[i] != 0:
                index.append(i)

        H = H[:, index]
        ################
        n_node = H.shape[0]
        n_edge = H.shape[1]

        # the weight of the hyperedge
        # W = np.ones(n_edge)
        W = torch.ones(n_edge).to(device)

        # the degree of the node
        # DV = np.sum(H * W, axis=1)
        DV = torch.sum(H * W, axis=1)

        # the degree of the hyperedge
        # DE = np.sum(H, axis=0)
        DE = torch.sum(H, axis=0)

        # invDE = np.mat(np.diag(np.power(DE, -1)))
        invDE = torch.diag(torch.pow(DE,-1))

        # DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        DV2 = torch.diag(torch.pow(DV, -0.5))

        # W = np.mat(np.diag(W))
        # H = np.mat(H)
        HT = H.T

        I = torch.eye(n_node, n_node).to(device)

        L = I - DV2 @ H @ W @ invDE @ HT @ DV2

        rank_L = torch.linalg.matrix_rank(L)

        rank_list.append(rank_L)

    print("===========================> Rank of L is: ", rank_list)


def train(model, optimizer, decoder, train_data, all_v_feat, train_unique_node, train_pos_edges, train_neg_edges, device, args):
    model.to(device)
    decoder.to(device)
    criterion = torch.nn.BCELoss()

    best_auroc = 0.0
    best_epoch = 0

    train_all_v_feat = all_v_feat.clone()

    for epoch in tqdm(range(args.epoch)):

        model.train()
        optimizer.zero_grad()

        args.stage = 'train'
        #------------------------ get updated node feature (x) -------------#
        updated_node_feat, H, H_raw = model(train_data, args)
        train_all_v_feat[train_unique_node] = updated_node_feat

        train_all_v_feat = train_all_v_feat.clone().detach()

        pos_preds = decoder(train_all_v_feat.to(device), train_pos_edges, 'Maxmin')
        pos_preds = pos_preds.squeeze()

        neg_preds = decoder(train_all_v_feat.to(device), train_neg_edges, 'Maxmin')
        neg_preds = neg_preds.squeeze()

        pos_labels = torch.ones_like(pos_preds)
        neg_labels = torch.zeros_like(neg_preds)

        preds = ( pos_preds.tolist() + neg_preds.tolist() )
        labels = ( pos_labels.tolist() + neg_labels.tolist() )

        real_loss = criterion(pos_preds, pos_labels)
        fake_loss = criterion(neg_preds, neg_labels)
    
        train_loss = real_loss + fake_loss 
        train_loss = train_loss/ 2
        
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_roc, train_ap = measure(labels, preds)

        '''if train_roc > best_auroc:
            patience = 0
            best_auroc = train_roc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_node_feat = train_all_v_feat
            if H is not None:
                # num_edges = hyperedge 수 
                args.num_edges = H[0].shape[1]
        else:
            patience = patience + 1

        if patience > args.patience:
            break'''
        
    best_model = copy.deepcopy(model.state_dict())
    best_node_feat = train_all_v_feat
    if H is not None:
        args.num_edges = H[0].shape[1]

    return best_model, best_node_feat

def evaluate(model, data, decoder, all_v_feat, test_unique_node,
              test_pos_edges, test_sns_edges, test_mns_edges, test_cns_edges, device, args):
    
    args.stage = 'test'
    stage = args.stage

    test_all_v_feat = all_v_feat.clone()
    model.eval()

    updated_node_feat, H, H_raw = model(data,args)
    test_all_v_feat[test_unique_node] = updated_node_feat

    test_all_v_feat = test_all_v_feat.clone().detach()

    test_pos_edges = test_pos_edges
    pos_preds = decoder(test_all_v_feat, test_pos_edges, 'Maxmin')
    pos_preds = pos_preds.squeeze()
    pos_labels = torch.ones_like(pos_preds)


    sns_neg_edges = test_sns_edges
    sns_neg_preds = decoder(test_all_v_feat, sns_neg_edges, 'Maxmin')
    sns_neg_preds = sns_neg_preds.squeeze()
    neg_labels = torch.zeros_like(sns_neg_preds)
    preds = ( pos_preds.tolist() + sns_neg_preds.tolist() )
    labels = ( pos_labels.tolist() + neg_labels.tolist() )

    sns_roc, sns_ap = measure(labels, preds) 

    mns_neg_edges = test_mns_edges
    mns_neg_preds = decoder(test_all_v_feat, mns_neg_edges, 'Maxmin')
    mns_neg_preds = mns_neg_preds.squeeze()
    neg_labels = torch.zeros_like(mns_neg_preds)
    preds = ( pos_preds.tolist() + mns_neg_preds.tolist() )
    labels = ( pos_labels.tolist() + neg_labels.tolist() )

    mns_roc, mns_ap = measure(labels, preds) 

    cns_neg_edges = test_cns_edges
    cns_neg_preds = decoder(test_all_v_feat, cns_neg_edges, 'Maxmin')
    cns_neg_preds = cns_neg_preds.squeeze()
    neg_labels = torch.zeros_like(cns_neg_preds)
    preds = ( pos_preds.tolist() + cns_neg_preds.tolist() )
    labels = ( pos_labels.tolist() + neg_labels.tolist() )

    cns_roc, cns_ap = measure(labels, preds) 

    d = len(test_pos_edges) // 3
    mixed_neg_hedge = test_sns_edges[0:d]+test_mns_edges[0:d]+test_cns_edges[0:d]
    mix_neg_preds = decoder(test_all_v_feat, mixed_neg_hedge, 'Maxmin')
    mix_neg_preds = mix_neg_preds.squeeze()
    neg_labels = torch.zeros_like(mix_neg_preds)
    preds = ( pos_preds.tolist() + mix_neg_preds.tolist() )
    labels = ( pos_labels.tolist() + neg_labels.tolist() )

    mix_roc, mix_ap = measure(labels, preds) 

    roc_average = (sns_roc+mns_roc+cns_roc+mix_roc)/4
    ap_average = (sns_ap+mns_ap+cns_ap+mix_ap)/4        

    print('*' * 20)
    print(f'Test AUROC: {roc_average} Test AP: {ap_average}')
    print('*' * 20)

    return roc_average, ap_average

def train_dhl(data, device, args):
    model = HGNN_classifier(args)
    cls_layers = [128, 128, 8, 1]
    decoder = Decoder(cls_layers)
    optimizer = torch.optim.Adam(list(model.parameters())+list(decoder.parameters()),lr=args.lrate,weight_decay=args.wdecay)

    snapshot_data, all_v_feat = load_our_data(args, device)
    all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = get_all_neg_samples(args)

    total_time = len(snapshot_data)
    auc_roc_list=[]
    ap_list=[]

    for t in tqdm(range(total_time-1)): 

        train_data, test_data, train_unique_node, test_unique_node, all_node_in_snap = load_snapshot_data(snapshot_data[t], all_v_feat, device)
        next_pos_hedge, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = get_next_samples(snapshot_data[t+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)
        
        if args.neg_mode == 'sns':
            next_neg_edges = sns_neg_hedge
        elif args.neg_mode == 'mns':
            next_neg_edges = mns_neg_hedge
        elif args.neg_mode == 'cns':
            next_neg_edges = cns_neg_hedge
        elif args.neg_mode == 'mix':
            d = len(next_pos_hedge) // 3
            next_neg_edges = sns_neg_hedge[0:d]+cns_neg_hedge[0:d]+mns_neg_hedge[0:d]
        
        total_size = len(next_pos_hedge)
        train_size = int(total_size * 0.8)
        train_idx = np.arange(0, train_size)
        test_idx = np.arange(train_size, total_size)

        train_pos_hedge = [next_pos_hedge[idx] for idx in train_idx]
        test_pos_hedge =[next_pos_hedge[idx] for idx in train_idx]

        train_neg_hedge = [next_neg_edges[idx] for idx in train_idx]

        test_sns_edges = [sns_neg_hedge[idx] for idx in test_idx]
        test_mns_edges =[mns_neg_hedge[idx] for idx in test_idx]
        test_cns_edges = [cns_neg_hedge[idx] for idx in test_idx]

        if t != 0 :
            model.load_state_dict(best_model)

        best_model, node_feat = train(model, optimizer, decoder, train_data, all_v_feat, train_unique_node, train_pos_hedge, train_neg_hedge, device, args)

        test_model = model.to(device)
        test_model.load_state_dict(best_model)
        time_auroc, time_ap = evaluate(test_model, test_data, decoder, all_v_feat, test_unique_node,
              test_pos_hedge, test_sns_edges, test_mns_edges, test_cns_edges, device, args)

        all_v_feat = node_feat

        auc_roc_list.append(time_auroc)
        ap_list.append(time_ap)

    final_roc = sum(auc_roc_list)/len(auc_roc_list)
    final_ap = sum(ap_list)/len(ap_list)

    print('[ FINAL ]')
    print('AUROC\t AP\t ')
    print(f'{final_roc:.4f}\t{final_ap:.4f}')
    
    return final_roc, final_ap

def train_gcn(data, device, args):
    in_dim = args.in_dim
    hid_dim = args.hid_dim 
    out_dim = args.out_dim
    model = GCN(args)
    cls_layers = [128, 128, 8, 1]
    decoder = Decoder(cls_layers)
    optimizer = torch.optim.Adam(list(model.parameters())+list(decoder.parameters()),lr=args.lrate,weight_decay=args.wdecay)

    snapshot_data, all_v_feat = load_our_data(args, device)
    all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = get_all_neg_samples(args)

    total_time = len(snapshot_data)
    auc_roc_list=[]
    ap_list=[]

    for t in tqdm(range(total_time-1)): 

        train_data, test_data, train_unique_node, test_unique_node, all_node_in_snap = load_snapshot_data(snapshot_data[t], all_v_feat, device)
        next_pos_hedge, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = get_next_samples(snapshot_data[t+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)
        
        if args.neg_mode == 'sns':
            next_neg_edges = sns_neg_hedge
        elif args.neg_mode == 'mns':
            next_neg_edges = mns_neg_hedge
        elif args.neg_mode == 'cns':
            next_neg_edges = cns_neg_hedge
        elif args.neg_mode == 'mix':
            d = len(next_pos_hedge) // 3
            next_neg_edges = sns_neg_hedge[0:d]+cns_neg_hedge[0:d]+mns_neg_hedge[0:d]
        
        total_size = len(next_pos_hedge)
        train_size = int(total_size * 0.8)
        train_idx = np.arange(0, train_size)
        test_idx = np.arange(train_size, total_size)

        train_pos_hedge = [next_pos_hedge[idx] for idx in train_idx]
        test_pos_hedge =[next_pos_hedge[idx] for idx in train_idx]

        train_neg_hedge = [next_neg_edges[idx] for idx in train_idx]

        test_sns_edges = [sns_neg_hedge[idx] for idx in test_idx]
        test_mns_edges =[mns_neg_hedge[idx] for idx in test_idx]
        test_cns_edges = [cns_neg_hedge[idx] for idx in test_idx]

        if t != 0 :
            model.load_state_dict(best_model)

        best_model, node_feat = train(model, optimizer, decoder, train_data, all_v_feat, train_unique_node, train_pos_hedge, train_neg_hedge, device, args)

        test_model = model.to(device)
        test_model.load_state_dict(best_model)
        time_auroc, time_ap = evaluate(test_model, test_data, decoder, all_v_feat, test_unique_node,
              test_pos_hedge, test_sns_edges, test_mns_edges, test_cns_edges, device, args)

        all_v_feat = node_feat

        auc_roc_list.append(time_auroc)
        ap_list.append(time_ap)

    final_roc = sum(auc_roc_list)/len(auc_roc_list)
    final_ap = sum(ap_list)/len(ap_list)

    print('[ FINAL ]')
    print('AUROC\t AP\t ')
    print(f'{final_roc:.4f}\t{final_ap:.4f}')
    
    return final_roc, final_ap

def train_gat(data, device, args):
    model = GAT(args)
    cls_layers = [128, 128, 8, 1]
    decoder = Decoder(cls_layers)
    optimizer = torch.optim.Adam(list(model.parameters())+list(decoder.parameters()),lr=args.lrate,weight_decay=args.wdecay)

    snapshot_data, all_v_feat = load_our_data(args, device)
    all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = get_all_neg_samples(args)

    total_time = len(snapshot_data)
    auc_roc_list=[]
    ap_list=[]

    for t in tqdm(range(total_time-1)): 

        train_data, test_data, train_unique_node, test_unique_node, all_node_in_snap = load_snapshot_data(snapshot_data[t], all_v_feat, device)
        next_pos_hedge, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = get_next_samples(snapshot_data[t+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)
        
        if args.neg_mode == 'sns':
            next_neg_edges = sns_neg_hedge
        elif args.neg_mode == 'mns':
            next_neg_edges = mns_neg_hedge
        elif args.neg_mode == 'cns':
            next_neg_edges = cns_neg_hedge
        elif args.neg_mode == 'mix':
            d = len(next_pos_hedge) // 3
            next_neg_edges = sns_neg_hedge[0:d]+cns_neg_hedge[0:d]+mns_neg_hedge[0:d]
        
        total_size = len(next_pos_hedge)
        train_size = int(total_size * 0.8)
        train_idx = np.arange(0, train_size)
        test_idx = np.arange(train_size, total_size)

        train_pos_hedge = [next_pos_hedge[idx] for idx in train_idx]
        test_pos_hedge =[next_pos_hedge[idx] for idx in train_idx]

        train_neg_hedge = [next_neg_edges[idx] for idx in train_idx]

        test_sns_edges = [sns_neg_hedge[idx] for idx in test_idx]
        test_mns_edges =[mns_neg_hedge[idx] for idx in test_idx]
        test_cns_edges = [cns_neg_hedge[idx] for idx in test_idx]

        if t != 0 :
            model.load_state_dict(best_model)

        best_model, node_feat = train(model, optimizer, decoder, train_data, all_v_feat, train_unique_node, train_pos_hedge, train_neg_hedge, device, args)

        test_model = model.to(device)
        test_model.load_state_dict(best_model)
        time_auroc, time_ap = evaluate(test_model, test_data, decoder, all_v_feat, test_unique_node,
              test_pos_hedge, test_sns_edges, test_mns_edges, test_cns_edges, device, args)

        all_v_feat = node_feat

        auc_roc_list.append(time_auroc)
        ap_list.append(time_ap)

    final_roc = sum(auc_roc_list)/len(auc_roc_list)
    final_ap = sum(ap_list)/len(ap_list)

    print('[ FINAL ]')
    print('AUROC\t AP\t ')
    print(f'{final_roc:.4f}\t{final_ap:.4f}')
    
    return final_roc, final_ap