import torch 
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torchmetrics import AveragePrecision
import os
from tqdm import tqdm

import utils
from data_load_re import gen_data, gen_DGLGraph, load_train, load_val, load_test
import models
from sampler import *
from aggregator import *
from generator import MLPgenerator
from training import model_train, model_eval
import random
import pandas as pd
import ast
import data_load_5, preprocess
from batch import HEBatchGenerator

def live_update(args):
    os.makedirs(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}", exist_ok=True)
    f_log = open(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/logs/{args.dataset_name}/{args.folder_name}_train.log", "w")
    f_log.write(f"args: {args}\n")

    train_DG = args.train_DG.split(":")
    # args.train_DG = [int(train_DG[0]), int(train_DG[1]), int(train_DG[0])+int(train_DG[1])]  
    args.train_DG = [1, 1, 2]    
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    
    if args.fix_seed:
        np.random.seed(516)
        torch.manual_seed(516)
        torch.cuda.manual_seed_all(516)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(516)
    
    for j in tqdm(range(args.exp_num)):  
        DATA = args.dataset_name
        if args.eval_mode == 'live_update':
            snapshot_data, snapshot_time, num_node = data_load_5.load_snapshot(args, DATA)
        elif args.eval_mode == 'fixed_split':
            full_data, full_time, num_node = data_load_5.load_fulldata(args, DATA)
        
        snapshot_data, snapshot_time, num_node = data_load_5.load_snapshot(args, DATA)
        full_data, full_time, num_node = data_load_5.load_fulldata(args, DATA)    
         
        args = data_load_5.gen_init_data(args, num_node)
        
        model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                        args.n_layers, memory_dim=args.nv)
        model.to(device)        
        Aggregator = None    
        cls_layers = [args.dim_vertex, 128, 8, 1]
        Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
        Aggregator.to(device)  
        size_dist = utils.gen_size_dist(full_data)
        if args.gen == "MLP":
            dim = [64, 256, 256, args.nv]
            if args.dataset_name == "pubmed":
                dim = [128, 512, 512, args.nv]
            elif "dblp" in args.dataset_name:
                dim = [256, 1024, 2048, args.nv]
            print(f"{args.dataset_name} generator dimension: "+str(dim))
            Generator =  MLPgenerator(dim, args.nv, device, size_dist)
        Generator.to(device)
        average_precision = AveragePrecision(task='binary')

        best_roc = 0
        best_epoch = 0 
        optim_D = torch.optim.RMSprop(list(model.parameters())+list(Aggregator.parameters()), lr=args.D_lr)
        optim_G = torch.optim.RMSprop(Generator.parameters(), lr=args.G_lr)
        total_time = len(snapshot_data) 
        
        all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = preprocess.get_all_neg_samples(args)        
        average_avg_roc, average_avg_ap = [], []
        
        for t in tqdm(range(len(snapshot_data)-1)):                       
            # Load data
            args = gen_data(args, args.dataset_name)
            dataset = args.dataset_name

            # load t snapshot (for training)    
            train_dataloader, valid_dataloader, test_dataloader = preprocess.get_dataloaders(snapshot_data[t], snapshot_time[t], args.bs, device) 
            
            # load t+1 snapshot (for link prediction)
            # next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.make_and_get_samples(snapshot_data[t+1], snapshot_data[t+1:])
            next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(snapshot_data[t+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)
            no_improvement_count = 0   
            for epoch in tqdm(range(args.epochs), leave=False):            
                train_pred, train_label = [], []
                d_loss_sum, g_loss_sum, count  = 0.0, 0.0, 0
                    
                # [Train]      
                while True:            
                    train_snapshot_edges, train_snapshot_times, time_end = train_dataloader.next()      

                    reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(train_snapshot_edges)
                    g = data_load_5.gen_DGLGraph(reindexed_snapshot_edges).to(device)
                    next_pos_labels =  torch.ones(len(next_pos_edges)).to(device)
                    d_loss, g_loss, train_pred, train_label = model_train(args, g, model, Aggregator, Generator, optim_D, optim_G, next_pos_edges, next_pos_labels, train_pred, train_label, device, epoch, snapshot_node_index)
                    
                    count += 1
                    d_loss_sum = d_loss_sum + d_loss
                    g_loss_sum = g_loss_sum + g_loss
                    
                    if time_end:
                        break       
                
                train_pred = torch.stack(train_pred)
                train_pred = train_pred.squeeze()
                train_label = torch.round(torch.cat(train_label, dim=0)) 
                train_label = train_label.type(torch.int64)       
                train_roc = metrics.roc_auc_score(np.array(train_label.cpu()), np.array(train_pred.cpu()))
                train_ap = average_precision(torch.tensor(train_pred), torch.tensor(train_label))            
                
                f_log.write(f'{epoch} epoch: Training d_loss : {d_loss_sum / count} / Training g_loss : {g_loss_sum / count} /')
                f_log.write(f'Training roc : {train_roc} / Training ap : {train_ap} \n')
        
                # Eval validation
                reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(valid_dataloader.hyperedges)
                g = data_load_5.gen_DGLGraph(reindexed_snapshot_edges).to(device)                
                
                val_label = torch.ones(len(next_pos_edges)).to(device)
                val_batchloader_pos = HEBatchGenerator(next_pos_edges, val_label, args.bs, device, test_generator=True)            
                val_pred_pos, total_label_pos = model_eval(args, val_batchloader_pos, g, model, Aggregator,snapshot_node_index)
                
                val_label = torch.zeros(len(sns_neg_hedge)).to(device)
                val_batchloader_sns = HEBatchGenerator(sns_neg_hedge, val_label, args.bs, device, test_generator=True)   
                val_pred_sns, total_label_sns = model_eval(args, val_batchloader_sns, g, model, Aggregator,snapshot_node_index)
                auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, val_pred_pos+val_pred_sns)
                f_log.write(f"{epoch} epoch, SNS : Val AP : {ap_sns} / AUROC : {auc_roc_sns}\n")
                
                val_label = torch.zeros(len(mns_neg_hedge)).to(device)
                val_batchloader_mns = HEBatchGenerator(mns_neg_hedge, val_label, args.bs, device, test_generator=True)   
                val_pred_mns, total_label_mns = model_eval(args, val_batchloader_mns, g, model, Aggregator, snapshot_node_index)
                auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, val_pred_pos+val_pred_mns)
                f_log.write(f"{epoch} epoch, MNS : Val AP : {ap_mns} / AUROC : {auc_roc_mns}\n")
                
                val_label = torch.zeros(len(cns_neg_hedge)).to(device)
                val_batchloader_cns = HEBatchGenerator(cns_neg_hedge, val_label, args.bs, device, test_generator=True)  
                val_pred_cns, total_label_cns = model_eval(args, val_batchloader_cns, g, model, Aggregator, snapshot_node_index)
                auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, val_pred_pos+val_pred_cns)
                f_log.write(f"{epoch} epoch, CNS : Val AP : {ap_cns} / AUROC : {auc_roc_cns}\n")
                            
                l = len(val_pred_pos)//3
                val_pred_all = val_pred_pos + val_pred_sns[0:l] + val_pred_mns[0:l] + val_pred_cns[0:l]
                total_label_all = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
                auc_roc_all, ap_all = utils.measure(total_label_all, val_pred_all)
                f_log.write(f"{epoch} epoch, ALL : Val AP : {ap_all} / AUROC : {auc_roc_all}\n")
                f_log.flush()
                # Save best checkpoint
                if best_roc < (auc_roc_sns+auc_roc_mns+auc_roc_cns)/3:
                    best_roc = (auc_roc_sns+auc_roc_mns+auc_roc_cns)/3
                    best_epoch=epoch
                    torch.save(model.state_dict(), f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}/model_{j}.pkt")
                    torch.save(Aggregator.state_dict(), f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}/Aggregator_{j}.pkt")
                    torch.save(Generator.state_dict(), f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}/Generator_{j}.pkt")            
                    no_improvement_count = 0          
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= 10: 
                        break                   
            with open(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}/best_epochs.logs", "a") as e_log:  
                e_log.write(f"exp {j} best epochs: {best_epoch}\n")
            
            roc_average, ap_average = test(args, test_dataloader, next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge,j)     
            average_avg_roc.append(roc_average)
            average_avg_ap.append(ap_average)  
        f_log.close()
        
        final_average_roc = sum(average_avg_roc)/len(average_avg_roc)
        final_average_ap = sum(average_avg_ap)/len(average_avg_ap)
    print('============================================ Test End =================================================')
    # print('[ BEST ]')
    # print('AUROC\t AP\t ')
    # print(f'{max(final_average_roc):.4f}\t{max(final_average_roc):.4f}')
    print('[ AVG ]')
    print('AUROC\t AP\t ')
    print(f'{final_average_roc:.4f}\t{final_average_ap:.4f}')
    print(f'===============================================================================================================')
    
    return args, final_average_roc, final_average_ap

def fixed_split(args):
    os.makedirs(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}", exist_ok=True)
    f_log = open(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/logs/{args.dataset_name}/{args.folder_name}_train.log", "w")
    f_log.write(f"args: {args}\n")

    train_DG = args.train_DG.split(":")
    # args.train_DG = [int(train_DG[0]), int(train_DG[1]), int(train_DG[0])+int(train_DG[1])]  
    args.train_DG = [1, 1, 2]    
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    
    if args.fix_seed:
        np.random.seed(516)
        torch.manual_seed(516)
        torch.cuda.manual_seed_all(516)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(516)
    
    for j in tqdm(range(args.exp_num)):  
        DATA = args.dataset_name
        
        full_data, full_time, num_node = data_load_5.load_fulldata(args, DATA)    
         
        args = data_load_5.gen_init_data(args, num_node)
        args = gen_data(args, args.dataset_name)
        
        model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                        args.n_layers, memory_dim=args.nv)
        model.to(device)        
        Aggregator = None    
        cls_layers = [args.dim_vertex, 128, 8, 1]
        Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
        Aggregator.to(device)  
        size_dist = utils.gen_size_dist(full_data)
        if args.gen == "MLP":
            dim = [64, 256, 256, args.nv]
            if args.dataset_name == "pubmed":
                dim = [128, 512, 512, args.nv]
            elif "dblp" in args.dataset_name:
                dim = [256, 1024, 2048, args.nv]
            print(f"{args.dataset_name} generator dimension: "+str(dim))
            Generator =  MLPgenerator(dim, args.nv, device, size_dist)
        Generator.to(device)
        average_precision = AveragePrecision(task='binary')

        best_roc = 0
        best_epoch = 0 
        optim_D = torch.optim.RMSprop(list(model.parameters())+list(Aggregator.parameters()), lr=args.D_lr)
        optim_G = torch.optim.RMSprop(Generator.parameters(), lr=args.G_lr)
        
        all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = preprocess.get_all_neg_samples(args)  
        
        train_hyperedge, train_time, valid_hyperedge, valid_time, test_hyperedge, test_time, test_label = utils.split_edges(full_data, full_time)
        snapshot_data, snapshot_time = preprocess.split_in_snapshot(args, train_hyperedge, train_time)      
        average_avg_roc, average_avg_ap = [], []
        no_improvement_count = 0    
                                
        for epoch in tqdm(range(args.epochs), leave=False):      
            train_pred, train_label = [], []      
            d_loss_sum, g_loss_sum, count  = 0.0, 0.0, 0
            
            for t in tqdm(range(len(snapshot_data)-1)):           
                # load t snapshot (for training)    
                train_dataloader = preprocess.BatchDataloader(snapshot_data[t], snapshot_time[t], args.bs, device, is_Train=True)     
            
                # load t+1 snapshot (for link prediction)
                # next_snapshot_pos_edges, next_snapshot_neg_edges = preprocess.load_hedge_pos_neg(snapshot_data[t+1], args.neg_mode)
                next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(snapshot_data[t+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)
            
                # [Train]      
                while True:            
                    train_snapshot_edges, _, time_end = train_dataloader.next()      

                    reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(train_snapshot_edges)
                    g = data_load_5.gen_DGLGraph(reindexed_snapshot_edges).to(device)
                    next_pos_labels =  torch.ones(len(next_pos_edges)).to(device)
                    d_loss, g_loss, train_pred, train_label = model_train(args, g, model, Aggregator, Generator, optim_D, optim_G, next_pos_edges, next_pos_labels, train_pred, train_label, device, epoch, snapshot_node_index)
                    
                    count += 1
                    d_loss_sum = d_loss_sum + d_loss
                    g_loss_sum = g_loss_sum + g_loss
                    
                    if time_end:
                        break       
                
            #     train_pred = torch.stack(train_pred)
            #     train_pred = train_pred.squeeze()
            #     train_label = torch.round(torch.cat(train_label, dim=0)) 
            #     train_label = train_label.type(torch.int64)                     
            #     train_pred = train_pred.tolist()
            #     train_label = train_label.tolist()          
            # train_roc = metrics.roc_auc_score(np.array(train_label.cpu()), np.array(train_pred.cpu()))
            # train_ap = average_precision(torch.tensor(train_pred), torch.tensor(train_label))            
                
            # f_log.write(f'{epoch} epoch: Training d_loss : {d_loss_sum / count} / Training g_loss : {g_loss_sum / count} /')
            # f_log.write(f'Training roc : {train_roc} / Training ap : {train_ap} \n')
            
            next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(snapshot_data[t+1], all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)
                    
            valid_dataloader = preprocess.BatchDataloader(valid_hyperedge, valid_time, args.bs, device, is_Train=False)    
        
            # Eval validation
            reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(valid_dataloader.hyperedges)
            g = data_load_5.gen_DGLGraph(reindexed_snapshot_edges).to(device)                
            
            val_label = torch.ones(len(next_pos_edges)).to(device)
            val_batchloader_pos = HEBatchGenerator(next_pos_edges, val_label, args.bs, device, test_generator=True)            
            val_pred_pos, total_label_pos = model_eval(args, val_batchloader_pos, g, model, Aggregator,snapshot_node_index)
            
            val_label = torch.zeros(len(sns_neg_hedge)).to(device)
            val_batchloader_sns = HEBatchGenerator(sns_neg_hedge, val_label, args.bs, device, test_generator=True)   
            val_pred_sns, total_label_sns = model_eval(args, val_batchloader_sns, g, model, Aggregator,snapshot_node_index)
            auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, val_pred_pos+val_pred_sns)
            f_log.write(f"{epoch} epoch, SNS : Val AP : {ap_sns} / AUROC : {auc_roc_sns}\n")
            
            val_label = torch.zeros(len(mns_neg_hedge)).to(device)
            val_batchloader_mns = HEBatchGenerator(mns_neg_hedge, val_label, args.bs, device, test_generator=True)   
            val_pred_mns, total_label_mns = model_eval(args, val_batchloader_mns, g, model, Aggregator, snapshot_node_index)
            auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, val_pred_pos+val_pred_mns)
            f_log.write(f"{epoch} epoch, MNS : Val AP : {ap_mns} / AUROC : {auc_roc_mns}\n")
            
            val_label = torch.zeros(len(cns_neg_hedge)).to(device)
            val_batchloader_cns = HEBatchGenerator(cns_neg_hedge, val_label, args.bs, device, test_generator=True)  
            val_pred_cns, total_label_cns = model_eval(args, val_batchloader_cns, g, model, Aggregator, snapshot_node_index)
            auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, val_pred_pos+val_pred_cns)
            f_log.write(f"{epoch} epoch, CNS : Val AP : {ap_cns} / AUROC : {auc_roc_cns}\n")
                        
            l = len(val_pred_pos)//3
            val_pred_all = val_pred_pos + val_pred_sns[0:l] + val_pred_mns[0:l] + val_pred_cns[0:l]
            total_label_all = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
            auc_roc_all, ap_all = utils.measure(total_label_all, val_pred_all)
            f_log.write(f"{epoch} epoch, ALL : Val AP : {ap_all} / AUROC : {auc_roc_all}\n")
            f_log.flush()
            # Save best checkpoint
            if best_roc < (auc_roc_sns+auc_roc_mns+auc_roc_cns)/3:
                best_roc = (auc_roc_sns+auc_roc_mns+auc_roc_cns)/3
                best_epoch=epoch
                no_improvement_count = 0          
                torch.save(model.state_dict(), f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}/model_{j}.pkt")
                torch.save(Aggregator.state_dict(), f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}/Aggregator_{j}.pkt")
                torch.save(Generator.state_dict(), f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}/Generator_{j}.pkt")            
            else:
                no_improvement_count += 1
                if no_improvement_count >= 10: 
                    break                   
            with open(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}/best_epochs.logs", "a") as e_log:  
                e_log.write(f"exp {j} best epochs: {best_epoch}\n")
            
        # [Test]  
        next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(test_label, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge)
            
        test_dataloader = preprocess.BatchDataloader(test_hyperedge, test_time, args.bs, device, is_Train=False)   
        roc_average, ap_average = test(args, test_dataloader, next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge,j)     
        average_avg_roc.append(roc_average)
        average_avg_ap.append(ap_average)  
        f_log.close()
        
        final_average_roc = sum(average_avg_roc)/len(average_avg_roc)
        final_average_ap = sum(average_avg_ap)/len(average_avg_ap)
    print('============================================ Test End =================================================')
    # print('[ BEST ]')
    # print('AUROC\t AP\t ')
    # print(f'{max(final_average_roc):.4f}\t{max(final_average_roc):.4f}')
    print('[ AVG ]')
    print('AUROC\t AP\t ')
    print(f'{final_average_roc:.4f}\t{final_average_ap:.4f}')
    print(f'===============================================================================================================')
    
    return args, final_average_roc, final_average_ap

def test(args, test_dataloader, next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge,j):    
    args.checkpoint = f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}"
    f_log = open(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/logs/{args.dataset_name}/{args.folder_name}_results.log", "w")    
    f_log.write(f"{args}\n")  
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'   

    # Load data
    args = gen_data(args, args.dataset_name)
    dataset = args.dataset_name
    dataset = args.dataset_name

    # data_dict = torch.load(f'/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/splits/{dataset}split{j}.pt')
    
    # g = gen_DGLGraph(args, data_dict['HE'])  
    # test_batchloader_pos = load_test(data_dict['test_data'], args.bs, device, label="pos")
    # test_batchloader_sns = load_test(data_dict['test_data'], args.bs, device, label="sns")
    # test_batchloader_mns = load_test(data_dict['test_data'], args.bs, device, label="mns")
    # test_batchloader_cns = load_test(data_dict['test_data'], args.bs, device, label="cns")

    reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(test_dataloader.hyperedges)
    g = data_load_5.gen_DGLGraph(reindexed_snapshot_edges).to(device)                
   
    # Initialize models
    model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                    args.n_layers, memory_dim=args.nv)
    model.to(device)
    model.load_state_dict(torch.load(f"{args.checkpoint}/model_{j}.pkt"))
    cls_layers = [args.dim_vertex, 128, 8, 1]
    Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
    Aggregator.to(device)
    Aggregator.load_state_dict(torch.load(f"{args.checkpoint}/Aggregator_{j}.pkt"))
    
    model.eval()
    Aggregator.eval()

    with torch.no_grad():
        val_label = torch.ones(len(next_pos_edges)).to(device)
        test_batchloader_pos = HEBatchGenerator(next_pos_edges, val_label, args.bs, device, test_generator=True)  
        test_pred_pos, total_label_pos = model_eval(args, test_batchloader_pos, g, model, Aggregator, snapshot_node_index)
        

        val_label = torch.zeros(len(next_pos_edges)).to(device)
        test_batchloader_sns = HEBatchGenerator(sns_neg_hedge, val_label, args.bs, device, test_generator=True)  
        test_pred_sns, total_label_sns = model_eval(args, test_batchloader_sns, g, model, Aggregator, snapshot_node_index)
        auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, test_pred_pos+test_pred_sns)
        f_log.write(f"SNS : Test AP : {ap_sns} / AUROC : {auc_roc_sns}\n")
        
        val_label = torch.zeros(len(next_pos_edges)).to(device)
        test_batchloader_mns = HEBatchGenerator(mns_neg_hedge, val_label, args.bs, device, test_generator=True)  
        test_pred_mns, total_label_mns = model_eval(args, test_batchloader_mns, g, model, Aggregator, snapshot_node_index)
        auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, test_pred_pos+test_pred_mns)
        f_log.write(f"MNS : Test AP : {ap_mns} / AUROC : {auc_roc_mns}\n")        
        
        val_label = torch.zeros(len(next_pos_edges)).to(device)
        test_batchloader_cns = HEBatchGenerator(cns_neg_hedge, val_label, args.bs, device, test_generator=True)  
        test_pred_cns, total_label_cns = model_eval(args, test_batchloader_cns, g, model, Aggregator, snapshot_node_index)
        auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, test_pred_pos+test_pred_cns)
        f_log.write(f"CNS : Test AP : {ap_cns} / AUROC : {auc_roc_cns}\n")
        
        l = len(test_pred_pos)//3
        test_pred_all = test_pred_pos + test_pred_sns[0:l] + test_pred_mns[0:l] + test_pred_cns[0:l]
        total_label_all = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
        auc_roc_all, ap_all = utils.measure(total_label_all, test_pred_all)       
        
        f_log.write(f"ALL : Test AP : {ap_all} / AUROC : {auc_roc_all}\n")
        f_log.flush()
        
        return auc_roc_all, ap_all
    
def split_method(args):
    dataset = args.dataset_name
    
    for split in range(args.exp_num):      
        
        r_data = pd.read_csv('/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/{}/new_hyper_{}.csv'.format(dataset, dataset))
        HE = r_data['node_set'].apply(ast.literal_eval).tolist()
        
        # HE 데이터셋을 무작위로 섞음
        random.shuffle(HE)

        # 데이터셋의 길이
        total_length = len(HE)

        # 비율에 따라 데이터셋을 나눔
        ground_train_num = int(0.6 * total_length)
        ground_valid_num = int(0.2 * total_length)
        test_num = int(0.2 * total_length)
        
        # 데이터셋을 비율에 따라 잘라냄
        ground_train_data = HE[:ground_train_num]
        ground_valid_data = HE[ground_train_num:ground_train_num + ground_valid_num]
        test_data = HE[ground_train_num + ground_valid_num:]
        # TODO) test_data --> pt 파일로 저장
        
        torch.save({'HE':HE, 'ground_train_data': ground_train_data, 'ground_valid_data': ground_valid_data, \
                    'test_data': test_data},
                    f'/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/splits/{dataset}split{split}.pt')
    
if __name__ == "__main__":
    args = utils.parse_args()  
    args.folder_name = "exp1"
    os.makedirs(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/logs/{args.dataset_name}/{args.folder_name}", exist_ok=True)
    os.makedirs(f"/home/dake/workspace/AHP_TEST/SIGIR22-AHP_NEW/data/checkpoints/{args.dataset_name}/{args.folder_name}", exist_ok=True)
   
    # split_method(args)    
    if args.eval_mode == 'live_update': 
        live_update(args)
    elif args.eval_mode == 'fixed_split': 
        fixed_split(args)

    # args = utils.parse_args()
    # args.folder_name = "exp1"
    # for j in range(args.exp_num):
    #     test(args, j)

    