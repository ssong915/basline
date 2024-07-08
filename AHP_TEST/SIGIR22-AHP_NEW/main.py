import os
import numpy as np
import statistics
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn import metrics
import utils, data_load_5, train, preprocess, train_fixed_split
from models.decoder import *
from models.updater import *
from models.HNHN7 import *
from models.HGNN import *
from models.time_interval import *

warnings.simplefilter("ignore")
    
def run(args):
    #---------------- get args -------------------#
    print()
    print('============================================ New Data Start =================================================')
    DATA = args.dataset_name
    print(f'[ ARGS ]')
    print(args)
    print()
                
    os.makedirs(f"/data/checkpoints/{args.folder_name}", exist_ok=True)
    test_env = f'dim:{args.dim_vertex}_snapshot:{args.freq_size}_neg_mode:{args.use_contrastive}_{args.neg_mode}_'
    module_env = f"node_aware:{args.node_aware}_edge_aware:{args.edge_aware}"
    if args.node_aware == 'true':
        if args.edge_aware == 'true':
            module_env = f"node_aware:O  edge_aware:O"
            concat_env = f"time:{args.time_concat_mode} edge:{args.edge_concat_mode}"
        if args.edge_aware == 'false':
            module_env = f"node_aware:O  edge_aware:X"
            concat_env = f"time:{args.time_concat_mode}  edge: X"
    elif args.node_aware == 'false':
        if args.edge_aware == 'true':
            module_env = f"node_aware:X  edge_aware:O"
            concat_env = f"time:X  edge:{args.edge_concat_mode}"
        if args.edge_aware == 'false':
            module_env = f"node_aware:X  edge_aware:X"
            concat_env = f"time:X  edge: X"
            
    os.makedirs(f"logs/{DATA}/{args.eval_mode}/{test_env}/{module_env}/{concat_env}", exist_ok=True)    
    args.folder_name = f"logs/{DATA}/{args.eval_mode}/{test_env}/{module_env}/{concat_env}"
    
    if args.time_split_type == 'sec': 
        f_log = open(f"{args.folder_name}.log", "w")
        f_log.write(f"args: {args}\n")
        
        
    roc_list = []
    ap_list = []   
    mrr_list = []   

    for i in range(args.exp_num): # number of splits (default: 5)

        # change seed
        utils.set_random_seeds(0)
        
        if args.eval_mode == 'live_update':
            snapshot_data, snapshot_time, num_node = data_load_5.load_snapshot(args, DATA)
        elif args.eval_mode == 'fixed_split':
            full_data, full_time, num_node = data_load_5.load_fulldata(args, DATA)
                    
        args = data_load_5.gen_init_data(args, num_node)
        device = args.device

        print(f'============================================ Experiments {i} ==================================================')
        f_log.write(f'============================================ Experiments {i} ==================================================')
        # Initialize models
        # 0. Args needed
        time_layer = args.time_layers
        node_aware = args.node_aware
        time_concat_mode = args.time_concat_mode
        edge_aware = args.edge_aware
        edge_concat_mode = args.edge_concat_mode
        
        # 1. Hypergraph encoder
        if args.model == 'hgnn':
            HypergraphEncoder = HGNN(in_ch= args.input_dim, n_hid=args.dim_vertex, dropout=0.5)
        elif args.model == 'hnhn':   
            HypergraphEncoder = HNHN(time_layer, node_aware, edge_aware, time_concat_mode, edge_concat_mode, 
                                        args.dim_vertex, args.dim_hidden, args.dim_edge, args.dim_time)  
        encoder = HypergraphEncoder.to(device)
        
        # 2. Spatio temporalLayer
        updater = SpatioTemporalLayer(dim_in= args.dim_vertex, dim_out=args.dim_vertex)
        updater = updater.to(device)
        
        #3. Decoder (classifier) for hyperedge prediction
        cls_layers = [args.dim_vertex, 128, 8, 1]
        decoder = Decoder(cls_layers)
        decoder = decoder.to(device)

        optimizer = torch.optim.RMSprop(list(encoder.parameters())+ list(updater.parameters()) + list(decoder.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        if args.eval_mode == 'live_update':
            auc_roc, ap = train.live_update(args, encoder, updater, decoder, optimizer, scheduler, 
                                                snapshot_data, snapshot_time,  f_log, i)
        elif args.eval_mode == 'fixed_split':
            auc_roc, ap = train.fixed_split(args, encoder, updater, decoder, optimizer, scheduler, 
                                                full_data, full_time,  f_log, i)
        
        roc_list.append(auc_roc)
        ap_list.append(ap)
        
        print()  
        
    # exp 여러번 할 경우
    final_roc = sum(roc_list)/len(roc_list)
    final_ap = sum(ap_list)/len(ap_list)

    if args.exp_num > 1:
        std_roc = statistics.stdev(roc_list)
        std_ap = np.std(ap_list)
    else:
        std_roc = 0.0 
        std_ap = 0.0 

    print('============================================ Test End =================================================')
    print(f"[ SETTING ]")
    print(f"logs/{DATA}/{test_env}/\n")
    print(f"MODULE: {module_env}\n")
    print(f"CONCAT: {concat_env}\n")
    
    print('[ BEST ]')
    print('AUROC\t AP\t ')
    print(f'{max(roc_list):.4f}\t{max(ap_list):.4f}')
    print('[ AVG ]')
    print('AUROC\t AP\t ')
    print(f'{final_roc:.4f}\t{final_ap:.4f}')
    print(f'===============================================================================================================')
    
    
    f_log.write(f'============================================ Test End =================================================\n')
    f_log.write(f"[ SETTING ]\n")
    f_log.write(f"logs/{DATA}/{test_env}/\n")
    f_log.write(f"MODULE: {module_env}\n")
    f_log.write(f"CONCAT: {concat_env}\n")
    
    f_log.write('[ BEST ]\n')
    f_log.write('AUROC\t AP\t \n')
    f_log.write(f'{max(roc_list):.4f}\t{max(ap_list):.4f}\n')
    f_log.write('[ AVG ]\n')
    f_log.write('AUROC\t AP\t \n')
    f_log.write(f'{final_roc:.4f}\t{final_ap:.4f}\n')
    f_log.write(f'===============================================================================================================')
    
    f_log.close
    
    wandb.log({'final_roc':final_roc, 'final_ap':final_ap})
    
import wandb
if __name__ == '__main__':
    
    # Project and sweep configuration
    project_name = 'st-han_experiments'
    wandb.init(project = project_name)

    args = utils.parse_args()
    run(args)
    