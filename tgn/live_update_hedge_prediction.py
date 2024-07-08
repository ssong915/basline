import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, get_snapshot_data, get_new_data
from itertools import groupby
#from utils.hyperdata_processing import get_data2

import random

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='email-Eu')
parser.add_argument('--bs', type=int, default=32, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tgn-attn', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs') 

parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', type=str, default="true",
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=50, help='Dimensions of the messages')
parser.add_argument('--random_seed', type=int, default=0, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
#parser.add_argument('--different_new_nodes', action='store_true',
#                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
#parser.add_argument('--randomize_features', action='store_true',
#                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument("--time_split_type", default='sec', type=str, help='snapshot split type')
parser.add_argument("--freq_size", default=2628000, type=int, help='batch size')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

n_degree = int(args.n_degree)
message_dim = int(args.message_dim)
random_seed = int(args.random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

MODEL_SAVE_PATH = f'./saved_models/random_seed={args.random_seed}/n_{args.n_degree},{args.message_dim}_{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'
  
### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

TOTAL_DATA, node_features, edge_features, n_nodes = get_snapshot_data(args, DATA)

time_length = len(TOTAL_DATA)
auc_roc_list = []
ap_list = []
#nn_auc_roc_list = []
#nn_ap_list = []

for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    full_data, train_data, val_data, test_data = get_new_data(TOTAL_DATA[0], future=1)
    # Initialize training neighbor finder to retrieve temporal grap        
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform, n_nodes)

    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                    edge_features=edge_features, device=device,
                    n_layers=NUM_LAYER,
                    n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                    message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                    memory_update_at_start=not args.memory_update_at_end,
                    embedding_module_type=args.embedding_module,
                    message_function=args.message_function,
                    aggregator_type=args.aggregator,
                    memory_updater_type=args.memory_updater,
                    n_neighbors=NUM_NEIGHBORS,
                    mean_time_shift_src=0, std_time_shift_src=1,
                    mean_time_shift_dst=0, std_time_shift_dst=1,
                    use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                    use_source_embedding_in_message=args.use_source_embedding_in_message,
                    dyrep=args.dyrep)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    for curr_time in tqdm(range(len(TOTAL_DATA))):

        if curr_time > 0:
            tgn.load_state_dict(torch.load(best_model_path))
            tgn.train()

        logger.info('start {} time'.format(curr_time))

    
        if (curr_time+1) > (time_length-1) :
            break
        '''      
        n_full_data, n_train_data, n_val_data, n_test_data, n_new_node_val_data, \
        n_new_node_test_data = get_data(TOTAL_DATA[curr_time], future=0,
                                      different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)
        
        ### Extract data for training, validation and testing
        full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data = get_data(TOTAL_DATA[curr_time+1], future=1,
                                    different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)
        '''

    
        n_full_data, n_train_data, n_val_data, n_test_data = get_new_data(TOTAL_DATA[:curr_time+1], future=0)
        ### Extract data for training, validation and testing
        full_data, train_data, val_data, test_data = get_new_data(TOTAL_DATA[curr_time+1], future=1)
        # Initialize training neighbor finder to retrieve temporal grap        
        train_ngh_finder = get_neighbor_finder(n_full_data, args.uniform, n_nodes)
        # Initialize validation and test neighbor finder to retrieve temporal graph

        full_ngh_finder = get_neighbor_finder(n_full_data, args.uniform, n_nodes)

        # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
        # across different runs
        # NB: in the inductive setting, negatives are sampled only amongst other new nodes
        train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
        val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
        #nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
        test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
        #nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations,seed=3)

        # Compute time statistics
        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

        tgn.mean_time_shift_src = mean_time_shift_src
        tgn.std_time_shift_src = std_time_shift_src
        tgn.mean_time_shift_dst = mean_time_shift_dst
        tgn.std_time_shift_dst = std_time_shift_dst

        tgn.neighbor_finder = train_ngh_finder
        tgn.set_neighbor_finder(train_ngh_finder)

        train_hyperedges = np.unique(train_data.timestamps).shape[0]
        group_lengths = [len(list(group)) for _, group in groupby(train_data.timestamps)]

        num_instance = train_hyperedges
        BATCH_SIZE = args.bs
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        new_nodes_val_aps = []
        val_aps = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []

        early_stopper = EarlyStopMonitor(max_round=args.patience)

        batch_sources = []
        batch_destinations = []
        batch_edge_idx = []
        batch_timestamps = []

        start = 0
        for length in group_lengths:
            end = start + length
            batch_sources.append(train_data.sources[start:end])
            batch_destinations.append(train_data.destinations[start:end])
            batch_edge_idx.append(train_data.edge_idxs[start:end])
            batch_timestamps.append(train_data.timestamps[start:end])
            start = end

        for epoch in range(NUM_EPOCH):
            start_epoch = time.time()
            ### Training

            # Reinitialize memory of the model at the start of each epoch
            if USE_MEMORY:
                tgn.memory.__init_memory__()

            # Train using only training graph
            tgn.set_neighbor_finder(train_ngh_finder)
            m_loss = []

            #logger.info('start {} epoch'.format(epoch))

            # snapshot 별로 돌기
            for k in range(0, num_batch, args.backprop_every):
                loss = 0
                optimizer.zero_grad()

                # Custom loop to allow to perform backpropagation only every a certain number of batches
                start_idx = k * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)


                sources_batch, destinations_batch = batch_sources[start_idx:end_idx], \
                                                    batch_destinations[start_idx:end_idx]
                edge_idxs_batch = batch_edge_idx[start_idx: end_idx]
                timestamps_batch = batch_timestamps[start_idx:end_idx]

                sources_batch = np.concatenate(sources_batch).tolist()
                destinations_batch = np.concatenate(destinations_batch).tolist()
                edge_idxs_batch = np.concatenate(edge_idxs_batch).tolist()
                timestamps_batch = np.concatenate(timestamps_batch).tolist()


                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)
            
                num_hyperedges = np.unique(timestamps_batch).shape[0]

                with torch.no_grad():
                    pos_label = torch.ones(num_hyperedges, dtype=torch.float, device=device)
                    neg_label = torch.zeros(num_hyperedges, dtype=torch.float, device=device)

                tgn = tgn.train()
                pos_prob, neg_prob = tgn.compute_hyperedge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                                    timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
                pos_prob = pos_prob.squeeze(1)
                neg_prob = neg_prob.squeeze(1)
                loss += criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)

                loss /= args.backprop_every

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if USE_MEMORY:
                    tgn.memory.detach_memory()

            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)

            ### Validation
            # Validation uses the full graph
            tgn.set_neighbor_finder(full_ngh_finder)

            if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
                train_memory_backup = tgn.memory.backup_memory()

            val_ap, val_auc = eval_edge_prediction(args,model=tgn, 
                                                negative_edge_sampler=val_rand_sampler,
                                                data=val_data,
                                                n_neighbors=NUM_NEIGHBORS)
            if USE_MEMORY:
                val_memory_backup = tgn.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
                tgn.memory.restore_memory(train_memory_backup)
        

            if USE_MEMORY:
            # Restore memory we had at the end of validation
                tgn.memory.restore_memory(val_memory_backup)

            #new_nodes_val_aps.append(nn_val_ap)
            val_aps.append(val_ap)
            train_losses.append(np.mean(m_loss))

            # Save temporary results to disk
            pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            # Early stopping
            if early_stopper.early_stop_check(val_ap):
                #logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                #logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                tgn.load_state_dict(torch.load(best_model_path))
                #logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                tgn.eval()
                break
            else:
                torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

        # Training has finished, we have loaded the best model, and we want to backup its current
        # memory (which has seen validation edges) so that it can also be used when testing on unseen
        # nodes
        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()

        ### Test
        tgn.embedding_module.neighbor_finder = full_ngh_finder
        test_ap, test_auc = eval_edge_prediction(args, model=tgn,
                                                                    negative_edge_sampler=test_rand_sampler,
                                                                    data=test_data,
                                                                    n_neighbors=NUM_NEIGHBORS)
        if USE_MEMORY:
            tgn.memory.restore_memory(val_memory_backup)

        logger.info(
            'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
        
        auc_roc_list.append(test_auc)
        ap_list.append(test_ap)
    
        pickle.dump({
            "val_aps": val_aps,
            "test_ap": test_ap,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))
        logger.info('Saving TGN model')
        if USE_MEMORY:
            # Restore memory at the end of validation (save a model which is ready for testing)
            tgn.memory.restore_memory(val_memory_backup)
        torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
        logger.info('TGN model saved')

    logger.info(f'============================================ Test End =================================================\n')

    final_auc = sum(auc_roc_list)/len(auc_roc_list)
    final_ap = sum(ap_list)/len(ap_list)
    logger.info('Total avg Test statistics: Old nodes -- auc: {}, ap: {}'.format(final_auc, final_ap))

    '''nn_final_auc = sum(nn_auc_roc_list)/len(nn_auc_roc_list)
    nn_final_ap = sum(nn_ap_list)/len(nn_ap_list)
    logger.info('Total avg Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_final_auc, nn_final_ap))'''