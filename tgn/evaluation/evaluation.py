import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from itertools import groupby


def eval_edge_prediction(args, model, negative_edge_sampler, data, n_neighbors, batch_size=32):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []

  group_lengths = [len(list(group)) for _, group in groupby(data.timestamps)]
  test_hyperedges = np.unique(data.timestamps).shape[0]

  tbatch_sources = []
  tbatch_destinations = []
  tbatch_edge_idx = []
  tbatch_timestamps = []

  start = 0
  for length in group_lengths:
      end = start + length
      tbatch_sources.append(data.sources[start:end])
      tbatch_destinations.append(data.destinations[start:end])
      tbatch_edge_idx.append(data.edge_idxs[start:end])
      tbatch_timestamps.append(data.timestamps[start:end])
      start = end
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    num_test_instance = test_hyperedges
    TEST_BATCH_SIZE = batch_size
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    '''
    hyperedge_unit = [list(group) for _, group in groupby(data.timestamps)]
    num_test_instance = len(hyperedge_unit)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
  
    timestamps = data.timestamps
    freq_sec = args.freq_size 
    split_criterion = timestamps // freq_sec
    groups = np.unique(split_criterion)
    groups = np.sort(groups)
    num_test_batch = groups.shape[0]'''


    for k in range(num_test_batch):
      
      start_idx = k * TEST_BATCH_SIZE
      if num_test_instance < (start_idx + TEST_BATCH_SIZE):
        checking =0
      end_idx = min(num_test_instance, start_idx + TEST_BATCH_SIZE)

      '''if e_idx - s_idx < 100 :
        TEST_BATCH_SIZE = e_idx - s_idx
        e_idx = s_idx + TEST_BATCH_SIZE

      start_time = hyperedge_unit[s_idx][0]
      edge_time = hyperedge_unit[e_idx-1][0]
        
      batch_mask = np.logical_and(data.timestamps <= edge_time, data.timestamps >= start_time)
      
      masking_value = data.timestamps//freq_sec
      batch_mask = (masking_value == groups[k])

      sources_batch = data.sources[batch_mask]
      destinations_batch = data.destinations[batch_mask]
      edge_idxs_batch = data.edge_idxs[batch_mask]
      timestamps_batch = data.timestamps[batch_mask]'''

      sources_batch, destinations_batch = tbatch_sources[start_idx:end_idx], \
                                                    tbatch_destinations[start_idx:end_idx]
      edge_idxs_batch = tbatch_edge_idx[start_idx: end_idx]
      timestamps_batch = tbatch_timestamps[start_idx:end_idx]

      sources_batch = np.concatenate(sources_batch).tolist()
      destinations_batch = np.concatenate(destinations_batch).tolist()
      edge_idxs_batch = np.concatenate(edge_idxs_batch).tolist()
      timestamps_batch = np.concatenate(timestamps_batch).tolist()

      #num_hyperedges = len([list(group) for _, group in groupby(timestamps_batch)])

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      num_hyperedges = np.unique(timestamps_batch).shape[0]

      pos_prob, neg_prob = model.compute_hyperedge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), 1 - (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(num_hyperedges), np.zeros(num_hyperedges)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()
  import pandas as pd
  print(pd.DataFrame(data.labels.tolist()).value_counts())
  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
