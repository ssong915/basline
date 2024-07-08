import json
import numpy as np
import pandas as pd
import random

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            
            ts = float(e[2])
            label = int(e[3])
            
            feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)



def reindex(df):
    assert(df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert(df.i.max() - df.i.min() + 1 == len(df.i.unique()))
    
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u
    
    new_df = df.copy()
    print(new_df.u.max())
    print(new_df.i.max())
    
    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    
    print(new_df.u.max())
    print(new_df.i.max())
    
    return new_df



def run(data_name):
    PATH = './processed/{}.csv'.format(data_name)
    OUT_DF = './processed/ml_{}.csv'.format(data_name)
    OUT_FEAT = './processed/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
    
    df, feat = preprocess(PATH)
    new_df = reindex(df)
    
    print(feat.shape)
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])
    
    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    
    print(feat.shape)
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)

#run('reddit')

def get_snapshot_data(args, DATA):
    graph_df = pd.read_csv('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}.csv'.format(DATA))
    e_feat = np.load('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}.npy'.format(DATA))
    n_feat = np.load('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}_node.npy'.format(DATA))

    # clique expansion 된 정보를 기반으로 source, destination, edge_idx, labels, timestamps 저장
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    time = full_data.timestamps
    ts_start = time.min()
    ts_end = time.max() 
    filter_indexes = (full_data.timestamps >= ts_start) & (full_data.timestamps<=ts_end)

    all_sources = sources[filter_indexes]
    all_destinations = destinations[filter_indexes]
    all_edge_idxs = edge_idxs[filter_indexes]
    all_labels = labels[filter_indexes]
    all_timestamps = timestamps[filter_indexes]
    
    all_nodes = all_sources + all_destinations
    max_node_idx = all_nodes.max()
    snapshot_list = list()
    
    if args.time_split_type == 'sec':
        freq_sec = args.freq_size 
        split_criterion = all_timestamps // freq_sec
        groups = np.unique(split_criterion)
        groups = np.sort(groups)
        
        merge_time_data = []
        merge_time_data = []
        merge_source_data = []
        merge_destination_data = []
        merge_label_data = []
        merge_edge_idxs_data = []

        for t in groups:
            period_members = (split_criterion == t) # t시점에 있는 아이 
            source_data = list(all_sources[period_members])
            destination_data = list(all_destinations[period_members])
            edge_idxs_data = list(all_edge_idxs[period_members])
            label_data = list(all_labels[period_members])
            time_data  = list(all_timestamps[period_members]) 
            
            if len(set(time_data)) < 3 :
                if len(merge_source_data) != 0 :
                  merge_source_data = merge_source_data + source_data
                  merge_destination_data = merge_destination_data + destination_data
                  merge_edge_idxs_data = merge_edge_idxs_data + edge_idxs_data
                  merge_label_data = merge_label_data + label_data
                  merge_time_data = merge_time_data + time_data

                  if len(set(merge_time_data)) >= 3 :
                    source_data = merge_source_data
                    destination_data = merge_destination_data
                    time_data = merge_time_data
                    edge_idxs_data = merge_edge_idxs_data
                    label_data = merge_label_data
                    snapshot_data = Data(source_data, destination_data, time_data, edge_idxs_data, label_data)

                    merge_time_data = []
                    merge_source_data = []
                    merge_destination_data = []
                    merge_label_data = []
                    merge_edge_idxs_data = []
                    snapshot_list.append(snapshot_data)

                else:
                  merge_source_data = source_data
                  merge_destination_data = destination_data
                  merge_edge_idxs_data = edge_idxs_data
                  merge_label_data = label_data
                  merge_time_data = time_data

            else :
                if len(merge_source_data) != 0 :
                    source_data = merge_source_data + source_data
                    destination_data = merge_destination_data + destination_data
                    edge_idxs_data = merge_edge_idxs_data + edge_idxs_data
                    label_data = merge_label_data + label_data
                    time_data = merge_time_data + time_data
                
                snapshot_data = Data(source_data, destination_data, time_data, edge_idxs_data, label_data)

                merge_time_data = []
                merge_source_data = []
                merge_destination_data = []
                merge_label_data = []
                merge_edge_idxs_data = []
                snapshot_list.append(snapshot_data)
                    
    print('snapshot 수 : ',len(snapshot_list))

    return snapshot_list, n_feat, e_feat, max_node_idx

def get_data(dataset_name):
  ### Load data and train val test split

  # graph_df는 이미 clique expansion 된 dataset
  graph_df = pd.read_csv('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}.npy'.format(dataset_name))
  node_features = np.load('/home/dake/workspace/N_TGAT/Inductive-representation-learning-on-temporal-graphs/processed/ml_{}_node.npy'.format(dataset_name))

  # clique expansion 된 정보를 기반으로 source, destination, edge_idx, labels, timestamps 저장
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  sources = np.array(sources)
  destinations = np.array(destinations)
  edge_idxs = np.array(edge_idxs)
  labels = np.array(labels)
  timestamps = np.array(timestamps)


  val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(42)

  node_set = set(sources) | set(destinations)
  all_nodes = sources + destinations
  max_node_idx = all_nodes.max()

  # Compute nodes which appear at test time
  #filter_idx = (timestamps > val_time)
  test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
  
  train_mask = np.logical_and(timestamps <= val_time, True)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])
  
 
  return node_features, edge_features, full_data, train_data, val_data, test_data, max_node_idx

def get_new_data(graph_df, future):
  
  if future == 1:
    sources = np.array(graph_df.sources)
    destinations = np.array(graph_df.destinations)
    edge_idxs = np.array(graph_df.edge_idxs)
    labels = np.array(graph_df.labels)
    timestamps = np.array(graph_df.timestamps)
    
  else:
    sources = []
    destinations = []
    edge_idxs = []
    labels = []
    timestamps = []

    for i in range(len(graph_df)):    
      sources.extend(graph_df[i].sources)
      destinations.extend(graph_df[i].destinations)
      edge_idxs.extend(graph_df[i].edge_idxs)
      labels.extend(graph_df[i].labels)
      timestamps.extend(graph_df[i].timestamps)
 
    sources = np.array(sources)
    destinations = np.array(destinations)
    edge_idxs = np.array(edge_idxs)
    labels = np.array(labels)
    timestamps = np.array(timestamps)
    '''sources = np.array(graph_df.sources)
    destinations = np.array(graph_df.destinations)
    edge_idxs = np.array(graph_df.edge_idxs)
    labels = np.array(graph_df.labels)
    timestamps = np.array(graph_df.timestamps)'''


  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(42)

  node_set = set(sources) | set(destinations)
  snapshot_length = len(sources)
  min_each_set = 1
  n_total_unique_nodes = len(node_set)

  # 남은 데이터 길이 계산
  remaining_length = snapshot_length - 3

  # 비율 계산
  train_ratio = 0.7
  val_ratio = 0.2
  test_ratio = 0.1

  base_train_split = int(snapshot_length * train_ratio)
  base_val_split = int(snapshot_length * val_ratio)
  base_test_split = int(snapshot_length * test_ratio)

  # boolean 마스크 생성
  train_mask = np.zeros(snapshot_length, dtype=bool)
  val_mask = np.zeros(snapshot_length, dtype=bool)
  test_mask = np.zeros(snapshot_length, dtype=bool)

  if base_train_split < 1 or base_val_split < 1 or base_test_split < 1 :
    min_each_set = 1
    remaining_length = snapshot_length - 3 * min_each_set  # 각 세트에 1개씩 할당하고 남은 데이터 길이

     # 각 세트에 최소 하나의 데이터 할당
    train_mask[:min_each_set] = True
    val_mask[min_each_set:min_each_set * 2] = True
    test_mask[min_each_set * 2:min_each_set * 3] = True

    if base_train_split > remaining_length :
      train_mask[min_each_set * 3:min_each_set * 3 + remaining_length] = True
    else :
      add_data = remaining_length - (base_train_split-1)
      train_split = min_each_set * 3 + add_data
      train_mask[min_each_set * 3: train_split] = True
      add_data = remaining_length - add_data
      val_split = train_split + add_data
      val_mask[train_split:val_split] = True
      test_mask[val_split:] = True

  else:
    # 남은 데이터를 비율에 따라 분할
    train_mask[:base_train_split] = True
    base_val_split = base_train_split + base_val_split
    val_mask[base_train_split:base_val_split] = True
    test_mask[base_val_split:] = True
  # 마스크를 사용하여 데이터 분할
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idxs[val_mask], labels[val_mask])
  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idxs[test_mask], labels[test_mask])

  return full_data, train_data, val_data, test_data