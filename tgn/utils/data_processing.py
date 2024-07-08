import numpy as np
import random
import pandas as pd


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

def new_get_data(dataset_name):
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  random.seed(0)

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  return full_data, node_features, edge_features


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  random.seed(0)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data

def get_snapshot_data(args, dataset_name):
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  edge_features = np.pad(edge_features, ((0, 1), (0, 0)), mode='constant', constant_values=0)
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 

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
              snapshot_list.append(snapshot_data)

              merge_time_data = []
              merge_source_data = []
              merge_destination_data = []
              merge_label_data = []
              merge_edge_idxs_data = []
            
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

  return snapshot_list, node_features, edge_features, max_node_idx


def get_data(dataset_name):
  ### Load data and train val test split

  # graph_df는 이미 clique expansion 된 dataset
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  edge_features = np.pad(edge_features, ((0, 1), (0, 0)), mode='constant', constant_values=0)
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))
    

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
  
  ''''
  # Samplenodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = np.isin(sources, list(new_test_node_set))
  new_test_destination_mask = np.isin(destinations, list(new_test_node_set))

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)'''

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  # train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  train_mask = np.logical_and(timestamps <= val_time, True)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  #assert len(train_node_set & new_test_node_set) == 0
  #new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time
  '''
  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)'''

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])
  '''
  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])'''

  '''print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))'''

  #return full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data
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



def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
