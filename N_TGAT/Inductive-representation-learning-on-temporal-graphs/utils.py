import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()

        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))

        self.classifier = nn.Sequential(*Layers)


    def aggregate(self, embeddings, mode):
        if mode == 'Maxmin':
            max_val, _ = torch.max(embeddings, dim=0)
            min_val, _ = torch.min(embeddings, dim=0)
            return max_val - min_val
        elif mode == 'Avg' :
            embedding = embeddings.mean(dim=0).squeeze()
            return embedding
            
                
    def classify(self, embedding):
        return torch.sigmoid(self.classifier(embedding))
    
    def forward(self, v_feat, hedge_info, mode, device):
        preds =[]
        # hedge_info의 각 키(key)에 대해 반복합니다.
        for key, values in hedge_info.items():
            # 현재 키(key)에 해당하는 hedge_feat 값을 리스트 컴프리헨션을 사용하여 구축합니다.
            # 이때, v_feat 사전에서 각 value에 해당하는 값을 찾아 리스트로 만듭니다.
            embeddings = torch.tensor([v_feat[v] for v in values if v in v_feat])
            embedding = self.aggregate(embeddings, mode).to(device)
            pred = self.classify(embedding)
            preds.append(pred)
        preds = torch.stack(preds)

        return preds 
    
### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val[0] - self.last_best[0]) / np.abs(self.last_best[0]) > self.tolerance:
            self.last_best[0] = curr_val[0]
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]