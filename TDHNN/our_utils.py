import random
import argparse
import numpy as np
import torch
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics
from matplotlib import rcParams

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import pandas as pd

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='data sources to use tags-ask-ubuntu, tags-math-sx',
                        choices=['tags-ask-ubuntu', 'tags-math-sx', 'threads-ask-ubuntu', 'threads-math-sx', 'email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school'],
                        default='threads-ask-ubuntu')
    parser.add_argument('--f_dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--wdecay', type=float, default=0.00)
    parser.add_argument("--freq_size", default=2628000, type=int, help='batch size')
    parser.add_argument("--neg_mode", type=str, default='sns', help='negative sampling: mns, sns, cns')

    parser.add_argument('--in_dim', type=int, default=128)
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--num_edges', type=int, default=100)
    parser.add_argument('--min_num_edges', type=int, default=30)

    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--cuda', type=str, default='0', help='0/1/2/3')
    parser.add_argument("--gpu", type=int, default=1, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='dhl')

    parser.add_argument('--edges', type=str, default='h')
    parser.add_argument('--mask', type=int, default=1)
    parser.add_argument('--cf', type=str, default='x')
    parser.add_argument('--merge', type=str, default='cat', help='cat/plus')
    parser.add_argument('--stage', type=str, default='train', help='train/val')

    parser.add_argument('--conv_number', type=int, default=1)
    parser.add_argument('--k_n', type=int, default=10, help='number of nodes to choose')
    parser.add_argument('--k_e', type=int, default=10, help='number of edges to choose')

    parser.add_argument('--low_bound', type=float, default=0.9)
    parser.add_argument('--up_bound', type=float, default=0.95)

    parser.add_argument('--backbone', type=str, default='linear')
    parser.add_argument('--namuda', type=int, default=30)
    parser.add_argument('--namuda2', type=float, default=10)

    parser.add_argument('--splits', type=int, default=1)
    parser.add_argument('--fts', type=str, default='MVCNN', help='MVCNN/GVCNN')

    parser.add_argument('--split_ratio', type=float, default=0.8)

    parser.add_argument('--transfer', type=int, default=1)

    args = parser.parse_args()

    return args



def mscatter(x, y,  ax=None, m=None, **kw):
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)

    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
                
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)

    return sc

def plot_embedding_2d(X, labels, fname, title=None):
    config = {
    "font.size": 20,
    "mathtext.fontset":'stix'
}

    rcParams.update(config)


    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    num_of_labels = labels.max().item() + 1

    # fig = plt.figure(figsize=(16,10))

    plt.figure(figsize=(16,10))

    # plt.margins(0.001)

    colors_space = np.linspace(0, 1, num_of_labels)           # 生成颜色空间
    label_to_color = {}                                 # 将标签对应为颜色
    for i in range(num_of_labels):
        label_to_color[i] = colors_space[i]

    colors = []
    for label in labels:
        colors.append(label_to_color[label.item()])

    sc = plt.scatter(X[:, 0], X[:, 1], c=colors, s=20)
    # scatter = mscatter(X[:, 0], X[:, 1], c='r', ax=ax)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    cb=plt.colorbar(sc)
    cb.ax.tick_params(labelsize=32)  #设置色标刻度字体大小。
    
    plt.tight_layout(rect=(0, 0, 1.06, 1))

    plt.savefig(f'./{fname}.eps')
    plt.savefig(f'./{fname}.png')
    # plt.show()

def draw_TSNE(X, labels, fname, title=None):
    tsne2d = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_2d = tsne2d.fit_transform(X)
    plot_embedding_2d(X_tsne_2d, labels, fname, title)

def visualization(model, data, args, title=None):
    mask = data['train_idx']

    out, x, H, H_raw = model(data,args)

    # X = x.detach().to('cpu')
    # labels = data['lbls'].detach().to('cpu')

    X = x[mask].detach().to('cpu')
    labels = data['lbls'][mask].detach().to('cpu')

    fname = f'{args.model}_{args.dataset}_{args.fts}'
    draw_TSNE(X, labels, fname, title=None)
    Silhouette_score = metrics.silhouette_score(X, labels)
    print("Silhouette_score is: ", Silhouette_score)

def get_all_neg_samples(args):
    DATA = args.data
    feat_folder = f'/home/dake/workspace/DHGNN/hypergraph/{DATA}/'
    sns_neg_hedge = torch.load(feat_folder + f'sns_{DATA}.pt')
    cns_neg_hedge = torch.load(feat_folder + f'cns_{DATA}.pt')
    mns_neg_hedge = torch.load(feat_folder + f'mns_{DATA}.pt')
    
    # all_snapshot_edges = [element for row in snapshot_data for element in row]
    # sns_neg_hedge = neg_generator('sns', all_snapshot_edges, len(all_snapshot_edges))
    # cns_neg_hedge = neg_generator('cns', all_snapshot_edges, len(all_snapshot_edges))
    # mns_neg_hedge = neg_generator('mns', all_snapshot_edges, len(all_snapshot_edges))
    
    return sns_neg_hedge, cns_neg_hedge, mns_neg_hedge

def get_next_samples(next_snapshot,all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge):    
    
    pos_hedge = next_snapshot
    pos_size = len(pos_hedge)
    sns_neg_hedge = random.sample(all_sns_neg_hedge, pos_size)
    cns_neg_hedge = random.sample(all_cns_neg_hedge, pos_size)
    mns_neg_hedge = random.sample(all_mns_neg_hedge, pos_size)
    
    return pos_hedge, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge
 
def measure(label, pred):
    auc_roc = roc_auc_score(np.array(label), np.array(pred))
    ap = average_precision_score(np.array(label), np.array(pred))
                    
    return round(auc_roc,3), round(ap,3)

