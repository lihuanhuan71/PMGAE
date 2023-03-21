import argparse
import time
import torch
from dgl.nn.pytorch.factory import KNNGraph
import numpy as np
import torch as th
import torch.nn as nn
import  torch.nn.functional as F
import warnings

from model import PMGAE
from model import LogReg
from GraphAugmentation import random_aug
from dataset import load
from visual import visual

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='GAE')
parser.add_argument('--dataname', type=str, default='photo', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of DCGAE.')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay of DCGAE.')
parser.add_argument('--lr2', type=float, default=1e-3, help='Learning rate of linear evaluator.')
parser.add_argument('--wd2', type=float, default=1e-5, help='Weight decay of linear evaluator.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=512, help='Output layer dim.')
parser.add_argument('--der', type=float, default=0.5, help='Drop edge ratio.')
parser.add_argument('--dfr', type=float, default=0.5, help='Drop feature ratio.')
parser.add_argument("--scale", type=int, default=100, help='Output layer dim.')
parser.add_argument("--multiple", type=float, default=1, help='the multiple of degree.')
parser.add_argument('--ratio', type=float, default=0.2, help='ratio.')
args = parser.parse_args()
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'



if __name__ == '__main__':

    print(args)
    graph, feat, labels, num_class, train_idx, val_idx, test_idx ,degree= load(args.dataname)
    in_dim = feat.shape[1]

    model = PMGAE(in_dim, args.hid_dim, args.out_dim, args.n_layers)


    model = model.to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    k=degree*args.multiple
    k=k.long()
    knnGraph = KNNGraph(k)
    if k>0:

        z= graph.adjacency_matrix(transpose=True).to_dense()+knnGraph(feat,dist='cosine').adjacency_matrix(transpose=True).to_dense()
    else:
        z = graph.adjacency_matrix(transpose=True).to_dense()

    z = torch.where(z > 1, torch.full_like(z, 1), z)
    z=z*args.scale
    z = torch.triu(z, diagonal=0)
    N = graph.number_of_nodes()
    dur_time=[]
    dur_mem=[]
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        eids = np.arange(N)
        eids = np.random.permutation(eids)
        r0 = int(N * args.ratio)
        index = torch.LongTensor(np.array(eids[:r0]))

        pred = torch.index_select(z, 0, index)
        pred = torch.index_select(pred, 1, index)

        pred = pred.to(args.device)
        index = index.to(args.device)

        graph_aug, feat_aug= random_aug(graph, feat, args.dfr, args.der)


        graph_orgin= graph.add_self_loop()
        graph_orgin= graph_orgin.to(args.device)
        feat_aug = feat_aug.to(args.device)

        feat_orgin = feat.to(args.device)
        graph_aug = graph_aug.add_self_loop()
        graph_aug = graph_aug.to(args.device)



        out,mem= model(graph_orgin,feat_aug,graph_aug,feat_orgin,index)

        loss =F.mse_loss(out,pred)

        loss.backward()
        optimizer.step()
        dur_time.append(time.time() - t0)
        dur_mem.append(mem)
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    print("=== Evaluation ===")
    graph =graph.to(args.device)
    graph = graph.remove_self_loop().add_self_loop()
    feat = feat.to(args.device)

    embeds = model.get_embedding(graph,feat)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    #visual(feat.cpu(), label.cpu(),args.dataname,'orgin')
    visual(embeds.cpu(),label.cpu(),args.dataname,'PMGAE')


    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    train_feat = feat[train_idx]
    val_feat = feat[val_idx]
    test_feat = feat[test_idx]

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class)

    opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0
    best_epoch=0


    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc
                    best_epoch=epoch

            print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))

    print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
    print('best_epoch:'+str(best_epoch))
    print('Time(s) {:.4f},  Memory(s) {:.4f}MB'.format(np.mean(dur_time),np.mean(dur_mem)/(1024*1024)))