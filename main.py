import os
import sys
import time
import argparse
import torch
import torch_sparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import networkx as nx
import numpy as np

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from utils import *
from models import *
from cf_utils import calc_disc, sample_nodepairs, load_t_files


def get_args():
    parser = argparse.ArgumentParser(description='CFLP')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--datapath', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--embraw', type=str, default='mvgrl')
    parser.add_argument('--t', type=str, default='kcore', help='choice of the treatment function')
    parser.add_argument('--k', type=int, default=2, help='parameter for the treatment function (if needed)')
    parser.add_argument('--selfloopT', default=False, action='store_true', help='whether to add selfloop when getting T')
    parser.add_argument('--dist', type=str, default='euclidean', help='distant metric used when finding nearest neighbors')
    parser.add_argument('--alpha', type=float, default=1, help='weight of cf loss')
    parser.add_argument('--beta', type=float, default=1, help='weight of discrepancy loss')
    parser.add_argument('--gamma', type=float, default=30.0, help='maximum distance thresold for finding nearest neighbors')
    parser.add_argument('--neg_rate', type=int, default=1, help='rate of negative samples during training')
    parser.add_argument('--dec', type=str, default='hadamard', choices=['innerproduct','hadamard','mlp'], help='choice of decoder')
    parser.add_argument('--seed', type=int, default=-1, help='fix random seed if needed')
    parser.add_argument('--verbose', type=int, default=1, help='whether to print per-epoch logs')
    parser.add_argument('--trails', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=-1, help='-2 for CPU, -1 for default GPU, >=0 for specific GPU')
    parser.add_argument('--n_workers', type=int, default=20, help='number of CPU processes for finding counterfactual links in the first run')
    parser.add_argument('--gnn_type', type=str, default='GCN',choices=['SAGE','GCN'])
    parser.add_argument('--jk_mode', type=str, default='mean',choices=['max','cat','mean','lstm','sum','none'])
    parser.add_argument('--dim_h', type=int, default=256)
    parser.add_argument('--dim_z', type=int, default=256)
    parser.add_argument('--num_np', type=int, default=1000, help='number of sampled node pairs when calculating disc loss')
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--epochs_ft', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024*64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_ft', type=float, default=5e-3)
    parser.add_argument('--l2reg', type=float, default=5e-6)
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of edges for validation set (and same number of no-edges)')
    parser.add_argument('--test_frac', type=float, default=0.2, help='fraction of edges for testing set (and same number of no-edges)')
    parser.add_argument('--disc_func', type=str, default='lin', choices=['lin', 'kl', 'w'], help='distance function for discrepancy loss')
    parser.add_argument('--lr_scheduler', type=str, default='zigzag', choices=['sgdr', 'cos', 'zigzag', 'none'], help='lr scheduler')
    parser.add_argument('--metric', type=str, default='auc', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda:0' if args.gpu >= -1 else 'cpu')

    return args


def train(args, logger):
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # load data
    adj_label, features, dim_feat, adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false = load_data(args, logger)
    # load n by n treatment matrix
    T_file_path = f'{args.datapath}T_files/'
    if not os.path.exists(T_file_path):
        os.makedirs(T_file_path, exist_ok=True)
    T_file = f'{T_file_path}{args.dataset}_{args.t}{args.k}-{args.dist}{args.gamma}-{args.embraw}.pkl'
    T_f, edges_cf_t1, edges_cf_t0, T_cf, adj_cf = load_t_files(args, T_file, logger, adj_train)

    # get the factual node pairs
    edges_f_t1 = np.asarray((sp.triu(T_f, 1) > 0).nonzero()).T
    edges_f_t0 = np.asarray(sp.triu(T_f==0, 1).nonzero()).T
    assert edges_f_t1.shape[0] + edges_f_t0.shape[0] == np.arange(adj_label.shape[0]).sum()
    logger.info('Number of edges: F: t=1: {} t=0: {}, CF: t=1: {} t=0: {}'.format(
        edges_f_t1.shape[0], edges_f_t0.shape[0], edges_cf_t1.shape[0], edges_cf_t0.shape[0]))

    # get train_edges and train_edges_false
    logger.info('...getting train splitting...')
    trainsplit_dir_name = 'data/train_split/'
    if not os.path.exists(trainsplit_dir_name):
        os.makedirs(trainsplit_dir_name, exist_ok=True)
    try:
        train_edges, train_edges_false = pickle.load(open(f'{trainsplit_dir_name}{args.dataset}.pkl', 'rb'))
    except:
        train_edges = np.asarray(sp.triu(adj_train, 1).nonzero()).T
        all_set = set([tuple(x) for x in train_pairs])
        edge_set = set([tuple(x) for x in train_edges])
        noedge_set = all_set - edge_set
        train_edges_false = np.asarray(list(noedge_set))
        pickle.dump((train_edges, train_edges_false), open(f'{trainsplit_dir_name}{args.dataset}.pkl', 'wb'))

    assert train_edges.shape[0] + train_edges_false.shape[0] == train_pairs.shape[0]
    logger.info(f'train_edges len: {train_edges.shape[0]}, batch size: {args.batch_size}')
    logger.info('...finishing train splitting...')

    max_neg_rate = train_edges_false.shape[0] // train_edges.shape[0] - 1
    if args.neg_rate > max_neg_rate:
        args.neg_rate = max_neg_rate
        logger.info(f'negative rate change to: {max_neg_rate}')
    val_pairs = np.concatenate((val_edges, val_edges_false), axis=0)
    val_labels = np.concatenate((np.ones(val_edges.shape[0]), np.zeros(val_edges_false.shape[0])), axis=0)
    test_pairs = np.concatenate((test_edges, test_edges_false), axis=0)
    test_labels = np.concatenate((np.ones(test_edges.shape[0]), np.zeros(test_edges_false.shape[0])), axis=0)

    # cast everything to proper type
    adj_train_coo = adj_train.tocoo()
    edge_index = np.concatenate((adj_train_coo.row[np.newaxis,:],adj_train_coo.col[np.newaxis,:]), axis=0)
    adj_norm = torch_sparse.SparseTensor.from_edge_index(torch.LongTensor(edge_index))

    # move everything to device
    device = args.device
    adj_norm = adj_norm.to(device)
    T_f = torch.FloatTensor(T_f.toarray()).to(device)
    T_cf = torch.FloatTensor(T_cf.toarray()).to(device)
    T_f_val = T_f[val_pairs.T]
    T_f_test = T_f[test_pairs.T]
    adj_cf = torch.FloatTensor(adj_cf.toarray()).to(device)
    features = features.to(device)
    pos_w_f = torch.FloatTensor([args.neg_rate]).to(device)

    model = CFLP(dim_feat, args.dim_h, args.dim_z, args.dropout, args.gnn_type, args.jk_mode, args.dec)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.l2reg)
    optims = MultipleOptimizer(args.lr_scheduler, optim)

    logger.info(f'Using evaluation metric: {args.metric}')
    best_val_res = 0.0
    pretrained_params = None
    cnt_wait = 0
    for epoch in range(args.epochs):
        total_examples = 0
        total_loss = 0
        for perm in DataLoader(range(train_edges.shape[0]), args.batch_size, shuffle=True):
            # sample no_edges for this batch
            pos_edges =  train_edges[perm]
            neg_sample_idx = np.random.choice(train_edges_false.shape[0], args.neg_rate * len(perm), replace=False)
            neg_edges = train_edges_false[neg_sample_idx]
            train_edges_batch = np.concatenate((pos_edges, neg_edges), axis=0)
            # move things to device
            labels_f_batch = torch.cat((torch.ones(pos_edges.shape[0]), torch.zeros(neg_edges.shape[0]))).to(device)
            labels_cf_batch = adj_cf[train_edges_batch.T]
            T_f_batch = T_f[train_edges_batch.T]
            T_cf_batch = T_cf[train_edges_batch.T]
            pos_w_cf = (labels_cf_batch.shape[0] - labels_cf_batch.sum()) / labels_cf_batch.sum()

            model.train()
            lr = optims.update_lr(args.lr)
            optims.zero_grad()
            # forward pass
            z, logits_f, logits_cf = model(adj_norm, features, train_edges_batch, T_f_batch, T_cf_batch)
            # loss
            nodepairs_f, nodepairs_cf = sample_nodepairs(args.num_np, edges_f_t1, edges_f_t0, edges_cf_t1, edges_cf_t0)
            loss_disc = calc_disc(args.disc_func, z, nodepairs_f, nodepairs_cf)
            loss_f = F.binary_cross_entropy_with_logits(logits_f, labels_f_batch, pos_weight=pos_w_f)
            loss_cf = F.binary_cross_entropy_with_logits(logits_cf, labels_cf_batch, pos_weight=pos_w_cf)
            loss = loss_f + args.alpha * loss_cf + args.beta * loss_disc
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optims.step()

            total_loss += loss.item() * pos_edges.shape[0]
            total_examples += pos_edges.shape[0]

        total_loss /= total_examples
        #evaluation
        model.eval()
        with torch.no_grad():
            z = model.encoder(adj_norm, features)
            logits_val = model.decoder(z[val_pairs.T[0]], z[val_pairs.T[1]], T_f_val).detach().cpu()
            logits_test = model.decoder(z[test_pairs.T[0]], z[test_pairs.T[1]], T_f_test).detach().cpu()
        val_res = eval_ep_batched(logits_val, val_labels, val_edges.shape[0])
        if val_res[args.metric] >= best_val_res:
            cnt_wait = 0
            best_val_res = val_res[args.metric]
            pretrained_params = parameters_to_vector(model.parameters())
            test_res = eval_ep_batched(logits_test, test_labels, test_edges.shape[0])
            test_res['best_val'] = val_res[args.metric]
            if args.verbose:
                logger.info('Epoch {} Loss: {:.4f} lr: {:.4f} val: {:.4f} test: {:.4f}'.format(
                    epoch+1, total_loss, lr, val_res[args.metric], test_res[args.metric]))
        else:
            cnt_wait += 1
            if args.verbose:
                logger.info('Epoch {} Loss: {:.4f} lr: {:.4f} val: {:.4f}'.format(
                    epoch+1, total_loss, lr, val_res[args.metric]))

        if cnt_wait >= args.patience:
            if args.verbose:
                print('Early stopping!')
            break

    """Fine-tune"""
    if args.epochs_ft:
        optim_ft = torch.optim.Adam(model.decoder.parameters(),
                                    lr=args.lr_ft,
                                    weight_decay=args.l2reg)
        vector_to_parameters(pretrained_params, model.parameters())
        model.encoder.eval()
        with torch.no_grad():
            z = model.encoder(adj_norm, features).detach()

        # best_val_res = 0.0
        best_params = None
        cnt_wait = 0

        # model.decoder.reset_parameters()
        for epoch in range(args.epochs_ft):
            total_examples = 0
            total_loss = 0
            # sample no_edges for this epoch
            for perm in DataLoader(range(train_edges.shape[0]), args.batch_size, shuffle=True):
                pos_edges =  train_edges[perm]
                neg_sample_idx = np.random.choice(train_edges_false.shape[0], args.neg_rate * len(perm), replace=False)
                neg_edges = train_edges_false[neg_sample_idx]
                train_edges_batch = np.concatenate((pos_edges, neg_edges), axis=0)
                labels_f_batch = torch.cat((torch.ones(pos_edges.shape[0]), torch.zeros(neg_edges.shape[0]))).to(device)
                T_f_batch = T_f[train_edges_batch.T]

                model.decoder.train()
                optim_ft.zero_grad()

                logits = model.decoder(z[train_edges_batch.T[0]], z[train_edges_batch.T[1]], T_f_batch)
                loss = F.binary_cross_entropy_with_logits(logits, labels_f_batch, pos_weight=pos_w_f)

                loss.backward()
                optim_ft.step()

                total_loss += loss.item() * pos_edges.shape[0]
                total_examples += pos_edges.shape[0]

            total_loss /= total_examples
            model.decoder.eval()
            with torch.no_grad():
                logits_val = model.decoder(z[val_pairs.T[0]], z[val_pairs.T[1]], T_f_val).detach().cpu()
                logits_test = model.decoder(z[test_pairs.T[0]], z[test_pairs.T[1]], T_f_test).detach().cpu()

            val_res = eval_ep_batched(logits_val, val_labels, val_edges.shape[0])
            if val_res[args.metric] > best_val_res:
                cnt_wait = 0
                best_val_res = val_res[args.metric]
                best_params = parameters_to_vector(model.parameters())
                test_res = eval_ep_batched(logits_test, test_labels, test_edges.shape[0])
                test_res['best_val'] = val_res[args.metric]
                if args.verbose:
                    logger.info('FT epoch {} loss: {:.4f} val: {:.4f} test: {:.4f}'.format(
                        epoch+1, total_loss, val_res[args.metric], test_res[args.metric]))
            else:
                cnt_wait += 1
                if args.verbose:
                    logger.info('FT epoch {} loss: {:.4f} val: {:.4f}'.format(
                        epoch+1, total_loss, val_res[args.metric]))

            if cnt_wait >= args.patience:
                if args.verbose:
                    print('Early stopping!')
                break

    ate_dict = None
    if args.cal_ate:
        if best_params is not None:
            vector_to_parameters(best_params, model.parameters())
        model.decoder.eval()
        with torch.no_grad():
            all_pairs = np.concatenate((train_pairs, val_pairs, test_pairs), axis=0)
            y_f_estimated = []
            y_cf_estimated = []
            for perm in DataLoader(range(all_pairs.shape[0]), args.batch_size):
                perm_pairs = all_pairs[perm]
                T_f_perm = T_f[perm_pairs.T]
                T_cf_perm = 1 - T_f_perm
                f_estimated = torch.sigmoid(model.decoder(z[perm_pairs.T[0]], z[perm_pairs.T[1]], T_f_perm))
                cf_estimated = torch.sigmoid(model.decoder(z[perm_pairs.T[0]], z[perm_pairs.T[1]], T_cf_perm))
                y_f_estimated.append(f_estimated.detach().cpu().numpy())
                y_cf_estimated.append(cf_estimated.detach().cpu().numpy())
            T_f_all = T_f[all_pairs.T].cpu().numpy()
            y_f_estimated = np.concatenate(y_f_estimated, axis=None)
            y_cf_estimated = np.concatenate(y_cf_estimated, axis=None)

            ite_est = (2 * T_f_all - 1) * (y_f_estimated - y_cf_estimated)

            y_f_ground = np.concatenate((np.ones(train_edges.shape[0]), np.zeros(train_edges_false.shape[0]), val_labels, test_labels))
            y_cf_nns = adj_cf[all_pairs.T].cpu().numpy()
            T_cf_all = T_cf[all_pairs.T].cpu().numpy()
            effect_pairs = (T_f_all != T_cf_all)
            ite_obs = (2 * T_f_all - 1) * (y_f_ground - y_cf_nns)
            ate_obs = ite_obs[effect_pairs].mean()
            ate_dict = {'ate_obs': ate_obs, 'ate_est': ite_est.mean()}

    if ate_dict is not None:
        test_res = {**test_res, **ate_dict}
    return test_res


def main(args):
    log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{args.t}_{args.embraw}_{time.strftime("%m-%d_%H-%M")}'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(log_name)

    logger.info(f'Input argument vector: {args.argv[1:]}')
    logger.info(f'args: {args}')
    args.cal_ate = True
    if args.cal_ate:
        results = {'auc': [], 'ap': [], 'hits@20': [], 'hits@50': [], 'hits@100': [], 'best_val': [], 'ate_obs': [], 'ate_est': []}
    else:
        results = {'auc': [], 'ap': [], 'hits@20': [], 'hits@50': [], 'hits@100': [], 'best_val': []}
    for _ in range(args.trails):
        res = train(args, logger)
        for metric in results.keys():
            results[metric].append(res[metric])
    logger.info('final results:')
    for metric, nums in results.items():
        logger.info('{}: {:.4f}+-{:.4f} {}'.format(
            metric, np.mean(nums), np.std(nums), nums))


if __name__ == "__main__":
    args = get_args()
    main(args)

