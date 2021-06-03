import os
import copy
import pickle
from multiprocessing import Pool
from itertools import combinations
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import inv, eigs
import networkx as nx
from sknetwork.embedding import Spectral
from sknetwork.utils import membership_matrix
from sknetwork.hierarchy import Ward, cut_straight
from sknetwork.clustering import Louvain, KMeans, PropagationClustering
from geomloss import SamplesLoss
import pysbm


def load_t_files(args, T_file, logger, adj_train):
    # raw node embeddings for nearest neighbor finding: numpy.ndarray
    node_embs_raw = pickle.load(open(f'{args.datapath}{args.dataset}_embs-raw{args.embraw}.pkl', 'rb'))
    # print('cf distance threshold: ', np.percentile(cdist(node_embs_raw, node_embs_raw, 'euclidean'), args.gamma))
    if os.path.exists(T_file):
        T_f, T_cf, adj_cf, edges_cf_t0, edges_cf_t1 = pickle.load(open(T_file, 'rb'))
        logger.info(f'loaded cached T files: {args.t} {args.k}')
    else:
        T_f = get_t(adj_train, args.t, args.k, args.selfloopT)
        T_cf, adj_cf, edges_cf_t0, edges_cf_t1 = get_CF(adj_train, node_embs_raw, T_f, args.dist, args.gamma, args.n_workers)
        T_cf = sp.csr_matrix(T_cf)
        adj_cf = sp.csr_matrix(adj_cf)
        pickle.dump((T_f, T_cf, adj_cf, edges_cf_t0, edges_cf_t1), open(T_file, 'wb'))
        logger.info(f'calculated and cached T files: {args.t} {args.k}')
    return T_f, edges_cf_t1, edges_cf_t0, T_cf, adj_cf

def get_t(adj_mat, method, k, selfloop=False):
    adj = copy.deepcopy(adj_mat)
    if not selfloop:
        adj.setdiag(0)
        adj.eliminate_zeros()
    if method == 'anchor_nodes':
        T = anchor_nodes(adj, k)
    elif method == 'common_neighbors':
        T = common_neighbors(adj, k)
    elif method == 'louvain':
        T = louvain(adj)
    elif method == 'spectral_clustering':
        T = spectral_clustering(adj, k)
    elif method == 'propagation':
        T = propagation(adj)
    elif method == 'kcore':
        T = kcore(adj)
    elif method == 'katz':
        T = katz(adj, k)
    elif method == 'hierarchy':
        T = ward_hierarchy(adj, k)
    elif method == 'jaccard':
        T = jaccard_index(adj, k)
    elif method == 'sbm':
        T = SBM(adj, k)
    return T

def SBM(adj, k):
    nx_g = nx.from_scipy_sparse_matrix(adj)
    standard_partition = pysbm.NxPartition(graph=nx_g, number_of_blocks=k)
    rep = standard_partition.get_representation()
    labels = np.asarray([v for k, v in sorted(rep.items(), key=lambda item: item[0])])
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def ward_hierarchy(adj, k):
    ward = Ward()
    dendrogram = ward.fit_transform(adj)
    labels = cut_straight(dendrogram, k)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def jaccard_index(adj, k):
    adj = adj.astype(int)
    intrsct = adj.dot(adj.T)
    row_sums = intrsct.diagonal()
    unions = row_sums[:,None] + row_sums - intrsct
    sim_matrix = intrsct / unions
    thre = np.percentile(sim_matrix, (100-10*k))
    thre = max(thre, np.percentile(sim_matrix, 0.5))
    thre = min(thre, np.percentile(sim_matrix, 0.8))
    T = np.asarray((sim_matrix >= thre).astype(int))
    T = T - np.diag(T.diagonal())
    return sp.csr_matrix(T)

def katz(adj, k):
    max_eigvalue = eigs(adj.astype(float), k=1)[0][0]
    beta = min(1/max_eigvalue/2, 0.003)
    sim_matrix = inv(sp.identity(adj.shape[0]) - beta *  adj) - sp.identity(adj.shape[0])
    sim_matrix = sim_matrix.toarray()
    size = sim_matrix.shape[0]
    thre = 2 * k * sim_matrix.sum() / (size*size-1)
    T = np.asarray((sim_matrix > thre).astype(int))
    T = T - np.diag(T.diagonal())
    return sp.csr_matrix(T)

def anchor_nodes(adj, k):
    row_sum = np.asarray(adj.sum(axis = 1)).reshape(-1)
    dist = dijkstra(csgraph=adj, indices=np.argmax(row_sum), directed=False, limit=k+1, return_predecessors=False)
    res = dist < (k+1)
    T = np.zeros(adj.shape)
    T[res] += 1
    T[:,res] += 1
    T = (T > 1).astype(int)
    return sp.csr_matrix(T)

def common_neighbors(adj, k):
    mul_hop_adj = adj
    for i in range(2):
        mul_hop_adj += adj ** (i+2)
    mul_hop_adj = (mul_hop_adj>0).astype(int)
    T = (mul_hop_adj @ mul_hop_adj.T) >= k
    T = T.astype(int)
    return T

def louvain(adj):
    louvain = Louvain()
    labels = louvain.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def propagation(adj):
    propagation = PropagationClustering()
    labels = propagation.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def spectral_clustering(adj, k):
    kmeans = KMeans(n_clusters = k, embedding_method=Spectral(256))
    labels = kmeans.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def kcore(adj):
    G = nx.from_scipy_sparse_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    labels = np.array(list(nx.algorithms.core.core_number(G).values()))-1
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def sample_nodepairs(num_np, edges_f_t1, edges_f_t0, edges_cf_t1, edges_cf_t0):
    # TODO: add sampling with separated treatments
    nodepairs_f = np.concatenate((edges_f_t1, edges_f_t0), axis=0)
    f_idx = np.random.choice(len(nodepairs_f), min(num_np,len(nodepairs_f)), replace=False)
    np_f = nodepairs_f[f_idx]
    nodepairs_cf = np.concatenate((edges_cf_t1, edges_cf_t0), axis=0)
    cf_idx = np.random.choice(len(nodepairs_cf), min(num_np,len(nodepairs_f)), replace=False)
    np_cf = nodepairs_cf[cf_idx]
    return np_f, np_cf

def calc_disc(disc_func, z, nodepairs_f, nodepairs_cf):
    X_f = torch.cat((z[nodepairs_f.T[0]], z[nodepairs_f.T[1]]), axis=1)
    X_cf = torch.cat((z[nodepairs_cf.T[0]], z[nodepairs_cf.T[1]]), axis=1)
    if disc_func == 'lin':
        mean_f = X_f.mean(0)
        mean_cf = X_cf.mean(0)
        loss_disc = torch.sqrt(F.mse_loss(mean_f, mean_cf) + 1e-6)
    elif disc_func == 'kl':
        # TODO: kl divergence
        pass
    elif disc_func == 'w':
        # Wasserstein distance
        dist = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        loss_disc = dist(X_cf, X_f)
    else:
        raise Exception('unsupported distance function for discrepancy loss')
    return loss_disc

def get_CF(adj, node_embs, T_f, dist='euclidean', thresh=50, n_workers=20):
    if dist == 'cosine':
        # cosine similarity (flipped to use as a distance measure)
        embs = normalize(node_embs, norm='l1', axis=1)
        simi_mat = embs @ embs.T
        simi_mat = 1 - simi_mat
    elif dist == 'euclidean':
        # Euclidean distance
        simi_mat = cdist(node_embs, node_embs, 'euclidean')
    thresh = np.percentile(simi_mat, thresh)
    # give selfloop largest distance
    np.fill_diagonal(simi_mat, np.max(simi_mat)+1)
    # nearest neighbor nodes index for each node
    node_nns = np.argsort(simi_mat, axis=1)
    # find nearest CF node-pair for each node-pair
    node_pairs = list(combinations(range(adj.shape[0]), 2))
    print('This step may be slow, please adjust args.n_workers according to your machine')
    pool = Pool(n_workers)
    batches = np.array_split(node_pairs, n_workers)
    results = pool.map(get_CF_single, [(adj, simi_mat, node_nns, T_f, thresh, np_batch, True) for np_batch in batches])
    results = list(zip(*results))
    T_cf = np.add.reduce(results[0])
    adj_cf = np.add.reduce(results[1])
    edges_cf_t0 = np.concatenate(results[2])
    edges_cf_t1 = np.concatenate(results[3])
    return T_cf, adj_cf, edges_cf_t0, edges_cf_t1,

def get_CF_single(params):
    """ single process for getting CF edges """
    adj, simi_mat, node_nns, T_f, thresh, node_pairs, verbose = params

    T_cf = np.zeros(adj.shape)
    adj_cf = np.zeros(adj.shape)
    edges_cf_t0 = []
    edges_cf_t1 = []
    c = 0
    for a, b in node_pairs:
        # for each node pair (a,b), find the nearest node pair (c,d)
        nns_a = node_nns[a]
        nns_b = node_nns[b]
        i, j = 0, 0
        while i < len(nns_a)-1 and j < len(nns_b)-1:
            if simi_mat[a, nns_a[i]] + simi_mat[b, nns_b[j]] > 2 * thresh:
                T_cf[a, b] = T_f[a, b]
                adj_cf[a, b] = adj[a, b]
                break
            if T_f[nns_a[i], nns_b[j]] != T_f[a, b]:
                T_cf[a, b] = 1 - T_f[a, b] # T_f[nns_a[i], nns_b[j]] when treatment not binary
                adj_cf[a, b] = adj[nns_a[i], nns_b[j]]
                if T_cf[a, b] == 0:
                    edges_cf_t0.append([nns_a[i], nns_b[j]])
                else:
                    edges_cf_t1.append([nns_a[i], nns_b[j]])
                break
            if simi_mat[a, nns_a[i+1]] < simi_mat[b, nns_b[j+1]]:
                i += 1
            else:
                j += 1
        c += 1
        if verbose and c % 20000 == 0:
            print(f'{c} / {len(node_pairs)} done')
    edges_cf_t0 = np.asarray(edges_cf_t0)
    edges_cf_t1 = np.asarray(edges_cf_t1)
    return T_cf, adj_cf, edges_cf_t0, edges_cf_t1

