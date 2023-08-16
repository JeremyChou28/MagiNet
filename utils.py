import os
import torch
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import pandas as pd
from scipy.sparse.linalg import eigs
import math


def missed_eval_torch(predict, true, mask):
    mae = torch.sum(torch.absolute(predict - true) *
                    (1 - mask)) / torch.sum(1 - mask)
    rmse = torch.sqrt(
        torch.sum((predict - true)**2 * (1 - mask)) / torch.sum(1 - mask))
    mape_mask = torch.where(true > 5, 1, 0)
    mape_mask = mape_mask * (1 - mask)
    mape = torch.sum(
        torch.absolute(
            (predict - true) /
            (true + 1e-5)) * mape_mask) / (torch.sum(mape_mask) + 1e-5)
    return mae, rmse, mape


def missed_eval_np(predict, true, mask):
    predict, true = np.asarray(predict), np.asarray(true)
    mae = np.sum(np.absolute(predict - true) * (1 - mask)) / np.sum(1 - mask)
    rmse = np.sqrt(np.sum((predict - true)**2 * (1 - mask)) / np.sum(1 - mask))
    mape_mask = np.where(true > 5, 1, 0)
    mape_mask = mape_mask * (1 - mask)
    mape = np.sum(np.absolute(
        (predict - true) /
        (true + 1e-5)) * mape_mask) / np.sum(mape_mask + 1e-5)
    return mae, rmse, mape


def unnormalization(data, mean, std):
    return data * std + mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(
        np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(
        d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_graph_missdata(adj_mx, adjtype):
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [
            calculate_normalized_laplacian(adj_mx).astype(
                np.float32).todense()
        ]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj


def weight_matrix(adj_mx, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = adj_mx
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(
            -W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W


def load_PA(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.float64(df > 0)
    return df


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] -
                                cheb_polynomials[i - 2])

    return cheb_polynomials


def load_weighted_adjacency_matrix(file_path, num_v):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.float64(df > 0)
    return df


def batch_cosine_similarity(x, y):
    # 计算分母
    l2_x = torch.norm(
        x, dim=2, p=2
    ) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_y = torch.norm(
        y, dim=2, p=2
    ) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_m = torch.matmul(l2_x.unsqueeze(dim=2),
                        l2_y.unsqueeze(dim=2).transpose(1, 2))
    # 计算分子
    l2_z = torch.matmul(x, y.transpose(1, 2))
    # cos similarity affinity matrix
    cos_affnity = l2_z / l2_m
    adj = cos_affnity
    return adj


def batch_dot_similarity(x, y):
    QKT = torch.bmm(x, y.transpose(-1, -2)) / math.sqrt(x.shape[2])
    W = torch.softmax(QKT, dim=-1)
    return W
