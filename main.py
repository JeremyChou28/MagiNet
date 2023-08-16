import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import pickle as pk
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse
import wandb
import math
import time
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='PEMS-BAY',
                    type=str,
                    help='dataset name')
parser.add_argument('--miss_mechanism',
                    default='MCAR',
                    type=str,
                    help='miss mechanism')
parser.add_argument('--miss_ratio', default=0.5, type=float, help='miss ratio')
parser.add_argument('--hidden_size', default=32, type=int)
parser.add_argument('--num_feature_in',
                    default=2,
                    type=int,
                    help='numbers of input features')
parser.add_argument('--num_nodes', default=325, type=int)
parser.add_argument('--seqlen', default=12, type=int)
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--save_path', default='checkpoints/', type=str)
parser.add_argument('--result_path', default='results/', type=str)
parser.add_argument('--cuda', default=1, type=int)
args = parser.parse_args()

datapath = './datasets'
dataset = args.dataset
miss_mechanism = args.miss_mechanism
miss_ratio = args.miss_ratio
batch_size = args.batch_size
epochs = args.epochs
seqlen = args.seqlen
seed = args.seed

if dataset == 'didi-chengdu' or dataset == 'didi-shenzhen':
    from models.didi import *
else:
    from models.pems import *

device = torch.device(
    'cuda:{}'.format(args.cuda) if torch.cuda.is_available() else "cpu")


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


seed_torch(seed)
torch.set_num_threads(10)


def load_data(datapath, dataset, miss_mechanism, miss_ratio, seqlen):
    '''
    read traffic data: 读取数据中不包含时间点在时间序列中的位置信息
    '''
    import pickle as pk
    # get adjacency matrix
    with open(datapath + "/{}/adj_mx.pkl".format(dataset), 'rb') as fb:
        A = pk.load(fb).astype(np.float32)
    # get normalization metric: mean, std
    with open(datapath + "/{}/processed/normalization.pkl".format(dataset),
              'rb') as fb:
        mean, std = pk.load(fb)['args'].values()

    # train
    with open(
            datapath + "/{}/processed/{}/train_{}_ms{}_seqlen_{}.pkl".format(
                dataset, miss_mechanism, miss_mechanism, miss_ratio, seqlen),
            'rb') as fb:
        train_data = pk.load(fb)
    # train_X = np.expand_dims(train_data['data'].astype(np.float32), 3)
    train_X = np.expand_dims(np.nan_to_num(train_data['data']),
                             3).astype(np.float32)  # (B,N,L,1)
    train_M = np.expand_dims(train_data['mask'].astype(np.float32),
                             3)  # (B,N,L,1)
    train_Y = np.expand_dims(train_data['target'].astype(np.float32),
                             3)  # (B,N,L,1)

    # valid
    with open(
            datapath + "/{}/processed/{}/valid_{}_ms{}_seqlen_{}.pkl".format(
                dataset, miss_mechanism, miss_mechanism, miss_ratio, seqlen),
            'rb') as fb:
        valid_data = pk.load(fb)
    # valid_X = np.expand_dims(valid_data['data'].astype(np.float32), 3)
    valid_X = np.expand_dims(np.nan_to_num(valid_data['data']),
                             3).astype(np.float32)  # (B,N,L,1)
    valid_M = np.expand_dims(valid_data['mask'].astype(np.float32),
                             3)  # (B,N,L,1)
    valid_Y = np.expand_dims(valid_data['target'].astype(np.float32),
                             3)  # (B,N,L,1)

    # test
    with open(
            datapath + "/{}/processed/{}/test_{}_ms{}_seqlen_{}.pkl".format(
                dataset, miss_mechanism, miss_mechanism, miss_ratio, seqlen),
            'rb') as fb:
        test_data = pk.load(fb)
    # test_X = np.expand_dims(test_data['data'].astype(np.float32), 3)
    test_X = np.expand_dims(np.nan_to_num(test_data['data']),
                            3).astype(np.float32)  # (B,N,L,1)
    test_M = np.expand_dims(test_data['mask'].astype(np.float32),
                            3)  # (B,N,L,1)
    test_Y = np.expand_dims(test_data['target'].astype(np.float32),
                            3)  # (B,N,L,1)

    return train_X, train_M, train_Y, valid_X, valid_M, valid_Y, test_X, test_M, test_Y, A, mean, std


def generate_miss_loader():
    # split to train,valid,test
    with open(datapath + "/{}/processed/index_{}.pkl".format(dataset, seqlen),
              'rb') as fb:
        index = pk.load(fb)

    def add_pos_emb(mode, X, Y):
        pos_emb = []
        for item in index[mode]:
            pos = datapos[item[0]:item[1], :, 1]
            pos_emb.append(pos)
        pos_emb = np.array(pos_emb).transpose(0, 2, 1)
        pos_emb = np.expand_dims(pos_emb, 2)  # (B,N,1,L)
        X = X.transpose(0, 1, 3, 2)  # (B,N,1,L)
        Y = Y.transpose(0, 1, 3, 2)  # (B,N,1,L)
        X = np.concatenate([X, pos_emb], axis=2).astype(np.float32)
        Y = np.concatenate([Y, pos_emb], axis=2).astype(np.float32)
        return X, Y

    data = {}

    train_X, train_M, train_Y, valid_X, valid_M, valid_Y, test_X, test_M, test_Y, A, mean, std = load_data(
        datapath, dataset, miss_mechanism, miss_ratio, seqlen)

    # get data with position information
    with open(datapath + "/{0}/data_pos.pkl".format(dataset), 'rb') as fb:
        datapos = pk.load(fb)

    # train
    train_M = np.repeat(train_M, 2, axis=-1)
    train_M = train_M.transpose(0, 1, 3, 2)  # (B,N,2,L)
    train_X, train_Y = add_pos_emb('train', train_X, train_Y)  # (B,N,2,L)

    # valid
    valid_M = np.repeat(valid_M, 2, axis=-1)
    valid_M = valid_M.transpose(0, 1, 3, 2)  # (B,N,2,L)
    valid_X, valid_Y = add_pos_emb('valid', valid_X, valid_Y)  # (B,N,2,L)

    # test
    test_M = np.repeat(test_M, 2, axis=-1)
    test_M = test_M.transpose(0, 1, 3, 2)  # (B,N,2,L)
    test_X, test_Y = add_pos_emb('test', test_X, test_Y)  # (B,N,2,L)

    print('train X: {} train M: {} train Y: {}'.format(train_X.shape,
                                                       train_M.shape,
                                                       train_Y.shape))
    print('valid X: {} valid M: {} valid Y: {}'.format(valid_X.shape,
                                                       valid_M.shape,
                                                       valid_Y.shape))
    print('test X: {} test M: {} test Y: {}'.format(test_X.shape, test_M.shape,
                                                    test_Y.shape))

    train_dataset = TensorDataset(torch.from_numpy(train_X),
                                  torch.from_numpy(train_M),
                                  torch.from_numpy(train_Y))

    valid_dataset = TensorDataset(torch.from_numpy(valid_X),
                                  torch.from_numpy(valid_M),
                                  torch.from_numpy(valid_Y))

    test_dataset = TensorDataset(torch.from_numpy(test_X),
                                 torch.from_numpy(test_M),
                                 torch.from_numpy(test_Y))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, mean, std, A


class TransformerLayers(nn.Module):

    def __init__(self, hidden_dim, nlayers, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads,
                                                 hidden_dim * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.reshape(B * N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)

        return output  # [B, N, L, D]


class SpatialBlock(nn.Module):

    def __init__(self, spatial_channels, out_channels, num_nodes):
        super(SpatialBlock, self).__init__()
        self.norm = nn.LayerNorm(num_nodes)
        self.fc = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, A_hat):
        lfs = torch.einsum("ij,jklm->kilm",
                           [A_hat, X.permute(1, 0, 2, 3)])  # B,N,L,D
        t = self.fc(lfs).permute(0, 2, 3, 1)  # B,L,D,N
        t1 = self.norm(t).permute(0, 3, 1, 2)  # B,N,L,D
        return self.dropout(t1)


class LearnableMaskEmb(nn.Module):

    def __init__(self, hidden_dim):
        super(LearnableMaskEmb, self).__init__()
        self.hidden_dim = hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim, 1))
        nn.init.uniform_(self.mask_token, -0.02, 0.02)

    def forward(self, input_emb, m):
        B, N, D, L = input_emb.shape  # B,N,D,L

        mask = m[:, :, :1, :].expand(B, N, D, L)

        observed_token = input_emb * mask

        missed_token = self.mask_token.expand(B, N, D, L) * (1 - mask)

        learnable_input_emb = observed_token + missed_token
        learnable_input_emb = learnable_input_emb.transpose(-1,
                                                            -2)  # (B,N,L,D)
        return learnable_input_emb


class LearnablePositionEmb(nn.Module):

    def __init__(self, hidden_dim, max_len=1000, dropout=0.1):
        super(LearnablePositionEmb, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.pe = nn.Parameter(torch.empty(max_len, hidden_dim),
                               requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, input_emb, position):
        B, N, L, D = input_emb.shape  # B,N,L,D

        # position emb
        pe = self.pe[position.view(B * N, L, -1).long(), :]  # (B*N,L,1,D)
        learnable_pos_emb = input_emb + pe.view(B, N, L, -1)
        learnable_pos_emb = self.dropout(learnable_pos_emb)  # (B,N,L,D)
        return learnable_pos_emb


class MissSTImputer(nn.Module):

    def __init__(self,
                 num_nodes,
                 seqlen,
                 in_channel,
                 hidden_dim,
                 adj_mx,
                 adj_pa,
                 num_heads,
                 num_layers,
                 learnable=True,
                 pe_learnable=True,
                 max_len=1000,
                 dropout=0.1):
        super(MissSTImputer, self).__init__()

        self.num_nodes = num_nodes
        self.seqlen = seqlen
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim
        self.adj_pa = torch.FloatTensor(adj_pa).to(device)
        self.learnable = learnable
        self.pe_learnable = pe_learnable

        self.input_embedding = nn.Conv2d(in_channel,
                                         hidden_dim,
                                         kernel_size=(1, 1),
                                         stride=(1, 1))
        self.encoder = TransformerLayers(hidden_dim, seqlen)
        self.norm = nn.LayerNorm(hidden_dim)
        # self.decoder = TransformerLayers(hidden_dim, 1)
        # self.encoder_2_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.maskemb = LearnableMaskEmb(hidden_dim)
        self.posemb = LearnablePositionEmb(hidden_dim)
        self.spatial = SpatialBlock(spatial_channels=16,
                                    out_channels=hidden_dim,
                                    num_nodes=num_nodes)

        self.STGNNs = make_model(DEVICE=device,
                                 num_of_d=1,
                                 nb_block=4,
                                 in_channels=1,
                                 K=3,
                                 nb_chev_filter=32,
                                 nb_time_filter=32,
                                 time_strides=1,
                                 adj_mx=adj_mx,
                                 adj_pa=adj_pa,
                                 num_for_predict=seqlen,
                                 len_input=seqlen,
                                 num_of_vertices=num_nodes,
                                 d_model=512,
                                 d_k=32,
                                 d_v=32,
                                 n_heads=3)

    def forward(self, x, m, A_hat):
        if self.pe_learnable:
            position = x[:, :, 1, :].unsqueeze(-2)
        x = x[:, :, :self.in_channel, :]
        B, N, F_in, L = x.shape  # B,N,F,L

        input = x.unsqueeze(-1)  # B, N, F, L, 1
        input = input.reshape(B * N, F_in, L, 1)  # B*N, F, L, 1

        # learnable mask emb
        input_emb = self.input_embedding(input)  # B*N,  d, L, 1
        input_emb = input_emb.squeeze(-1).view(B, N, self.hidden_dim,
                                               -1)  # B,N,d,L
        learnablemaskemb = self.maskemb(input_emb, m)  # B,N,L,D

        # learnable position emb
        learnableposemb = self.posemb(learnablemaskemb,
                                      position.view(B * N, L,
                                                    -1).long())  # B,N,L,D

        # # spatial learning
        # H_spa = self.spatial(learnableposemb, A_hat)  # B,N,L,D

        # # temporal attention
        # H_tem = self.encoder(H_spa)  # (B,N,L,D)
        # H_norm = self.norm(H_tem).transpose(-1, -2)  # (B,N,D,L)

        # downstream STGNNs
        output = self.STGNNs(learnableposemb.transpose(-1, -2), m,
                             learnableposemb.transpose(-1, -2))

        return output  # (B,N,1,L)


def main():
    train_loader, valid_loader, test_loader, mean, std, A = generate_miss_loader(
    )

    wandb.init(project="learnable_mask_emb",
               name="{}_lr{}_hiddensize{}_batchsize{}_seed{}".format(
                   dataset, args.lr, args.hidden_size, args.batch_size,
                   args.seed))

    # adj_pa = load_PA('../DSTAGNN-ICML2022/data/{}/strg_001_{}.csv'.format(
    #     args.dataset, args.dataset))
    # adj_TMD = load_weighted_adjacency_matrix(
    #     '../DSTAGNN-ICML2022/data/{}/stag_001_{}.csv'.format(
    #         args.dataset, args.dataset), args.num_nodes)
    adj_mx = weight_matrix(A)
    A_wave = torch.from_numpy(adj_mx).float().to(device)
    adj_pa = adj_mx
    # L_tilde = scaled_Laplacian(adj_TMD)
    # cheb_polynomials = [
    #     torch.from_numpy(i).type(torch.FloatTensor).to(device)
    #     for i in cheb_polynomial(L_tilde, K=3)
    # ]

    AEmodel = MissSTImputer(num_nodes=args.num_nodes,
                            seqlen=args.seqlen,
                            in_channel=1,
                            hidden_dim=args.hidden_size,
                            adj_mx=adj_mx,
                            adj_pa=adj_pa,
                            num_heads=4,
                            num_layers=2,
                            learnable=True).to(device)  # 生成AE模型，并转移到GPU上去
    print('The structure of our model is shown below: \n')
    print(AEmodel)
    loss_function = nn.SmoothL1Loss()  # 生成损失函数
    # loss_function = nn.L1Loss()  # 生成损失函数
    optimizer = optim.Adam(AEmodel.parameters(),
                           lr=args.lr)  # 生成优化器，需要优化的是model的参数，学习率为0.001
    # optimizer = optim.Adam(AEmodel.parameters(), lr=args.lr,
    #                        weight_decay=1e-4)  # 生成优化器，需要优化的是model的参数，学习率为0.001

    # 开始迭代
    patience = 0
    best_val_mae = 999
    for epoch in range(epochs):
        epoch_time = time.time()
        # train
        loss_epoch = []
        AEmodel.train()
        for _, (x, m, y) in enumerate(train_loader):
            x = x.to(device)  # (B,N,2,L)
            m = m.to(device)  # (B,N,2,L)
            y = y.to(device)  # (B,N,2,L)
            # 前向传播
            x_hat = AEmodel(x, m, A_wave)  # 模型的输出，在这里会自动调用model中的forward函数
            loss = loss_function(x_hat, y[:, :, :1, :])  # 计算损失值，即目标函数
            # 后向传播
            optimizer.zero_grad()  # 梯度清零，否则上一步的梯度仍会存在
            loss.backward()  # 后向传播计算梯度，这些梯度会保存在model.parameters里面
            optimizer.step()  # 更新梯度，这一步与上一步主要是根据model.parameters联系起来了
            loss_epoch.append(loss.item())

        # valid
        valid_maes, valid_rmses, valid_mapes = [], [], []
        AEmodel.eval()
        with torch.no_grad():
            for _, (x, m, y) in enumerate(valid_loader):
                x = x.to(device)  # (B,N,2,L)
                m = m.to(device)  # (B,N,2,L)
                y = y[:, :, :1, :].detach().cpu().numpy()
                # 前向传播
                x_hat = AEmodel(x, m, A_wave).detach().cpu().numpy(
                )  # 模型的输出，在这里会自动调用model中的forward函数
                unnorm_x_hat = unnormalization(x_hat, mean, std)
                unnorm_y = unnormalization(y, mean, std)
                mask = m.detach().cpu().numpy()
                mae, rmse, mape = missed_eval_np(unnorm_x_hat, unnorm_y, mask)
                valid_maes.append(mae)
                valid_rmses.append(rmse)
                valid_mapes.append(mape)
            valid_mae = np.mean(valid_maes)
            valid_rmse = np.mean(valid_rmses)
            valid_mape = np.mean(valid_mapes)
            if valid_mae < best_val_mae:
                patience = 0
                best_val_mae = valid_mae
                if not os.path.exists(args.save_path + '{}'.format(dataset)):
                    os.makedirs(args.save_path + '{}'.format(dataset))
                best_save_path = args.save_path + '{}'.format(
                    dataset) + '/best_model_ms{}_seed{}.pth'.format(
                        args.miss_ratio, args.seed)
                torch.save(AEmodel.state_dict(), best_save_path)
            # else:
            #     patience += 1
            #     if patience > 10:
            #         print("Early Stop!")
            #         break
        wandb.log({"train loss": loss.item(), "valid loss": valid_mae})
        print('Epoch [{}/{}] : '.format(epoch, epochs), 'loss = ',
              np.mean(loss_epoch),
              "epoch_time: {}".format(time.time() -
                                      epoch_time))  # loss是Tensor类型
    wandb.finish()

    predict(AEmodel, best_save_path, test_loader, mean, std, A_wave)


def predict(model, best_save_path, test_loader, mean, std, A_wave):
    model.load_state_dict(torch.load(best_save_path))
    # test
    test_maes, test_rmses, test_mapes = [], [], []
    model.eval()
    miss_data = []
    predict_results = []
    groundtruths = []
    with torch.no_grad():
        for _, (x, m, y) in enumerate(test_loader):
            x = x.to(device)  # (B,N,2,L)
            m = m.to(device)  # (B,N,2,L)
            y = y[:, :, :1, :].detach().cpu().numpy()
            # 前向传播
            x_hat = model(x, m, A_wave).detach().cpu().numpy(
            )  # 模型的输出，在这里会自动调用model中的forward函数
            unnorm_x = unnormalization(x[:, :, :1, :].detach().cpu().numpy(),
                                       mean, std)
            unnorm_x_hat = unnormalization(x_hat, mean, std)
            unnorm_y = unnormalization(y, mean, std)
            mask = m.detach().cpu().numpy()
            mae, rmse, mape = missed_eval_np(unnorm_x_hat, unnorm_y, mask)
            predict_data = unnorm_x_hat * (
                1 - mask[:, :, :1, :]) + unnorm_x * mask[:, :, :1, :]
            unnorm_x = np.where(mask[:, :, :1, :] == 0, np.nan, unnorm_x)
            miss_data.append(unnorm_x)
            predict_results.append(predict_data)
            groundtruths.append(unnorm_y)
            test_maes.append(mae)
            test_rmses.append(rmse)
            test_mapes.append(mape)
        test_mae = np.mean(test_maes)
        test_rmse = np.mean(test_rmses)
        test_mape = np.mean(test_mapes)
    print("Test result: MAE {} RMSE {} MAPE {}".format(test_mae, test_rmse,
                                                       test_mape * 100))
    result = {}
    result['missed_data'] = np.concatenate(miss_data, axis=0)  # B,N,1,L
    result['imputed_data'] = np.concatenate(predict_results, axis=0)  # B,N,1,L
    result['groundtruth'] = np.concatenate(groundtruths, axis=0)  # B,N,1,L
    print(result['missed_data'].shape)
    print(result['imputed_data'].shape)
    print(result['groundtruth'].shape)
    result_path = args.result_path + '{}/'.format(args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(
            result_path +
            'result_ms{}_seed{}.pkl'.format(args.miss_ratio, args.seed),
            'wb') as fb:
        pk.dump(result, fb)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print('Spend Time: {}'.format(time.time() - start_time))
