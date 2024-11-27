import os
import time
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.nn as nn
from pretrain_model import ppi_model
from utils import Metrictor_PPI, print_file
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import datetime

parser = argparse.ArgumentParser(description='Pre-training')

parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default='./protein_info/vec5_CTC.txt', type=str,
                    help='protein sequence vector path')
parser.add_argument('--p_feat_matrix', default=None, type=str,
                    help="protein feature matrix")
parser.add_argument('--p_adj_matrix', default=None, type=str,
                    help="protein adjacency matrix")
parser.add_argument('--split', default=None, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--save_path', default=None, type=str,
                    help="save folder")
parser.add_argument('--epoch_num', default=None, type=int,
                    help='train epoch number')
seed_num = 2
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)


def multi2big_x(x_ori, node_num):
    x_cat = torch.zeros(1, 7)
    x_num_index = torch.zeros(node_num)
    for i in range(node_num):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index


def multi2big_batch(x_num_index, node_num):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1, node_num):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch


def multi2big_edge(edge_ori, num_index, node_num):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(node_num)
    for i in range(node_num):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.device = device
        self.loss_computer = FlatNCE(0.1)

    def forward(self, z_i, z_j):
        logits, labels  = self.loss_computer(z_i, z_j)
        return logits, labels


class FlatNCE(nn.Module):
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        features = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        logits = (negatives - positives) / self.temperature
        labels = torch.zeros(positives.shape[0], dtype=torch.long).to(device)
        clogits = torch.logsumexp(logits, dim=1, keepdim=True)
        return logits, labels

def train(batch, tensor_data, p_x_all, p_edge_all, edge_num_index, model, contrastive_loss, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=512, epochs=1000, scheduler=None):
    global_step = 0
    global_best_loss = 9999.0
    criterion = torch.nn.CrossEntropyLoss().to(device)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f'./pretrain_opt/model_best_{current_time}.pth'
    for epoch in range(epochs):
        loss_sum = 0.0
        model.train()
        seq, stru = model(batch, p_x_all, p_edge_all, edge_num_index, tensor_data)

        logits, labels  = contrastive_loss(seq, stru)

        v = torch.logsumexp(logits, dim=1, keepdim=True)
        loss_vec = torch.exp(v - v.detach()).to(device)
        assert loss_vec.shape == (len(logits), 1)
        dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).to(device), logits], 1)
        loss = loss_vec.mean() - 1 + criterion(dummy_logits, labels).detach()
        if loss < 0:
            return

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        summary_writer.add_scalar('train/loss', loss.item(), global_step)

        global_step += 1
        print_file("epoch: {}, step: {}, Train: label_loss: {},best loss :{}".format(epoch, 999, loss.item()
                                                                                     , global_best_loss))



        if global_best_loss > loss.item():
            global_best_loss = loss.item()
            torch.save(model.state_dict(), model_save_path)

def embed_normal(self, seq, dim):
    if len(seq) > self.max_len:
        return seq[:self.max_len]
    elif len(seq) < self.max_len:
        less_len = self.max_len - len(seq)
        return np.concatenate((seq, np.zeros((less_len, dim))))
    return seq

def main():
    args = parser.parse_args()

    pseq_path = args.pseq_path
    found_proteins_file = args.pseq_path
    vec_path = args.vec_path
    pseq_dict = {}
    protein_len = []
    with open(found_proteins_file, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) > 1:
                pseq_dict[parts[0]] = parts[1]
                protein_len.append(len(parts[1]))

    acid2vec = {}
    dim = None
    with open(vec_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                temp = np.array([float(x) for x in parts[1].split()])
                acid2vec[parts[0]] = temp
                if dim is None:
                    dim = len(temp)
    print(f"acid vector dimension: {dim}")

    def embed_normal(temp_vec, dim):
        if temp_vec.shape[0] < 2000:
            padding = np.zeros((2000 - temp_vec.shape[0], dim))
            temp_vec = np.vstack((temp_vec, padding))
        elif temp_vec.shape[0] > 2000:
            temp_vec = temp_vec[:2000, :]
        return temp_vec

    pvec_dict = {}
    for p_name in tqdm(pseq_dict.keys()):
        temp_seq = pseq_dict[p_name]
        temp_vec = []
        for acid in temp_seq:
            if acid in acid2vec:
                temp_vec.append(acid2vec[acid])
            else:
                temp_vec.append(np.zeros(dim))
        temp_vec = np.array(temp_vec)

        temp_vec = embed_normal(temp_vec, dim)

        pvec_dict[p_name] = temp_vec

    n = len(pvec_dict)
    tensor_data = np.zeros((n, 2000, dim))
    for i, (p_name, vec) in enumerate(pvec_dict.items()):
        tensor_data[i] = vec

    tensor_data = torch.tensor(tensor_data)
    p_x_all = torch.load(args.p_feat_matrix)

    p_edge_all = np.load(args.p_adj_matrix, allow_pickle=True)

    p_x_all, x_num_index = multi2big_x(p_x_all, n)

    p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index, n)

    batch = multi2big_batch(x_num_index, n) + 1



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = ppi_model()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           verbose=True)
    save_path = args.save_path
    contrastive_loss = ContrastiveLoss().to(device)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    time_stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    save_path = os.path.join(save_path, "gnn_{}".format('training_seed_1' + time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    print(save_path)
    os.mkdir(save_path)

    summary_writer = SummaryWriter(save_path)

    train(batch, tensor_data, p_x_all, p_edge_all, edge_num_index, model, contrastive_loss, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=1024, epochs=args.epoch_num, scheduler=scheduler)

    summary_writer.close()


if __name__ == "__main__":
    main()
