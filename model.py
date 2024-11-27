import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_dim=13, output_dim=128):
        super(ConvFeatureExtractor, self).__init__()
        self.conv1d_1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self._calculate_conv_output_dim(input_dim)

        self.fc_input_dim = self.conv_output_dim * (2000 // 2 // 2)
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def _calculate_conv_output_dim(self, input_dim):
        input_tensor = torch.randn(1, input_dim, 2000)
        x = F.relu(self.conv1d_1(input_tensor))
        x = self.pool(x)
        x = F.relu(self.conv1d_2(x))
        x = self.pool(x)
        self.conv_output_dim = x.size(1)

    def forward(self, protein_vec):
        x = protein_vec.permute(0, 2, 1)
        x = F.relu(self.conv1d_1(x))
        x = self.pool(x)
        x = F.relu(self.conv1d_2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class GIN(torch.nn.Module):
    def __init__(self, hidden=512, train_eps=True, class_num=7):
        super(GIN, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(128, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(128, hidden)
        self.fc1 = nn.Linear(2 * hidden, 7)  # clasifier for concat
        self.fc2 = nn.Linear(hidden, 7)  # classifier for inner product

    def reset_parameters(self):
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        hidden = 128
        self.conv1 = GCNConv(7, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden, 0.5)
        self.sag2 = SAGPooling(hidden, 0.5)
        self.sag3 = SAGPooling(hidden, 0.5)
        self.sag4 = SAGPooling(hidden, 0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        # x = self.dropout(x)
        y = self.sag1(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        # x = self.dropout(x)
        y = self.sag2(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.dropout(x)
        y = self.sag3(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.dropout(x)
        y = self.sag4(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        return global_mean_pool(y[0], y[3])



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class mfc(nn.Module):
    def __init__(self):
        super(mfc, self).__init__()
        # Structure Feature Extractor
        self.SFE = GCN()
        # PPI Feature Extractor
        self.PFE = GIN()
        # Sequence Feature Extractor
        self.CFE = ConvFeatureExtractor()
        # self.ATT = AttentionFusion(input_size=128) 1:0.6
        self.seq_weight = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.stru_weight = nn.Parameter(torch.Tensor([0.6]), requires_grad=False)

        # 1:0.5
        self.ppi_weight = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.mlp_weight = nn.Parameter(torch.Tensor([0.5]), requires_grad=False)

        self.fc = nn.Linear(512, 7)
        self.fe_mlp = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                    nn.Dropout(), nn.BatchNorm1d(256))



    def forward(self, batch, p_x_all, p_edge_all, edge_num_index, edge_index, train_edge_id, protein_vec, p=0.5):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        edge_num_index = edge_num_index.to(torch.int64).to(device)


        protein_vec = protein_vec.to(torch.float32).to(device)

        batch_size = 256
        num_batches = len(edge_num_index) // batch_size
        if len(edge_num_index) % batch_size != 0:
            num_batches += 1

        output_features = []
        output_features_seq = []

        start_edge_indices = 0

        x_batch_all = 0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(batch))

            batch_indices = torch.where((batch > start_idx) & (batch <= end_idx))[0]

            batch_batch = batch[batch_indices]

            x_batch = x[batch_indices]

            batch_batch = batch_batch - i * batch_size

            edge_indices = torch.sum(edge_num_index[start_idx:end_idx]).item()

            edge_batch = edge[:, start_edge_indices:(edge_indices + start_edge_indices)]

            edge_batch = edge_batch - x_batch_all

            x_batch_all = x_batch_all + len(x_batch)

            start_edge_indices = edge_indices + start_edge_indices

            output_batch = self.SFE(x_batch, edge_batch, batch_batch - 1)

            seq_batch = protein_vec[start_idx:end_idx, :, :]

            seq_embs_batch = self.CFE(seq_batch)

            output_features.append(output_batch)

            output_features_seq.append(seq_embs_batch)

            # torch.cuda.empty_cache()


        struct_embs = torch.cat(output_features, dim=0)

        seq_embs = torch.cat(output_features_seq, dim=0)



        unit_embs = self.seq_weight * seq_embs + self.stru_weight * struct_embs


        seq_embs = self.fe_mlp(seq_embs)
        struct_embs = self.fe_mlp(struct_embs)

        mlp_feature = torch.cat([self.seq_weight * seq_embs, self.stru_weight * struct_embs], dim=1)


        graph_feature = self.PFE(unit_embs, edge_index)


        feature = self.mlp_weight * mlp_feature + self.ppi_weight * graph_feature

        node_id = edge_index[:, train_edge_id]
        x1 = feature[node_id[0]]
        x2 = feature[node_id[1]]
        feature = torch.mul(x1, x2)
        feature = self.fc(feature)

        return feature
