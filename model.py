'''

'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,GCNConv, GATConv, global_max_pool as gmp

class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=32, output_dim=128,
                 dropout=0.2):
        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output

        self.conv1 = GATConv(32, 32)
        self.conv2 = GATConv(32, 64)
        self.conv3 = GATConv(64, 128)
        self.fc_g1 = torch.nn.Linear(128,1024)
        self.fc_g2 = torch.nn.Linear(1024,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #self.l1 = torch.nn.Linear(128+2048, 512)
        self.l1 = torch.nn.Linear(128, 512)
        self.l2 = torch.nn.Linear(512, 1024)
        self.l3 = torch.nn.Linear(1024, 1)



    def forward(self,data):
        # get graph

        x, edge_index, fp, batch = data.x, data.edge_index, data.fp, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        #
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling

        #flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = torch.sigmoid(x)

        t = torch.cat((x,fp),dim=1)
        t = self.l1(x)
        t = self.relu(t)
        t = self.l2(t)
        t = self.relu(t)
        t = self.l3(t)
        t = self.relu(t)
        t = torch.sigmoid(t)

        out = t

        return out
