"""
Graph Convolutional Network (GCN)
Learn representations on graph-structured data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer

    H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

    where:
    - A: Adjacency matrix (with self-loops)
    - D: Degree matrix
    - H: Node features
    - W: Learnable weights
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        Args:
            x: (N, in_features) - node features
            adj: (N, N) - normalized adjacency matrix

        Returns:
            output: (N, out_features)
        """
        support = torch.mm(x, self.weight)  # (N, out_features)
        output = torch.spmm(adj, support)   # (N, out_features)

        if self.bias is not None:
            output = output + self.bias

        return output


class GCN(nn.Module):
    """
    Graph Convolutional Network

    Architecture:
        Input Features
          ↓
        GCN Layer 1 + ReLU + Dropout
          ↓
        GCN Layer 2 + ReLU + Dropout
          ↓
        ...
          ↓
        GCN Layer N
          ↓
        Output (node embeddings / classifications)
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=2):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # First layer
        self.gc1 = GraphConvolution(nfeat, nhid)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            GraphConvolution(nhid, nhid)
            for _ in range(num_layers - 2)
        ]) if num_layers > 2 else nn.ModuleList()

        # Output layer
        self.gc_out = GraphConvolution(nhid, nclass)

    def forward(self, x, adj):
        """
        Args:
            x: (N, nfeat) - node features
            adj: (N, N) - normalized adjacency matrix

        Returns:
            output: (N, nclass) - node predictions/embeddings
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        for layer in self.hidden_layers:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc_out(x, adj)

        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """
    Graph Attention Network

    Use attention mechanism instead of fixed adjacency
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        super().__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ])

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention Layer"""

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return self.leakyrelu(torch.matmul(all_combinations_matrix, self.a).view(N, N))


def normalize_adj(adj):
    """
    Normalize adjacency matrix: D^(-1/2) A D^(-1/2)

    Args:
        adj: (N, N) - adjacency matrix

    Returns:
        norm_adj: (N, N) - normalized adjacency
    """
    # Add self-loops
    adj = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute D^(-1/2)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    # Normalize: D^(-1/2) A D^(-1/2)
    norm_adj = adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)

    return norm_adj


def create_synthetic_graph(num_nodes=1000, num_edges=5000, num_classes=7, feat_dim=1433):
    """Create synthetic graph data for testing"""
    # Random features
    features = torch.randn(num_nodes, feat_dim)

    # Random edges
    edge_list = torch.randint(0, num_nodes, (2, num_edges))
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_list[0], edge_list[1]] = 1
    adj[edge_list[1], edge_list[0]] = 1  # Undirected

    # Normalize
    adj = normalize_adj(adj)

    # Random labels
    labels = torch.randint(0, num_classes, (num_nodes,))

    # Train/val/test split
    idx_train = torch.arange(0, int(0.6 * num_nodes))
    idx_val = torch.arange(int(0.6 * num_nodes), int(0.8 * num_nodes))
    idx_test = torch.arange(int(0.8 * num_nodes), num_nodes)

    return features, adj, labels, idx_train, idx_val, idx_test


if __name__ == "__main__":
    print("=" * 60)
    print("Graph Convolutional Network (GCN)")
    print("=" * 60)

    # Create synthetic graph
    features, adj, labels, idx_train, idx_val, idx_test = create_synthetic_graph(
        num_nodes=1000, num_edges=5000, num_classes=7, feat_dim=1433
    )

    print(f"\nSynthetic Graph:")
    print(f"Nodes: {features.size(0)}")
    print(f"Features per node: {features.size(1)}")
    print(f"Edges: {(adj > 0).sum() // 2}")
    print(f"Classes: {labels.max().item() + 1}")
    print(f"Train/Val/Test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    # Create model
    model = GCN(nfeat=features.size(1), nhid=16, nclass=labels.max().item() + 1, dropout=0.5)

    print(f"\nGCN Model:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Forward pass
    output = model(features, adj)
    print(f"Output shape: {output.shape}")

    # Training loop
    print("\n" + "=" * 60)
    print("Training GCN")
    print("=" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = (output[idx_train].argmax(1) == labels[idx_train]).float().mean()
        loss_train.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                output = model(features, adj)
                loss_val = criterion(output[idx_val], labels[idx_val])
                acc_val = (output[idx_val].argmax(1) == labels[idx_val]).float().mean()

            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {loss_train:.4f} Acc: {acc_train:.4f} | "
                  f"Val Loss: {loss_val:.4f} Acc: {acc_val:.4f}")
            model.train()

    # Test
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test = (output[idx_test].argmax(1) == labels[idx_test]).float().mean()

    print(f"\nTest Loss: {loss_test:.4f} | Test Accuracy: {acc_test:.4f}")

    print("\n" + "=" * 60)
    print("GCN Complete!")
    print("=" * 60)
    print("\nUse Cases:")
    print("✓ Node classification (e.g., classify users in social network)")
    print("✓ Link prediction (predict edges)")
    print("✓ Graph classification (classify entire graphs)")
    print("✓ Community detection")
    print("✓ Recommendation systems")
