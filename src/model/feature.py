from typing import Optional
from jittor import nn

import jittor as jt

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def execute(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # (B, H, N, D)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        attn = jt.matmul(q, k.transpose(2, 3)) * self.scale
        attn = jt.nn.softmax(attn, dim=-1)
        out = jt.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__()
        
        if activation == 'ReLU':
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU()
            )
            self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise Exception("Please assign valid activation to MLP!")
    
    def execute(self, x, edge_index):
        """
        x: (N, C)
        edge_index: (2, E)
        """
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        
        # gather
        x_i = x[dst]  # (E, C)
        x_j = x[src]  # (E, C)
        
        # message
        tmp = jt.concat([x_i, x_j - x_i], dim=1)  # (E, 2C)
        msg = self.mlp(tmp)  # (E, out_channels)
        
        N = x.shape[0]
        out = jt.full((N, msg.shape[1]), 0)
        cnt = jt.full((N, msg.shape[1]), 0)
        
        # scatter mean
        out = out.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), msg, reduce='add')
        cnt = cnt.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), jt.ones_like(msg), reduce='add')
        out = out / (cnt + 1)
        out_2 = self.lin(x)
        return out + out_2

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__(in_channels, out_channels, activation)
    
    def execute(self, x, edge_index):
        return super().execute(x, edge_index)

class FeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=0,
        embedding_dim=512,
        distance_estimation=False,
        num_layers=3,
        use_residual=True,
        use_attention=False,
        attention_heads=4,
        use_global=False,
    ):
        super().__init__()

        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.distance_estimation = distance_estimation
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_global = use_global
        
        hidden_dims = [embedding_dim // 8, embedding_dim // 4, embedding_dim // 2, embedding_dim]
        hidden_dims = hidden_dims[:max(1, num_layers)]
        self.convs = nn.ModuleList()
        in_dim = self.input_dim
        for i, out_dim in enumerate(hidden_dims):
            activation = None if i == len(hidden_dims) - 1 else 'ReLU'
            self.convs.append(DynamicEdgeConv(in_dim, out_dim, activation=activation))
            in_dim = out_dim
        self.fuse = nn.Linear(sum(hidden_dims), embedding_dim)
        if self.use_global:
            self.global_proj = nn.Linear(embedding_dim * 3, embedding_dim)
        if self.use_attention:
            self.attention = SelfAttention(embedding_dim, num_heads=attention_heads)

    # ========= edge_index 构建 =========
    def get_edge_index(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape
        knn_idx = get_knn_idx(x, x, self.k + 1)  # (B, N, k+1)
        knn_idx = knn_idx[:, :, 1:]
        base = jt.arange(B) * N  # (B,)
        base = base.reshape(B, 1, 1)
        
        knn_idx = knn_idx + base  # (B, N, k)
        
        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, self.k))
        dst = dst + base
        
        src = knn_idx.reshape(-1)
        dst = dst.reshape(-1)
        
        edge_index = jt.stack([src, dst], dim=0)  # (2, E)
        
        return edge_index
    
    def normalize_patch(self, pcl):
        scale = jt.sqrt((pcl ** 2).sum(-1, keepdims=True))
        scale = scale.max(dim=-2, keepdims=True)
        return pcl / (scale + 1e-8) # type: ignore
    
    def execute(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape
        
        if self.distance_estimation:
            x = self.normalize_patch(x)
        
        features = []
        x_in = x
        for conv in self.convs:
            edge_index = self.get_edge_index(x_in)
            x_flat = x_in.reshape(B * N, -1)
            x_out = conv(x_flat, edge_index)
            x_out = x_out.reshape(B, N, -1)
            if self.use_residual and x_out.shape[-1] == x_in.shape[-1]:
                x_out = x_out + x_in
            features.append(x_out)
            x_in = x_out
        x_cat = jt.concat(features, dim=-1)
        x_fused = self.fuse(x_cat.reshape(B * N, -1)).reshape(B, N, -1)
        if self.use_global:
            global_feat = jt.concat([jt.max(x_fused, dim=1), jt.mean(x_fused, dim=1)], dim=-1)
            global_feat = global_feat.unsqueeze(1).broadcast((B, N, global_feat.shape[-1]))
            x_fused = self.global_proj(jt.concat([x_fused, global_feat], dim=-1))
        if self.use_attention:
            x_fused = x_fused + self.attention(x_fused)
        return x_fused

class Decoder(nn.Module):
    
    def __init__(self, z_dim, dim, out_dim, hidden_size):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        c_dim = z_dim
        self.lin_1 = nn.Linear(c_dim, c_dim)
        self.bn_1_out = nn.BatchNorm1d(c_dim)
        
        self.lin_2 = nn.Linear(c_dim, hidden_size)
        self.bn_2_out = nn.BatchNorm1d(hidden_size)
        
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        
        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def execute(self, c, B=None, N=None):
        """
        c: (B*N, F)
        """
        net = self.lin_1(c)
        net = self.bn_1_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        net = self.lin_2(net)
        net = self.bn_2_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        if self.out_dim == 1:
            net = net.reshape(B, N, -1)
            net = jt.max(net, dim=1, keepdims=True)
            net = self.lin_3(net)
            net = jt.sigmoid(net)
        else:
            net = self.lin_3(net)
        return net

def get_knn_idx(x, y, k, offset=0):
    """
    x: (B, N, d)
    y: (B, M, d)
    return: (B, N, k)
    """
    K = k + offset
    if x.shape[-1] == 3:
        _, idx = jt.misc.knn(x, y, K)
    else:
        dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
        _, idx = jt.topk(dist, k=K, dim=-1, largest=False)
    return idx[:, :, offset:]