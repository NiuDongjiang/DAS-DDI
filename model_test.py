import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (GATConv,
                                TransformerConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Linear,
                                )
from layers import (
                    CoAttentionLayer,
                    RESCAL,
                    IntraGraphAttention,
                    InterGraphAttention,
                    MergeFD,
                    )
import time
import math
from torch.nn.parameter import Parameter
import numpy as np
from scipy import sparse as sp
import dgl.function as fn
from torch_geometric.utils import degree,softmax

#Constructing k-nearest neighbor graph
def knn_graph(disMat, k):
    k_neighbor = np.argpartition(-disMat, kth=k, axis=1)[:, :k]
    row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
    col_index = k_neighbor.reshape(-1)
    edges = np.array([row_index, col_index]).astype(int).T
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(disMat.shape[0], disMat.shape[0]),
                        dtype=np.float32)
    # Remove diagonal elements
    # drug_adj = drug_adj - sp.dia_matrix((drug_adj.diagonal()[np.newaxis, :], [0]), shape=drug_adj.shape)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def generate_feat_graph(similarity,num_neighbor):
    # drug feature graph
    '''
    Generate feature map
    Parameters: similarity: similarity matrix
        similarity: similarity matrix
        num_neighbor: number of neighbors
    Returns.
        Torch sparse tensor representation of the feature map.
    '''

    drug_sim = similarity
    drug_num_neighbor = num_neighbor
    if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
        drug_num_neighbor = drug_sim.shape[0]

    drug_adj = knn_graph(drug_sim, drug_num_neighbor)
    drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
    drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)

    return drug_graph
#Calculate the dot product of the source and target nodes and store it in the attributes of the edges
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func

#Scaling the properties of an edge
def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func

#Calculation of indices for numerical stability of softmax calculations
def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


class GraphConvolution1(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight.double())
        output = torch.spmm(adj, support.float())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GraphConvolution2(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
#Build GCN model
class GCN(nn.Module):
    def __init__(self, features, nhid, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution1(features, nhid)
        self.gc2 = GraphConvolution2(nhid, nhid2)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

#intra-view in DA encoder
class FGCN(nn.Module):
    def __init__(self, fdim_drug, nhid1, nhid2, dropout):
        super(FGCN, self).__init__()
        self.FGCN1 = GCN(fdim_drug, nhid1, nhid2, dropout)

        self.dropout = dropout

    def forward(self, drug_graph, drug_sim_feat):
        emb1 = self.FGCN1(drug_sim_feat, drug_graph)

        return emb1
class GCN1(nn.Module):
    def __init__(self, features, nhid, nhid2, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution2(features, nhid)
        self.gc2 = GraphConvolution2(nhid, nhid2)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
class FGCN1(nn.Module):
    def __init__(self, fdim_drug, nhid1, nhid2, dropout):
        super(FGCN1, self).__init__()
        self.FGCN2 = GCN1(fdim_drug, nhid1, nhid2, dropout)

        self.dropout = dropout

    def forward(self, drug_graph, drug_sim_feat):
        emb1 = self.FGCN2(drug_sim_feat, drug_graph)

        return emb1

#inter-view in DA encoder
class InteractionEmbedding(nn.Module):
    def __init__(self, n_drug1, n_drug2, embedding_dim, dropout=0.5):
        super(InteractionEmbedding, self).__init__()
        self.drug_project1 = nn.Linear(n_drug1, embedding_dim, bias=False)
        self.drug_project2 = nn.Linear(n_drug2, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = embedding_dim

    def forward(self, association_pairs, drug_embedding1, drug_embedding2):
        drug_embedding1 = torch.diag(torch.ones(drug_embedding1.shape[0], device=drug_embedding1.device))
        drug_embedding2 = torch.diag(torch.ones(drug_embedding2.shape[0], device=drug_embedding2.device))

        drug_embedding1 = self.drug_project1(drug_embedding1)
        drug_embedding2 = self.drug_project2(drug_embedding2)

        drug_embedding1 = F.embedding(association_pairs[0,:], drug_embedding1)
        drug_embedding2 = F.embedding(association_pairs[1,:], drug_embedding2)

        associations = drug_embedding1*drug_embedding2

        associations = F.normalize(associations)
        associations = self.dropout(associations)
        return associations

#attention mechanism
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):

        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))

        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
        e_out = g.edata['e_out']

        return h_out, e_out

#DS encoder layer
class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e):

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h,e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)

#Build DAS-DDI model
class MVN_DDI(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params, edge_feature,dp):
        super().__init__()
        self.in_node_features = in_node_features[0]
        self.in_node_features_fp = in_node_features[1]
        self.in_node_features_desc = in_node_features[2]
        self.in_edge_features = in_edge_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.n_blocks = len(blocks_params)
        self.initial_node_feature = Linear(self.in_node_features, self.hidd_dim ,bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, edge_feature ,bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.hidd_dim)

        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(self.hidd_dim, n_heads, head_out_feats, edge_feature, dp)
            # block = DeeperGCN(self.hidd_dim, n_heads, head_out_feats)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
        self.interaction_encoder = InteractionEmbedding(n_drug=1706, n_disease=1706,
                                                        embedding_dim=128, dropout=0.4)
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)
        self.fdmer = MergeFD(self.in_node_features_fp, self.in_node_features_desc, self.kge_dim)
        # self.fdmer = MergeFD_trans(self.in_node_features_fp, self.in_node_features_desc, self.kge_dim, "Transformer")
        self.merge_all = nn.Sequential(nn.Linear(self.kge_dim * 2, self.kge_dim),
                                   nn.ReLU())
        self.num_neighbor = 4
        self.FGCN = FGCN(1706,
                         1024,
                         128,
                         0.3)

        self.attention = Attention(128)
    def forward(self, triples):
        h_data,h_dgl, t_data,t_dgl, rels,b_graph, h_data_edge, t_data_edge, similarity, similarity_graph,association,association_graph = triples
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels = triples

        #DA encoder
        drug_sim_out = self.FGCN(similarity_graph, similarity)
        #drug_ass_out = self.FGCN1(association_graph,association.float())
        drug_pairs_h = torch.stack([h_data.id,t_data.id],0)
        interaction_embedding_h = self.interaction_encoder(drug_pairs_h, similarity, similarity.T)
        drug_pairs_t = torch.stack([t_data.id, h_data.id], 0)
        interaction_embedding_t = self.interaction_encoder(drug_pairs_t, similarity.T, similarity.T)

        # 线性变换 55-64/128
        h_data.x = self.initial_node_feature(h_data.x)
        t_data.x = self.initial_node_feature(t_data.x)
        h_data.x = self.initial_node_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_node_norm(t_data.x, t_data.batch)
        h_data.x = F.elu(h_data.x)
        t_data.x = F.elu(t_data.x)

        h_data.edge_attr = self.initial_edge_feature(h_data.edge_attr)
        t_data.edge_attr = self.initial_edge_feature(t_data.edge_attr)
        h_data.edge_attr = F.elu(h_data.edge_attr)
        t_data.edge_attr = F.elu(t_data.edge_attr)
        h_sim = drug_sim_out[h_data.id]
        t_sim = drug_sim_out[t_data.id]
        h_net = torch.stack([interaction_embedding_h, h_sim], dim=1)
        t_net = torch.stack([interaction_embedding_t, t_sim], dim=1)
        h_net, h_att = self.attention(h_net)
        t_net, t_att = self.attention(t_net)

        # DS encoder
        repr_h = []
        repr_t = []
        for i, block in enumerate(self.blocks):
            out = block(h_data,h_dgl, t_data, t_dgl, h_data_edge, t_data_edge,b_graph)
            # out = block(h_data,t_data,h_data_desc,t_data_desc)
            h_data = out[0]
            t_data = out[1]
            h_global_graph_emb = out[2]
            t_global_graph_emb = out[3]
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)

        #_, _, h_data_fin, h_data_desc, t_data_fin, t_data_desc = self.fdmer(h_data_fin,h_data_desc,t_data_fin,t_data_desc)
        repr_h_fd = []
        repr_t_fd = []
        for i in range(len(self.blocks)):
            h = torch.stack([repr_h[i], h_net], dim=1)
            t = torch.stack([repr_t[i], t_net], dim=1)
            h, h_a = self.attention(h)
            t, t_a = self.attention(t)
            repr_h_fd.append(F.normalize(h))
            repr_t_fd.append(F.normalize(t))

        repr_h = torch.stack((repr_h_fd), dim=-2)
        repr_t = torch.stack((repr_t_fd), dim=-2)
        kge_heads = repr_h  # 1024,4,128
        kge_tails = repr_t  # 1024,4,128
        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores


class MVN_DDI_Block(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dp):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.feature_conv = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.lin_up = Linear(128, 128, bias=True, weight_initializer='glorot')
        self.feature_conv3 = GraphTransformerLayer(128, 128, 8, 0.0, True, False, True)
        self.feature_conv2 = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.feature_conv4 = GATConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature, dropout=dp)
        self.lin_up2 = Linear(128, 128, bias=True, weight_initializer='glorot')
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads, dp, n_heads, edge_feature, head_out_feats)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads, dp, n_heads, edge_feature, head_out_feats)

        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.re_shape = Linear(64 + 128, 128, bias=True, weight_initializer='glorot')
        self.norm = LayerNorm(n_heads * head_out_feats)
        self.norm2 = LayerNorm(n_heads * head_out_feats)
        self.attention = Attention(128)
        self.re_shape_e = Linear(128, 64, bias=True, weight_initializer='glorot')
        self.w_j = Linear(128, 128, bias=True, weight_initializer='glorot')
        self.w_i = Linear(128, 128, bias=True, weight_initializer='glorot')

        self.prj_j = Linear(128, 128, bias=True, weight_initializer='glorot')
        self.prj_i = Linear(128, 128, bias=True, weight_initializer='glorot')

    def forward(self, h_data,h_dgl,t_data,t_dgl,h_data_edge, t_data_edge,b_graph):
        # node_update
        h_data, t_data = self.ne_update(h_data,h_dgl, t_data, t_dgl,h_data_edge,t_data_edge,b_graph)

        # global
        #h_data.x = self.feature_conv2(h_data.x, h_data.edge_index, h_data.edge_attr)
        #t_data.x = self.feature_conv2(t_data.x, t_data.edge_index, t_data.edge_attr)
        #h_data.edge_attr = self.lin_up2(h_data.edge_attr)
        #t_data.edge_attr = self.lin_up2(t_data.edge_attr)

        h_global_graph_emb, t_global_graph_emb = self.GlobalPool(h_data,h_dgl, t_data,t_dgl, h_data_edge, t_data_edge)

        # node_shortcut
        h_data.x = F.elu(self.norm2(h_data.x, h_data.batch))
        t_data.x = F.elu(self.norm2(t_data.x, t_data.batch))

        h_data.edge_attr = F.elu(h_data.edge_attr)
        t_data.edge_attr = F.elu(t_data.edge_attr)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb

    def ne_update(self, h_data,h_dgl, t_data,t_dgl,h_data_edge,t_data_edge,b_graph):
        h, e_h = self.feature_conv3(h_dgl, h_data.x, h_data.edge_attr)
        t, e_t = self.feature_conv3(t_dgl, t_data.x, t_data.edge_attr)
        #h_data.x = self.feature_conv4(h_data.x, h_data.edge_index, h_data.edge_attr)
        #t_data.x = self.feature_conv4(t_data.x, t_data.edge_index, t_data.edge_attr)
        h_data.x = F.elu(self.norm(h, h_data.batch))
        t_data.x = F.elu(self.norm(t, t_data.batch))

        h_data.edge_attr = F.elu(self.norm(e_h,h_data_edge.batch))
        t_data.edge_attr = F.elu(self.norm(e_t,t_data_edge.batch))

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep

        return h_data, t_data

    def GlobalPool(self, h_data,h_dgl, t_data,t_dgl,h_data_edge, t_data_edge):
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readout(h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readout(t_data.x, t_data.edge_index, edge_attr=t_data.edge_attr, batch=t_data.batch)
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        g_h_align = h_global_graph_emb.repeat_interleave(degree(t_data.batch, dtype=t_data.batch.dtype), dim=0)
        g_t_align = t_global_graph_emb.repeat_interleave(degree(h_data.batch, dtype=h_data.batch.dtype), dim=0)

        # Equation (7)
        h_scores = (self.w_i(h_att_x) * self.prj_i(g_t_align)).sum(-1)
        h_scores = softmax(h_scores, h_data.batch, dim=0)
        # Equation (7)
        t_scores = (self.w_j(t_att_x) * self.prj_j(g_h_align)).sum(-1)
        t_scores = softmax(t_scores, t_data.batch, dim=0)

        h_global_graph_emb = global_add_pool(h_att_x * g_t_align * h_scores.unsqueeze(-1), h_data.batch)
        t_global_graph_emb = global_add_pool(t_att_x * g_h_align * t_scores.unsqueeze(-1), t_data.batch)

        h_data_edge.x = h_data.edge_attr
        t_data_edge.x = t_data.edge_attr
        h_global_graph_emb_edge = global_add_pool(h_data_edge.x, batch=h_data_edge.batch)
        t_global_graph_emb_edge = global_add_pool(t_data_edge.x, batch=t_data_edge.batch)

        e_h_align = h_global_graph_emb_edge.repeat_interleave(degree(t_data_edge.batch, dtype=t_data_edge.batch.dtype), dim=0)
        e_t_align = t_global_graph_emb_edge.repeat_interleave(degree(h_data_edge.batch, dtype=h_data_edge.batch.dtype), dim=0)

        h_scores_e = (self.w_i(h_data_edge.x) * self.prj_i(e_t_align)).sum(-1)
        h_scores_e = softmax(h_scores_e, h_data_edge.batch, dim=0)

        t_scores_e = (self.w_j(t_data_edge.x) * self.prj_j(e_h_align)).sum(-1)
        t_scores_e = softmax(t_scores_e, t_data_edge.batch, dim=0)

        h_global_graph_emb_edge = global_add_pool(h_data_edge.x * e_t_align * h_scores_e.unsqueeze(-1), h_data_edge.batch)
        t_global_graph_emb_edge = global_add_pool(t_data_edge.x * e_h_align * t_scores_e.unsqueeze(-1), t_data_edge.batch)
        #h_global_graph_emb_edge = F.elu(h_global_graph_emb_edge)
        #t_global_graph_emb_edge = F.elu(t_global_graph_emb_edge)
        if h_global_graph_emb.shape[0] == h_global_graph_emb_edge.shape[0]:
            h_global_graph_emb = torch.stack([h_global_graph_emb,h_global_graph_emb_edge],dim=1)
            h_global_graph_emb,h_t = self.attention(h_global_graph_emb)

        if t_global_graph_emb.shape[0] == t_global_graph_emb_edge.shape[0]:
            t_global_graph_emb = torch.stack([t_global_graph_emb, t_global_graph_emb_edge], dim=1)
            t_global_graph_emb,t_t = self.attention(t_global_graph_emb)

        #h_global_graph_emb = h_global_graph_emb * h_global_graph_emb_edge
        #t_global_graph_emb = t_global_graph_emb * t_global_graph_emb_edge
        return h_global_graph_emb, t_global_graph_emb

