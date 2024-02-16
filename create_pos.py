from operator import index
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import DataStructs
from tqdm import tqdm
import pickle
import os
import torch
import pickle

import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib
from data_preprocessing_t import atom_features,bond_features
class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''

    def __inc__(self, key, value, *args, **kwargs):
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"


def one_of_k_encoding(k, possible_values):
    '''
    Convert integer to one-hot representation.
    '''
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    '''
    Convert integer to one-hot representation.
    '''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s,
                    allowable_set))
def save_data(data, filename):
    dirname = f'{"/tmp/PEB-DDI-main/PEB-DDI/dataset/trans"}/{"drugbank"}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    return lap_pos_enc

def generate_drug_data(mol_graph, atom_symbols):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *bond_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                    edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T
    node_feature = features.long()
    edge_feature = edge_feats.long()
    max_atoms = 9
    if node_feature.shape[0] < max_atoms:
        padding_atoms = max_atoms - node_feature.shape[0]
        padding_features = torch.zeros(padding_atoms, node_feature.shape[1])
        node_feature = torch.cat([node_feature, padding_features], dim=0)
    g = dgl.DGLGraph()
    g.add_nodes(node_feature.shape[0])
    g.ndata['feat'] = node_feature
    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    g.edata['feat'] = edge_feature

    lap_pos = laplacian_positional_encoding(g, 8)

    return lap_pos
def generate_drug_data_dgl(mol_graph, atom_symbols):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *bond_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                    edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T
    node_feature = features.float()
    edge_feature = edge_feats.float()
    max_atoms = 9
    if node_feature.shape[0] < max_atoms:
        padding_atoms = max_atoms - node_feature.shape[0]
        padding_features = torch.zeros(padding_atoms, node_feature.shape[1])
        node_feature = torch.cat([node_feature, padding_features], dim=0)
    g = dgl.DGLGraph()
    g.add_nodes(node_feature.shape[0])
    g.ndata['feat'] = node_feature
    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    g.edata['feat'] = edge_feature
    return g
drug_smiles_path = pd.read_csv('/tmp/PEB-DDI-main/PEB-DDI/dataset/trans/drugbank/drug_smiles.csv')
drug_pairs_path = pd.read_csv('/tmp/PEB-DDI-main/PEB-DDI/dataset/trans/drugbank/ddis.csv')
drug_smiles = [(h, t) for h,t in zip(drug_smiles_path['drug_id'],drug_smiles_path['smiles'])]
drug_dict = {}
drug_id_mol_tup = []
smiles_rdkit_list = []
symbols = list()
smiles_list = []
ID_list=[]
for i in drug_smiles:
    ID_list.append(i[0])
    smiles_list.append(i[1])
moleculse = [Chem.MolFromSmiles(smile) for smile in smiles_list]
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in moleculse]
similarity_matrix = np.zeros((len(fingerprints),len(fingerprints)))
for i in range(len(fingerprints)):
    similarities = BulkTanimotoSimilarity(fingerprints[i],fingerprints)
    similarity_matrix[i] = similarities

for id in drug_smiles:
    drug_dict[id[0]] = id[1]

for id,smiles in drug_dict.items():
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is not None:
        drug_id_mol_tup.append((id,mol))
        symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())

for m in drug_id_mol_tup:
    smiles_rdkit_list.append(m[-1])
symbols = list(set(symbols))
#drug_pos_data = {id: generate_drug_data(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_lap')}
#drug_dgl_data = {id: generate_drug_data_dgl(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_dgl')}
drug_dgl_data_padding = {id: generate_drug_data_dgl(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_dgl')}
drug_pairs = [(h, t) for h, t in zip(drug_pairs_path['d1'], drug_pairs_path['d2'])]
association_matrix = np.zeros((1706,1706),dtype=int)
ID_index = {drug : idx for idx, drug in enumerate(ID_list)}
for drug1,drug2 in drug_pairs:
    idx1 = ID_index[drug1]
    idx2 = ID_index[drug2]
    association_matrix[idx1, idx2] = 1
#save_data(drug_pos_data,'drug_pos_data.pkl')
#save_data(drug_dgl_data,'drug_dgl_data.pkl')
save_data(drug_dgl_data_padding,'drug_dgl_data_padding.pkl')
drug_similarity=[ID_list, similarity_matrix]
save_data(drug_similarity, 'drug_similarity.pkl')
association = [ID_index, association_matrix]
save_data(association,'drugbank_association_matrix.pkl')