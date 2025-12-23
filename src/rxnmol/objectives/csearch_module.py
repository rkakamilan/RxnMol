"""
CSearch GNN Models for Docking Energy Prediction.

Adapted from CSearch: Chemical Space Search via Virtual Synthesis and Global Optimization
GitHub: https://github.com/seoklab/CSearch
Reference: Kim et al. J Cheminform (2024)

Contains:
    - MyModel: GNN model for docking score prediction
    - MyDataset: Dataset class for molecular graphs
    - my_collate_fn: Collate function for DataLoader
    - Graph convolution layers (GCN, GIN, GIE, GAT)
    - PMA readout layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

# inserted below
# from libs.layers import GraphConvolution
# from libs.layers import GraphIsomorphism
# from libs.layers import GraphIsomorphismEdge
# from libs.layers import GraphAttention
# from libs.layers import PMALayer


class MyModel(nn.Module):
	def __init__(
			self,
			model_type,
			num_layers=4,
			hidden_dim=64,
			num_heads=4, # Only used for GAT
			dropout_prob=0.2,
			bias_mlp=True,
			out_dim=1,
			readout='sum',
			act=F.relu,
			initial_node_dim=58,
			initial_edge_dim=6,
			apply_sigmoid=False,
			norm_features=False,
		):
		super().__init__()

		self.num_layers = num_layers
		self.embedding_node = nn.Linear(initial_node_dim, hidden_dim, bias=False)
		self.embedding_edge = nn.Linear(initial_edge_dim, hidden_dim, bias=False)
		self.readout = readout

		self.mp_layers = torch.nn.ModuleList()
		for _ in range(self.num_layers):
			mp_layer = None
			if model_type == 'gcn':
				mp_layer = GraphConvolution(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
				)
			elif model_type == 'gin':
				mp_layer = GraphIsomorphism(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			elif model_type == 'gin':
				mp_layer = GraphIsomorphismEdge(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			elif model_type == 'gat':
				mp_layer = GraphAttention(
					hidden_dim=hidden_dim,
					num_heads=num_heads,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			else:
				raise ValueError('Invalid model type: you should choose model type in [gcn, gin, gin, gat, ggnn]')
			self.mp_layers.append(mp_layer)

		if self.readout == 'pma':
			self.pma = PMALayer(
				k=1,
				hidden_dim=hidden_dim,
				num_heads=num_heads,
				norm_features=norm_features,
			)

		self.linear_out = nn.Linear(hidden_dim, out_dim, bias=True)

		self.apply_sigmoid = apply_sigmoid
		if self.apply_sigmoid:
			self.sigmoid = F.sigmoid


	def forward(
			self,
			graph,
			training=False,
		):
		h = self.embedding_node(graph.ndata['h'].float())
		e_ij = self.embedding_edge(graph.edata['e_ij'].float())
		graph.ndata['h'] = h
		graph.edata['e_ij'] = e_ij

		# Update the node features
		for i in range(self.num_layers):
			graph = self.mp_layers[i](
				graph=graph,
				training=training
			)

		# Aggregate the node features and apply the last linear layer to compute the logit
		alpha = None
		if self.readout in ['sum', 'mean', 'max']:
			out = dgl.readout_nodes(graph, 'h', op=self.readout)
		elif self.readout == 'pma':
			out, alpha = self.pma(graph)
		out = self.linear_out(out)

		if self.apply_sigmoid:
			out = self.sigmoid(out)
		return out, alpha


class MLP_model(nn.Module):
	def __init__(
			self,
			num_layers=3,
			inp_dim=1024,
			hidden_dim=512,
			dropout_prob=0.2,
			out_dim=1,
			act=F.relu,
			apply_sigmoid=False,
		):
		super().__init__()

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.dropout_prob = dropout_prob
		self.apply_sigmoid = apply_sigmoid

		self.linear1 = nn.Linear(inp_dim, hidden_dim, bias=True)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
		self.linear3 = nn.Linear(hidden_dim, out_dim, bias=True)
		self.act = act

		if self.apply_sigmoid:
			self.sigmoid = F.sigmoid

	def forward(self, x, training=False):
		x = x.float()

		out = self.linear1(x)
		out = self.act(out)
		out = F.dropout(out, p=self.dropout_prob, training=training)
		out = self.linear2(out)
		out = self.act(out)
		out = F.dropout(out, p=self.dropout_prob, training=training)
		out = self.linear3(out)

		if self.apply_sigmoid:
			out = self.sigmoid(out)
		return out

# https://github.com/seoklab/CSearch/blob/main/libs/layers.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.backend import pytorch as F_dgl
from dgl.nn.functional import edge_softmax


class MLP(nn.Module):
	def __init__(
		self,
		input_dim,
		hidden_dim,
		output_dim,
		bias=True,
		act=F.relu,
	):
		super().__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.act = act

		self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
		self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)

	def forward(self, h):
		h = self.linear1(h)
		h = self.act(h)
		h = self.linear2(h)
		return h


class GraphConvolution(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			dropout_prob=0.2,
		):
		super().__init__()

		self.act = act
		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob
		self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

	def forward(
			self,
			graph,
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'u_'))
		h = self.act(self.linear(graph.ndata['u_'])) + h0
		h = self.norm(h)

		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphIsomorphism(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			bias_mlp=True,
			dropout_prob=0.2,
		):
		super().__init__()

		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=4*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act
		)
		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob

	def forward(
			self,
			graph,
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'u_'))
		h = self.mlp(graph.ndata['u_']) + h0
		h = self.norm(h)

		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphIsomorphismEdge(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			bias_mlp=True,
			dropout_prob=0.2,
		):
		super().__init__()

		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob
		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=4*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act,
		)

	def forward(
			self,
			graph,
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
		graph.update_all(fn.copy_edge('e_ij', 'm_e'), fn.sum('m_e', 'u_'))
		u_ = graph.ndata['neigh'] + graph.ndata['u_']
		h = self.mlp(u_) + h0
		h = self.norm(h)

		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphAttention(nn.Module):
	def __init__(
			self,
			hidden_dim,
			num_heads=4,
			bias_mlp=True,
			dropout_prob=0.2,
			act=F.relu,
		):
		super().__init__()

		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=2*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act,
		)
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.splitted_dim = hidden_dim // num_heads

		self.prob = dropout_prob

		self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w6 = nn.Linear(hidden_dim, hidden_dim, bias=False)

		self.act = F.elu
		self.norm = nn.LayerNorm(hidden_dim)

	def forward(
			self,
			graph,
			training=False
		):
		h0 = graph.ndata['h']
		e_ij = graph.edata['e_ij']

		graph.ndata['u'] = self.w1(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.ndata['v'] = self.w2(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.edata['x_ij'] = self.w3(e_ij).view(-1, self.num_heads, self.splitted_dim)

		graph.apply_edges(fn.v_add_e('v', 'x_ij', 'm'))
		graph.apply_edges(fn.u_mul_e('u', 'm', 'attn'))
		graph.edata['attn'] = edge_softmax(graph, graph.edata['attn'] / math.sqrt(self.splitted_dim))


		graph.ndata['k'] = self.w4(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.edata['x_ij'] = self.w5(e_ij).view(-1, self.num_heads, self.splitted_dim)
		graph.apply_edges(fn.v_add_e('k', 'x_ij', 'm'))

		graph.edata['m'] = graph.edata['attn'] * graph.edata['m']
		graph.update_all(fn.copy_edge('m', 'm'), fn.sum('m', 'h'))

		h = self.w6(h0) + graph.ndata['h'].view(-1, self.hidden_dim)
		h = self.norm(h)

		# Add and Norm module
		h = h + self.mlp(h)
		h = self.norm(h)

		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class PMALayer(nn.Module):
	def __init__(
			self,
			k,
			hidden_dim,
			num_heads,
			norm_features=False,
		):
		super().__init__()

		self.k = k
		self.hidden_dim = hidden_dim,
		self.num_heads = num_heads
		self.norm_features = norm_features

		self.mha = MultiHeadAttention(
			hidden_dim,
			num_heads,
		)
		self.seed_vec = torch.ones(1)
		self.w_seed = nn.Linear(1, k*hidden_dim)

	def forward(
			self,
			graph,
		):
		h = graph.ndata['h']
		if self.norm_features:
			h = h / torch.norm(h)

		lengths = graph.batch_num_nodes()
		batch_size = len(lengths)

		device = h.device
		self.seed_vec = self.seed_vec.to(device)
		if self.k == 1:
			query = self.w_seed(self.seed_vec).repeat(batch_size, 1)
		else:
			query = self.w_seed(self.seed_vec).reshape(self.k, -1).repeat(batch_size, 1)

		out, alpha = self.mha(
			query,
			h,
			[self.k] * batch_size,
			lengths,
		)
		return out, alpha


class MultiHeadAttention(nn.Module):
	def __init__(
			self,
			hidden_dim,
			num_heads,
		):
		super().__init__()

		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.d_heads = hidden_dim // num_heads

		self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=False)


	def forward(
			self,
			q,
			v,
			lengths_q,
			lengths_v,
		):
		batch_size = len(lengths_q)
		max_len_q = max(lengths_q)
		max_len_v = max(lengths_v)

		q = self.w_q(q).view(-1, self.num_heads, self.d_heads)
		k = self.w_k(v).view(-1, self.num_heads, self.d_heads)
		v = self.w_v(v).view(-1, self.num_heads, self.d_heads)

		q = F_dgl.pad_packed_tensor(q, lengths_q, 0)
		k = F_dgl.pad_packed_tensor(k, lengths_v, 0)
		v = F_dgl.pad_packed_tensor(v, lengths_v, 0)

		#e = torch.einsum('bxhd,byhd->bhxy', q, k)
		q = q.tile(1, max_len_v, 1, 1)
		e = q*v
		e = torch.sum(e, -1)
		e = e.permute(0, 2, 1)
		e = e.unsqueeze(2)
		e = e / math.sqrt(self.d_heads)

		mask = torch.zeros(batch_size, max_len_q, max_len_v).to(e.device)
		for i in range(batch_size):
			mask[i, :lengths_q[i], :lengths_v[i]].fill_(1)
		mask = mask.unsqueeze(1)
		e.masked_fill_(mask == 0, -float('inf'))

		alpha = torch.softmax(e, dim=-1)

		#out = torch.einsum('bhxy,bhyd->bxhd', alpha, v)
		alpha = alpha.permute(0,1,3,2)
		v = v.permute(0,2,1,3)
		out = alpha * v
		out = torch.sum(out, 2)
		out = out.unsqueeze(1)

		out = self.w_o(
			out.contiguous().view(batch_size, max_len_q, self.hidden_dim)
		)
		out = F_dgl.pack_padded_tensor(out, lengths_q)

		#out = out * torch.tensor(lengths_v).unsqueeze(-1).repeat(1, self.hidden_dim)
		return out, alpha


# https://github.com/seoklab/CSearch/blob/main/libs/io_inference.py
import pandas as pd

import torch
import dgl

from rdkit import Chem

from tdc.single_pred import ADME
from tdc.single_pred import HTS
from tdc.single_pred import Tox


ATOM_VOCAB = [
	'C', 'N', 'O', 'S', 'F',
	'H', 'Si', 'P', 'Cl', 'Br',
	'Li', 'Na', 'K', 'Mg', 'Ca',
	'Fe', 'As', 'Al', 'I', 'B',
	'V', 'Tl', 'Sb', 'Sn', 'Ag',
	'Pd', 'Co', 'Se', 'Ti', 'Zn',
	'Ge', 'Cu', 'Au', 'Ni', 'Cd',
	'Mn', 'Cr', 'Pt', 'Hg', 'Pb'
]


def one_of_k_encoding(x, vocab):
	if x not in vocab:
		x = vocab[-1]
	return list(map(lambda s: float(x==s), vocab))


def get_atom_feature(atom):
	atom_feature = one_of_k_encoding(atom.GetSymbol(), ATOM_VOCAB)
	atom_feature += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
	atom_feature += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
	atom_feature += one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
	atom_feature += [atom.GetIsAromatic()]
	return atom_feature


def get_bond_feature(bond):
	bt = bond.GetBondType()
	bond_feature = [
		bt == Chem.rdchem.BondType.SINGLE,
		bt == Chem.rdchem.BondType.DOUBLE,
		bt == Chem.rdchem.BondType.TRIPLE,
		bt == Chem.rdchem.BondType.AROMATIC,
		bond.GetIsConjugated(),
		bond.IsInRing()
	]
	return bond_feature


def get_molecular_graph(smi):
    # Handle both SMILES strings and RDKit Mol objects
    if isinstance(smi, Chem.Mol):
        mol = smi
    else:
        mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    graph = dgl.DGLGraph()

    atom_list = mol.GetAtoms()
    num_atoms = len(atom_list)
    graph.add_nodes(num_atoms)

    atom_feature_list = [get_atom_feature(atom) for atom in atom_list]
    atom_feature_list = torch.tensor(atom_feature_list, dtype=torch.float64)
    graph.ndata['h'] = atom_feature_list

    bond_list = mol.GetBonds()
    bond_feature_list = []
    for bond in bond_list:
        bond_feature = get_bond_feature(bond)

        src = bond.GetBeginAtom().GetIdx()
        dst = bond.GetEndAtom().GetIdx()

		# DGL graph is undirectional
		# Thus, we have to add edge pair of both (i,j) and (j, i)
		# i --> j
        graph.add_edges(src, dst)
        bond_feature_list.append(bond_feature)

		# j --> i
        graph.add_edges(dst, src)
        bond_feature_list.append(bond_feature)
    bond_feature_list = torch.tensor(bond_feature_list, dtype=torch.float64)
    graph.edata['e_ij'] = bond_feature_list
    return graph


def get_smi_and_label(dataset):
	smi_list = list(dataset['Drug'])
	label_list = list(dataset['Y'])
	return smi_list, label_list


def my_collate_fn(batch):
	graph_list = []
	smi_list = []
	for i, smi in enumerate(batch):
		graph = get_molecular_graph(smi)
		graph_list.append(graph)
		smi_list.append(smi)
	graph_list = dgl.batch(graph_list)
	return graph_list, smi_list


def get_dataset(
		path,
		smi_column='SMILES',
		dropna=False,
	):
	df = pd.read_csv(path)
	if dropna:
		df = df.dropna()
	smi_list = list(df[smi_column])
	return smi_list


class MyDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			smi_list
		):
		self.smi_list = smi_list

	def __len__(self):
		return len(self.smi_list)

	def __getitem__(
			self,
			idx
		):
		return self.smi_list[idx]


def debugging():
	data = ADME(
		name='BBB_Martins'
	)
	split = data.get_split(
		method='random',
		seed=999,
		frac=[0.7, 0.1, 0.2],
	)
	train_set = split['train']
	valid_set = split['valid']
	test_set = split['test']

	smi_train, label_train = get_smi_and_label(train_set)
	graph = get_molecular_graph(smi_train[0])


if __name__ == '__main__':
	debugging()