# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining a layer that is applying graph attention mechanism to node features
class GraphAttentionLayer(nn.Module):
	def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
		super(GraphAttentionLayer, self).__init__()
		self.dropout = dropout
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat

		# Learning parameters for the linear transformation and attention mechanism
		self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
		self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
		
		# Initializing transformation and attention parameters for stable training
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		nn.init.xavier_uniform_(self.a.data, gain=1.414)
		
		self.leakyrelu = nn.LeakyReLU(self.alpha)

	def forward(self, h, adj):
		batch_size, N, _ = h.size()

		# Applying linear transformation to input node features
		Wh = torch.matmul(h, self.W)
		
		# Constructing pairwise concatenation for attention computation
		a_input = torch.cat([
			Wh.repeat(1, 1, N).view(batch_size, N * N, self.out_features),
			Wh.repeat(1, N, 1)
		], dim=2).view(batch_size, N, N, 2 * self.out_features)
		
		# Computing raw attention scores with leaky ReLU activation
		e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
		
		# Masking out non-existent edges to ignore them in attention
		zero_vec = -9e15 * torch.ones_like(e)
		attention = torch.where(adj > 0, e, zero_vec)
		
		# Normalizing attention scores and applying dropout for regularization
		attention = F.softmax(attention, dim=2)
		attention = F.dropout(attention, self.dropout, training=self.training)
		
		# Aggregating transformed features using computed attention coefficients
		h_prime = torch.matmul(attention, Wh)
		
		# Returning the output with or without non-linear activation based on configuration
		if self.concat:
			return F.elu(h_prime)
		else:
			return h_prime

# Defining a GAT model that is pooling attention-based node embeddings into a graph representation
class CitationGAT(nn.Module):
	def __init__(self, in_features, hidden_dim, out_features, dropout=0.1, alpha=0.2, n_heads=4):
		super(CitationGAT, self).__init__()
		self.dropout = dropout
		
		# Creating multiple attention heads for the first graph attention layer
		self.attention_heads = nn.ModuleList([
			GraphAttentionLayer(in_features, hidden_dim, dropout=dropout, alpha=alpha, concat=True)
			for _ in range(n_heads)
		])
		
		# Defining a single-headed attention layer for producing final node embeddings
		self.out_att = GraphAttentionLayer(hidden_dim * n_heads, out_features, dropout=dropout, alpha=alpha, concat=False)
		
		# Defining a simple feedforward network to pool node embeddings into a graph-level vector
		self.graph_pooling = nn.Sequential(
			nn.Linear(out_features, out_features),
			nn.ReLU(),
			nn.Dropout(dropout)
		)

	def forward(self, x, adj, mask=None):
		# Applying dropout to the input features before attention
		x = F.dropout(x, self.dropout, training=self.training)
		
		# Running multiple attention heads in parallel and concatenating their outputs
		x = torch.cat([att(x, adj) for att in self.attention_heads], dim=2)
		
		# Applying dropout after the first attention layer to prevent overfitting
		x = F.dropout(x, self.dropout, training=self.training)
		
		# Generating final node-level embeddings using the output attention head
		x = self.out_att(x, adj)
		
		# Pooling node embeddings into a fixed-size graph embedding
		if mask is not None:
			x = x * mask.unsqueeze(2)
			pooled = x.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-10)
		else:
			pooled = x.mean(dim=1)
		
		# Transforming pooled representation to obtain the final graph-level embedding
		graph_embedding = self.graph_pooling(pooled)
		
		return graph_embedding, x
