import math
from typing import Optional, List, Union, Tuple
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from tcn import TemporalConvNet

from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian, softmax


class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j \, \Vert \, \mathbf{e}_{i,j}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k \, \Vert \, \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                # spatial_attention: torch.FloatTensor,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # row, col = edge_index
        # edge_attr = spatial_attention[0, row, col]
        # edge_attr = edge_attr / edge_attr.sum()

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class ChebConvAttention(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator with attention from the
    `Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_ paper
    :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = None,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(ChebConvAttention, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, "sym", "rw"], "Invalid normalization"

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._normalization = normalization
        self._weight = Parameter(torch.Tensor(K, in_channels, out_channels)) 

        if bias:
            self._bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("_bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self._weight)
        if self._bias is not None:
            nn.init.uniform_(self._bias)

    #--forward pass-----
    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes
        )

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float("inf"), 0)

        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight is not None

        return edge_index, edge_weight #for example 307 nodes as deg, 340 edges , 307 nodes as self connections

    def forward(
        self,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        spatial_attention: torch.FloatTensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the ChebConv Attention layer (Chebyshev graph convolution operation).

        Arg types:
            * x (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in).
            * edge_index (Tensor array) - Edge indices.
            * spatial_attention (PyTorch Float Tensor) - Spatial attention weights, with shape (B, N_nodes, N_nodes).
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * batch (PyTorch Tensor, optional) - Batch labels for each edge.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.

        Return types:
            * out (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, F_out).
        """
        if self._normalization != "sym" and lambda_max is None:
            raise ValueError(
                "You need to pass `lambda_max` to `forward() in`"
                "case the normalization is non-symmetric."
            )

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype, device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self._normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )
        row, col = edge_index # refer to the index of each note each is a list of nodes not a number # (954, 954)
        Att_norm = norm * spatial_attention[:, row, col] # spatial_attention for example (32, 307, 307), -> (954) * (32, 954) -> (32, 954)
        num_nodes = x.size(self.node_dim) #for example 307
        # (307, 307) * (32, 307, 307) -> (32, 307, 307) -permute-> (32, 307,307) * (32, 307, 1) -> (32, 307, 1)
        TAx_0 = torch.matmul(
            (torch.eye(num_nodes).to(edge_index.device) * spatial_attention).permute(
                0, 2, 1
            ),
            x,
        ) #for example (32, 307, 1)
        out = torch.matmul(TAx_0, self._weight[0]) #for example (32, 307, 1) * [1, 64] -> (32, 307, 64)
        edge_index_transpose = edge_index[[1, 0]]
        if self._weight.size(0) > 1:
            TAx_1 = self.propagate(
                edge_index_transpose, x=TAx_0, norm=Att_norm, size=None
            )
            out = out + torch.matmul(TAx_1, self._weight[1])

        for k in range(2, self._weight.size(0)):
            TAx_2 = self.propagate(edge_index_transpose, x=TAx_1, norm=norm, size=None)
            TAx_2 = 2.0 * TAx_2 - TAx_0
            out = out + torch.matmul(TAx_2, self._weight[k])
            TAx_0, TAx_1 = TAx_1, TAx_2

        if self._bias is not None:
            out += self._bias

        return out #? (b, N, F_out) (32, 307, 64)

    def message(self, x_j, norm):
        if norm.dim() == 1:  # true
            return norm.view(-1, 1) * x_j  # (954, 1) * (32, 954, 1) -> (32, 954, 1)
        else:
            d1, d2 = norm.shape
            return norm.view(d1, d2, 1) * x_j

    def __repr__(self):
        return "{}({}, {}, K={}, normalization={})".format(
            self.__class__.__name__,
            self._in_channels,
            self._out_channels,
            self._weight.size(0),
            self._normalization,
        )

class SpatialAttention(nn.Module):
    r"""An implementation of the Spatial Attention Module (i.e compute spatial attention scores). For details see this paper:
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_
    
    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super(SpatialAttention, self).__init__()

        self._W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))  #for example (12)
        self._W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps)) #for example (1, 12)
        self._W3 = nn.Parameter(torch.FloatTensor(in_channels)) #for example (1)
        self._bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices)) #for example (1,307, 307)
        self._Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices)) #for example (307, 307)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial attention layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **S** (PyTorch FloatTensor) - Spatial attention score matrices, with shape (B, N_nodes, N_nodes).
        """
        # lhs = left hand side embedding;
        # to calculcate it : 
        # multiply with W1 (B, N, F_in, T)(T) -> (B,N,F_in)
        # multiply with W2 (B,N,F_in)(F_in,T)->(B,N,T)
        # for example (32, 307, 1, 12) * (12) -> (32, 307, 1) * (1, 12) -> (32, 307, 12) 
        LHS = torch.matmul(torch.matmul(X, self._W1), self._W2)
        
        # rhs = right hand side embedding
        # to calculcate it : 
        # mutliple W3 with X (F)(B,N,F,T)->(B, N, T) 
        # transpose  (B, N, T)  -> (B, T, N)
        # for example (1)(32, 307, 1, 12) -> (32, 307, 12) -transpose-> (32, 12, 307)
        RHS = torch.matmul(self._W3, X).transpose(-1, -2)
        
        # Then, we multiply LHS with RHS : 
        # (B,N,T)(B,T, N)->(B,N,N)
        # for example (32, 307, 12) * (32, 12, 307) -> (32, 307, 307) 
        # Then multiply Vs(N,N) with the output
        # (N,N)(B, N, N)->(B,N,N) (32, 307, 307)
        # for example (307, 307) *  (32, 307, 307) ->   (32, 307, 307)
        S = torch.matmul(self._Vs, torch.sigmoid(torch.matmul(LHS, RHS) + self._bs))
        S = F.softmax(S, dim=1)
        # now_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        # if now_time > '2023-01-31 17-11-20':
        #     plt.matshow(S[0].detach().numpy())
        #     plt.savefig(now_time)
        return S # (B,N,N) for example (32, 307, 307)

class TemporalAttention(nn.Module):
    r"""An implementation of the Temporal Attention Module( i.e. compute temporal attention scores). For details see this paper:
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_
    
    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super(TemporalAttention, self).__init__()

        self._U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))  # for example 307
        self._U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices)) #for example (1, 307)
        self._U3 = nn.Parameter(torch.FloatTensor(in_channels))  # for example (1)
        self._be = nn.Parameter(
            torch.FloatTensor(1, num_of_timesteps, num_of_timesteps)
        ) # for example (1,12,12)
        self._Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))  #for example (12, 12)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the temporal attention layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **E** (PyTorch FloatTensor) - Temporal attention score matrices, with shape (B, T_in, T_in).
        """
        # lhs = left hand side embedding;
        # to calculcate it : 
        # permute x:(B, N, F_in, T) -> (B, T, F_in, N)  
        # multiply with U1 (B, T, F_in, N)(N) -> (B,T,F_in)
        # multiply with U2 (B,T,F_in)(F_in,N)->(B,T,N)
        # for example (32, 307, 1, 12) -premute-> (32, 12, 1, 307) * (307) -> (32, 12, 1) * (1, 307) -> (32, 12, 307) 
        LHS = torch.matmul(torch.matmul(X.permute(0, 3, 2, 1), self._U1), self._U2) # (32, 12, 307) 
        
        
        #rhs = right hand side embedding
        # to calculcate it : 
        # mutliple U3 with X (F)(B,N,F,T)->(B, N, T)
        # for example (1)(32, 307, 1, 12) -> (32, 307, 12)
        RHS = torch.matmul(self._U3, X) # (32, 307, 12)
        
        # Them we multiply LHS with RHS : 
        # (B,T,N)(B,N,T)->(B,T,T)
        # for example (32, 12, 307) * (32, 307, 12) -> (32, 12, 12) 
        # Then multiply Ve(T,T) with the output
        # (T,T)(B, T, T)->(B,T,T)
        # for example (12, 12) *  (32, 12, 12) ->   (32, 12, 12)
        E = torch.matmul(self._Ve, torch.sigmoid(torch.matmul(LHS, RHS) + self._be))
        # E = torch.matmul(self._Ve, torch.matmul(LHS, RHS) + self._be)
        E = F.softmax(E, dim=1) #  (B, T, T)  for example (32, 12, 12)
        return E

class ASTGCNBlock(nn.Module):
    r"""An implementation of the Attention Based Spatial-Temporal Graph Convolutional Block.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        num_of_vertices: int,
        num_of_timesteps: int,
        nb_gatv2conv: int,
        dropout_gatv2conv: float = 0.5,
        head_gatv2conv: int = 4,
        normalization: Optional[str] = None,
        bias: bool = True,
    ):
        super(ASTGCNBlock, self).__init__()

        # self._temporal_attention = TemporalAttention(
        #     in_channels, num_of_vertices, num_of_timesteps
        # )
        # self._spatial_attention = SpatialAttention(
        #     in_channels, num_of_vertices, num_of_timesteps
        # )
        # self._chebconv_attention = ChebConvAttention(
        #     in_channels, nb_chev_filter, K, normalization, bias
        # )
        self._gatv2conv_attention = GATv2Conv(
            in_channels, out_channels=nb_gatv2conv, dropout=dropout_gatv2conv, heads=head_gatv2conv
        )
        self._time_convolution = nn.Conv2d(
            nb_gatv2conv * head_gatv2conv,
            # nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self._residual_convolution = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides)
        )
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self._normalization = normalization

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: Union[torch.LongTensor, List[torch.LongTensor]],
    ) -> torch.FloatTensor:
        """
        Making a forward pass with the ASTGCN block.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * **edge_index** (LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = X.shape # (32, 307, 1, 12)

        # X_tilde = self._temporal_attention(X) # (b, T, T)  (32, 12, 12) * reshaped x(32, 307, 12)  -reshape> (32, 307, 1, 12)
        # # xreshaped is e.g. (32, 307, 12) * (32, 12, 12) -then_reshaped> (32, 307, 1, 12)
        # X_tilde = torch.matmul(X.reshape(batch_size, -1, num_of_timesteps), X_tilde)
        # X_tilde = X_tilde.reshape(
        #     batch_size, num_of_vertices, num_of_features, num_of_timesteps
        # )
        # X_tilde = self._spatial_attention(X_tilde)  # (B,N,N) for example (32, 307, 307)

        if not isinstance(edge_index, list):
            data = Data(
                edge_index=edge_index, edge_attr=None, num_nodes=num_of_vertices
            )
            # if self._normalization != "sym":
            #     lambda_max = LaplacianLambdaMax()(data).lambda_max
            # else:
            #     lambda_max = None
            X_hat = []
            for t in range(num_of_timesteps):
                X_hat.append(
                    torch.unsqueeze(
                        self._gatv2conv_attention(x=X[0, :, :, t], edge_index=edge_index),
                        # self._chebconv_attention(
                        #     X[:, :, :, t], edge_index, X_tilde, lambda_max=lambda_max
                        # ),
                        -1,
                    )
                )

            X_hat = F.relu(torch.cat(X_hat, dim=-1))
        else:
            X_hat = []
            for t in range(num_of_timesteps):
                data = Data(
                    edge_index=edge_index[t], edge_attr=None, num_nodes=num_of_vertices
                )
                # if self._normalization != "sym":
                #     lambda_max = LaplacianLambdaMax()(data).lambda_max
                # else:
                #     lambda_max = None
                X_hat.append(
                    torch.unsqueeze(
                        self._gatv2conv_attention(x=X[0, :, :, t], edge_index=edge_index[t]),
                        # self._chebconv_attention(
                        #     X[:, :, :, t], edge_index[t], X_tilde, lambda_max=lambda_max
                        # ),
                        -1,
                    )
                )
            X_hat = F.relu(torch.cat(X_hat, dim=-1))

        # (b,N,F,T)->(b,F,N,T) for example (32, 307, 64, 12) -premute->(32, 64, 307,12)
        # then convolution along the time axis is applied
        X_hat = X_hat[None, ...]
        X_hat = self._time_convolution(X_hat.permute(0, 2, 1, 3)) # will give (32, 64, 307,12)
        # (b,N,F,T)-permute>(b,F,N,T) (1,1)->(b,F,N,T)  (32, 64, 307, 12)
        X = self._residual_convolution(X.permute(0, 2, 1, 3))   # will also give (32, 64, 307,12)
        #-adding X + X_hat->(32, 64, 307, 12)-premuting-> (32, 12, 307, 64)-layer_normalization_-premuting->(32, 307, 64,12) 
        X = self._layer_norm(F.relu(X + X_hat).permute(0, 3, 2, 1))
        X = X.permute(0, 2, 3, 1)
        return X # (b,N,F,T) for example (32, 307, 64,12) 


class GATv2TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        len_input: int,
        len_output: int,
        temporal_filter: int,
        # kernel_tcn: int,
        # kernel_conv2d: int,
        out_gatv2conv: int,
        dropout_tcn: float = 0.5,
        dropout_gatv2conv: float = 0.5,
        head_gatv2conv: int = 1
    ):
        super(GATv2TCN, self).__init__()
        self._gatv2conv_attention = GATv2Conv(
            in_channels, out_channels=out_gatv2conv, dropout=dropout_gatv2conv, heads=head_gatv2conv
        )
        # self._tcn = TemporalConvNet(num_inputs=len_input,
        #                             num_channels=[8, 4, 2, 1],
        #                             kernel_size=kernel_tcn,
        #                             dropout=0.25)
        self._time_convolution = nn.Conv2d(
            out_gatv2conv*head_gatv2conv,
            temporal_filter,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self._residual_convolution = nn.Conv2d(
            in_channels, temporal_filter, kernel_size=(1, 1), stride=(1, 1)
        )
        self._layer_norm = nn.LayerNorm(temporal_filter)
        self._final_conv = nn.Conv2d(
            len_input,
            out_channels,
            kernel_size=(1, temporal_filter//len_output),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: Union[torch.LongTensor, List[torch.LongTensor]],
    ) -> torch.FloatTensor:
        assert isinstance(edge_index, list)
        X_hat = []
        for t in range(len(edge_index)):
            X_hat.append(
                torch.unsqueeze(
                    self._gatv2conv_attention(x=X[0, :, :, t], edge_index=edge_index[t]),
                    -1,
                )
            )
        # X_hat = torch.cat(X_hat, dim=-1) # list [10] -> tensor [582, 64, 10]
        # X_hat = X_hat[None, ...].reshape(1, X_hat.shape[0], -1)
        # X_hat = self._tcn(X_hat)

        # x = X_hat[-1] # torch.cat(X_hat, dim=-1)
        # x = F.relu(x.reshape(582, -1))
        # att = torch.matmul(x, x.T)
        # att_matrix = F.softmax(att, dim=1).detach().numpy()
        # fig, ax = plt.subplots()
        # cax = ax.matshow(att_matrix)
        # fig.colorbar(cax)
        # fig.savefig('attention.png')
        #
        # att_rank_by_index = att_matrix.sum(axis=0).argsort()[::-1]
        # player_id_to_name = pd.read_pickle('player_id2name.pkl')
        # print(np.array(list(player_id_to_name.values()))[att_rank_by_index])

        X_hat = F.relu(torch.cat(X_hat, dim=-1))[None, ...]
        X_hat = self._time_convolution(X_hat.permute(0, 2, 1, 3))
        X = self._residual_convolution(X.permute(0, 2, 1, 3))   # will also give (32, 64, 307,12)
        #-adding X + X_hat->(32, 64, 307, 12)-premuting-> (32, 12, 307, 64)-layer_normalization_-premuting->(32, 307, 64,12)
        X = self._layer_norm(F.relu(X + X_hat).permute(0, 3, 2, 1))
        # X = X.permute(0, 2, 3, 1)
        X = self._final_conv(X)
        return X.permute(0, 2, 1, 3)[..., -1] # (b,N,F,T) for example (32, 307, 64,12)



class ASTGCN(nn.Module):
    r"""An implementation of the Attention Based Spatial-Temporal Graph Convolutional Cell.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        edge_index (array): edge indices.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
        num_of_vertices (int): Number of vertices in the graph.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        nb_block: int,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        num_for_predict: int,
        len_input: int,
        num_of_vertices: int,
        nb_gatv2conv: int,
        dropout_gatv2conv: float = 0.5,
        head_gatv2conv: int = 4,
        normalization: Optional[str] = None,
        bias: bool = True,
    ):

        super(ASTGCN, self).__init__()

        self._blocklist = nn.ModuleList(
            [
                ASTGCNBlock(
                    in_channels,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    time_strides,
                    num_of_vertices,
                    len_input,
                    nb_gatv2conv,
                    dropout_gatv2conv,
                    head_gatv2conv,
                    normalization,
                    bias,
                )
            ]
        )

        self._blocklist.extend(
            [
                ASTGCNBlock(
                    nb_time_filter,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    1,
                    num_of_vertices,
                    len_input // time_strides,
                    nb_gatv2conv,
                    dropout_gatv2conv,
                    head_gatv2conv,
                    normalization,
                    bias,
                )
                for _ in range(nb_block - 1)
            ]
        )

        self._final_conv = nn.Conv2d(
            int(len_input / time_strides),
            num_for_predict,
            kernel_size=(1, nb_time_filter),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, X: torch.FloatTensor, edge_index: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * **edge_index** (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * **X** (PyTorch FloatTensor)* - Hidden state tensor for all nodes, with shape (B, N_nodes, T_out).
        """
        for block in self._blocklist:
            # original x is (B,N,F_in,T) will give (B,N,F_out,T) for example (32, 307, 1, 12) -> (32, 307, 64, 12) 
            X = block(X, edge_index) 

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1) 
        # for example (32, 307, 64, 12) -permute-> (32, 12, 307,64) -final_conv-> (32, 12, 307, 1)
        X = self._final_conv(X.permute(0, 3, 1, 2))
        # (b,c_out*T,N)->(b,N,T)
        X = X[:, :, :, -1] # (b,c_out*T,N) for example (32, 12, 307)
        X = X.permute(0, 2, 1) # (b,T,N)-> (b,N,T)
        return X #(b,N,T) for exmaple (32, 307,12)
