import torch
import torch.nn as nn
import math
from typing import Optional


#
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.

    Computes attention of the input tensor on itself, where queries, keys, and values
    are all derived from the same input.
    """

    #
    def __init__(self, embed_dim: int, num_heads: int, dtype: torch.dtype = torch.float32):
        """
        Initialize the MultiHeadSelfAttention module.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.

        Raises:
            AssertionError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.d_k: int = embed_dim // num_heads  # Dimension per head for queries and keys
        self.d_v: int = embed_dim // num_heads  # Dimension per head for values

        # Linear projections for queries, keys, values, and output
        self.q_linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.k_linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.out_linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)


    #
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-head self-attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (Optional[torch.Tensor]): Boolean mask tensor of shape broadcastable to
                                          (batch_size, seq_len, seq_len), where True
                                          indicates positions to mask (set to -inf).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """

        #
        nb_dims: int = len(x.size())

        #
        if nb_dims == 2:
            #
            seq_len, embed_dim = x.size()

        #
        elif nb_dims == 3:
            #
            batch_size, seq_len, embed_dim = x.size()

        #
        elif nb_dims == 4:
            #
            batch0_size, batch1_size, seq_len, embed_dim = x.size()

        #
        elif nb_dims == 5:
            #
            batch0_size, batch1_size, batch2_size, seq_len, embed_dim = x.size()

        # Compute queries, keys, and values
        Q: torch.Tensor = self.q_linear(x)  # (batch_size, seq_len, embed_dim)
        K: torch.Tensor = self.k_linear(x)  # (batch_size, seq_len, embed_dim)
        V: torch.Tensor = self.v_linear(x)  # (batch_size, seq_len, embed_dim)

        # Reshape and transpose for multi-head attention: split embed_dim into num_heads * d_k
        if nb_dims == 2:
            #
            Q = Q.view(seq_len, self.num_heads, self.d_k).transpose(-3, -2)
            K = K.view(seq_len, self.num_heads, self.d_k).transpose(-3, -2)
            V = V.view(seq_len, self.num_heads, self.d_v).transpose(-3, -2)
            # Shapes: (num_heads, seq_len, d_k)
            # or      (num_heads, seq_len, d_v)

        #
        elif nb_dims == 3:
            #
            Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
            K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
            V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
            # Shapes: (batch_size, num_heads, seq_len, d_k)
            # or      (batch_size, num_heads, seq_len, d_v)

        #
        elif nb_dims == 4:
            #
            Q = Q.view(batch0_size, batch1_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
            K = K.view(batch0_size, batch1_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
            V = V.view(batch0_size, batch1_size, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
            # Shapes: (batch0_size, batch1_size, num_heads, seq_len, d_k)
            # or      (batch0_size, batch1_size, num_heads, seq_len, d_v)

        #
        elif nb_dims == 5:
            #
            Q = Q.view(batch0_size, batch1_size, batch2_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
            K = K.view(batch0_size, batch1_size, batch2_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
            V = V.view(batch0_size, batch1_size, batch2_size, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
            # Shapes: (batch0_size, batch1_size, batch2_size, num_heads, seq_len, d_k)
            # or      (batch0_size, batch1_size, batch2_size, num_heads, seq_len, d_v)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply mask if provided (broadcasts over heads)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3), float('-inf'))

        # Apply softmax to get attention weights
        attention_weights: torch.Tensor = torch.softmax(scores, dim=-1)
        # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Compute attention output: attention_weights @ V
        attention_output: torch.Tensor = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len, d_v)

        #
        if nb_dims == 2:

            # Transpose and reshape back to (batch_size, seq_len, embed_dim)
            attention_output = attention_output.transpose(-3, -2).contiguous().view(seq_len, embed_dim)

        #
        elif nb_dims == 3:

            # Transpose and reshape back to (batch_size, seq_len, embed_dim)
            attention_output = attention_output.transpose(-3, -2).contiguous().view(batch_size, seq_len, embed_dim)

        #
        elif nb_dims == 4:

            # Transpose and reshape back to (batch_size0, batch1_size, seq_len, embed_dim)
            attention_output = attention_output.transpose(-3, -2).contiguous().view(batch0_size, batch1_size, seq_len, embed_dim)

        #
        elif nb_dims == 4:

            # Transpose and reshape back to (batch_size0, batch1_size, batch2_size, seq_len, embed_dim)
            attention_output = attention_output.transpose(-3, -2).contiguous().view(batch0_size, batch1_size, batch2_size, seq_len, embed_dim)

        # Apply final linear projection
        output: torch.Tensor = self.out_linear(attention_output)

        # Shape: (batch_size, seq_len, embed_dim)

        #
        return output


#
class MultiHeadBinaryAttention(nn.Module):
    """
    Multi-head binary attention module.

    Computes attention of input X on input Y, where queries are derived from X,
    and keys and values are derived from Y.
    """

    #
    def __init__(self, embed_dim: int, num_heads: int, dtype: torch.dtype = torch.float32):
        """
        Initialize the MultiHeadBinaryAttention module.

        Args:
            embed_dim (int): Dimension of the input embeddings (same for X and Y).
            num_heads (int): Number of attention heads.

        Raises:
            AssertionError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.d_k: int = embed_dim // num_heads  # Dimension per head for queries and keys
        self.d_v: int = embed_dim // num_heads  # Dimension per head for values

        # Linear projections for queries (from X), keys (from Y), values (from Y), and output
        self.q_linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.k_linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.out_linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)


    #
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-head binary attention mechanism.

        Args:
            x (torch.Tensor): Input tensor for queries of shape (batch_size, seq_len_x, embed_dim).
            y (torch.Tensor): Input tensor for keys and values of shape (batch_size, seq_len_y, embed_dim).
            mask (Optional[torch.Tensor]): Boolean mask tensor of shape broadcastable to
                                          (batch_size, seq_len_x, seq_len_y), where True
                                          indicates positions to mask (set to -inf).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len_x, embed_dim).
        """

        #
        nb_dims: int = len(x.shape)

        #
        if nb_dims == 2:

            #
            seq_len_x, embed_dim = x.size()
            seq_len_y, _ = y.size()

        #
        elif nb_dims == 3:

            batch_size, seq_len_x, embed_dim = x.size()
            _, seq_len_y, _ = y.size()

        # Compute queries from x, keys and values from y
        Q: torch.Tensor = self.q_linear(x)  # (batch_size, seq_len_x, embed_dim)
        K: torch.Tensor = self.k_linear(y)  # (batch_size, seq_len_y, embed_dim)
        V: torch.Tensor = self.v_linear(y)  # (batch_size, seq_len_y, embed_dim)

        # Reshape and transpose for multi-head attention: split embed_dim into num_heads * d_k
        if nb_dims == 2:

            Q = Q.view(seq_len_x, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(seq_len_y, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(seq_len_y, self.num_heads, self.d_v).transpose(1, 2)
            # Shapes: Q: (num_heads, seq_len_x, d_k)
            #         K: (num_heads, seq_len_y, d_k)
            #         V: (num_heads, seq_len_y, d_v)

        #
        elif nb_dims == 3:

            # Reshape and transpose for multi-head attention: split embed_dim into num_heads * d_k
            Q = Q.view(batch_size, seq_len_x, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, seq_len_y, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, seq_len_y, self.num_heads, self.d_v).transpose(1, 2)
            # Shapes: Q: (batch_size, num_heads, seq_len_x, d_k)
            #         K: (batch_size, num_heads, seq_len_y, d_k)
            #         V: (batch_size, num_heads, seq_len_y, d_v)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Shape: (batch_size, num_heads, seq_len_x, seq_len_y)

        # Apply mask if provided (broadcasts over heads)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # Apply softmax to get attention weights
        attention_weights: torch.Tensor = torch.softmax(scores, dim=-1)
        # Shape: (batch_size, num_heads, seq_len_x, seq_len_y)

        # Compute attention output: attention_weights @ V
        attention_output: torch.Tensor = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len_x, d_v)

        # Transpose and reshape back to (batch_size, seq_len_x, embed_dim)
        if nb_dims == 2:
            #
            attention_output = attention_output.transpose(1, 2).contiguous().view(seq_len_x, embed_dim)
        #
        elif nb_dims == 3:
            #
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len_x, embed_dim)

        # Apply final linear projection
        output: torch.Tensor = self.out_linear(attention_output)
        # Shape: (batch_size, seq_len_x, embed_dim)

        #
        return output