#
import torch
from torch import Tensor
from torch import nn
#
from lib_model_attentions import MultiHeadSelfAttention, MultiHeadBinaryAttention   # type: ignore
from lib_model_feed_forward import FeedForward, BilinearFeedForward                 # type: ignore


#
class TransformerEncoderBlock(nn.Module):

    #
    def __init__(self, embedding_dim: int, attention_num_head: int, hidden_dim: int, dtype: torch.dtype = torch.float32) -> None:

        #
        super().__init__()

        #
        self.attention: MultiHeadSelfAttention = MultiHeadSelfAttention(
            embed_dim=embedding_dim,
            num_heads=attention_num_head,
            dtype=dtype
        )

        #
        self.feed_forward: FeedForward = FeedForward(
            in_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_dim=embedding_dim
        )

        #
        self.layer_norm_1: nn.LayerNorm = nn.LayerNorm( normalized_shape=embedding_dim )

        #
        self.layer_norm_2: nn.LayerNorm = nn.LayerNorm( normalized_shape=embedding_dim )


    #
    def forward(self, X: Tensor) -> Tensor:

        #
        X2: Tensor = self.attention(X)
        #
        X = X + X2
        #
        X = self.layer_norm_1( X )
        #
        X2 = self.feed_forward(X)
        #
        X = X + X2
        #
        X = self.layer_norm_2( X )
        #
        return X




#
class TransformerDecoderBlock(nn.Module):

    #
    def __init__(self, embedding_dim: int, attention_num_head: int, hidden_dim: int, dtype: torch.dtype = torch.float32) -> None:

        #
        super().__init__()

        #
        self.attention: MultiHeadBinaryAttention = MultiHeadBinaryAttention(
            embed_dim=embedding_dim,
            num_heads=attention_num_head,
            dtype=dtype
        )

        #
        self.feed_forward: BilinearFeedForward = BilinearFeedForward(
            in1_dim=embedding_dim,
            in2_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_dim=embedding_dim
        )

        #
        self.layer_norm_1: nn.LayerNorm = nn.LayerNorm( normalized_shape=embedding_dim )

        #
        self.layer_norm_2: nn.LayerNorm = nn.LayerNorm( normalized_shape=embedding_dim )


    #
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:

        #
        X2: Tensor = self.attention(X, Y)
        #
        X = X + X2
        #
        X = self.layer_norm_1( X )
        #
        X2 = self.feed_forward(X, Y)
        #
        X = X + X2
        #
        X = self.layer_norm_2( X )
        #
        return X

