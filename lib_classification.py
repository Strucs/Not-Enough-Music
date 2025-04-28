#
import torch
from torch import nn
from torch import Tensor
#
from lib_model_feed_forward import FeedForward
from lib_model_transformer_block import TransformerEncoderBlock



#
class ClassificationModule(nn.Module):

    #
    def __init__(self, embedding_dim: int, nb_classes: int) -> None:

        #
        super().__init__()

        #
        self.ff: FeedForward = FeedForward(
            in_dim = embedding_dim,
            hidden_dim = embedding_dim // 2,
            out_dim = nb_classes
        )
        self.transformer_block: TransformerEncoderBlock = TransformerEncoderBlock(
            embedding_dim = embedding_dim,
            attention_num_head = 4,
            hidden_dim = embedding_dim // 2
        )

        #
        self.final_linear: nn.Linear = nn.Linear(in_features=embedding_dim, out_features=nb_classes)

    #
    def get_embedding(self, X: Tensor) -> Tensor:

        #
        X = self.transformer_block(X)

        #
        X = torch.mean(X, dim=-2)

        #
        return X

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        X = self.get_embedding(X)

        #
        X = self.final_linear(X)

        #
        return X
