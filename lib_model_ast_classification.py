#
import torch
from torch import nn
from torch import Tensor
#
from transformers import AutoProcessor, ASTModel
#
from lib_model_attentions import MultiHeadSelfAttention
from lib_model_feed_forward import FeedForward
from lib_model_transformer_block import TransformerEncoderBlock
from lib_device import get_device


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
    def forward(self, X: Tensor) -> Tensor:

        #
        X = self.transformer_block(X)

        #
        X = torch.mean(X, dim=-2)

        #
        X = self.final_linear(X)

        #
        return X




# AST stands for AudioSpectrogramTransformer
class ASTClassification(nn.Module):

    #
    def __init__(self, nb_classes: int) -> None:

        #
        super().__init__()

        #
        self.nb_classes: int = nb_classes

        #
        self.ast_processor: AutoProcessor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", use_fast=True)
        #
        self.ast_model: ASTModel = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        #
        # Freeze all layers of ast model
        for param in self.ast_model.parameters():
            param.requires_grad = False

        #
        self.classification_module: ClassificationModule = ClassificationModule(
            embedding_dim=768,
            nb_classes=10
        )

    #
    def forward(self, inputs: Tensor) -> Tensor:

        #
        pre_inputs = [ self.ast_processor(inputs[i].to("cpu"), sampling_rate=16000, return_tensors="pt") for i in range(len(inputs)) ]

        #
        inputs = torch.zeros( (len(pre_inputs), 1024, 128) ).to(get_device())

        #
        for i, inp in enumerate(pre_inputs):

            #
            inputs[i] = inp["input_values"][0]

        #
        outputs = self.ast_model(inputs)

        #
        last_hidden_states = outputs.last_hidden_state

        #
        X = self.classification_module(last_hidden_states)

        #
        return X
