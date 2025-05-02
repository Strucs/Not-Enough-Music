#
import torch
from torch import nn
from torch import Tensor
#
from transformers import AutoProcessor, ASTModel  # type: ignore
#
from lib_model_feed_forward import FeedForward
from lib_model_transformer_block import TransformerEncoderBlock
from lib_classification import ClassificationModule
from lib_device import get_device





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
            nb_classes=self.nb_classes
        )

    #
    def get_embedding(self, inputs: Tensor) -> Tensor:

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
        X = self.classification_module.get_embedding(last_hidden_states)

        #
        return X


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
