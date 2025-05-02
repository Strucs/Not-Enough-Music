#
from typing import Optional
#
import sys
import os
#
import torch
from torch import nn
#
import lib_dataset as ld
#
from matplotlib import pyplot as plt
import numpy as np
#
from lib_training import train_simple_epochs_loop
#
from lib_device import get_device



#
def load_dataset(model_name: str, give_train_dataset: bool = True, from_dataset: Optional[ld.Dataset] = None) -> ld.Dataset:

    #
    dataset: ld.Dataset

    #
    if model_name in ["AST_classif_1", "small_conv_1", "small_conv_2", "pre_train_AST_classif_1", "pre_train_small_conv_1", "pre_train_small_conv_2"]:
        #
        dataset = ld.DatasetAudios() if from_dataset is None else from_dataset
        #
        assert isinstance(dataset, ld.DatasetAudios), "type error dataset"
        #
        if give_train_dataset and model_name.startswith("pre_train_"):
            #
            dataset = ld.create_audio2vec_signal_dataset(in_dataset=dataset)

        #
        return dataset

    #
    elif model_name in ["vit", "simple_vit", "pre_train_vit", "pre_train_simple_vit"]:
        #
        dataset = ld.DatasetImagesFiltered(px_height_to_keep = -1, divisible_per = 64) if from_dataset is None else from_dataset
        #
        assert isinstance(dataset, ld.DatasetImagesFiltered) or isinstance(dataset, ld.DatasetImages), "type error dataset"
        #
        if give_train_dataset and model_name.startswith("pre_train_"):
            #
            dataset = ld.create_audio2vec_img_dataset(in_dataset=dataset)
        #
        return dataset
    #
    elif model_name.startswith("resnet") or model_name.startswith("pre_train_resnet"):
        #
        dataset = ld.DatasetImagesFiltered(px_height_to_keep = -1, divisible_per = 64) if from_dataset is None else from_dataset
        #
        assert isinstance(dataset, ld.DatasetImagesFiltered) or isinstance(dataset, ld.DatasetImages), "type error dataset"
        #
        if give_train_dataset and model_name.startswith("pre_train_"):
            #
            dataset = ld.create_audio2vec_img_dataset(in_dataset=dataset)
        #
        return dataset

    #
    raise UserWarning(f"Error: unknown model_name `{model_name}`")


#
def load_model(model_name: str, model_weights_path: str = "") -> nn.Module:

    #
    nb_classes: int = 2 if model_name.startswith("pre_train_") else 10

    #
    if model_name == "AST_classif_1" or model_name == "pre_train_AST_classif_1":

        #
        from lib_model_ast_classification import ASTClassification

        #
        model = ASTClassification( nb_classes = nb_classes )

    #
    elif model_name == "small_conv_1" or model_name == "pre_train_small_conv_1":

        #
        from model_angel1 import SimpleCNN1D

        #
        model = SimpleCNN1D(num_classes = nb_classes)  # type: ignore

    #
    elif model_name == "small_conv_2" or model_name == "pre_train_small_conv_2":

        #
        from model_angel2 import SimpleCNN1D  # type: ignore

        #
        model = SimpleCNN1D(num_classes = nb_classes)  # type: ignore

    #
    elif model_name.startswith("resnet") or model_name.startswith("pre_train_resnet"):

        #
        from model_resnet import Resnet

        #
        model = Resnet(num_classes = nb_classes, resnet_version = model_name, pretrained= True)  # type: ignore

    #
    elif model_name == "simple_vit" or model_name == "pre_train_simple_vit":

        #
        from model_vit import SimpleVitClassifier

        #
        model = SimpleVitClassifier(num_classes = nb_classes)  # type: ignore

    #
    elif model_name == "vit" or model_name == "pre_train_vit":

        #
        from model_vit import VitClassifier

        #
        model = VitClassifier(um_classes = nb_classes)  # type: ignore

    #
    else:

        #
        raise UserWarning(f"Unknown model class : `{model_name}`")


    #
    if model is None:
        #
        raise UserWarning(f"model not correctly loaded...\nModel = {model}")

    #
    if model_weights_path != "":

        #
        if not os.path.exists( model_weights_path ):

            #
            raise FileNotFoundError(f"Path doesn't exists : `{model_weights_path}`")

        #
        print(f"Loading model from path `{model_weights_path}` ...")
        model.load_state_dict( torch.load(model_weights_path, map_location=get_device()) )


    #
    return model
