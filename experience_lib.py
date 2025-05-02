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
def load_model(give_train_dataset: bool = True) -> tuple[nn.Module, ld.Dataset]:


    #
    if sys.argv[1] == "AST_classif_1":

        #
        from lib_model_ast_classification import ASTClassification

        #
        dataset = ld.DatasetAudios()
        #
        model = ASTClassification(nb_classes = 10)

    #
    elif sys.argv[1] == "small_conv_1":

        #
        from model_angel1 import SimpleCNN1D

        #
        dataset = ld.DatasetAudios()
        #
        model = SimpleCNN1D(input_channels = len(dataset.get_batch_train(1)[0]), num_classes = 10)  # type: ignore

    #
    elif sys.argv[1] == "small_conv_2":

        #
        from model_angel2 import SimpleCNN1D  # type: ignore

        #
        dataset = ld.DatasetAudios()
        #
        model = SimpleCNN1D(input_channels = len(dataset.get_batch_train(1)[0]), num_classes = 10)  # type: ignore

    #
    elif sys.argv[1].startswith("resnet"):

        #
        from model_resnet import Resnet

        #
        # dataset = ld.DatasetImages()  # type: ignore
        dataset = ld.DatasetImagesFiltered(px_height_to_keep = -1)  # type: ignore
        #
        model = Resnet(image_size = dataset.get_batch_train(1)[0].shape, num_classes = 10, resnet_version = sys.argv[1], pretrained= True)  # type: ignore

    #
    elif sys.argv[1].startswith("simple_vit"):

        #
        from model_vit import SimpleVitClassifier

        #
        # dataset = ld.DatasetImages()  # type: ignore
        dataset = ld.DatasetImagesFiltered(px_height_to_keep = -1, divisible_per = 64)  # type: ignore
        #
        model = SimpleVitClassifier(image_size = dataset.get_batch_train(1)[0].shape, num_classes = 10)  # type: ignore

    #
    elif sys.argv[1].startswith("vit"):

        #
        from model_vit import VitClassifier

        #
        # dataset = ld.DatasetImages()  # type: ignore
        dataset = ld.DatasetImagesFiltered(px_height_to_keep = -1, divisible_per = 64)  # type: ignore
        #
        model = VitClassifier(image_size = dataset.get_batch_train(1)[0].shape, num_classes = 10)  # type: ignore

    #
    elif sys.argv[1] == "pre_train_AST_classif_1":

        #
        from lib_model_ast_classification import ASTClassification

        #
        if give_train_dataset:
            dataset = ld.create_audio2vec_signal_dataset()
        else:
            dataset = ld.DatasetAudios()
        #
        model = ASTClassification(nb_classes = 2)

    #
    elif sys.argv[1] == "pre_train_small_conv_1":

        #
        from model_angel1 import SimpleCNN1D

        #
        if give_train_dataset:
            dataset = ld.create_audio2vec_signal_dataset()
        else:
            dataset = ld.DatasetAudios()
        #
        model = SimpleCNN1D(input_channels = len(dataset.get_batch_train(1)[0]), num_classes = 2)  # type: ignore

    #
    elif sys.argv[1] == "pre_train_small_conv_2":

        #
        from model_angel2 import SimpleCNN1D  # type: ignore

        #
        if give_train_dataset:
            dataset = ld.create_audio2vec_signal_dataset()
        else:
            dataset = ld.DatasetAudios()
        #
        model = SimpleCNN1D(input_channels = len(dataset.get_batch_train(1)[0]), num_classes = 2)  # type: ignore

    #
    elif sys.argv[1].startswith("pre_train_resnet"):

        #
        from model_resnet import Resnet

        #
        # dataset = ld.DatasetImages()  # type: ignore
        dataset = ld.DatasetImagesFiltered(px_height_to_keep = -1)  # type: ignore
        #
        if give_train_dataset:
            dataset = ld.create_audio2vec_img_dataset(in_dataset=dataset)
        #
        model = Resnet(image_size = dataset.get_batch_train(1)[0].shape, num_classes = 2, resnet_version = sys.argv[1][len("pre_train_"):], pretrained= True)  # type: ignore

    #
    elif sys.argv[1].startswith("pre_train_simple_vit"):

        #
        from model_vit import SimpleVitClassifier

        #
        # dataset = ld.DatasetImages()  # type: ignore
        dataset = ld.DatasetImagesFiltered(px_height_to_keep = -1, divisible_per = 64)  # type: ignore
        #
        if give_train_dataset:
            dataset = ld.create_audio2vec_img_dataset(in_dataset=dataset)
        #
        model = SimpleVitClassifier(image_size = dataset.get_batch_train(1)[0].shape, num_classes = 2)  # type: ignore

    #
    elif sys.argv[1].startswith("pre_train_vit"):

        #
        from model_vit import VitClassifier

        #
        # dataset = ld.DatasetImages()  # type: ignore
        dataset = ld.DatasetImagesFiltered(px_height_to_keep = -1, divisible_per = 64)  # type: ignore
        #
        if give_train_dataset:
            dataset = ld.create_audio2vec_img_dataset(in_dataset=dataset)
        #
        model = VitClassifier(image_size = dataset.get_batch_train(1)[0].shape, num_classes = 2)  # type: ignore

    #
    else:

        #
        raise UserWarning(f"Unknown model class : `{sys.argv[1]}`")


    #
    if model is None or dataset is None:
        #
        raise UserWarning(f"model or dataset not correctly loaded...\nModel = {model}\nDataset = {dataset}")

    #
    model_weights_path: str = ""

    #
    if len(sys.argv) > 2:

        #
        model_weights_path = sys.argv[2]

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
    return (model, dataset)
