#
from typing import Callable
#
import os
import json
#
from datetime import datetime
#
import torch
from torch import Tensor
from torch import optim
from torch import nn
#
from tqdm import tqdm    # type: ignore
#
from lib_dataset import Dataset
from lib_device import get_device


#
OPTIMIZERS: dict[str, Callable] = {
    "adam": optim.Adam
}


#
def save_model(model: nn.Module, folder_path: str, additional_txt: str) -> str:

    #
    if not os.path.exists(folder_path):
        #
        os.makedirs(folder_path)

    #
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%d_%m_%y-%H_%M_%S")

    #
    model_save_path: str = f"{folder_path}weights_{len(os.listdir(folder_path))}_{additional_txt}_{currentTime}.pth"

    #
    model = model.to("cpu")

    #
    torch.save(model.state_dict(), model_save_path)

    #
    model = model.to(get_device())

    #
    print(f"Model weights saved at : `{model_save_path}`")

    #
    return model_save_path



#
def train_simple_epochs_loop(
    dataset: Dataset,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer_type: str,
    optimizer_kwargs: dict,
    nb_epochs: int,
    batch_size: int,
    batch_parallel_calcul: int,
    model_saving_folder: str,
    model_save_epochs_steps: int = -1,
    device: str = get_device()
) -> dict[str, list[float]]:

    #
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    #
    optimizer: optim.Optimizer = OPTIMIZERS[optimizer_type](**optimizer_kwargs)

    #
    history: dict[str, list[float]] = {
        "loss_train": []
    }

    #
    x_batch: Tensor
    y_batch: Tensor

    #
    x_input: Tensor
    y_input: Tensor

    #
    for epoch in range(nb_epochs):

        #
        print(f"Epoch {epoch+1} / {nb_epochs}")

        #
        if batch_size > 0:
            #
            x_batch, y_batch = dataset.get_batch_train(batch_size)
        #
        else:
            #
            x_batch, y_batch = dataset.get_full_train()

        #
        losses: list[float] = []

        #
        nb_iter_batch: int = len(x_batch) // batch_parallel_calcul

        #
        p_bar = tqdm(total=nb_iter_batch)

        #
        for i in range(nb_iter_batch):

            #
            optimizer.zero_grad()
            model.zero_grad()

            #
            x_input = x_batch[ (i*batch_parallel_calcul) : ((i+1)*batch_parallel_calcul) ].to(dtype=torch.float32).to(device)
            y_input = y_batch[ (i*batch_parallel_calcul) : ((i+1)*batch_parallel_calcul) ].to(device)
            #
            if len(y_input.shape) > 1:
                y_input = y_input.squeeze()
            if len(list(y_input.shape)) < 1:
                y_input = Tensor( [y_input] ).to(device)

            #
            y_input = y_input.to(dtype=torch.int64)

            #
            pred: Tensor = model(x_input).to(dtype=torch.float32).to(device)

            #
            loss = loss_fn(pred, y_input).to(device)

            #
            loss_value: float = loss.item()
            losses.append(loss_value)
            #
            loss.backward()
            #
            optimizer.step()

            #
            p_bar.set_description(f"Epoch {epoch+1}/{nb_epochs}")
            p_bar.set_postfix(loss=loss_value)
            p_bar.update(n=1)

        #
        loss_mean: float = sum(losses) / len(losses) if len(losses) >= 1 else -1
        #
        history["loss_train"].append( loss_mean )

        #
        print(f"Epoch {i+1}: loss = {loss_mean}")

        #
        if model_save_epochs_steps > 0 and epoch % model_save_epochs_steps == 0:

            #
            save_model(
                model=model,
                folder_path=model_saving_folder,
                additional_txt=f"_epoch_{epoch}"
            )

    #
    model_save_path: str = save_model(
        model=model,
        folder_path=model_saving_folder,
        additional_txt=""
    )

    #
    with open(model_save_path + f"_loss_.json", "w", encoding="utf-8") as f:

        #
        json.dump( history, f )

    #
    return history
