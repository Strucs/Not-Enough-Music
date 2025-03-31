#
import torch
from torch import optim
from torch import nn
#
from tqdm import tqdm


#
OPTIMIZERS: dict[str, Callable] = {
    "adam": optim.Adam
}


#
def train_simple_epochs_loop(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer_type: str,
    optimizer_kwargs: dict,
    model_saving_folder: str,
    device: str = "cpu" if not torch.cuda.is_available() else "cuda"
) -> None:

    #
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    #
    optimizer: optim.Optimizer = OPTIMIZERS[optimizer_type](**optimizer_kwargs)

    #
    for epoch in range(nb_epochs):

        # TODO
        pass




