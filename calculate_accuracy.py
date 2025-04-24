#
import torch
from torch import nn
#
from tqdm import tqdm

#
from lib_dataset import Dataset



#
def calculate_accuracy(dataset: Dataset, model: nn.Module) -> None:

    #
    x_test, y_test = dataset.get_full_test()

    #
    tot: float = 0
    good: float = 0

    #
    i: int
    #
    for i in tqdm(range(len(x_test))):
        #
        pred = model(x_test[i].unsqueeze(0))

        #
        idx_pred = torch.argmax( pred, dim = -1 )

        if( idx_pred.item() == y_test[i].item() ):
            #
            good += 1

        #
        tot += 1

    #
    acc: float = good / tot if tot != 0 else 0

    #
    print(f"Accuracy = {good} / {tot} = {acc}")

    #
    return acc
