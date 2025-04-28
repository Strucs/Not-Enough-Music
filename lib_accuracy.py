#
import torch
from torch import nn
#
from tqdm import tqdm

#
from lib_dataset import Dataset

from lib_device import get_device

from lib_device import get_device

#
@torch.no_grad() # Disable gradient calculations for efficiency during inference
def calculate_accuracy(dataset: Dataset, model: nn.Module) -> None:

    #
    model.eval()
    model = model.to( get_device() )

    model = model.to(get_device())

    #
    x_test, y_test = dataset.get_full_test()

    x_test = x_test.to( get_device() )
    y_test = y_test.to( get_device() )

    x_test = x_test.to(get_device())
    y_test = y_test.to(get_device())

    #
    tot: float = 0
    good: float = 0

    #
    i: int
    #
    for i in tqdm(range(len(x_test))):
        #
        pred = model(x_test[i].unsqueeze(0)).to( get_device() )
        pred = model(x_test[i].unsqueeze(0)).to(get_device())

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




@torch.no_grad() # Disable gradient calculations for efficiency during inference
def calculate_top_k_accuracy(
    dataset: Dataset,
    model: nn.Module,
    k: int = 3, # Parameterize k for flexibility (Top-K)
    batch_size: int = 1 # Add batch_size argument for optimization
) -> float:

    """
    Calculates the Top-K accuracy of a model on the test set using batches.

    Args:
        dataset: An object with a 'get_full_test' method returning test features and labels.
        model: The PyTorch model to evaluate.
        k: The value for 'K' in Top-K accuracy (default: 3).
        batch_size: The number of samples to process in each batch (default: 64).

    Returns:
        The calculated Top-K accuracy (float).
    """

    model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates etc.)

    model = model.to( get_device() )

    try:
        x_test, y_test = dataset.get_full_test()
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return 0.0
    except Exception as e:
        print(f"Error getting test data: {e}")
        return 0.0

    x_test = x_test.to( get_device() )
    y_test = y_test.to( get_device() )

    total_samples: int = len(x_test)
    if total_samples == 0:
        print("Warning: Test dataset is empty.")
        return 0.0

    # Check if model and data are on the same device (optional but good practice)
    # Assumes model parameters() is not empty
    device = next(model.parameters()).device
    x_test = x_test.to(device)
    y_test = y_test.to(device)


    correct_predictions: float = 0.0

    # Process data in batches
    for i in tqdm(range(0, total_samples, batch_size), desc=f"Calculating Top-{k} Accuracy"):
        # Determine batch indices
        start_idx = i
        end_idx = min(i + batch_size, total_samples) # Handle the last batch correctly

        # Get the current batch
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]

        # Get model predictions for the batch
        # No need for unsqueeze(0) as we process a batch
        predictions = model(x_batch) # Output shape: (batch_size, num_classes)

        # Get the indices of the top K predictions for each sample in the batch
        # torch.topk returns a tuple (values, indices)
        # We only need the indices. Shape: (batch_size, k)
        _, top_k_indices = torch.topk(predictions, k, dim=-1)

        # Expand y_batch to compare with each of the top k predictions
        # y_batch shape: (batch_size) -> Unsqueeze to (batch_size, 1)
        # top_k_indices shape: (batch_size, k)
        # Comparison uses broadcasting: checks if y_batch element exists in the corresponding row of top_k_indices
        y_batch_expanded = y_batch.unsqueeze(1) # Shape: (batch_size, 1)

        # Check if the true label (y_batch_expanded) is present in the top K predicted indices (top_k_indices)
        # comparison shape: (batch_size, k) (boolean)
        comparison = (top_k_indices == y_batch_expanded)

        # Check if *any* of the top k predictions were correct for each sample
        # .any(dim=1) reduces shape from (batch_size, k) to (batch_size)
        # It yields True if at least one comparison was True in that row
        correct_in_batch_mask = comparison.any(dim=1) # Shape: (batch_size) (boolean)

        # Sum the number of correct predictions in this batch (True values count as 1)
        correct_predictions += correct_in_batch_mask.sum().item()

    # Calculate final accuracy
    accuracy: float = correct_predictions / total_samples if total_samples > 0 else 0.0

    print(f"Top-{k} accuracy = {int(correct_predictions)} / {total_samples} = {accuracy:.4f}")

    return accuracy
