#
import torch
from torch import Tensor
from torch import nn
#
import numpy as np   # type: ignore
from sklearn.decomposition import PCA   # type: ignore
from sklearn.manifold import TSNE   # type: ignore
import matplotlib.pyplot as plt
#
from tqdm import tqdm   # type: ignore

#
from lib_dataset import Dataset

from lib_device import get_device

from lib_device import get_device



#
@torch.no_grad() # Disable gradient calculations for efficiency during inference
def calculate_accuracy(dataset: Dataset, model: nn.Module) -> float:

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





@torch.no_grad() # Disable gradient calculations for efficiency during inference
def calculate_confusion_matrix(
    dataset: Dataset,
    model: nn.Module,
    k: int = 3, # Parameterize k for flexibility (Top-K)
    batch_size: int = 1, # Add batch_size argument for optimization,
    nb_classes: int = 10,   # nb classes
    plot: bool = True   # plot the confusion matrix
) -> Tensor:

    """
    Calculates the Confusion Matrix of a model on the test set using batches.

    Args:
        dataset: An object with a 'get_full_test' method returning test features and labels.
        model: The PyTorch model to evaluate.
        k: The value for 'K' in Top-K accuracy (default: 3).
        batch_size: The number of samples to process in each batch (default: 64).
        nb_classes: int = The total number of classes for the matrix dim (default: 10)


    Returns:
        The calculated confusion matrix (Tensor).
    """

    model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates etc.)

    model = model.to( get_device() )

    try:
        x_test, y_test = dataset.get_full_test()
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return torch.Tensor([0])
    except Exception as e:
        print(f"Error getting test data: {e}")
        return torch.Tensor([0])

    x_test = x_test.to( get_device() )
    y_test = y_test.to( get_device() )

    total_samples: int = len(x_test)
    if total_samples == 0:
        print("Warning: Test dataset is empty.")
        return torch.Tensor([0])

    # Check if model and data are on the same device (optional but good practice)
    # Assumes model parameters() is not empty
    device = next(model.parameters()).device
    x_test = x_test.to(device)
    y_test = y_test.to(device)


    #
    confusion_matrix: Tensor = torch.zeros( (nb_classes, nb_classes), dtype=torch.int )

    # Iterate over batches
    for start in tqdm(range(0, total_samples, batch_size), desc="Computing Confusion Matrix"):
        end = min(start + batch_size, total_samples)
        x_batch = x_test[start:end]
        y_batch = y_test[start:end]

        # Forward pass
        outputs = model(x_batch)
        # Get top-1 predictions
        _, preds = torch.topk(outputs, 1, dim=-1)
        preds = preds.squeeze(1)  # Shape: (batch_size,)

        # Update confusion matrix
        for true_label, pred_label in zip(y_batch.view(-1), preds.view(-1)):
            confusion_matrix[true_label.long(), pred_label.long()] += 1

    # Plot if requested
    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix.cpu().numpy(), interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return confusion_matrix



@torch.no_grad()
def calculate_pca_embeddings(
    dataset: Dataset,
    model: nn.Module,
    batch_size: int = 64,
    plot: bool = True
) -> Tensor:
    """
    Computes 2D PCA on model embeddings for the test set and optionally plots them.

    Args:
        dataset: An object with a 'get_full_test' method returning test features and labels.
        model: The PyTorch model with a 'get_embedding' method.
        batch_size: Number of samples per batch during extraction.
        plot: Whether to display the 2D PCA scatter.

    Returns:
        A Tensor of shape (num_samples, 2) with the PCA-reduced embeddings.
    """
    model.eval()
    device = get_device()
    model = model.to(device)

    #
    if not hasattr(model, "get_embedding"):
        print("Error: The 'model' object must have a 'get_embedding' method.")
        return torch.empty((0, 2))

    try:
        x_all, y_all = dataset.get_full_test()
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return torch.empty((0, 2))

    x_all = x_all.to(device)
    y_all = y_all.to(device)
    total_samples = len(x_all)
    if total_samples == 0:
        print("Warning: Test dataset is empty.")
        return torch.empty((0, 2))

    # Extract embeddings
    embeddings = []
    labels = []
    for start in tqdm(range(0, total_samples, batch_size), desc="Extracting Embeddings for PCA"):
        end = min(start + batch_size, total_samples)
        x_batch = x_all[start:end]
        # Assume model.get_embedding exists
        emb_batch = model.get_embedding(x_batch)  # type: ignore
        embeddings.append(emb_batch.cpu().numpy())
        labels.append(y_all[start:end].cpu().numpy())

    embeddings = np.vstack(embeddings)  # type: ignore
    labels = np.concatenate(labels)

    # PCA reduction
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=10, alpha=0.7)
        plt.title('PCA of Model Embeddings')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar(label='Class Label')
        plt.tight_layout()
        plt.show()

    return torch.from_numpy(reduced)


@torch.no_grad()
def calculate_tsne_embeddings(
    dataset: Dataset,
    model: nn.Module,
    batch_size: int = 64,
    plot: bool = True,
    perplexity: float = 30.0,
    learning_rate: float = 200.0
) -> Tensor:
    """
    Computes 2D t-SNE on model embeddings for the test set and optionally plots them.

    Args:
        dataset: An object with a 'get_full_test' method returning test features and labels.
        model: The PyTorch model with a 'get_embedding' method.
        batch_size: Number of samples per batch during extraction.
        plot: Whether to display the 2D t-SNE scatter.
        perplexity: The perplexity parameter for t-SNE.
        learning_rate: The learning rate parameter for t-SNE.

    Returns:
        A Tensor of shape (num_samples, 2) with the t-SNE embeddings.
    """
    model.eval()
    device = get_device()
    model = model.to(device)

    #
    if not hasattr(model, "get_embedding"):
        print("Error: The 'model' object must have a 'get_embedding' method.")
        return torch.empty((0, 2))

    try:
        x_all, y_all = dataset.get_full_test()
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return torch.empty((0, 2))

    x_all = x_all.to(device)
    y_all = y_all.to(device)
    total_samples = len(x_all)
    if total_samples == 0:
        print("Warning: Test dataset is empty.")
        return torch.empty((0, 2))

    # Extract embeddings
    embeddings = []
    labels = []
    for start in tqdm(range(0, total_samples, batch_size), desc="Extracting Embeddings for t-SNE"):
        end = min(start + batch_size, total_samples)
        x_batch = x_all[start:end]
        emb_batch = model.get_embedding(x_batch)  # type: ignore
        embeddings.append(emb_batch.cpu().numpy())
        labels.append(y_all[start:end].cpu().numpy())

    embeddings = np.vstack(embeddings)  # type: ignore
    labels = np.concatenate(labels)

    # t-SNE reduction
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=10, alpha=0.7)
        plt.title('t-SNE of Model Embeddings')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.colorbar(label='Class Label')
        plt.tight_layout()
        plt.show()

    return torch.from_numpy(reduced)
