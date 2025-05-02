#
from typing import Optional
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
    batch_size: int = 1, # Add batch_size argument for optimization
    dataset_part: str = "test"
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

    #
    try:
        if dataset_part == "test":
            x_test, y_test = dataset.get_full_test()
        elif dataset_part == "train":
            x_test, y_test = dataset.get_full_train()
        else:
            print(f"Error: No dataset part `{dataset_part}`.")
            return torch.empty((0, 2))
    #
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return 0.0
    #
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




@torch.no_grad()
def calculate_confusion_matrix(
    dataset: Dataset,
    model: nn.Module,
    k: int = 3,
    batch_size: int = 1,
    nb_classes: int = 10,
    plot: bool = True,
    class_names: Optional[list[str]] = None,
    save_plot: Optional[str] = None
) -> Tensor:

    model.eval()
    model = model.to(get_device())

    try:
        x_test, y_test = dataset.get_full_test()
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return torch.Tensor([0])
    except Exception as e:
        print(f"Error getting test data: {e}")
        return torch.Tensor([0])

    x_test = x_test.to(get_device())
    y_test = y_test.to(get_device())

    total_samples: int = len(x_test)
    if total_samples == 0:
        print("Warning: Test dataset is empty.")
        return torch.Tensor([0])

    device = next(model.parameters()).device
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    confusion_matrix: Tensor = torch.zeros((nb_classes, nb_classes), dtype=torch.int)

    for start in tqdm(range(0, total_samples, batch_size), desc="Computing Confusion Matrix"):
        end = min(start + batch_size, total_samples)
        x_batch = x_test[start:end]
        y_batch = y_test[start:end]

        outputs = model(x_batch)
        _, preds = torch.topk(outputs, 1, dim=-1)
        preds = preds.squeeze(1)

        for true_label, pred_label in zip(y_batch.view(-1), preds.view(-1)):
            confusion_matrix[true_label.long(), pred_label.long()] += 1

    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix.cpu().numpy(), interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()

        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, ha="right")
            plt.yticks(tick_marks, class_names)
        else:
            plt.xticks(np.arange(nb_classes))
            plt.yticks(np.arange(nb_classes))

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        #
        if save_plot is not None:
            plt.savefig(save_plot)
        #
        else:
            plt.show()

        #
        plt.close()

    return confusion_matrix


@torch.no_grad()
def calculate_pca_embeddings(
    dataset: Dataset,
    model: nn.Module,
    batch_size: int = 64,
    plot: bool = True,
    class_names: Optional[list[str]] = None,
    save_plot: Optional[str] = None,
    dataset_part: str = "test"
) -> Tensor:

    model.eval()
    device = get_device()
    model = model.to(device)

    if not hasattr(model, "get_embedding"):
        print("Error: The 'model' object must have a 'get_embedding' method.")
        return torch.empty((0, 2))

    #
    try:
        if dataset_part == "test":
            x_all, y_all = dataset.get_full_test()
        elif dataset_part == "train":
            x_all, y_all = dataset.get_full_train()
        else:
            print(f"Error: No dataset part `{dataset_part}`.")
            return torch.empty((0, 2))
    #
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return torch.empty((0, 2))

    x_all = x_all.to(device)
    y_all = y_all.to(device)

    total_samples = len(x_all)
    if total_samples == 0:
        print("Warning: Test dataset is empty.")
        return torch.empty((0, 2))

    embeddings = []
    labels = []
    for start in tqdm(range(0, total_samples, batch_size), desc="Extracting Embeddings for PCA"):
        end = min(start + batch_size, total_samples)
        x_batch = x_all[start:end]
        emb_batch = model.get_embedding(x_batch)    # type: ignore
        embeddings.append(emb_batch.cpu().numpy())
        labels.append(y_all[start:end].cpu().numpy())

    embeddings = np.vstack(embeddings)    # type: ignore
    labels = np.concatenate(labels)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    if plot:

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1],
            c=labels, s=10, alpha=0.7, cmap='tab10'
        )
        plt.title('PCA of Model Embeddings')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Création de la légende flottante si les noms de classes sont donnés
        if class_names is not None:
            unique_labels = np.unique(labels)
            handles = []
            for i in unique_labels:
                color = scatter.cmap(scatter.norm(i))
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker='o',
                        color='w',
                        label=class_names[int(i)],
                        markerfacecolor=color,
                        markersize=8
                    )
                )
            plt.legend(
                title="Classes",
                handles=handles,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )
        else:
            plt.colorbar(scatter, label='Class Label')

        #
        plt.tight_layout()

        #
        if save_plot is not None:
            plt.savefig(save_plot)
        #
        else:
            plt.show()

        #
        plt.close()

    return torch.from_numpy(reduced)

@torch.no_grad()
def calculate_tsne_embeddings(
    dataset: Dataset,
    model: nn.Module,
    batch_size: int = 64,
    plot: bool = True,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    class_names: Optional[list[str]] = None,
    save_plot: Optional[str] = None,
    dataset_part: str = "test"
) -> Tensor:

    model.eval()
    device = get_device()
    model = model.to(device)

    if not hasattr(model, "get_embedding"):
        print("Error: The 'model' object must have a 'get_embedding' method.")
        return torch.empty((0, 2))

    #
    try:
        if dataset_part == "test":
            x_all, y_all = dataset.get_full_test()
        elif dataset_part == "train":
            x_all, y_all = dataset.get_full_train()
        else:
            print(f"Error: No dataset part `{dataset_part}`.")
            return torch.empty((0, 2))
    #
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return torch.empty((0, 2))

    x_all = x_all.to(device)
    y_all = y_all.to(device)

    total_samples = len(x_all)
    if total_samples == 0:
        print("Warning: Test dataset is empty.")
        return torch.empty((0, 2))

    embeddings = []
    labels = []
    for start in tqdm(range(0, total_samples, batch_size), desc="Extracting Embeddings for t-SNE"):
        end = min(start + batch_size, total_samples)
        x_batch = x_all[start:end]
        emb_batch = model.get_embedding(x_batch)    # type: ignore
        embeddings.append(emb_batch.cpu().numpy())
        labels.append(y_all[start:end].cpu().numpy())

    embeddings = np.vstack(embeddings)    # type: ignore
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, max_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    if plot:

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=10, alpha=0.7, cmap='tab10')
        plt.title('t-SNE of Model Embeddings')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')

        # Si class_names est fourni, créer une légende manuelle
        if class_names is not None:
            unique_labels = np.unique(labels)
            handles = []
            for i in unique_labels:
                color = scatter.cmap(scatter.norm(i))
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker='o',
                        color='w',
                        label=class_names[int(i)],
                        markerfacecolor=color,
                        markersize=8
                    )
                )
            plt.legend(
                title="Classes",
                handles=handles,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )
        else:
            plt.colorbar(scatter, label='Class Label')

        #
        plt.tight_layout()

        #
        if save_plot is not None:
            plt.savefig(save_plot)
        #
        else:
            plt.show()

        #
        plt.close()


    return torch.from_numpy(reduced)



#
def distance_point_class( embeddings: list[np.ndarray], idx_class_pts: list[int], idx_unknown_pt: int, method: str = "avg" ) -> float:

    #
    distances: list[float] = []
    #
    for idx_class_pt in idx_class_pts:
        #
        distances.append(
            np.sqrt( np.sum( (embeddings[idx_class_pt] - embeddings[idx_unknown_pt]) ** 2 ) )
        )
    #
    if method == "min":
        return min( distances ) if len(distances) > 0 else 0
    elif method == "max":
        return max( distances ) if len(distances) > 0 else 0
    elif method == "avg":
        return sum( distances ) / float(len(distances)) if len(distances) > 0 else 0
    #
    return 0




@torch.no_grad()
def calculate_unsupervized_clusters(
    dataset: Dataset,
    model: nn.Module,
    batch_size: int = 64,
    plot: bool = True,
    class_names: Optional[list[str]] = None,
    save_plot: Optional[str] = None,
    dataset_part: str = "test",
    known_prct: float = 0.1,
    distances_to_each_class_method: str = "avg"
) -> None:

    model = model.eval()
    device = get_device()
    model = model.to(device)

    if not hasattr(model, "get_embedding"):
        print("Error: The 'model' object must have a 'get_embedding' method.")
        return

    #
    try:
        if dataset_part == "test":
            x_all, y_all = dataset.get_full_test()
        elif dataset_part == "train":
            x_all, y_all = dataset.get_full_train()
        else:
            print(f"Error: No dataset part `{dataset_part}`.")
            return
    #
    except AttributeError:
        print("Error: The 'dataset' object must have a 'get_full_test' method.")
        return

    x_all = x_all.to(device)
    y_all = y_all.to(device)

    total_samples = len(x_all)
    if total_samples == 0:
        print("Warning: Test dataset is empty.")
        return

    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for start in tqdm(range(0, total_samples, batch_size), desc="Extracting Embeddings for unsupervized clustering"):
        end = min(start + batch_size, total_samples)
        x_batch = x_all[start:end]
        emb_batch = model.get_embedding(x_batch)    # type: ignore
        embeddings.append(emb_batch.cpu().numpy())
        labels.append(y_all[start:end].cpu().numpy())

    #
    all_embeddings = np.vstack(embeddings)

    #
    nb_per_class: list[int] = []

    #
    cls: int

    #
    for i in range(len(y_all)):
        #
        cls = int(y_all[i].item())
        #
        if cls >= len(nb_per_class):
            #
            nb_per_class += [0] * ( 1 + cls - len(nb_per_class) )
        #
        nb_per_class[ cls ] += 1

    #
    nb_known_per_class: list[int] = [ int(float(nbpc) * known_prct) for nbpc in nb_per_class]
    #
    crt_nb_known_per_class: list[int] = [ 0 ] * len(nb_per_class)
    #
    known_cluster_pts: list[list[int]] = [ [] for _ in range(len(nb_per_class)) ]
    #
    unknown_pts: list[int] = []

    #
    for i in range(len(y_all)):
        #
        cls = int(y_all[i].item())
        #
        if crt_nb_known_per_class[ cls ] < nb_known_per_class[ cls ]:
            #
            crt_nb_known_per_class[ cls ] += 1
            #
            known_cluster_pts[ cls ].append( i )
        #
        else:
            unknown_pts.append( i )

    """
    #
    predicted_unknown_pt_classes: list[list[tuple[int, float]]] = []

    #
    for j in range( len(unknown_pts) ):

        #
        predicted_unknown_pt_classes.append( [] )

        #
        for c in range(len(nb_per_class)):

            #
            d: float = distance_point_class(
                            embeddings=all_embeddings,
                            idx_class_pts=known_cluster_pts[c],
                            idx_unknown_pt=unknown_pts[j],
                            method=distances_to_each_class_method
            )

            #
            predicted_unknown_pt_classes[j].append( (c, d) )

        #
        predicted_unknown_pt_classes[j] = sorted( predicted_unknown_pt_classes[j], key = lambda x: x[1] )

    # Evaluation:

    # 1. Build predicted labels array
    predicted_labels = [cls_dist[0][0] for cls_dist in predicted_unknown_pt_classes]
    """

    predicted_labels = [0] * len(unknown_pts)

    tmp: list[int] = [i for i in range(len(unknown_pts))]
    crt_cls: int = 0

    while len(tmp) > 0:

        #
        dsts: list[float] = [
            distance_point_class(
                embeddings=all_embeddings,
                idx_class_pts=known_cluster_pts[crt_cls],
                idx_unknown_pt=unknown_pts[p],
                method="min"
            )
            for p in tmp
        ]

        #
        ip: int = np.argmin(dsts)
        idx: int = tmp.pop(ip)

        #
        predicted_labels[ idx ] = crt_cls

        #
        crt_cls += 1
        #
        if crt_cls >= len(nb_per_class):
            crt_cls = 0

    # Evaluation:
    true_labels      = [ y_all[i].item() for i in unknown_pts ]

    # 2. Report accuracy
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    acc = correct / len(unknown_pts) if unknown_pts else 0.0
    print(f"Unsupervised clustering accuracy on unknown pts: {correct}/{len(unknown_pts)} = {acc:.4f}")

    if plot:

        # 2D projection using t-SNE
        tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=1000)
        proj2d = tsne.fit_transform(all_embeddings)

        plt.figure(figsize=(8,6))
        # Plot known seeds
        for c, idxs in enumerate(known_cluster_pts):
            pts = proj2d[idxs]
            label = class_names[c] if class_names else f"Known class {c}"
            plt.scatter(pts[:,0], pts[:,1], marker='o', alpha=0.3, label=f"Known: {label}")
        # Plot unknown points with predicted labels
        unknown_idx_arr = np.array(unknown_pts)
        pred_arr = np.array(predicted_labels)
        for c in set(pred_arr):
            sel = unknown_idx_arr[pred_arr == c]
            pts = proj2d[sel]
            label = class_names[c] if class_names else f"Predicted {c}"
            plt.scatter(pts[:,0], pts[:,1], marker='x', s=30, label=f"Predicted: {label}")

        plt.title("t-SNE Unsupervised Clustering: Known vs. Predicted")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_plot)
        else:
            plt.show()
        plt.close()
