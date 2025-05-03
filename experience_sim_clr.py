import torch
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm # type: ignore

# Assume get_device() exists as in your code
from lib_device import get_device
# Assume save_model exists
from lib_training import save_model, OPTIMIZERS

from experience_lib import load_model, load_dataset

from lib_dataset import SimCLRDataset, AudioAugmentation, Dataset, ImageAugmentation
from lib_sim_clr import YourSimCLRModel, NTXentLoss


def train_simclr_loop(
    train_dataset: SimCLRDataset, # Use the SimCLR dataset
    model: YourSimCLRModel, # The model with the projection head
    loss_fn: NTXentLoss, # The NT-Xent loss
    optimizer_type: str,
    optimizer_kwargs: dict,
    nb_epochs: int,
    batch_size: int, # Dataloader handles batching
    # batch_parallel_calcul is less relevant with DataLoader, but keep if needed for gradient accumulation
    model_saving_folder: str,
    model_save_epochs_steps: int = -1,
    num_workers: int = 0, # For DataLoader
    device: str = get_device()
) -> dict[str, list[float]]:

    #
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Use PyTorch DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # Usually good for GPU training
        drop_last=True # Important for SimCLR to avoid issues with last partial batch
    )

    #
    optimizer: optim.Optimizer = OPTIMIZERS[optimizer_type](model.parameters(), **optimizer_kwargs['kwargs']) # Adjusted for SimCLR model params

    #
    history: dict[str, list[float]] = {
        "loss_train": []
    }

    #
    for epoch in range(nb_epochs):
        #
        print(f"Epoch {epoch+1} / {nb_epochs}")
        model.train() # Set model to training mode
        epoch_losses: list[float] = []

        # Use DataLoader iterator
        p_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}")

        for view1, view2 in p_bar:
            #
            optimizer.zero_grad()

            # Move data to device
            view1 = view1.to(device, non_blocking=True, dtype=torch.float32)
            view2 = view2.to(device, non_blocking=True, dtype=torch.float32)

            # Get projections from the model
            proj1: Tensor = model(view1) # Shape: [batch_size, projection_dim]
            proj2: Tensor = model(view2) # Shape: [batch_size, projection_dim]

            # Calculate loss
            loss = loss_fn(proj1, proj2)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Logging
            loss_value: float = loss.item()
            epoch_losses.append(loss_value)
            p_bar.set_postfix(loss=loss_value)


        # End of Epoch
        loss_mean: float = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        history["loss_train"].append(loss_mean)
        print(f"Epoch {epoch+1} finished: Average Loss = {loss_mean:.4f}")

        # Save model checkpoint
        if model_save_epochs_steps > 0 and (epoch + 1) % model_save_epochs_steps == 0:
            save_model(
                model=model.base_model, # Save only the base encoder for inference
                folder_path=model_saving_folder,
                additional_txt=f"_epoch_{epoch+1}_simclr"
            )
            # Optionally save the full SimCLR model too if needed for resuming training
            # save_model(model=model, folder_path=model_saving_folder, additional_txt=f"_epoch_{epoch+1}_simclr_full")


    # Final save
    final_save_path = save_model(
        model=model.base_model, # Save only the base encoder
        folder_path=model_saving_folder,
        additional_txt="_final_simclr"
    )
    # save_model(model=model, folder_path=model_saving_folder, additional_txt="_final_simclr_full")

    # Save history (Consider using json or other formats)
    # ... saving history ...

    return history



#
if __name__ == "__main__":

    # --- Example Usage ---
    # 1. Load your base model (without projection head)
    model_name: str = "AST_classif_1"
    base_model: nn.Module = load_model(model_name) # Your function
    embedding_dim: int = 768
    # AST_classif_1 = 768 | vit = 1000

    # 2. Create the SimCLR model wrapper
    simclr_model: YourSimCLRModel = YourSimCLRModel(base_model, embedding_dim)

    # 3. Prepare the dataset and dataloader
    original_dataset: Dataset = load_dataset(model_name) # Your function
    augmentation: AudioAugmentation = AudioAugmentation(sample_rate=original_dataset.sampling_rate)
    # augmentation: ImageAugmentation = ImageAugmentation(input_size=(192, 320))
    simclr_dataset: SimCLRDataset = SimCLRDataset(original_dataset.x_train, augmentation)

    # 4. Define loss and optimizer
    simclr_loss = NTXentLoss(temperature=0.1, device=get_device())
    simclr_optimizer_kwargs = {"kwargs": {"lr": 1e-3, "weight_decay": 1e-6}} # Example

    # 5. Start training
    train_simclr_loop(
        train_dataset=simclr_dataset,
        model=simclr_model,
        loss_fn=simclr_loss,
        optimizer_type="adam", #
        optimizer_kwargs=simclr_optimizer_kwargs,
        nb_epochs=100,
        batch_size=8, # Adjust based on memory
        model_saving_folder=f"model_weights/simclr_{model_name}/",
        model_save_epochs_steps=2
    )
