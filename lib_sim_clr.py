import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Simple 2-layer MLP projection head
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)

# --- Inside your main model class ---
# Assume YourBaseModel has the get_embedding() method
class YourSimCLRModel(nn.Module):
    def __init__(self, base_model: nn.Module, embedding_dim: int, projection_hidden_dim: int = 512, projection_output_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        # Assuming base_model has a get_embedding method or you adapt it
        # If get_embedding is not part of forward pass, you might need to modify base_model
        # or call get_embedding explicitly. Here we assume forward() up to embedding exists.

        # Check if base_model has get_embedding method
        if not hasattr(base_model, 'get_embedding'):
             raise AttributeError("The base_model must have a 'get_embedding' method.")


        self.projection_head = SimCLRProjectionHead(embedding_dim, projection_hidden_dim, projection_output_dim)

    def get_embedding(self, x: Tensor) -> Tensor:
         # Get the embedding from the base model
         # This might involve calling base_model.forward(x) or base_model.get_embedding(x)
         # depending on how YourBaseModel is structured.
         # Let's assume we call the method directly if it exists

        # Check if base_model has get_embedding method
        if hasattr(self.base_model, 'get_embedding'):
            return self.base_model.get_embedding(x)
        #
        raise AttributeError("The base_model must have a 'get_embedding' method.")


    def forward(self, x: Tensor) -> Tensor:
        """
        During training, outputs the projection.
        For inference, use get_embedding().
        """
        embedding = self.get_embedding(x) # Get embedding from base model
        projection = self.projection_head(embedding) # Pass through projection head
        return projection



class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # Sum over batch dimension
        self.similarity_f = nn.CosineSimilarity(dim=2) # Compare across embedding dimension

    # def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
    #     """
    #     Calculates the NT-Xent loss.
    #     Args:
    #         z_i (Tensor): Projections of the first view of the batch. Shape: [batch_size, projection_dim]
    #         z_j (Tensor): Projections of the second view of the batch. Shape: [batch_size, projection_dim]
    #     Returns:
    #         Tensor: The NT-Xent loss value.
    #     """
    #     batch_size = z_i.shape[0]
    #     # Concatenate projections from both views
    #     projections = torch.cat([z_i, z_j], dim=0) # Shape: [2 * batch_size, projection_dim]

    #     # Calculate pairwise cosine similarity
    #     # sim[a, b] = cos_sim(projections[a], projections[b])
    #     sim = self.similarity_f(projections.unsqueeze(1), projections.unsqueeze(0)) / self.temperature
    #     # Shape: [2 * batch_size, 2 * batch_size]

    #     # --- Create labels and masks ---
    #     # Positive pairs: (z_i[k], z_j[k]) for k=0..batch_size-1
    #     # These correspond to indices (k, k + batch_size) and (k + batch_size, k) in the similarity matrix
    #     idx = torch.arange(batch_size, device=self.device)
    #     # labels = torch.cat([idx + batch_size, idx], dim=0) # Correct labels for CrossEntropyLoss
    #     labels = torch.cat([torch.arange(batch_size, device=self.device) + batch_size - 1,
    #                         torch.arange(batch_size, device=self.device)], dim=0)
    #     # labels[k] = k + batch_size (target for z_i[k])
    #     # labels[k + batch_size] = k (target for z_j[k])

    #     # Mask to remove self-comparisons (diagonal elements)
    #     mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
    #     sim = sim[~mask].view(2 * batch_size, -1) # Remove diagonal, Shape: [2*B, 2*B-1]

    #     # --- Calculate loss ---
    #     # CrossEntropyLoss(logits, labels)
    #     loss = self.criterion(sim, labels)
    #     loss = loss / (2 * batch_size) # Normalize by batch size (as per SimCLR paper)

    #     return loss

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """
        Calculates the NT-Xent loss.
        Args:
            z_i (Tensor): Projections of the first view of the batch. Shape: [batch_size, projection_dim]
            z_j (Tensor): Projections of the second view of the batch. Shape: [batch_size, projection_dim]
        Returns:
            Tensor: The NT-Xent loss value.
        """
        batch_size = z_i.shape[0]

        # --- Apply L2 normalization ---
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate normalized projections from both views
        projections = torch.cat([z_i, z_j], dim=0) # Shape: [2 * batch_size, projection_dim]

        # Calculate pairwise cosine similarity
        # sim[a, b] = cos_sim(projections[a], projections[b])
        sim = self.similarity_f(projections.unsqueeze(1), projections.unsqueeze(0)) / self.temperature
        # Shape: [2 * batch_size, 2 * batch_size]

        # --- Create masks and labels ---
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        sim = sim[~mask].view(2 * batch_size, -1) # Remove diagonal, Shape: [2*B, 2*B-1]

        # Correct labels for the masked similarity matrix
        labels = torch.cat([torch.arange(batch_size, device=self.device) + batch_size - 1,
                            torch.arange(batch_size, device=self.device)], dim=0)

        # --- Calculate loss ---
        loss = self.criterion(sim, labels)
        loss = loss / (2 * batch_size) # Normalize by batch size (as per SimCLR paper)

        return loss


