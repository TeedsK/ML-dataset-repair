# File: pytorch_model.py
# Defines the PyTorch model (TiedLinear) and training/inference logic (RepairModel)
# Adapted from the official HoloClean learn.py

import logging
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import softmax
from tqdm import tqdm # For progress bars
import numpy as np

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class TiedLinear(nn.Module):
    """
    Linear layer with shared parameters for features across output classes (domain values).
    Input X has dimensions (batch_size, output_dim, in_features).
    Output has dimensions (batch_size, output_dim), representing scores for each domain value.

    Adapted from HoloClean's learn.py.
    """
    def __init__(self, num_features, output_dim, bias=False):
        super(TiedLinear, self).__init__()
        self.in_features = num_features
        self.output_dim = output_dim
        self.use_bias = bias

        # Shared weight parameter: shape (1, in_features)
        # Applied across all output_dim possibilities for each item in the batch
        self.weight = Parameter(torch.Tensor(1, self.in_features))

        if self.use_bias:
            # Bias could also be shared (1, in_features) or per output class (output_dim, 1)
            # Let's assume shared bias for simplicity, similar to weight
            self.bias = Parameter(torch.Tensor(1, self.in_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Kaiming Uniform heuristic
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Calculate fan_in for bias initialization
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.unsqueeze(0).expand(self.output_dim, -1, -1).reshape(self.output_dim, -1))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, X, mask):
        """
        Performs the forward pass.
        X shape: (batch_size, output_dim, in_features)
        mask shape: (batch_size, output_dim)
        """
        # Expand weight to match input dimensions for element-wise multiplication
        # weight shape: (1, 1, in_features) -> expand to (batch_size, 1, in_features)
        expanded_weight = self.weight.unsqueeze(0).expand(X.size(0), 1, -1)

        # Element-wise multiply features with weights
        # (batch_size, output_dim, in_features) * (batch_size, 1, in_features) -> (batch_size, output_dim, in_features)
        # Broadcasting applies the same weight across the output_dim dimension
        output = X * expanded_weight

        if self.use_bias:
             expanded_bias = self.bias.unsqueeze(0).expand(X.size(0), 1, -1)
             output = output + expanded_bias # Apply shared bias

        # Sum features for each output class (domain value)
        # Sum along the last dimension (in_features) -> (batch_size, output_dim)
        output = output.sum(dim=2)

        # Apply the mask: Add a large negative number where mask is non-zero (-1e6)
        # This effectively zeros out probabilities after softmax for invalid domain values.
        output = output + mask # Add mask directly (-1e6 where invalid, 0 where valid)

        return output

class RepairModel:
    """
    Manages the TiedLinear model, training, and prediction.

    Adapted from HoloClean's learn.py.
    """
    def __init__(self, num_features, max_domain_size, lr=0.01, epochs=20, batch_size=64, weight_decay=1e-4, use_bias=False):
        self.num_features = num_features
        self.max_domain_size = max_domain_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.use_bias = use_bias

        # Instantiate the model
        self.model = TiedLinear(self.num_features, self.max_domain_size, bias=self.use_bias)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info(f"[RepairModel] Model initialized on device: {self.device}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def fit(self, X_train, Y_train, mask_train):
        """Trains the model."""
        if X_train is None or Y_train is None or mask_train is None:
            logging.error("[RepairModel] Training data is missing. Cannot fit model.")
            return

        logging.info(f"[RepairModel] Starting training for {self.epochs} epochs...")
        logging.info(f"  Training data shape: X={X_train.shape}, Y={Y_train.shape}, mask={mask_train.shape}")
        logging.info(f"  Batch size: {self.batch_size}, LR: {self.lr}, Weight Decay: {self.weight_decay}")

        # Create DataLoader for batching
        # Ensure Y_train is flattened if it's not already (N,) shape
        if Y_train.dim() > 1 and Y_train.shape[1] == 1:
             Y_train = Y_train.squeeze(1)
        if Y_train.dim() > 1:
             logging.error(f"Y_train has incorrect shape {Y_train.shape}, expected (N,) or (N,1).")
             return

        # Filter out samples where Y_train is -1 (indicating missing evidence)
        valid_indices = (Y_train != -1).nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
             logging.error("[RepairModel] No valid training labels found (all are -1). Cannot train.")
             return
        if len(valid_indices) < X_train.shape[0]:
             logging.warning(f"[RepairModel] Filtering out {X_train.shape[0] - len(valid_indices)} samples with invalid labels (-1).")

        X_train_filtered = X_train[valid_indices]
        Y_train_filtered = Y_train[valid_indices]
        mask_train_filtered = mask_train[valid_indices]

        train_dataset = TensorDataset(X_train_filtered, Y_train_filtered, mask_train_filtered)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train() # Set model to training mode
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch_X, batch_Y, batch_mask in pbar:
                # Move data to the appropriate device
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                batch_mask = batch_mask.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                # Output scores shape: (batch_size, max_domain_size)
                scores = self.model(batch_X, batch_mask)

                # Calculate loss
                # CrossEntropyLoss expects scores (N, C) and targets (N)
                # where C is number of classes (max_domain_size)
                # Targets should be class indices (0 to C-1)
                loss = self.criterion(scores, batch_Y)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0) # Accumulate loss weighted by batch size
                total_samples += batch_X.size(0)

                # Calculate accuracy (optional, for monitoring)
                _, predicted_indices = torch.max(scores.data, 1)
                correct_predictions += (predicted_indices == batch_Y).sum().item()

                pbar.set_postfix({'loss': loss.item()}) # Show current batch loss

            avg_epoch_loss = epoch_loss / total_samples
            epoch_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
            logging.info(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_epoch_loss:.6f}, Accuracy: {epoch_accuracy:.2f}%")

        train_time = time.time() - start_time
        logging.info(f"[RepairModel] Training finished in {train_time:.2f}s.")


    def predict_proba(self, X_pred, mask_pred):
        """Generates probability distributions for the inference data."""
        if X_pred is None or mask_pred is None:
            logging.error("[RepairModel] Inference data is missing. Cannot predict.")
            return None

        logging.info(f"[RepairModel] Starting prediction on {X_pred.shape[0]} samples...")
        pred_start_time = time.time()

        # Set model to evaluation mode (disables dropout, batch norm updates etc.)
        self.model.eval()

        # Create DataLoader for inference (batching can speed it up)
        pred_dataset = TensorDataset(X_pred, mask_pred)
        # Use larger batch size for prediction if memory allows
        pred_loader = DataLoader(pred_dataset, batch_size=self.batch_size * 4, shuffle=False)

        all_probabilities = []

        with torch.no_grad(): # Disable gradient calculations for inference
            pbar = tqdm(pred_loader, desc="  Predicting")
            for batch_X, batch_mask in pbar:
                batch_X = batch_X.to(self.device)
                batch_mask = batch_mask.to(self.device)

                # Forward pass to get scores
                scores = self.model(batch_X, batch_mask)

                # Apply softmax to get probabilities
                probabilities = softmax(scores, dim=1)

                # Move probabilities to CPU and store
                all_probabilities.append(probabilities.cpu())

        # Concatenate results from all batches
        if not all_probabilities:
             logging.error("[RepairModel] No probabilities were generated during prediction.")
             return None

        final_probabilities = torch.cat(all_probabilities, dim=0)
        predict_time = time.time() - pred_start_time
        logging.info(f"[RepairModel] Prediction finished in {predict_time:.2f}s.")

        return final_probabilities # Shape: (num_pred_samples, max_domain_size)

    def get_weights(self):
         """Returns the learned weights."""
         if self.model and hasattr(self.model, 'weight'):
              return self.model.weight.data.cpu().numpy()
         return None