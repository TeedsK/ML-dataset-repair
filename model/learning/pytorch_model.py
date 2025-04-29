# File: model/learning/pytorch_model.py
# Defines the PyTorch model and training logic.
# VERSION 2: Added ignore_index=-1 to loss and updated accuracy calc.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from torch.nn.functional import softmax # Removed cross_entropy import, using nn.CrossEntropyLoss instance
from torch.autograd import Variable # Older import, might not be needed if using tensors directly
import math
import logging
from tqdm import tqdm
import numpy as np
import time

class TiedLinear(nn.Module):
    """
    TiedLinear is a linear layer where the single weight vector is expanded
    and applied across the domain dimension.
    Input X: (batch_size, output_dim, num_features)
    Output: (batch_size, output_dim)
    """
    def __init__(self, num_features, output_dim, bias=False):
        super(TiedLinear, self).__init__()
        self.output_dim = output_dim
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(1, num_features)) # Single shared weight vector
        if bias:
            # Consider if a bias per output class (output_dim) is more appropriate
            # self.bias = Parameter(torch.Tensor(output_dim))
            self.bias = Parameter(torch.Tensor(1, num_features)) # Shared bias per feature (original)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Standard initialization for weights
        stdv = 1. / math.sqrt(self.num_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # Adjust bias init if changing its shape
            self.bias.data.uniform_(-stdv, stdv) # Original bias init

    def forward(self, X, mask):
        # X shape: (batch_size, output_dim, num_features)
        # self.weight shape: (1, num_features)

        # Expand weight to match output_dim (domain size)
        # expanded_weight shape: (output_dim, num_features)
        expanded_weight = self.weight.expand(X.size(1), -1)

        # Element-wise multiplication and sum over features
        # output shape: (batch_size, output_dim)
        output = torch.sum(X * expanded_weight.unsqueeze(0), dim=2)

        if self.bias is not None:
            # --- Original Bias Logic (per feature, summed) ---
            expanded_bias = self.bias.expand(X.size(1), -1)
            # Check if summing bias contributions this way is intended.
            # Usually bias is added per output class.
            output = output + torch.sum(expanded_bias.unsqueeze(0), dim=2)
            # --- Alternative: Bias per output class ---
            # if self.bias.shape[0] == self.output_dim: # Check if bias has shape (output_dim,)
            #     output = output + self.bias.unsqueeze(0) # Add bias per output class


        # Apply the mask *after* the linear computation
        # mask shape: (batch_size, output_dim)
        # Adding a large negative number simulates masking for softmax/cross_entropy
        output = output + mask

        return output


class RepairModel:
    """Wraps the TiedLinear model and handles training/prediction."""

    def __init__(self, num_features, output_dim,
                 learning_rate=0.01, weight_decay=1e-6, optimizer_type='adam',
                 device='cpu'):
        self.num_features = num_features
        self.output_dim = output_dim # max domain size
        self.lr = learning_rate
        self.wd = weight_decay
        self.optimizer_type = optimizer_type
        self.device = torch.device(device)

        self.model = TiedLinear(num_features, output_dim, bias=False).to(self.device)
        logging.info(f"[RepairModel] Model initialized on device: {self.device}")

        self.minimality_feature_idx = -1

    def set_minimality_feature_index(self, idx):
        self.minimality_feature_idx = idx
        if idx != -1:
             logging.info(f"[RepairModel] Will fix weight for minimality feature at index: {idx}")
        else:
             logging.warning("[RepairModel] Minimality feature index not set. Its weight will be learned.")


    def __train__(self, X_batch, Y_batch, mask_batch, loss_fn, optimizer):
        """Performs a single training step."""
        self.model.train() # Set model to training mode

        # Move data to the correct device
        X_var = X_batch.to(self.device)
        Y_var = Y_batch.to(self.device) # Shape: (batch_size,)
        mask_var = mask_batch.to(self.device) # Shape: (batch_size, output_dim)

        optimizer.zero_grad()
        # Forward pass
        fx = self.model(X_var, mask_var) # Shape: (batch_size, output_dim)
        # Calculate loss - loss_fn now ignores index -1
        loss = loss_fn(fx, Y_var)
        # Backward pass (only if loss is valid - avoid NaN gradients if all samples ignored)
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()

            # Zero out gradient for minimality feature weight
            if self.minimality_feature_idx != -1 and self.model.weight.grad is not None:
                with torch.no_grad():
                     # Ensure index is valid before trying to zero grad
                     if 0 <= self.minimality_feature_idx < self.model.weight.grad.shape[1]:
                          self.model.weight.grad[:, self.minimality_feature_idx] = 0.0
                     else:
                          logging.warning(f"Minimality feature index {self.minimality_feature_idx} out of bounds for weight grad shape {self.model.weight.grad.shape}. Not zeroing.")

            # Optimizer step
            optimizer.step()
            return loss.item()
        else:
            # Handle case where loss is NaN/Inf (e.g., all samples in batch were ignored)
            logging.debug("Skipping optimizer step due to invalid loss (NaN/Inf).")
            return 0.0 # Or return NaN? Returning 0.0 might skew epoch average loss slightly if it happens often


    def fit(self, X_train, Y_train, mask_train, epochs=10, batch_size=64):
        """Trains the model."""
        train_start_time = time.time()
        logging.info(f"[RepairModel] Starting training for {epochs} epochs...")
        logging.info(f"  Training data shape: X={X_train.shape}, Y={Y_train.shape}, mask={mask_train.shape}")
        logging.info(f"  Batch size: {batch_size}, LR: {self.lr}, Weight Decay: {self.wd}")

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_train, Y_train, mask_train)
        # Consider num_workers > 0 for faster data loading if I/O is bottleneck
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # --- MODIFIED: Set ignore_index=-1 ---
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        # --- END MODIFIED ---

        if self.optimizer_type.lower() == 'adam':
             optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer_type.lower() == 'sgd':
             optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0.9)
        else:
             logging.warning(f"Unknown optimizer type '{self.optimizer_type}'. Defaulting to Adam.")
             optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        epoch_losses = []
        epoch_accuracies = []

        for i in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.
            # total_samples counts batches processed where loss was valid
            valid_loss_batches = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {i+1}/{epochs}", unit="batch")
            for X_batch, Y_batch, mask_batch in progress_bar:
                batch_loss = self.__train__(X_batch, Y_batch, mask_batch, loss_fn, optimizer)
                # Only accumulate loss if it was valid
                if batch_loss > 0: # Check > 0 because __train__ returns 0.0 for invalid loss
                    total_loss += batch_loss
                    valid_loss_batches += 1

            # Average loss over batches where loss was computed
            avg_loss = total_loss / valid_loss_batches if valid_loss_batches > 0 else 0
            epoch_losses.append(avg_loss)

            # Calculate accuracy at end of epoch (using updated evaluate_accuracy)
            accuracy = self.evaluate_accuracy(X_train, Y_train, mask_train, batch_size)
            epoch_accuracies.append(accuracy)

            epoch_duration = time.time() - epoch_start_time
            # Report avg loss over batches with valid loss
            logging.info(f"  Epoch {i+1}/{epochs} - Avg Batch Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%, Time: {epoch_duration:.2f}s")


        train_end_time = time.time()
        logging.info(f"[RepairModel] Training finished in {train_end_time - train_start_time:.2f}s.")
        return epoch_losses, epoch_accuracies


    def predict_proba(self, X_pred, mask_pred, batch_size=256):
        """Predicts class probabilities for prediction data."""
        logging.info(f"[RepairModel] Starting prediction on {X_pred.shape[0]} samples...")
        pred_start_time = time.time()
        self.model.eval() # Set model to evaluation mode

        dataset = torch.utils.data.TensorDataset(X_pred, mask_pred)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_probas = []
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="  Predicting", unit="batch")
            for X_batch, mask_batch in progress_bar:
                X_var = X_batch.to(self.device)
                mask_var = mask_batch.to(self.device)
                fx = self.model(X_var, mask_var) # Logits + mask applied
                # Softmax converts logits to probabilities
                probas = softmax(fx, dim=1)
                all_probas.append(probas.cpu())

        pred_end_time = time.time()
        logging.info(f"[RepairModel] Prediction finished in {pred_end_time - pred_start_time:.2f}s.")

        if not all_probas:
             return torch.empty((0, self.output_dim), dtype=torch.float32)

        final_probas = torch.cat(all_probas, dim=0)
        logging.info(f"Prediction tensor shape: {final_probas.shape}")
        return final_probas

    def evaluate_accuracy(self, X, Y, mask, batch_size=256):
        """Evaluates accuracy on a given dataset, ignoring targets == -1."""
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X, Y, mask)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        correct = 0
        total_valid = 0 # Count only samples where target is not -1
        with torch.no_grad():
            for X_batch, Y_batch, mask_batch in dataloader:
                X_var = X_batch.to(self.device)
                Y_var = Y_batch.to(self.device) # Shape: (batch_size,)
                mask_var = mask_batch.to(self.device) # Shape: (batch_size, output_dim)

                fx = self.model(X_var, mask_var) # Shape: (batch_size, output_dim)
                predictions = fx.argmax(dim=1) # Shape: (batch_size,)

                # --- MODIFIED: Only evaluate accuracy where Y_var is not -1 ---
                valid_mask = (Y_var != -1)
                valid_targets = Y_var[valid_mask]
                valid_predictions = predictions[valid_mask]

                correct += (valid_predictions == valid_targets).sum().item()
                total_valid += valid_targets.size(0)
                # --- END MODIFIED ---

        # Calculate accuracy based only on valid samples
        return 100. * correct / total_valid if total_valid > 0 else 0.0

    def save_weights(self, filepath="repair_model_weights.pth"):
        """Saves the model's state dictionary."""
        try:
            torch.save(self.model.state_dict(), filepath)
            logging.info(f"[RepairModel] Weights saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model weights: {e}", exc_info=True)

    def load_weights(self, filepath="repair_model_weights.pth"):
        """Loads weights from a state dictionary file."""
        if not os.path.exists(filepath):
             logging.error(f"Model weights file not found: {filepath}")
             return False
        try:
            state_dict = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logging.info(f"[RepairModel] Weights loaded from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error loading model weights: {e}", exc_info=True)
            return False