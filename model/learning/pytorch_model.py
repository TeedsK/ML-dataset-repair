# File: model/learning/pytorch_model.py
# Defines the PyTorch model and training logic.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from torch.nn.functional import softmax, cross_entropy
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
            self.bias = Parameter(torch.Tensor(1, num_features)) # Shared bias per feature
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Standard initialization for weights
        stdv = 1. / math.sqrt(self.num_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

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
            expanded_bias = self.bias.expand(X.size(1), -1)
            output = output + torch.sum(expanded_bias.unsqueeze(0), dim=2) # Add bias contribution (check if this logic is intended)
            # Simpler bias: Add a bias per output class instead?
            # self.bias = Parameter(torch.Tensor(output_dim)) -> output = output + self.bias

        # Apply the mask *after* the linear computation
        # mask shape: (batch_size, output_dim)
        output = output + mask # Add large negative value for masked entries

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

        # --- ADDED: Placeholder for minimality index ---
        self.minimality_feature_idx = -1
        # --- END ADDED ---

    # --- ADDED: Method to set the minimality index ---
    def set_minimality_feature_index(self, idx):
        self.minimality_feature_idx = idx
        if idx != -1:
             logging.info(f"[RepairModel] Will fix weight for minimality feature at index: {idx}")
        else:
             logging.warning("[RepairModel] Minimality feature index not set. Its weight will be learned.")
    # --- END ADDED ---


    def __train__(self, X_batch, Y_batch, mask_batch, loss_fn, optimizer):
        """Performs a single training step."""
        self.model.train() # Set model to training mode

        # Move data to the correct device
        X_var = X_batch.to(self.device)
        Y_var = Y_batch.to(self.device)
        mask_var = mask_batch.to(self.device)

        optimizer.zero_grad()
        # Forward pass
        fx = self.model(X_var, mask_var)
        # Calculate loss
        loss = loss_fn(fx, Y_var) # Y_var should be shape (batch_size,)
        # Backward pass
        loss.backward()

        # --- ADDED: Zero out gradient for minimality feature weight ---
        if self.minimality_feature_idx != -1 and self.model.weight.grad is not None:
            with torch.no_grad(): # Ensure this operation isn't tracked
                 self.model.weight.grad[:, self.minimality_feature_idx] = 0.0
            # logging.debug(f"Zeroed gradient for feature index {self.minimality_feature_idx}") # Optional: very verbose
        # --- END ADDED ---

        # Optimizer step
        optimizer.step()
        return loss.item()


    def fit(self, X_train, Y_train, mask_train, epochs=10, batch_size=64):
        """Trains the model."""
        train_start_time = time.time()
        logging.info(f"[RepairModel] Starting training for {epochs} epochs...")
        logging.info(f"  Training data shape: X={X_train.shape}, Y={Y_train.shape}, mask={mask_train.shape}")
        logging.info(f"  Batch size: {batch_size}, LR: {self.lr}, Weight Decay: {self.wd}")

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_train, Y_train, mask_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
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
            correct_predictions = 0
            total_samples = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {i+1}/{epochs}", unit="batch")
            for X_batch, Y_batch, mask_batch in progress_bar:
                batch_loss = self.__train__(X_batch, Y_batch, mask_batch, loss_fn, optimizer)
                total_loss += batch_loss * len(Y_batch) # Accumulate loss weighted by batch size
                total_samples += len(Y_batch)

                # Calculate accuracy within the loop (optional, adds overhead)
                # with torch.no_grad():
                #    self.model.eval()
                #    fx_eval = self.model(X_batch.to(self.device), mask_batch.to(self.device))
                #    predictions = fx_eval.argmax(dim=1)
                #    correct_predictions += (predictions == Y_batch.to(self.device)).sum().item()

            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            epoch_losses.append(avg_loss)

            # Calculate accuracy at end of epoch for efficiency
            accuracy = self.evaluate_accuracy(X_train, Y_train, mask_train, batch_size)
            epoch_accuracies.append(accuracy)

            epoch_duration = time.time() - epoch_start_time
            logging.info(f"  Epoch {i+1}/{epochs} - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%, Time: {epoch_duration:.2f}s")


        train_end_time = time.time()
        logging.info(f"[RepairModel] Training finished in {train_end_time - train_start_time:.2f}s.")
        return epoch_losses, epoch_accuracies


    def predict_proba(self, X_pred, mask_pred, batch_size=256):
        """Predicts class probabilities for prediction data."""
        logging.info(f"[RepairModel] Starting prediction on {X_pred.shape[0]} samples...")
        pred_start_time = time.time()
        self.model.eval() # Set model to evaluation mode

        dataset = torch.utils.data.TensorDataset(X_pred, mask_pred)
        # No shuffle for prediction
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probas = []
        with torch.no_grad(): # Disable gradient calculation for inference
            progress_bar = tqdm(dataloader, desc="  Predicting", unit="batch")
            for X_batch, mask_batch in progress_bar:
                X_var = X_batch.to(self.device)
                mask_var = mask_batch.to(self.device)
                fx = self.model(X_var, mask_var)
                probas = softmax(fx, dim=1)
                all_probas.append(probas.cpu()) # Move result back to CPU

        pred_end_time = time.time()
        logging.info(f"[RepairModel] Prediction finished in {pred_end_time - pred_start_time:.2f}s.")

        if not all_probas:
             return torch.empty((0, self.output_dim), dtype=torch.float32)

        final_probas = torch.cat(all_probas, dim=0)
        logging.info(f"Prediction tensor shape: {final_probas.shape}")
        return final_probas

    def evaluate_accuracy(self, X, Y, mask, batch_size=256):
        """Evaluates accuracy on a given dataset."""
        self.model.eval() # Evaluation mode
        dataset = torch.utils.data.TensorDataset(X, Y, mask)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch, mask_batch in dataloader:
                X_var = X_batch.to(self.device)
                Y_var = Y_batch.to(self.device)
                mask_var = mask_batch.to(self.device)
                fx = self.model(X_var, mask_var)
                predictions = fx.argmax(dim=1)
                total += Y_var.size(0)
                correct += (predictions == Y_var).sum().item()
        return 100. * correct / total if total > 0 else 0
    
    # --- ADDED: Save/Load Methods ---
    def save_weights(self, filepath="repair_model_weights.pth"):
        """Saves the model's state dictionary."""
        try:
            torch.save(self.model.state_dict(), filepath)
            logging.info(f"[RepairModel] Weights saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model weights: {e}", exc_info=True)

    def load_weights(self, filepath="repair_model_weights.pth"):
        """Loads weights from a state dictionary file."""
        try:
            state_dict = torch.load(filepath, map_location=self.device) # Load to appropriate device
            self.model.load_state_dict(state_dict)
            self.model.eval() # Set model to evaluation mode after loading weights
            logging.info(f"[RepairModel] Weights loaded from {filepath}")
            return True
        except FileNotFoundError:
            logging.error(f"Model weights file not found: {filepath}")
            return False
        except Exception as e:
            logging.error(f"Error loading model weights: {e}", exc_info=True)
            return False
    # --- END ADDED ---