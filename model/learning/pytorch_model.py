# File: model/learning/pytorch_model.py
# Defines the PyTorch model and training logic.
# VERSION 7: Converts sparse batch to dense inside forward pass.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from torch.nn.functional import softmax
# Import DataLoader and TensorDataset
from torch.utils.data import TensorDataset, DataLoader
import math
import logging
from tqdm import tqdm
import numpy as np
import time

# Custom Collate Function is still needed for DataLoader
def sparse_collate_fn_train(batch):
    # Separates sparse X, dense Y, dense mask
    # Stacks dense Y and mask
    # Combines sparse X tensors into a single batched sparse tensor
    # (Implementation from previous version - VERSION 6)
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    masks = [item[2] for item in batch]
    y_batch = torch.stack([torch.as_tensor(y) for y in ys], dim=0)
    mask_batch = torch.stack([torch.as_tensor(m) for m in masks], dim=0)
    if not xs: raise ValueError("Collate function received an empty batch.")

    # Get dimensions from the first sample's sparse tensor shape
    # Assuming all samples have the same logical shape before slicing
    # We need the intended full dimensions (output_dim, num_features)
    # These should be consistent across the dataset.
    # Let's assume xs[0] represents a slice and get expected shape info
    # This part is tricky if the collate doesn't know the model's output_dim/num_features
    # For simplicity, let's assume the collate function can get these values
    # Ideally, pass output_dim and num_features to the collate function if needed
    # Or retrieve from the first tensor's shape if it's reliable
    try:
        # Attempt to get shape; this might fail if xs[0] is empty or dimensions differ
        # We rely on the model passing these dimensions correctly later
        output_dim = xs[0].size(0) # This assumes the slice retains domain dim
        num_features = xs[0].size(1) # This assumes the slice retains feature dim
        # Note: This inference based on the slice might be fragile.
        # A better approach passes expected dims.
    except IndexError:
         # Fallback or error if shape inference fails
         # This indicates an issue with how data is stored or sliced.
         # For now, let's try to proceed assuming dimensions are consistent
         # and will be provided by the model context later.
         # We'll define the final size based on model params anyway.
         # Just need batch size here.
         pass


    batch_size_local = len(xs)
    # *** We need the ACTUAL output_dim and num_features for the final shape ***
    # *** This collate function cannot know them reliably ***
    # *** The fix is to define it locally in fit/predict/evaluate ***
    # *** The code below assumes it's defined locally and has access ***
    # *** to output_dim and num_features variables from the outer scope ***
    # final_size = (batch_size_local, output_dim, num_features) # Placeholder

    all_sparse_indices = []
    all_sparse_values = []
    for k, x_sparse_sample in enumerate(xs):
        if not isinstance(x_sparse_sample, torch.Tensor) or not x_sparse_sample.is_sparse:
             logging.warning(f"Sample {k} in batch is not sparse, type: {type(x_sparse_sample)}. Skipping its features.")
             continue

        x_sparse = x_sparse_sample.coalesce()

        if x_sparse._nnz() > 0:
            indices = x_sparse.indices()
            values = x_sparse.values()
            if indices.shape[0] != 2:
                 logging.error(f"Unexpected sparse tensor dimension after slicing/coalescing: {indices.shape[0]} != 2. Sample index {k}. Skipping features.")
                 continue

            batch_dim_indices = torch.full_like(indices[0], k)
            new_indices = torch.stack([batch_dim_indices, indices[0], indices[1]], dim=0)
            all_sparse_indices.append(new_indices)
            all_sparse_values.append(values)

    # --- This part needs output_dim and num_features from the model ---
    # --- It MUST be defined locally within fit/predict/evaluate ---
    # if all_sparse_indices:
    #     final_indices = torch.cat(all_sparse_indices, dim=1)
    #     final_values = torch.cat(all_sparse_values, dim=0)
    #     # final_size needs correct output_dim, num_features
    #     x_batch_sparse = torch.sparse_coo_tensor(final_indices, final_values, final_size)
    # else:
    #     x_batch_sparse = torch.sparse_coo_tensor(torch.empty((3, 0), dtype=torch.long), torch.empty(0), final_size)
    # return x_batch_sparse, y_batch, mask_batch
    # --- End Placeholder ---
    # We will redefine this locally in the methods below.


class TiedLinear(nn.Module):
    """
    TiedLinear layer. Now expects DENSE input batch.
    Input X: Dense Tensor (batch_size, output_dim, num_features)
    Output: Dense Tensor (batch_size, output_dim)
    """
    def __init__(self, num_features, output_dim, bias=False):
        super(TiedLinear, self).__init__()
        self.output_dim = output_dim
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(1, num_features))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.num_features if self.num_features > 0 else 1.0)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
             stdv_bias = 1. / math.sqrt(self.output_dim if self.output_dim > 0 else 1.0)
             self.bias.data.uniform_(-stdv_bias, stdv_bias)

    def forward(self, X_dense, mask): # Changed input name
        # X_dense shape: (batch_size, output_dim, num_features), DENSE
        # self.weight shape: (1, num_features)
        # mask shape: (batch_size, output_dim), dense

        if X_dense.is_sparse:
             # This shouldn't happen if called correctly now
             raise ValueError("Input X to TiedLinear (Dense Batch Version) must be a dense tensor.")

        # --- Dense Computation ---
        # Ensure weight is on the same device as input
        w_expanded = self.weight.expand(self.output_dim, -1).unsqueeze(0).to(X_dense.device) # Shape: (1, output_dim, num_features)

        # Element-wise multiplication and sum over features
        # output shape: (batch_size, output_dim)
        output = torch.sum(X_dense * w_expanded, dim=2)
        # --- End Dense Computation ---

        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).to(output.device)

        output = output + mask.to(output.device)

        return output


class RepairModel:
    """Wraps the TiedLinear model. Converts sparse batch to dense in forward."""

    def __init__(self, num_features, output_dim,
                 learning_rate=0.01, weight_decay=1e-6, optimizer_type='adam',
                 device='cpu'):
        self.num_features = num_features
        self.output_dim = output_dim
        self.lr = learning_rate
        self.wd = weight_decay
        self.optimizer_type = optimizer_type
        self.device = torch.device(device)
        if self.num_features <= 0 or self.output_dim <= 0:
             raise ValueError(f"num_features ({self.num_features}) and output_dim ({self.output_dim}) must be positive.")
        # Model expects dense input now, but initialization is the same
        self.model = TiedLinear(num_features, output_dim, bias=False).to(self.device)
        logging.info(f"[RepairModel] Model initialized on device: {self.device} (Dense Batch Input Mode)")
        self.minimality_feature_idx = -1

    def set_minimality_feature_index(self, idx):
        self.minimality_feature_idx = idx
        if idx != -1:
             logging.info(f"[RepairModel] Will fix weight for minimality feature at index: {idx}")
        else:
             logging.warning("[RepairModel] Minimality feature index not set. Its weight will be learned.")

    def __train__(self, X_batch_sparse, Y_batch, mask_batch, loss_fn, optimizer):
        """Performs a single training step. Converts sparse X to dense."""
        self.model.train()
        # Move dense tensors to device
        Y_var = Y_batch.to(self.device)
        mask_var = mask_batch.to(self.device)

        # --- Convert Sparse X to Dense ---
        try:
            # Ensure X_batch_sparse is coalesced before converting
            if not X_batch_sparse.is_coalesced():
                X_batch_sparse = X_batch_sparse.coalesce()
            X_var_dense = X_batch_sparse.to_dense().to(self.device)
            logging.debug(f"Converted sparse batch to dense, shape: {X_var_dense.shape}")
        except Exception as e:
            logging.error(f"Error converting sparse batch to dense: {e}")
            logging.error(f"Sparse batch info: size={X_batch_sparse.size()}, nnz={X_batch_sparse._nnz()}, is_coalesced={X_batch_sparse.is_coalesced()}")
            # Skip this batch if conversion fails?
            return 0.0 # Return 0 loss to indicate failure for this batch
        # --- End Conversion ---

        optimizer.zero_grad()
        # Forward pass with DENSE X
        fx = self.model(X_var_dense, mask_var)
        loss = loss_fn(fx, Y_var)

        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            if self.minimality_feature_idx != -1 and self.model.weight.grad is not None:
                with torch.no_grad():
                     if 0 <= self.minimality_feature_idx < self.model.weight.grad.shape[1]:
                          self.model.weight.grad[:, self.minimality_feature_idx] = 0.0
                     else:
                          logging.warning(f"Minimality feature index {self.minimality_feature_idx} out of bounds for weight grad shape {self.model.weight.grad.shape}. Not zeroing.")
            optimizer.step()
            return loss.item()
        else:
            logging.debug("Skipping optimizer step due to invalid loss (NaN/Inf).")
            return 0.0

    def fit(self, X_train_sparse, Y_train, mask_train, epochs=10, batch_size=64):
        """Trains the model. Uses sparse collate, converts to dense in __train__."""
        train_start_time = time.time()
        logging.info(f"[RepairModel] Starting training for {epochs} epochs...")
        logging.info(f"  Training data shape: X(sparse)={X_train_sparse.shape} (nnz={X_train_sparse._nnz()}), Y={Y_train.shape}, mask={mask_train.shape}")
        logging.info(f"  Batch size: {batch_size}, LR: {self.lr}, Weight Decay: {self.wd}")

        # --- Define Collate Function Locally (Still needed for DataLoader) ---
        output_dim = self.output_dim
        num_features = self.num_features
        def sparse_collate_fn_train(batch):
            # (Same collate function as VERSION 6 - creates batched sparse tensor)
            xs = [item[0] for item in batch]
            ys = [item[1] for item in batch]
            masks = [item[2] for item in batch]
            y_batch = torch.stack([torch.as_tensor(y) for y in ys], dim=0)
            mask_batch = torch.stack([torch.as_tensor(m) for m in masks], dim=0)
            if not xs: raise ValueError("Collate function received an empty batch.")
            batch_size_local = len(xs)
            final_size = (batch_size_local, output_dim, num_features)
            all_sparse_indices = []
            all_sparse_values = []
            for k, x_sparse_sample in enumerate(xs):
                if not isinstance(x_sparse_sample, torch.Tensor) or not x_sparse_sample.is_sparse:
                     logging.warning(f"Sample {k} in batch is not sparse, type: {type(x_sparse_sample)}. Skipping features.")
                     continue
                x_sparse = x_sparse_sample.coalesce()
                if x_sparse._nnz() > 0:
                    indices = x_sparse.indices()
                    values = x_sparse.values()
                    if indices.shape[0] != 2:
                         logging.error(f"Unexpected sparse tensor dimension after slicing/coalescing: {indices.shape[0]} != 2. Sample index {k}. Skipping features.")
                         continue
                    batch_dim_indices = torch.full_like(indices[0], k)
                    new_indices = torch.stack([batch_dim_indices, indices[0], indices[1]], dim=0)
                    all_sparse_indices.append(new_indices)
                    all_sparse_values.append(values)
            if all_sparse_indices:
                final_indices = torch.cat(all_sparse_indices, dim=1)
                final_values = torch.cat(all_sparse_values, dim=0)
                x_batch_sparse = torch.sparse_coo_tensor(final_indices, final_values, final_size)
            else:
                x_batch_sparse = torch.sparse_coo_tensor(torch.empty((3, 0), dtype=torch.long), torch.empty(0), final_size)
            return x_batch_sparse, y_batch, mask_batch
        # --- End Local Collate Definition ---

        dataset = TensorDataset(X_train_sparse.cpu(), Y_train.cpu(), mask_train.cpu())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0,
                                collate_fn=sparse_collate_fn_train) # Still use sparse collate

        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

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
            valid_loss_batches = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {i+1}/{epochs}", unit="batch")
            for X_batch_sparse, Y_batch, mask_batch in progress_bar:
                # __train__ now handles conversion to dense internally
                batch_loss = self.__train__(X_batch_sparse, Y_batch, mask_batch, loss_fn, optimizer)
                if batch_loss > 0:
                    total_loss += batch_loss
                    valid_loss_batches += 1

            avg_loss = total_loss / valid_loss_batches if valid_loss_batches > 0 else 0
            epoch_losses.append(avg_loss)
            # Accuracy eval also needs to handle the dense conversion
            accuracy = self.evaluate_accuracy(X_train_sparse, Y_train, mask_train, batch_size, sparse_collate_fn_train)
            epoch_accuracies.append(accuracy)
            epoch_duration = time.time() - epoch_start_time
            logging.info(f"  Epoch {i+1}/{epochs} - Avg Batch Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%, Time: {epoch_duration:.2f}s")

        train_end_time = time.time()
        logging.info(f"[RepairModel] Training finished in {train_end_time - train_start_time:.2f}s.")
        return epoch_losses, epoch_accuracies

    def predict_proba(self, X_pred_sparse, mask_pred, batch_size=256):
        """Predicts class probabilities. Converts sparse batch to dense in forward."""
        logging.info(f"[RepairModel] Starting prediction on {X_pred_sparse.shape[0]} samples (Dense Batch)...")
        pred_start_time = time.time()
        self.model.eval()

        # --- Define Collate Function Locally for Prediction ---
        output_dim = self.output_dim
        num_features = self.num_features
        def sparse_pred_collate_fn(batch):
            # (Same collate function as VERSION 6 - creates batched sparse tensor)
            xs = [item[0] for item in batch]
            masks = [item[1] for item in batch]
            mask_batch = torch.stack([torch.as_tensor(m) for m in masks], dim=0)
            if not xs: raise ValueError("Collate function received an empty batch.")
            batch_size_local = len(xs)
            final_size = (batch_size_local, output_dim, num_features)
            all_sparse_indices = []
            all_sparse_values = []
            for k, x_sparse_sample in enumerate(xs):
                 if not isinstance(x_sparse_sample, torch.Tensor) or not x_sparse_sample.is_sparse:
                     logging.warning(f"Prediction Sample {k} in batch is not sparse, type: {type(x_sparse_sample)}. Skipping features.")
                     continue
                 x_sparse = x_sparse_sample.coalesce()
                 if x_sparse._nnz() > 0:
                    indices = x_sparse.indices()
                    values = x_sparse.values()
                    if indices.shape[0] != 2:
                         logging.error(f"Unexpected sparse tensor dimension in prediction slice: {indices.shape[0]} != 2. Sample index {k}. Skipping features.")
                         continue
                    batch_dim_indices = torch.full_like(indices[0], k)
                    new_indices = torch.stack([batch_dim_indices, indices[0], indices[1]], dim=0)
                    all_sparse_indices.append(new_indices)
                    all_sparse_values.append(values)
            if all_sparse_indices:
                final_indices = torch.cat(all_sparse_indices, dim=1)
                final_values = torch.cat(all_sparse_values, dim=0)
                x_batch_sparse = torch.sparse_coo_tensor(final_indices, final_values, final_size)
            else:
                x_batch_sparse = torch.sparse_coo_tensor(torch.empty((3, 0), dtype=torch.long), torch.empty(0), final_size)
            return x_batch_sparse, mask_batch
        # --- End Local Collate Definition ---

        dataset = TensorDataset(X_pred_sparse.cpu(), mask_pred.cpu())
        pred_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=0, collate_fn=sparse_pred_collate_fn)

        all_probas = []
        with torch.no_grad():
            pred_progress_bar = tqdm(pred_dataloader, desc="  Predicting", unit="batch")
            for X_batch_sparse, mask_batch in pred_progress_bar:
                mask_var = mask_batch.to(self.device)
                # --- Convert Sparse X to Dense ---
                try:
                    if not X_batch_sparse.is_coalesced():
                        X_batch_sparse = X_batch_sparse.coalesce()
                    X_var_dense = X_batch_sparse.to_dense().to(self.device)
                except Exception as e:
                    logging.error(f"Error converting sparse prediction batch to dense: {e}")
                    # Handle error - maybe append zeros or skip? For now, skip.
                    continue
                # --- End Conversion ---

                fx = self.model(X_var_dense, mask_var) # Forward pass with dense X
                probas = softmax(fx, dim=1)
                all_probas.append(probas.cpu())

        pred_end_time = time.time()
        logging.info(f"[RepairModel] Prediction finished in {pred_end_time - pred_start_time:.2f}s.")

        if not all_probas:
             return torch.empty((0, self.output_dim), dtype=torch.float32)

        final_probas = torch.cat(all_probas, dim=0)
        logging.info(f"Prediction tensor shape: {final_probas.shape}")
        return final_probas

    # Needs to handle the dense conversion internally as well
    def evaluate_accuracy(self, X_sparse, Y, mask, batch_size, collate_fn_to_use):
        """Evaluates accuracy. Converts sparse batch to dense internally."""
        self.model.eval()
        dataset = TensorDataset(X_sparse.cpu(), Y.cpu(), mask.cpu())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0,
                                collate_fn=collate_fn_to_use) # Use the passed sparse collate
        correct = 0
        total_valid = 0
        with torch.no_grad():
            for X_batch_sparse, Y_batch, mask_batch in dataloader:
                Y_var = Y_batch.to(self.device)
                mask_var = mask_batch.to(self.device)

                # --- Convert Sparse X to Dense ---
                try:
                    if not X_batch_sparse.is_coalesced():
                        X_batch_sparse = X_batch_sparse.coalesce()
                    X_var_dense = X_batch_sparse.to_dense().to(self.device)
                except Exception as e:
                    logging.error(f"Error converting sparse eval batch to dense: {e}")
                    continue # Skip batch if conversion fails
                # --- End Conversion ---

                fx = self.model(X_var_dense, mask_var) # Forward pass with dense X
                predictions = fx.argmax(dim=1)

                valid_mask = (Y_var != -1)
                valid_targets = Y_var[valid_mask]
                valid_predictions = predictions[valid_mask]

                correct += (valid_predictions == valid_targets).sum().item()
                total_valid += valid_targets.size(0)

        return 100. * correct / total_valid if total_valid > 0 else 0.0

    def save_weights(self, filepath="repair_model_weights.pth"):
        try:
            torch.save(self.model.state_dict(), filepath)
            logging.info(f"[RepairModel] Weights saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model weights: {e}", exc_info=True)

    def load_weights(self, filepath="repair_model_weights.pth"):
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
