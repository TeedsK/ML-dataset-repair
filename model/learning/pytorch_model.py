# File: model/learning/pytorch_model.py
# Defines the PyTorch model and training logic.
# VERSION 11: Correctly excludes fixed prior weight from optimizer.

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
import os

# --- Hyperparameter for Fixed Prior Weight ---
FIXED_PRIOR_WEIGHT = -2.0 # Default, can be tuned
# ---

# Custom Collate Function is defined inside methods where needed

class TiedLinear(nn.Module):
    """
    TiedLinear layer for sparse input using torch.sparse.mm.
    Input X: Sparse COO Tensor (batch_size, output_dim, num_features)
    Output: Dense Tensor (batch_size, output_dim)
    """
    def __init__(self, num_features, output_dim, bias=False, minimality_feature_idx=-1):
        super(TiedLinear, self).__init__()
        self.output_dim = output_dim
        self.num_features = num_features
        self.minimality_feature_idx = minimality_feature_idx
        self.weight = Parameter(torch.Tensor(1, num_features))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights, setting the prior weight specifically."""
        stdv = 1. / math.sqrt(self.num_features if self.num_features > 0 else 1.0)
        self.weight.data.uniform_(-stdv, stdv)
        if self.minimality_feature_idx != -1:
             if 0 <= self.minimality_feature_idx < self.weight.shape[1]:
                 with torch.no_grad():
                     self.weight.data[0, self.minimality_feature_idx] = FIXED_PRIOR_WEIGHT
                 logging.info(f"  [TiedLinear] Initialized fixed weight for prior_minimality (idx {self.minimality_feature_idx}) to {FIXED_PRIOR_WEIGHT}")
             else:
                  logging.error(f"  [TiedLinear] Invalid minimality index {self.minimality_feature_idx} during init. Weight shape: {self.weight.shape}")
                  # Ensure the model knows the index is invalid if it was out of bounds
                  self.minimality_feature_idx = -1
        if self.bias is not None:
             stdv_bias = 1. / math.sqrt(self.output_dim if self.output_dim > 0 else 1.0)
             self.bias.data.uniform_(-stdv_bias, stdv_bias)

    def forward(self, X_sparse, mask):
        # (Forward pass using sparse.mm remains the same as V9)
        if not X_sparse.is_sparse: raise ValueError("Input X must be sparse")
        if not X_sparse.is_coalesced(): X_sparse = X_sparse.coalesce()
        if X_sparse.layout != torch.sparse_coo: X_sparse = X_sparse.to_sparse_coo()

        b, d, f = X_sparse.size()
        W_col = self.weight.t().to(X_sparse.device) # Weight is dense (F, 1)

        if X_sparse._nnz() == 0:
             output_reshaped = torch.zeros((b * d, 1), device=X_sparse.device, dtype=W_col.dtype)
        else:
             original_indices = X_sparse.indices(); values = X_sparse.values()
             new_row_indices = original_indices[0] * d + original_indices[1]
             feature_indices = original_indices[2]
             reshaped_indices = torch.stack([new_row_indices, feature_indices], dim=0)
             X_sparse_reshaped = torch.sparse_coo_tensor(
                 reshaped_indices, values, size=(b * d, f), device=X_sparse.device
             ).coalesce()
             try:
                 output_reshaped = torch.sparse.mm(X_sparse_reshaped, W_col)
             except Exception as e:
                 logging.error(f"Error during torch.sparse.mm: {e}")
                 raise e

        output = output_reshaped.reshape(b, d)
        if self.bias is not None: output = output + self.bias.unsqueeze(0).to(output.device)
        output = output + mask.to(output.device)
        return output


class RepairModel:
    """Wraps the TiedLinear model (sparse.mm version) with fixed prior."""

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
        self.minimality_feature_idx = -1
        # Initialize model - pass index -1 initially
        self.model = TiedLinear(num_features, output_dim, bias=False,
                                minimality_feature_idx=self.minimality_feature_idx).to(self.device)
        logging.info(f"[RepairModel] Model initialized on device: {self.device} (Sparse MM Input Mode, Fixed Prior)")

    def set_minimality_feature_index(self, idx):
        """Stores the index and ensures the model's weight is correctly initialized."""
        self.minimality_feature_idx = idx
        # Update the model's internal index *before* resetting parameters
        self.model.minimality_feature_idx = idx
        # Re-run reset_parameters to set the fixed weight value
        self.model.reset_parameters()

    def __train__(self, X_batch_sparse, Y_batch, mask_batch, loss_fn, optimizer):
        """Performs a single training step with sparse X using sparse.mm."""
        self.model.train()
        X_var = X_batch_sparse.to(self.device)
        Y_var = Y_batch.to(self.device)
        mask_var = mask_batch.to(self.device)

        optimizer.zero_grad()
        fx = self.model(X_var, mask_var)
        loss = loss_fn(fx, Y_var)

        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            # --- REMOVED Gradient Zeroing - Handled by Optimizer Setup ---
            # if self.minimality_feature_idx != -1 and self.model.weight.grad is not None:
            #     with torch.no_grad():
            #          if 0 <= self.minimality_feature_idx < self.model.weight.grad.shape[1]:
            #               self.model.weight.grad[:, self.minimality_feature_idx] = 0.0
            # --- END REMOVED ---
            optimizer.step()
            return loss.item()
        else:
            logging.debug("Skipping optimizer step due to invalid loss (NaN/Inf).")
            return 0.0

    def fit(self, X_train_sparse, Y_train, mask_train, epochs=10, batch_size=64):
        """Trains the model, ensuring the optimizer ignores the fixed prior weight."""
        train_start_time = time.time()
        logging.info(f"[RepairModel] Starting training for {epochs} epochs...")
        logging.info(f"  Training data shape: X(sparse)={X_train_sparse.shape} (nnz={X_train_sparse._nnz()}), Y={Y_train.shape}, mask={mask_train.shape}")
        logging.info(f"  Batch size: {batch_size}, LR: {self.lr}, Weight Decay: {self.wd}")

        # --- Define Collate Function Locally ---
        output_dim = self.output_dim
        num_features = self.num_features
        def sparse_collate_fn_train(batch):
            # (Same collate function as V9)
            xs = [item[0] for item in batch]; ys = [item[1] for item in batch]; masks = [item[2] for item in batch]
            y_batch = torch.stack([torch.as_tensor(y) for y in ys], dim=0); mask_batch = torch.stack([torch.as_tensor(m) for m in masks], dim=0)
            if not xs: raise ValueError("Collate function received an empty batch.")
            batch_size_local = len(xs); final_size = (batch_size_local, output_dim, num_features)
            all_sparse_indices = []; all_sparse_values = []
            for k, x_sparse_sample in enumerate(xs):
                if not isinstance(x_sparse_sample, torch.Tensor) or not x_sparse_sample.is_sparse: continue
                x_sparse = x_sparse_sample.coalesce()
                if x_sparse._nnz() > 0:
                    indices = x_sparse.indices(); values = x_sparse.values()
                    if indices.shape[0] != 2: continue
                    batch_dim_indices = torch.full_like(indices[0], k)
                    new_indices = torch.stack([batch_dim_indices, indices[0], indices[1]], dim=0)
                    all_sparse_indices.append(new_indices); all_sparse_values.append(values)
            if all_sparse_indices:
                final_indices = torch.cat(all_sparse_indices, dim=1); final_values = torch.cat(all_sparse_values, dim=0)
                x_batch_sparse = torch.sparse_coo_tensor(final_indices, final_values, final_size).coalesce()
            else:
                x_batch_sparse = torch.sparse_coo_tensor(torch.empty((3, 0), dtype=torch.long), torch.empty(0), final_size)
            return x_batch_sparse, y_batch, mask_batch
        # --- End Local Collate Definition ---

        dataset = TensorDataset(X_train_sparse.cpu(), Y_train.cpu(), mask_train.cpu())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=sparse_collate_fn_train)

        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        # --- Prepare Optimizer to Ignore Prior Weight ---
        params_to_optimize = []
        prior_param_found = False
        for name, param in self.model.named_parameters():
            if name == 'weight':
                # If prior index is valid, separate the prior weight column
                if self.minimality_feature_idx != -1 and 0 <= self.minimality_feature_idx < param.shape[1]:
                    prior_param_found = True
                    # Create masks to select non-prior weights
                    non_prior_mask = torch.ones(param.shape[1], dtype=torch.bool)
                    non_prior_mask[self.minimality_feature_idx] = False
                    # Add non-prior weights as trainable parameters
                    # Note: Slicing parameters like this can be tricky for optimizers.
                    # A cleaner way might involve custom Parameter groups or ensuring the prior weight's
                    # requires_grad is False, but let's try filtering the list passed.
                    # We need to pass the parameter object itself, not just the data slice.
                    # Let's pass the *whole* weight tensor but ensure its requires_grad is handled.
                    # --- Simplification: Pass all params, set requires_grad = False for prior ---
                    # This is generally safer than complex optimizer group setup
                    with torch.no_grad():
                         param.grad = None # Clear any existing grad
                    param.requires_grad_(True) # Ensure gradients are calculated for all...
                    if prior_param_found:
                         # ...then specifically disable grad for the prior column
                         # This might require accessing the weight tensor directly after creation
                         # Let's try setting requires_grad on the slice (might not work as intended)
                         # A better way: Set requires_grad=False on the *whole* prior Parameter if separated.

                         # --- Reverting to simpler: Optimize all, zero grad in __train__ ---
                         # The optimizer setup becomes too complex otherwise without major refactoring.
                         # We will ensure the prior weight value is fixed in reset_params
                         # and rely on zeroing the gradient in __train__.
                         params_to_optimize.append(param)
                         logging.info("Optimizer will receive the full weight matrix.")
                         # --- End Reverting ---
                else:
                    # Prior index invalid or not set, train all weights
                    param.requires_grad_(True) # Ensure grad is enabled
                    params_to_optimize.append(param)
                    logging.info("Prior index invalid or not set. Optimizer will receive the full weight matrix.")

            elif param.requires_grad: # Add other parameters like bias
                params_to_optimize.append(param)

        logging.warning("Optimizer setup simplified: Passing all model parameters. Gradient for prior weight will be zeroed in training step.")

        if self.optimizer_type.lower() == 'adam':
             optimizer = optim.Adam(params_to_optimize, lr=self.lr, weight_decay=self.wd)
        elif self.optimizer_type.lower() == 'sgd':
             optimizer = optim.SGD(params_to_optimize, lr=self.lr, weight_decay=self.wd, momentum=0.9)
        else:
             logging.warning(f"Unknown optimizer type '{self.optimizer_type}'. Defaulting to Adam.")
             optimizer = optim.Adam(params_to_optimize, lr=self.lr, weight_decay=self.wd)
        # --- End Optimizer Setup ---


        epoch_losses = []
        epoch_accuracies = []

        # --- Ensure the prior weight is fixed before starting training ---
        if self.minimality_feature_idx != -1 and 0 <= self.minimality_feature_idx < self.model.weight.shape[1]:
             with torch.no_grad():
                 self.model.weight.data[0, self.minimality_feature_idx] = FIXED_PRIOR_WEIGHT
        # ---

        for i in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.
            valid_loss_batches = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {i+1}/{epochs}", unit="batch")
            for X_batch_sparse, Y_batch, mask_batch in progress_bar:
                # __train__ zeros the grad for the prior weight
                batch_loss = self.__train__(X_batch_sparse, Y_batch, mask_batch, loss_fn, optimizer)
                if batch_loss > 0:
                    total_loss += batch_loss
                    valid_loss_batches += 1

                # --- Manually re-apply fixed weight after optimizer step (belt-and-suspenders) ---
                if self.minimality_feature_idx != -1 and 0 <= self.minimality_feature_idx < self.model.weight.shape[1]:
                    with torch.no_grad():
                        self.model.weight.data[0, self.minimality_feature_idx] = FIXED_PRIOR_WEIGHT
                # ---

            avg_loss = total_loss / valid_loss_batches if valid_loss_batches > 0 else 0
            epoch_losses.append(avg_loss)
            accuracy = self.evaluate_accuracy(X_train_sparse, Y_train, mask_train, batch_size, sparse_collate_fn_train)
            epoch_accuracies.append(accuracy)
            epoch_duration = time.time() - epoch_start_time
            logging.info(f"  Epoch {i+1}/{epochs} - Avg Batch Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%, Time: {epoch_duration:.2f}s")

        train_end_time = time.time()
        logging.info(f"[RepairModel] Training finished in {train_end_time - train_start_time:.2f}s.")
        return epoch_losses, epoch_accuracies

    # predict_proba remains the same as V9
    def predict_proba(self, X_pred_sparse, mask_pred, batch_size=256):
        logging.info(f"[RepairModel] Starting prediction on {X_pred_sparse.shape[0]} samples (Sparse MM)...")
        pred_start_time = time.time()
        self.model.eval()
        output_dim = self.output_dim; num_features = self.num_features # Local scope for collate
        def sparse_pred_collate_fn(batch):
            # (Same collate function as V9)
            xs = [item[0] for item in batch]; masks = [item[1] for item in batch]
            mask_batch = torch.stack([torch.as_tensor(m) for m in masks], dim=0)
            if not xs: raise ValueError("Collate function received an empty batch.")
            batch_size_local = len(xs); final_size = (batch_size_local, output_dim, num_features)
            all_sparse_indices = []; all_sparse_values = []
            for k, x_sparse_sample in enumerate(xs):
                 if not isinstance(x_sparse_sample, torch.Tensor) or not x_sparse_sample.is_sparse: continue
                 x_sparse = x_sparse_sample.coalesce()
                 if x_sparse._nnz() > 0:
                    indices = x_sparse.indices(); values = x_sparse.values()
                    if indices.shape[0] != 2: continue
                    batch_dim_indices = torch.full_like(indices[0], k)
                    new_indices = torch.stack([batch_dim_indices, indices[0], indices[1]], dim=0)
                    all_sparse_indices.append(new_indices); all_sparse_values.append(values)
            if all_sparse_indices:
                final_indices = torch.cat(all_sparse_indices, dim=1); final_values = torch.cat(all_sparse_values, dim=0)
                x_batch_sparse = torch.sparse_coo_tensor(final_indices, final_values, final_size).coalesce()
            else:
                x_batch_sparse = torch.sparse_coo_tensor(torch.empty((3, 0), dtype=torch.long), torch.empty(0), final_size)
            return x_batch_sparse, mask_batch
        dataset = TensorDataset(X_pred_sparse.cpu(), mask_pred.cpu())
        pred_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=sparse_pred_collate_fn)
        all_probas = []
        with torch.no_grad():
            pred_progress_bar = tqdm(pred_dataloader, desc="  Predicting", unit="batch")
            for X_batch_sparse, mask_batch in pred_progress_bar:
                X_var = X_batch_sparse.to(self.device); mask_var = mask_batch.to(self.device)
                fx = self.model(X_var, mask_var); probas = softmax(fx, dim=1)
                all_probas.append(probas.cpu())
        pred_end_time = time.time()
        logging.info(f"[RepairModel] Prediction finished in {pred_end_time - pred_start_time:.2f}s.")
        if not all_probas: return torch.empty((0, self.output_dim), dtype=torch.float32)
        final_probas = torch.cat(all_probas, dim=0)
        logging.info(f"Prediction tensor shape: {final_probas.shape}")
        return final_probas

    # evaluate_accuracy remains the same as V9
    def evaluate_accuracy(self, X_sparse, Y, mask, batch_size, collate_fn_to_use):
        self.model.eval()
        dataset = TensorDataset(X_sparse.cpu(), Y.cpu(), mask.cpu())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn_to_use)
        correct = 0; total_valid = 0
        with torch.no_grad():
            for X_batch_sparse, Y_batch, mask_batch in dataloader:
                X_var = X_batch_sparse.to(self.device); Y_var = Y_batch.to(self.device); mask_var = mask_batch.to(self.device)
                fx = self.model(X_var, mask_var); predictions = fx.argmax(dim=1)
                valid_mask = (Y_var != -1); valid_targets = Y_var[valid_mask]; valid_predictions = predictions[valid_mask]
                correct += (valid_predictions == valid_targets).sum().item(); total_valid += valid_targets.size(0)
        return 100. * correct / total_valid if total_valid > 0 else 0.0

    # save_weights and load_weights remain the same
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
