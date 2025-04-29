# File: model/learning/tensor_builder.py
# Builds PyTorch tensors from database data.
# VERSION 2: Corrected _load_metadata signature, added Save/Load state methods.

import pandas as pd
import torch
import time
import logging
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pickle # Added for saving/loading
import os # Added for path checks

# Assuming config.py has NULL_REPR_PLACEHOLDER
try:
    import config
    NULL_REPR_PLACEHOLDER = config.NULL_REPR_PLACEHOLDER
except (ImportError, AttributeError):
    logging.warning("Could not import NULL_REPR_PLACEHOLDER from config.py. Using default '__NULL__'.")
    NULL_REPR_PLACEHOLDER = "__NULL__"


class TensorBuilder:
    """Builds feature tensors for training and prediction."""

    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.feature_to_idx = {}
        self.idx_to_feature = {}
        self.num_features = 0
        self.minimality_feature_idx = -1 # Initialize to -1 (not found)
        self.cell_to_var_idx = {} # (tid, attr) -> var_idx (0 to N-1)
        self.var_idx_to_cell = {} # var_idx -> (tid, attr)
        self.num_vars = 0
        self.var_idx_to_domain = defaultdict(dict) # var_idx -> {domain_val_idx: candidate_val_str}
        self.var_idx_to_val_to_idx = defaultdict(dict) # var_idx -> {candidate_val_str: domain_val_idx}
        self.max_domain_size = 0
        self.evidence_var_indices = []
        self.query_var_indices = []
        self.initial_value_idx = {} # var_idx -> index of initial value in its domain (for Y_train)
        self.state_features = defaultdict(list)
        self.X_train = None
        self.Y_train = None
        self.mask_train = None
        self.X_pred = None
        self.mask_pred = None
        self.pred_indices = None
        self.loaded_state = False # Flag to track if state was loaded
        logging.info("[TensorBuilder] Initializing...")
        # Do not call _load_metadata here, call it explicitly or via build_tensors/get_data

    def _get_state_idx(self, var_idx, domain_val_idx):
        """Helper to create a unique index for each state."""
        if self.max_domain_size == 0:
             # This can happen if called before max_domain_size is set (e.g., during initial load)
             # Handle appropriately, maybe raise error or calculate on the fly if possible/needed?
             # For now, let's assume max_domain_size is set before use.
             logging.warning("_get_state_idx called before max_domain_size was determined.")
             # Fallback or error? Let's raise error to catch logic issues.
             raise ValueError("max_domain_size must be set before calculating state index.")
        return var_idx * self.max_domain_size + domain_val_idx

    # --- MODIFIED Signature: Added reuse_* arguments ---
    def _load_metadata(self, reuse_feature_map=None, reuse_max_domain=None):
        """
        Loads metadata. Optionally reuses feature map and domain size from training.
        reuse_feature_map: dict (feature_name -> idx)
        reuse_max_domain: int
        """
    # --- END MODIFIED Signature ---
        # Clear previous state if loading fresh metadata
        if reuse_feature_map is None:
             self.feature_to_idx = {}
             self.idx_to_feature = {}
             self.num_features = 0
             self.minimality_feature_idx = -1

        logging.info("[TensorBuilder] Loading metadata from DB (cells, domains, features)...")
        load_start = time.time()

        # 1. Load/Reuse distinct features and create mapping
        logging.info("  Loading features...")
        if reuse_feature_map is not None:
            # (Keep reuse logic as is)
            logging.info("  Reusing feature map from training phase.")
            self.feature_to_idx = reuse_feature_map
            self.num_features = len(self.feature_to_idx)
            self.idx_to_feature = {v: k for k, v in self.feature_to_idx.items()}
            self.minimality_feature_idx = self.feature_to_idx.get('prior_minimality', -1)
            if self.minimality_feature_idx != -1:
                 logging.info(f"    Found 'prior_minimality' feature at index: {self.minimality_feature_idx}")
            else:
                 logging.warning("  'prior_minimality' feature not found in reused feature map!")
        else:
            # (Keep discover features logic as is)
            features_df = pd.read_sql("SELECT DISTINCT feature FROM features", self.db_conn)
            for idx, feature_name in enumerate(features_df['feature']):
                self.feature_to_idx[feature_name] = idx
                self.idx_to_feature[idx] = feature_name
                if feature_name == 'prior_minimality':
                    self.minimality_feature_idx = idx
                    logging.info(f"    Found 'prior_minimality' feature at index: {idx}")
            self.num_features = len(self.feature_to_idx)
            logging.info(f"  Found {self.num_features} unique features.")
            if self.minimality_feature_idx == -1:
                logging.warning("  'prior_minimality' feature was not found in the database features table!")


        # 2. Load cell info (variables) and separate evidence/query
        # (Keep cell loading logic as is)
        logging.info("  Loading cells info...")
        cells_df = pd.read_sql("SELECT tid, attr, val, is_noisy FROM cells ORDER BY tid, attr", self.db_conn)
        self.num_vars = len(cells_df)
        initial_values_map = {}
        self.evidence_var_indices = []
        self.query_var_indices = []
        self.cell_to_var_idx = {}
        self.var_idx_to_cell = {}
        for idx, row in cells_df.iterrows():
            cell = (int(row['tid']), row['attr'])
            self.cell_to_var_idx[cell] = idx
            self.var_idx_to_cell[idx] = cell
            initial_values_map[cell] = row['val']
            if row['is_noisy']:
                self.query_var_indices.append(idx)
            else:
                self.evidence_var_indices.append(idx)
        logging.info(f"  Processed {self.num_vars} total cells (variables).")
        logging.info(f"  Identified {len(self.evidence_var_indices)} evidence variables and {len(self.query_var_indices)} query variables.")


        # 3. Load domains and determine max size
        # (Keep domain loading logic as is)
        logging.info("  Loading domains...")
        domain_dfs = pd.read_sql("SELECT tid, attr, candidate_val FROM domains ORDER BY tid, attr, candidate_val",
                                self.db_conn, chunksize=50000)
        temp_domains = defaultdict(list)
        self.var_idx_to_domain = defaultdict(dict)
        self.var_idx_to_val_to_idx = defaultdict(dict)
        self.initial_value_idx = {}
        row_count = 0
        pbar_domains = tqdm(desc="  Processing domains", unit=" candidates")
        for chunk in domain_dfs:
            for _, row in chunk.iterrows():
                cell = (int(row['tid']), row['attr'])
                temp_domains[cell].append(str(row['candidate_val']))
                row_count+=1
            pbar_domains.update(len(chunk))
        pbar_domains.close()

        current_max_domain = 0
        for cell, domain_list in temp_domains.items():
            if cell not in self.cell_to_var_idx: continue
            var_idx = self.cell_to_var_idx[cell]
            domain_size = len(domain_list)
            if domain_size > current_max_domain:
                current_max_domain = domain_size
            initial_val = initial_values_map.get(cell)
            initial_val_str = NULL_REPR_PLACEHOLDER if initial_val is None else str(initial_val)
            found_initial_idx = -1
            for domain_idx, candidate_val_str in enumerate(domain_list):
                self.var_idx_to_domain[var_idx][domain_idx] = candidate_val_str
                self.var_idx_to_val_to_idx[var_idx][candidate_val_str] = domain_idx
                if candidate_val_str == initial_val_str:
                    found_initial_idx = domain_idx
            if found_initial_idx != -1:
                 self.initial_value_idx[var_idx] = found_initial_idx
            elif var_idx in self.evidence_var_indices:
                 logging.warning(f"Initial value '{initial_val_str}' for evidence cell {cell} (var_idx {var_idx}) not found in its domain: {domain_list}. Assigning index 0.")
                 self.initial_value_idx[var_idx] = 0

        # Use reused max domain size if provided
        self.max_domain_size = reuse_max_domain if reuse_max_domain is not None else current_max_domain
        if self.max_domain_size == 0 and self.num_vars > 0:
             logging.warning("Max domain size calculated as 0. Check if 'domains' table is populated correctly.")
             # Set to 1 as a fallback to prevent division by zero etc.? Or raise error?
             # Setting to 1 might hide issues. Let's warn and keep it 0.
        logging.info(f"  Using Max domain size: {self.max_domain_size}")


        # 4. Load features per state
        # (Keep feature loading logic as is)
        logging.info("  Loading features per state...")
        self.state_features = defaultdict(list) # Reset
        # Ensure max_domain_size is set before attempting _get_state_idx
        if self.max_domain_size == 0 and self.num_vars > 0:
             logging.error("Cannot proceed with feature mapping as max_domain_size is 0.")
             # Optionally raise an error here
             raise ValueError("max_domain_size is 0, cannot map features to states.")

        feature_chunks = pd.read_sql(
            "SELECT tid, attr, candidate_val, feature FROM features",
            self.db_conn, chunksize=100000
        )
        pbar_features = tqdm(desc="  Mapping features", unit=" features")
        processed_features = 0
        unknown_feature_count = 0
        for chunk in feature_chunks:
            chunk_processed_count = 0
            for _, row in chunk.iterrows():
                cell = (int(row['tid']), row['attr'])
                if cell not in self.cell_to_var_idx: continue
                var_idx = self.cell_to_var_idx[cell]
                candidate_val_str = str(row['candidate_val'])
                feature_name = row['feature']

                if candidate_val_str not in self.var_idx_to_val_to_idx[var_idx]: continue
                domain_val_idx = self.var_idx_to_val_to_idx[var_idx][candidate_val_str]

                if feature_name not in self.feature_to_idx:
                     unknown_feature_count += 1
                     continue

                feature_idx = self.feature_to_idx[feature_name]
                state_idx = self._get_state_idx(var_idx, domain_val_idx) # Requires self.max_domain_size
                self.state_features[state_idx].append(feature_idx)
                chunk_processed_count += 1
            # Use total processed in this chunk for update if tqdm needs it
            pbar_features.update(chunk_processed_count)
            # processed_features += chunk_processed_count # Keep track if needed

        pbar_features.close()
        if unknown_feature_count > 0:
             logging.warning(f"  Skipped {unknown_feature_count} features not present in the loaded/reused feature map.")
        logging.info("  Built feature map for states.")

        load_end = time.time()
        logging.info(f"[TensorBuilder] Metadata loaded and mappings built in {load_end - load_start:.2f}s.")


    # ... (build_tensors, get_training_data, get_prediction_data etc. remain unchanged) ...
    # ... (Make sure build_tensors checks if metadata is loaded before proceeding) ...
    def build_tensors(self):
        """Builds the final PyTorch tensors."""
        if self.X_train is not None:
            logging.warning("Tensors already built. Skipping.")
            return
        # --- Check if metadata seems loaded ---
        if self.num_features == 0 or self.num_vars == 0 or self.max_domain_size == 0:
             # Try loading metadata if not already done explicitly
             logging.warning("Metadata not loaded or empty. Attempting to load now...")
             # Pass loaded state if available, else discover from DB
             if self.loaded_state: # Use flag set during load_state
                  self._load_metadata(reuse_feature_map=self.feature_to_idx, reuse_max_domain=self.max_domain_size)
             else:
                  self._load_metadata()
             # Re-check after loading
             if self.num_features == 0 or self.num_vars == 0 or self.max_domain_size == 0:
                  raise ValueError("Metadata still empty after attempting load. Cannot build tensors.")
        # --- End Check ---

        logging.info("[TensorBuilder] Building PyTorch tensors...")
        # ... (Rest of build_tensors remains unchanged) ...
        build_start = time.time()
        num_evidence = len(self.evidence_var_indices)
        num_query = len(self.query_var_indices)
        self.X_train = torch.zeros(num_evidence, self.max_domain_size, self.num_features, dtype=torch.float32)
        self.Y_train = torch.zeros(num_evidence, dtype=torch.long)
        self.mask_train = torch.full((num_evidence, self.max_domain_size), -1e6, dtype=torch.float32)
        self.X_pred = torch.zeros(num_query, self.max_domain_size, self.num_features, dtype=torch.float32)
        self.mask_pred = torch.full((num_query, self.max_domain_size), -1e6, dtype=torch.float32)
        self.pred_indices = torch.tensor(self.query_var_indices, dtype=torch.long)
        logging.info(f"  Populating training tensors ({num_evidence} variables)...")
        evidence_map = {var_idx: i for i, var_idx in enumerate(self.evidence_var_indices)}
        for train_idx, var_idx in enumerate(tqdm(self.evidence_var_indices, desc="  Training tensors", unit=" vars")):
            self.Y_train[train_idx] = self.initial_value_idx.get(var_idx, 0)
            for domain_val_idx, _ in self.var_idx_to_domain[var_idx].items():
                 if domain_val_idx >= self.max_domain_size: continue
                 self.mask_train[train_idx, domain_val_idx] = 0.0
                 state_idx = self._get_state_idx(var_idx, domain_val_idx)
                 feature_indices_for_state = self.state_features.get(state_idx, [])
                 if feature_indices_for_state:
                     valid_feature_indices = [idx for idx in feature_indices_for_state if idx < self.num_features]
                     if valid_feature_indices:
                          self.X_train[train_idx, domain_val_idx, valid_feature_indices] = 1.0
        logging.info(f"  Populating prediction tensors ({num_query} variables)...")
        pred_map = {var_idx: i for i, var_idx in enumerate(self.query_var_indices)}
        for pred_idx, var_idx in enumerate(tqdm(self.query_var_indices, desc="  Prediction tensors", unit=" vars")):
            for domain_val_idx, _ in self.var_idx_to_domain[var_idx].items():
                 if domain_val_idx >= self.max_domain_size: continue
                 self.mask_pred[pred_idx, domain_val_idx] = 0.0
                 state_idx = self._get_state_idx(var_idx, domain_val_idx)
                 feature_indices_for_state = self.state_features.get(state_idx, [])
                 if feature_indices_for_state:
                     valid_feature_indices = [idx for idx in feature_indices_for_state if idx < self.num_features]
                     if valid_feature_indices:
                          self.X_pred[pred_idx, domain_val_idx, valid_feature_indices] = 1.0
        build_end = time.time()
        logging.info(f"[TensorBuilder] Tensors built in {build_end - build_start:.2f}s.")

    # ... (get_training_data, get_prediction_data etc. remain unchanged) ...
    def get_training_data(self):
        if self.X_train is None:
            # logging.warning("Training tensors not built yet. Call build_tensors() first.")
            self.build_tensors() # Call build_tensors if not ready
        return self.X_train, self.Y_train, self.mask_train

    def get_prediction_data(self):
        if self.X_pred is None:
            # logging.warning("Prediction tensors not built yet. Call build_tensors() first.")
            self.build_tensors() # Call build_tensors if not ready
        return self.X_pred, self.mask_pred, self.pred_indices

    def get_feature_info(self):
        return self.num_features, self.max_domain_size

    def get_minimality_feature_index(self):
        return self.minimality_feature_idx

    def get_var_idx_to_domain_map(self):
        return self.var_idx_to_domain

    def get_var_idx_to_cell_map(self):
         return self.var_idx_to_cell


    # --- Save/Load Methods remain unchanged ---
    def save_state(self, filepath="tensor_builder_state.pkl"):
        """Saves essential state needed for prediction."""
        state = {
            'feature_to_idx': self.feature_to_idx,
            'num_features': self.num_features,
            'max_domain_size': self.max_domain_size,
            'minimality_feature_idx': self.minimality_feature_idx
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logging.info(f"[TensorBuilder] State saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving TensorBuilder state: {e}", exc_info=True)

    def load_state(self, filepath="tensor_builder_state.pkl"):
        """Loads state, typically before building tensors for prediction."""
        if not os.path.exists(filepath):
             logging.error(f"TensorBuilder state file not found: {filepath}")
             return False
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.feature_to_idx = state.get('feature_to_idx', {})
            self.num_features = state.get('num_features', 0)
            self.max_domain_size = state.get('max_domain_size', 0)
            self.minimality_feature_idx = state.get('minimality_feature_idx', -1)
            self.idx_to_feature = {v: k for k, v in self.feature_to_idx.items()}
            logging.info(f"[TensorBuilder] State loaded from {filepath}")
            logging.info(f"  Loaded num_features: {self.num_features}")
            logging.info(f"  Loaded max_domain_size: {self.max_domain_size}")
            logging.info(f"  Loaded minimality_feature_idx: {self.minimality_feature_idx}")
            self.loaded_state = True # Set flag
            return True
        except Exception as e:
            logging.error(f"Error loading TensorBuilder state: {e}", exc_info=True)
            return False