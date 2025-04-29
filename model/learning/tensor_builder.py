# File: model/learning/tensor_builder.py
# Builds PyTorch tensors from database data.

import pickle
import pandas as pd
import torch
import time
import logging
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Make sure this matches the one used elsewhere!
NULL_REPR_PLACEHOLDER = "__NULL__"

class TensorBuilder:
    """Builds feature tensors for training and prediction."""

    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.feature_to_idx = {}
        self.idx_to_feature = {}
        self.num_features = 0
        # Store the index for the minimality prior feature
        self.minimality_feature_idx = -1 # Initialize to -1 (not found)

        # Mappings for variables (cells)
        self.cell_to_var_idx = {} # (tid, attr) -> var_idx (0 to N-1)
        self.var_idx_to_cell = {} # var_idx -> (tid, attr)
        self.num_vars = 0

        # Domain info
        self.var_idx_to_domain = defaultdict(dict) # var_idx -> {domain_val_idx: candidate_val_str}
        self.var_idx_to_val_to_idx = defaultdict(dict) # var_idx -> {candidate_val_str: domain_val_idx}
        self.max_domain_size = 0

        # Evidence/Query separation
        self.evidence_var_indices = []
        self.query_var_indices = []
        self.initial_value_idx = {} # var_idx -> index of initial value in its domain (for Y_train)

        # Feature map { state_idx : [feature_idx, feature_idx,...] }
        # where state_idx is derived from (var_idx, domain_val_idx)
        self.state_features = defaultdict(list)

        # Tensors
        self.X_train = None
        self.Y_train = None
        self.mask_train = None
        self.X_pred = None
        self.mask_pred = None
        self.pred_indices = None # Original var_indices corresponding to X_pred rows

        logging.info("[TensorBuilder] Initializing...")
        self._load_metadata()

    def _get_state_idx(self, var_idx, domain_val_idx):
        """Helper to create a unique index for each state."""
        # Assumes self.max_domain_size is already determined
        return var_idx * self.max_domain_size + domain_val_idx

    def _load_metadata(self):
        """Loads features, cells, and domains from DB to build mappings."""
        logging.info("[TensorBuilder] Loading metadata from DB (cells, domains, features)...")
        load_start = time.time()

        # 1. Load distinct features and create mapping
        logging.info("  Loading features...")
        features_df = pd.read_sql("SELECT DISTINCT feature FROM features", self.db_conn)
        for idx, feature_name in enumerate(features_df['feature']):
            self.feature_to_idx[feature_name] = idx
            self.idx_to_feature[idx] = feature_name
            # --- ADDED: Find and store the minimality feature index ---
            if feature_name == 'prior_minimality':
                self.minimality_feature_idx = idx
                logging.info(f"    Found 'prior_minimality' feature at index: {idx}")
            # --- END ADDED ---
        self.num_features = len(self.feature_to_idx)
        logging.info(f"  Found {self.num_features} unique features.")
        if self.minimality_feature_idx == -1:
            logging.warning("  'prior_minimality' feature was not found in the database features table!")

        # 2. Load cell info (variables) and separate evidence/query
        logging.info("  Loading cells info...")
        cells_df = pd.read_sql("SELECT tid, attr, val, is_noisy FROM cells ORDER BY tid, attr", self.db_conn)
        self.num_vars = len(cells_df)
        initial_values_map = {} # (tid, attr) -> initial_val (str or None)
        for idx, row in cells_df.iterrows():
            cell = (int(row['tid']), row['attr'])
            self.cell_to_var_idx[cell] = idx
            self.var_idx_to_cell[idx] = cell
            initial_values_map[cell] = row['val'] # Store None directly
            if row['is_noisy']:
                self.query_var_indices.append(idx)
            else:
                self.evidence_var_indices.append(idx)
        logging.info(f"  Processed {self.num_vars} total cells (variables).")
        logging.info(f"  Identified {len(self.evidence_var_indices)} evidence variables and {len(self.query_var_indices)} query variables.")

        # 3. Load domains and determine max size
        logging.info("  Loading domains...")
        # Use chunking for potentially large domains table
        domain_dfs = pd.read_sql("SELECT tid, attr, candidate_val FROM domains ORDER BY tid, attr, candidate_val",
                                self.db_conn, chunksize=50000)
        temp_domains = defaultdict(list) # (tid, attr) -> [candidate_val_str]
        row_count = 0
        pbar_domains = tqdm(desc="  Processing domains", unit=" candidates")
        for chunk in domain_dfs:
            for _, row in chunk.iterrows():
                cell = (int(row['tid']), row['attr'])
                temp_domains[cell].append(str(row['candidate_val']))
                row_count+=1
            pbar_domains.update(len(chunk))
        pbar_domains.close()

        # Build final domain maps and find initial value index
        current_max_domain = 0
        for cell, domain_list in temp_domains.items():
            if cell not in self.cell_to_var_idx: continue # Should not happen if DB consistent
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
            elif not cells_df.loc[var_idx, 'is_noisy']: # Only warn if it's a training example
                 logging.warning(f"Initial value '{initial_val_str}' for evidence cell {cell} (var_idx {var_idx}) not found in its domain: {domain_list}. Y_train may be incorrect.")
                 # Decide handling: error, skip, assign default? Assigning 0 for now.
                 self.initial_value_idx[var_idx] = 0


        self.max_domain_size = current_max_domain
        logging.info(f"  Max domain size across all variables: {self.max_domain_size}")

        # 4. Load features per state
        logging.info("  Loading features per state...")
        feature_chunks = pd.read_sql(
            "SELECT tid, attr, candidate_val, feature FROM features",
            self.db_conn, chunksize=100000 # Read in chunks
        )
        pbar_features = tqdm(desc="  Mapping features", unit=" features")
        processed_features = 0
        for chunk in feature_chunks:
            for _, row in chunk.iterrows():
                cell = (int(row['tid']), row['attr'])
                if cell not in self.cell_to_var_idx: continue # Skip features for cells not loaded
                var_idx = self.cell_to_var_idx[cell]
                candidate_val_str = str(row['candidate_val'])
                feature_name = row['feature']

                if candidate_val_str not in self.var_idx_to_val_to_idx[var_idx]: continue # Skip features for candidates not in domain
                domain_val_idx = self.var_idx_to_val_to_idx[var_idx][candidate_val_str]

                if feature_name not in self.feature_to_idx: continue # Skip unknown features

                feature_idx = self.feature_to_idx[feature_name]
                state_idx = self._get_state_idx(var_idx, domain_val_idx)
                self.state_features[state_idx].append(feature_idx)
                processed_features += 1
            pbar_features.update(processed_features)
            processed_features = 0 # Reset counter for next update
        pbar_features.close()
        logging.info("  Built feature map for states.")

        load_end = time.time()
        logging.info(f"[TensorBuilder] Metadata loaded and mappings built in {load_end - load_start:.2f}s.")


    def build_tensors(self):
        """Builds the final PyTorch tensors."""
        if self.X_train is not None: # Avoid rebuilding
            logging.warning("Tensors already built. Skipping.")
            return
        if self.num_features == 0 or self.num_vars == 0 or self.max_domain_size == 0:
             raise ValueError("Metadata not loaded or empty. Cannot build tensors.")

        logging.info("[TensorBuilder] Building PyTorch tensors...")
        build_start = time.time()

        # Initialize tensors (using float32 for features, long for indices/labels)
        num_evidence = len(self.evidence_var_indices)
        num_query = len(self.query_var_indices)

        self.X_train = torch.zeros(num_evidence, self.max_domain_size, self.num_features, dtype=torch.float32)
        self.Y_train = torch.zeros(num_evidence, dtype=torch.long)
        self.mask_train = torch.full((num_evidence, self.max_domain_size), -1e6, dtype=torch.float32) # Mask with large negative

        self.X_pred = torch.zeros(num_query, self.max_domain_size, self.num_features, dtype=torch.float32)
        self.mask_pred = torch.full((num_query, self.max_domain_size), -1e6, dtype=torch.float32)
        self.pred_indices = torch.tensor(self.query_var_indices, dtype=torch.long)

        # Populate training tensors
        logging.info(f"  Populating training tensors ({num_evidence} variables)...")
        evidence_map = {var_idx: i for i, var_idx in enumerate(self.evidence_var_indices)}
        for train_idx, var_idx in enumerate(tqdm(self.evidence_var_indices, desc="  Training tensors", unit=" vars")):
            # Set Y_train (initial value index)
            self.Y_train[train_idx] = self.initial_value_idx.get(var_idx, 0) # Default to 0 if missing (warning already shown)

            # Set mask and features (X_train)
            for domain_val_idx, _ in self.var_idx_to_domain[var_idx].items():
                 self.mask_train[train_idx, domain_val_idx] = 0.0 # Unmask valid domain entries
                 state_idx = self._get_state_idx(var_idx, domain_val_idx)
                 feature_indices_for_state = self.state_features.get(state_idx, [])
                 if feature_indices_for_state:
                     # Use advanced indexing to set features to 1.0
                     self.X_train[train_idx, domain_val_idx, feature_indices_for_state] = 1.0

        # Populate prediction tensors
        logging.info(f"  Populating prediction tensors ({num_query} variables)...")
        pred_map = {var_idx: i for i, var_idx in enumerate(self.query_var_indices)}
        for pred_idx, var_idx in enumerate(tqdm(self.query_var_indices, desc="  Prediction tensors", unit=" vars")):
            # Set mask and features (X_pred)
            for domain_val_idx, _ in self.var_idx_to_domain[var_idx].items():
                 self.mask_pred[pred_idx, domain_val_idx] = 0.0 # Unmask valid domain entries
                 state_idx = self._get_state_idx(var_idx, domain_val_idx)
                 feature_indices_for_state = self.state_features.get(state_idx, [])
                 if feature_indices_for_state:
                     self.X_pred[pred_idx, domain_val_idx, feature_indices_for_state] = 1.0

        build_end = time.time()
        logging.info(f"[TensorBuilder] Tensors built in {build_end - build_start:.2f}s.")

    def get_training_data(self):
        if self.X_train is None:
            logging.warning("Training tensors not built yet. Call build_tensors() first.")
            self.build_tensors()
        return self.X_train, self.Y_train, self.mask_train

    def get_prediction_data(self):
        if self.X_pred is None:
            logging.warning("Prediction tensors not built yet. Call build_tensors() first.")
            self.build_tensors()
        return self.X_pred, self.mask_pred, self.pred_indices

    def get_feature_info(self):
        """Returns info needed by the model."""
        return self.num_features, self.max_domain_size

    def get_minimality_feature_index(self):
        """Returns the index of the minimality feature."""
        return self.minimality_feature_idx

    def get_var_idx_to_domain_map(self):
        """Returns the map from var_idx to its domain {idx: val}."""
        return self.var_idx_to_domain

    def get_var_idx_to_cell_map(self):
         """Returns the map from var_idx to cell (tid, attr)."""
         return self.var_idx_to_cell
    
    # --- ADDED: Save/Load Methods ---
    def save_state(self, filepath="tensor_builder_state.pkl"):
        """Saves essential state needed for prediction."""
        state = {
            'feature_to_idx': self.feature_to_idx,
            'num_features': self.num_features,
            'max_domain_size': self.max_domain_size,
            'minimality_feature_idx': self.minimality_feature_idx
            # Add other mappings if needed for prediction/evaluation later
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logging.info(f"[TensorBuilder] State saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving TensorBuilder state: {e}", exc_info=True)

    def load_state(self, filepath="tensor_builder_state.pkl"):
        """Loads state, typically before building tensors for prediction."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.feature_to_idx = state.get('feature_to_idx', {})
            self.num_features = state.get('num_features', 0)
            self.max_domain_size = state.get('max_domain_size', 0)
            self.minimality_feature_idx = state.get('minimality_feature_idx', -1)
            # Rebuild reverse map
            self.idx_to_feature = {v: k for k, v in self.feature_to_idx.items()}
            logging.info(f"[TensorBuilder] State loaded from {filepath}")
            logging.info(f"  Loaded num_features: {self.num_features}")
            logging.info(f"  Loaded max_domain_size: {self.max_domain_size}")
            logging.info(f"  Loaded minimality_feature_idx: {self.minimality_feature_idx}")
            return True
        except FileNotFoundError:
             logging.error(f"TensorBuilder state file not found: {filepath}")
             return False
        except Exception as e:
            logging.error(f"Error loading TensorBuilder state: {e}", exc_info=True)
            return False
    # --- END ADDED ---
