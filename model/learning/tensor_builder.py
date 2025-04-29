# File: model/learning/tensor_builder.py
# Builds PyTorch tensors from database data.
# VERSION 4: Corrected Y_train / mask_train generation for missing initial values.

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
        self.initial_values_map_str = {} # Stores initial values as strings (tid, attr) -> val_str
        self.var_idx_to_domain = defaultdict(dict) # var_idx -> {domain_val_idx: candidate_val_str}
        self.var_idx_to_val_to_idx = defaultdict(dict) # var_idx -> {candidate_val_str: domain_val_idx}
        self.max_domain_size = 0
        self.evidence_var_indices = []
        self.query_var_indices = []
        # self.initial_value_idx = {} # No longer needed - Y_train created directly
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
        if self.max_domain_size <= 0: # Check if non-positive
             raise ValueError(f"max_domain_size must be positive before calculating state index (was {self.max_domain_size}).")
        # Ensure domain_val_idx is within bounds conceptually, though map lookup handles it
        # if domain_val_idx < 0 or domain_val_idx >= self.max_domain_size:
        #     logging.warning(f"domain_val_idx {domain_val_idx} seems out of bounds for max_domain_size {self.max_domain_size}")
        return var_idx * self.max_domain_size + domain_val_idx

    def _load_metadata(self, reuse_feature_map=None, reuse_max_domain=None):
        """
        Loads metadata. Optionally reuses feature map and domain size from training.
        reuse_feature_map: dict (feature_name -> idx)
        reuse_max_domain: int
        """
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
        logging.info("  Loading cells info...")
        cells_df = pd.read_sql("SELECT tid, attr, val, is_noisy FROM cells ORDER BY tid, attr", self.db_conn)
        self.num_vars = len(cells_df)
        self.initial_values_map_str = {} # Reset
        self.evidence_var_indices = [] # Reset
        self.query_var_indices = [] # Reset
        self.cell_to_var_idx = {} # Reset
        self.var_idx_to_cell = {} # Reset
        for idx, row in cells_df.iterrows():
            cell = (int(row['tid']), row['attr'])
            self.cell_to_var_idx[cell] = idx
            self.var_idx_to_cell[idx] = cell
            # Store initial value as string using placeholder
            self.initial_values_map_str[cell] = NULL_REPR_PLACEHOLDER if pd.isna(row['val']) else str(row['val'])
            if row['is_noisy']:
                self.query_var_indices.append(idx)
            else:
                self.evidence_var_indices.append(idx)
        logging.info(f"  Processed {self.num_vars} total cells (variables).")
        logging.info(f"  Identified {len(self.evidence_var_indices)} evidence variables and {len(self.query_var_indices)} query variables.")

        # 3. Load domains and determine max size
        logging.info("  Loading domains from 'domains' table...")
        domain_dfs = pd.read_sql("SELECT tid, attr, candidate_val FROM domains ORDER BY tid, attr, candidate_val",
                                self.db_conn, chunksize=100000) # Increased chunksize
        temp_domains = defaultdict(list)
        self.var_idx_to_domain = defaultdict(dict) # Reset
        self.var_idx_to_val_to_idx = defaultdict(dict) # Reset
        # self.initial_value_idx = {} # Reset (no longer needed)
        row_count = 0
        pbar_domains = tqdm(desc="  Reading domains from DB", unit=" candidates")
        for chunk in domain_dfs:
            for _, row in chunk.iterrows():
                cell = (int(row['tid']), row['attr'])
                # Convert candidate value to string immediately
                candidate_str = str(row['candidate_val'])
                # Ensure NULL placeholder is never added to domain candidates directly
                if candidate_str == NULL_REPR_PLACEHOLDER:
                     logging.debug(f"    Skipping NULL placeholder '{NULL_REPR_PLACEHOLDER}' found in domains table for cell {cell}")
                     continue
                logging.debug(f"    Read domain candidate: Cell={cell}, Candidate='{candidate_str}'")
                temp_domains[cell].append(candidate_str)
                row_count+=1
            pbar_domains.update(len(chunk))
        pbar_domains.close()
        logging.info(f"  Finished reading {row_count} non-null candidate entries from 'domains' table.")
        logging.info(f"  Processing domains for {len(temp_domains)} unique cells found in 'domains' table.")

        current_max_domain = 0
        processed_domain_count = 0
        logging.info("  Calculating max domain size and mapping domains...")
        for cell, domain_list in temp_domains.items():
            if cell not in self.cell_to_var_idx:
                logging.debug(f"    Skipping cell {cell} from domains table as it's not in the main 'cells' table variable list.")
                continue

            var_idx = self.cell_to_var_idx[cell]
            # Filter out duplicates just in case DB had them
            unique_domain_list = sorted(list(set(domain_list)))
            domain_size = len(unique_domain_list)
            logging.debug(f"    Processing Cell={cell} (VarIdx={var_idx}): Unique Domain={unique_domain_list}, Size={domain_size}")

            if domain_size > current_max_domain:
                logging.info(f"    New max domain size found: {domain_size} (Previous: {current_max_domain}) for Cell={cell} (VarIdx={var_idx})")
                current_max_domain = domain_size

            # Map domain values to indices for this variable
            for domain_idx, candidate_val_str in enumerate(unique_domain_list):
                self.var_idx_to_domain[var_idx][domain_idx] = candidate_val_str
                self.var_idx_to_val_to_idx[var_idx][candidate_val_str] = domain_idx
                logging.debug(f"      Mapping: VarIdx={var_idx}, DomainIdx={domain_idx} -> Candidate='{candidate_val_str}'")

            processed_domain_count += 1

        logging.info(f"  Finished processing domains for {processed_domain_count} variables.")
        logging.info(f"  Calculated current_max_domain from data: {current_max_domain}")
        logging.info(f"  Reuse_max_domain provided: {reuse_max_domain}")

        # Use reused max domain size if provided
        self.max_domain_size = reuse_max_domain if reuse_max_domain is not None else current_max_domain
        if self.max_domain_size <= 0 and self.num_vars > 0: # Check if non-positive
             logging.warning("Max domain size determined to be <= 0. Check if 'domains' table is populated correctly and if candidates were read.")
        logging.info(f"  Final Max domain size being used: {self.max_domain_size}")


        # 4. Load features per state
        logging.info("  Loading features per state...")
        self.state_features = defaultdict(list) # Reset
        if self.max_domain_size <= 0 and self.num_vars > 0: # Check if non-positive
             logging.error("Cannot proceed with feature mapping as max_domain_size is <= 0.")
             raise ValueError(f"max_domain_size is {self.max_domain_size}, cannot map features to states.")

        feature_chunks = pd.read_sql(
            "SELECT tid, attr, candidate_val, feature FROM features",
            self.db_conn, chunksize=100000
        )
        pbar_features = tqdm(desc="  Mapping features", unit=" features")
        processed_features_count = 0
        unknown_feature_count = 0
        skipped_candidate_feature_count = 0
        for chunk in feature_chunks:
            chunk_mapped_count = 0
            for _, row in chunk.iterrows():
                cell = (int(row['tid']), row['attr'])
                if cell not in self.cell_to_var_idx: continue
                var_idx = self.cell_to_var_idx[cell]
                candidate_val_str = str(row['candidate_val'])
                feature_name = row['feature']

                # Check if candidate is actually in the mapped domain for this variable
                if candidate_val_str not in self.var_idx_to_val_to_idx[var_idx]:
                     if skipped_candidate_feature_count < 10: # Log first few
                         logging.debug(f"    Skipping feature '{feature_name}' for Cell={cell}, Candidate='{candidate_val_str}' because candidate is not in the mapped domain for VarIdx={var_idx}.")
                     skipped_candidate_feature_count += 1
                     continue

                domain_val_idx = self.var_idx_to_val_to_idx[var_idx][candidate_val_str]

                if feature_name not in self.feature_to_idx:
                     if unknown_feature_count < 10: # Log first few unknowns
                         logging.debug(f"    Skipping unknown feature '{feature_name}' for Cell={cell}, Candidate='{candidate_val_str}'.")
                     unknown_feature_count += 1
                     continue

                feature_idx = self.feature_to_idx[feature_name]
                # Ensure state_idx calculation uses the correct domain_val_idx (local index within the variable's domain)
                state_idx = self._get_state_idx(var_idx, domain_val_idx) # Requires self.max_domain_size > 0
                self.state_features[state_idx].append(feature_idx)
                chunk_mapped_count += 1
            pbar_features.update(chunk_mapped_count)
            processed_features_count += chunk_mapped_count

        pbar_features.close()
        if unknown_feature_count > 0:
             logging.warning(f"  Skipped {unknown_feature_count} features not present in the loaded/reused feature map.")
        if skipped_candidate_feature_count > 0:
             logging.warning(f"  Skipped {skipped_candidate_feature_count} features because their associated candidate value was not found in the mapped domain for that cell.")
        logging.info(f"  Built feature map for {len(self.state_features)} states from {processed_features_count} feature entries.")

        load_end = time.time()
        logging.info(f"[TensorBuilder] Metadata loaded and mappings built in {load_end - load_start:.2f}s.")

    def build_tensors(self):
        """Builds the final PyTorch tensors."""
        if self.X_train is not None:
            logging.warning("Tensors already built. Skipping.")
            return

        # Check if metadata seems loaded
        needs_load = False
        if self.num_features == 0 or self.num_vars == 0:
             needs_load = True
        elif self.max_domain_size <= 0 and self.num_vars > 0: # Check non-positive
             needs_load = True

        if needs_load:
             logging.warning("Metadata not loaded or empty (or max_domain_size is <= 0). Attempting to load now...")
             if self.loaded_state:
                  logging.info("  Loading metadata using previously loaded state...")
                  self._load_metadata(reuse_feature_map=self.feature_to_idx, reuse_max_domain=self.max_domain_size)
             else:
                  logging.info("  Loading metadata from database (discovery)...")
                  self._load_metadata()
             # Re-check after loading
             if self.num_features == 0 or self.num_vars == 0 or (self.max_domain_size <= 0 and self.num_vars > 0):
                  raise ValueError("Metadata still empty or max_domain_size is <= 0 after attempting load. Cannot build tensors.")

        logging.info("[TensorBuilder] Building PyTorch tensors...")
        build_start = time.time()
        num_evidence = len(self.evidence_var_indices)
        num_query = len(self.query_var_indices)

        # Ensure max_domain_size > 0 before allocating tensors
        if self.max_domain_size <= 0:
            raise ValueError(f"Cannot build tensors with max_domain_size={self.max_domain_size}. Check domain loading.")

        # Allocate tensors
        self.X_train = torch.zeros(num_evidence, self.max_domain_size, self.num_features, dtype=torch.float32)
        self.Y_train = torch.full((num_evidence,), -1, dtype=torch.long) # Initialize Y with -1 (ignore index)
        self.mask_train = torch.full((num_evidence, self.max_domain_size), -1e6, dtype=torch.float32) # Initialize mask to block all

        self.X_pred = torch.zeros(num_query, self.max_domain_size, self.num_features, dtype=torch.float32)
        self.mask_pred = torch.full((num_query, self.max_domain_size), -1e6, dtype=torch.float32) # Initialize mask to block all
        self.pred_indices = torch.tensor(self.query_var_indices, dtype=torch.long)

        logging.info(f"  Populating training tensors ({num_evidence} variables)...")
        valid_training_samples = 0
        skipped_training_samples = 0

        for train_idx, var_idx in enumerate(tqdm(self.evidence_var_indices, desc="  Training tensors", unit=" vars")):
            cell = self.var_idx_to_cell.get(var_idx)
            initial_val_str = self.initial_values_map_str.get(cell)

            # Check if domain exists and initial value is valid
            if var_idx not in self.var_idx_to_domain or not self.var_idx_to_domain[var_idx]:
                 logging.warning(f"No domain found or empty domain for evidence VarIdx {var_idx} (Cell: {cell}). Skipping training sample.")
                 skipped_training_samples += 1
                 continue # Keep Y_train as -1 and mask_train as -1e6

            if initial_val_str is None:
                 logging.warning(f"Initial value string not found for evidence VarIdx {var_idx} (Cell: {cell}). Skipping training sample.")
                 skipped_training_samples += 1
                 continue # Keep Y_train as -1 and mask_train as -1e6

            # --- MODIFIED LOGIC FOR Y_train ---
            initial_idx_in_domain = self.var_idx_to_val_to_idx[var_idx].get(initial_val_str)

            if initial_idx_in_domain is None:
                 # Initial value (likely NULL or just not included in generated domain) is NOT in the domain.
                 # We cannot train on this sample reliably. Keep Y=-1 and mask=-1e6.
                 logging.debug(f"Initial value '{initial_val_str}' for evidence VarIdx {var_idx} (Cell: {cell}) not found in its domain. Masking out sample.")
                 skipped_training_samples += 1
                 # No need to populate X_train or change mask_train (already -1e6)
            else:
                 # Initial value IS in the domain. Set Y_train and populate X_train/mask_train.
                 self.Y_train[train_idx] = initial_idx_in_domain
                 valid_training_samples += 1

                 # Populate features and unblock mask for valid domain entries
                 for domain_val_idx, _ in self.var_idx_to_domain[var_idx].items():
                     # The domain_val_idx is the local index (0 to k-1) for this variable's domain
                     # Check against actual domain size for safety, though map iteration handles it
                     # if domain_val_idx >= len(self.var_idx_to_domain[var_idx]): continue

                     # Unblock this position in the mask
                     self.mask_train[train_idx, domain_val_idx] = 0.0

                     # Populate features
                     state_idx = self._get_state_idx(var_idx, domain_val_idx)
                     feature_indices_for_state = self.state_features.get(state_idx, [])
                     if feature_indices_for_state:
                         # Ensure feature indices are within bounds [0, num_features-1]
                         valid_feature_indices = [idx for idx in feature_indices_for_state if 0 <= idx < self.num_features]
                         if valid_feature_indices:
                              self.X_train[train_idx, domain_val_idx, valid_feature_indices] = 1.0
                         elif feature_indices_for_state: # Log if non-empty list became empty
                             logging.debug(f"All feature indices {feature_indices_for_state} for state {state_idx} (VarIdx {var_idx}, DomainIdx {domain_val_idx}) were out of bounds (NumFeatures={self.num_features}).")
            # --- END MODIFIED LOGIC ---

        logging.info(f"  Finished populating training tensors. Valid samples: {valid_training_samples}, Skipped samples: {skipped_training_samples}.")

        # --- Populate Prediction Tensors (Logic largely unchanged, just ensure mask init is correct) ---
        logging.info(f"  Populating prediction tensors ({num_query} variables)...")
        for pred_idx, var_idx in enumerate(tqdm(self.query_var_indices, desc="  Prediction tensors", unit=" vars")):
            # Check if domain exists
            if var_idx not in self.var_idx_to_domain or not self.var_idx_to_domain[var_idx]:
                 logging.warning(f"No domain found or empty domain for query VarIdx {var_idx} (Cell: {self.var_idx_to_cell.get(var_idx)}). Predictions will be based on blocked mask.")
                 continue # Keep mask_pred as -1e6

            # Populate features and unblock mask for valid domain entries
            for domain_val_idx, _ in self.var_idx_to_domain[var_idx].items():
                 # Unblock this position in the mask
                 self.mask_pred[pred_idx, domain_val_idx] = 0.0

                 # Populate features
                 state_idx = self._get_state_idx(var_idx, domain_val_idx)
                 feature_indices_for_state = self.state_features.get(state_idx, [])
                 if feature_indices_for_state:
                     valid_feature_indices = [idx for idx in feature_indices_for_state if 0 <= idx < self.num_features]
                     if valid_feature_indices:
                          self.X_pred[pred_idx, domain_val_idx, valid_feature_indices] = 1.0
                     elif feature_indices_for_state:
                         logging.debug(f"All feature indices {feature_indices_for_state} for state {state_idx} (Query VarIdx {var_idx}, DomainIdx {domain_val_idx}) were out of bounds (NumFeatures={self.num_features}).")
        # --- End Prediction Population ---

        build_end = time.time()
        logging.info(f"[TensorBuilder] Tensors built in {build_end - build_start:.2f}s.")

    def get_training_data(self):
        if self.X_train is None:
            self.build_tensors()
        return self.X_train, self.Y_train, self.mask_train

    def get_prediction_data(self):
        if self.X_pred is None:
            self.build_tensors()
        return self.X_pred, self.mask_pred, self.pred_indices

    def get_feature_info(self):
        if self.max_domain_size <= 0: # Check non-positive
             logging.warning("get_feature_info called but max_domain_size is not set or zero. Trying to determine it.")
             if not self.loaded_state and self.X_train is None:
                  self.build_tensors()
             if self.max_domain_size <= 0:
                  logging.error("Failed to determine positive max_domain_size in get_feature_info.")
                  return self.num_features, 0 # Return 0 for size if it couldn't be determined
        return self.num_features, self.max_domain_size

    def get_minimality_feature_index(self):
        return self.minimality_feature_idx

    def get_var_idx_to_domain_map(self):
        # Returns dict: var_idx -> {domain_idx: domain_val_str}
        return self.var_idx_to_domain

    def get_var_idx_to_cell_map(self):
         # Returns dict: var_idx -> (tid, attr)
         return self.var_idx_to_cell


    def save_state(self, filepath="tensor_builder_state.pkl"):
        """Saves essential state needed for prediction."""
        if self.max_domain_size <= 0: # Check non-positive
             logging.warning(f"Attempting to save state but max_domain_size is not positive ({self.max_domain_size}). State may be incomplete or invalid.")

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
             self.loaded_state = False
             return False
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.feature_to_idx = state.get('feature_to_idx', {})
            self.num_features = state.get('num_features', 0)
            self.max_domain_size = state.get('max_domain_size', 0) # Load saved max_domain_size
            self.minimality_feature_idx = state.get('minimality_feature_idx', -1)
            self.idx_to_feature = {v: k for k, v in self.feature_to_idx.items()}
            logging.info(f"[TensorBuilder] State loaded from {filepath}")
            logging.info(f"  Loaded num_features: {self.num_features}")
            logging.info(f"  Loaded max_domain_size: {self.max_domain_size}")
            logging.info(f"  Loaded minimality_feature_idx: {self.minimality_feature_idx}")
            self.loaded_state = True # Set flag
            if self.max_domain_size <= 0 and self.num_features > 0: # Check non-positive
                 logging.warning("Loaded state has max_domain_size <= 0. This might cause issues downstream.")
            return True
        except Exception as e:
            logging.error(f"Error loading TensorBuilder state: {e}", exc_info=True)
            self.loaded_state = False
            return False