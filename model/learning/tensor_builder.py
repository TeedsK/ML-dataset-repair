import pandas as pd
import torch
import time
import logging
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pickle
import os

try:
    import config
    NULL_REPR_PLACEHOLDER = config.NULL_REPR_PLACEHOLDER
except (ImportError, AttributeError):
    logging.warning("Could not import NULL_REPR_PLACEHOLDER from config.py. Using default '__NULL__'.")
    NULL_REPR_PLACEHOLDER = "__NULL__"


#builds feature tensors (sparse) for training and prediction.
class TensorBuilder:

    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.feature_to_idx = {}
        self.idx_to_feature = {}
        self.num_features = 0
        self.minimality_feature_idx = -1
        self.cell_to_var_idx = {}
        self.var_idx_to_cell = {}
        self.num_vars = 0
        self.initial_values_map_str = {}
        self.var_idx_to_domain = defaultdict(dict)
        self.var_idx_to_val_to_idx = defaultdict(dict)
        self.max_domain_size = 0
        self.evidence_var_indices = []
        self.query_var_indices = []
        self.state_features = defaultdict(list)
        self.X_train_sparse = None
        self.Y_train = None
        self.mask_train = None
        self.X_pred_sparse = None
        self.mask_pred = None
        self.pred_indices = None

        self.loaded_state = False
        logging.info("[TensorBuilder] Initializing (Sparse Mode)...")

    def _get_state_idx(self, var_idx, domain_val_idx):
        if self.max_domain_size <= 0:
             raise ValueError(f"max_domain_size must be positive before calculating state index (was {self.max_domain_size}).")
        return var_idx * self.max_domain_size + domain_val_idx

    def _load_metadata(self, reuse_feature_map=None, reuse_max_domain=None):
        if reuse_feature_map is None:
             self.feature_to_idx = {}
             self.idx_to_feature = {}
             self.num_features = 0
             self.minimality_feature_idx = -1

        logging.info("[TensorBuilder] Loading metadata from DB (cells, domains, features)...")
        load_start = time.time()

        logging.info("  Loading features...")
        if reuse_feature_map is not None:
            logging.info("  Reusing feature map from training phase.")

            self.feature_to_idx = reuse_feature_map
            self.num_features = len(self.feature_to_idx)
            self.idx_to_feature = {v: k for k, v in self.feature_to_idx.items()}
            self.minimality_feature_idx = -1

            for name, idx in self.feature_to_idx.items():
                 if name == 'prior_minimality':
                     self.minimality_feature_idx = idx
                     break
            if self.minimality_feature_idx != -1:
                 logging.info(f"    Found 'prior_minimality' feature at index: {self.minimality_feature_idx}")
            else:
                 logging.warning("  'prior_minimality' feature not found in reused feature map!")
        else:
            features_df = pd.read_sql("SELECT DISTINCT feature FROM features", self.db_conn)
            feature_list = features_df['feature'].tolist()

            if 'prior_minimality' not in feature_list and any(f.startswith('h_') for f in feature_list):
                 logging.warning("Adding 'prior_minimality' to feature list as it was missing.")
                 feature_list.append('prior_minimality')

            self.feature_to_idx = {name: idx for idx, name in enumerate(feature_list)}
            self.idx_to_feature = {idx: name for idx, name in enumerate(feature_list)}
            self.num_features = len(self.feature_to_idx)
            self.minimality_feature_idx = self.feature_to_idx.get('prior_minimality', -1)

            logging.info(f"  Found {self.num_features} unique features.")
            if self.minimality_feature_idx != -1:
                 logging.info(f"    Found 'prior_minimality' feature at index: {self.minimality_feature_idx}")
            else:
                 logging.warning("  'prior_minimality' feature was not found in the database features table!")


        #load cell info
        logging.info("  Loading cells info...")
        cells_df = pd.read_sql("SELECT tid, attr, val, is_noisy FROM cells ORDER BY tid, attr", self.db_conn)
        self.num_vars = len(cells_df)
        self.initial_values_map_str = {}
        self.evidence_var_indices = []
        self.query_var_indices = []
        self.cell_to_var_idx = {}
        self.var_idx_to_cell = {}
        for idx, row in cells_df.iterrows():
            cell = (int(row['tid']), row['attr'])
            self.cell_to_var_idx[cell] = idx
            self.var_idx_to_cell[idx] = cell
            self.initial_values_map_str[cell] = NULL_REPR_PLACEHOLDER if pd.isna(row['val']) else str(row['val'])
            if row['is_noisy']:
                self.query_var_indices.append(idx)
            else:
                self.evidence_var_indices.append(idx)
        logging.info(f"  processed {self.num_vars} total cells (variables).")
        logging.info(f"  Identified {len(self.evidence_var_indices)} evidence variables and {len(self.query_var_indices)} query variables.")


        #load domains and determine max size
        logging.info("  Loading domains from 'domains' table...")
        domain_dfs = pd.read_sql("SELECT tid, attr, candidate_val FROM domains ORDER BY tid, attr, candidate_val",
                                self.db_conn, chunksize=100000)
        temp_domains = defaultdict(list)
        self.var_idx_to_domain = defaultdict(dict)
        self.var_idx_to_val_to_idx = defaultdict(dict)
        row_count = 0
        pbar_domains = tqdm(desc="  Reading domains from DB", unit=" candidates")
        for chunk in domain_dfs:
            for _, row in chunk.iterrows():
                cell = (int(row['tid']), row['attr'])
                candidate_str = str(row['candidate_val'])
                if candidate_str == NULL_REPR_PLACEHOLDER: continue
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
            if cell not in self.cell_to_var_idx: continue
            var_idx = self.cell_to_var_idx[cell]
            unique_domain_list = sorted(list(set(domain_list)))
            domain_size = len(unique_domain_list)
            if domain_size > current_max_domain:
                logging.info(f"    New max domain size found: {domain_size} (Previous: {current_max_domain}) for Cell={cell} (VarIdx={var_idx})")
                current_max_domain = domain_size

            for domain_idx, candidate_val_str in enumerate(unique_domain_list):
                self.var_idx_to_domain[var_idx][domain_idx] = candidate_val_str
                self.var_idx_to_val_to_idx[var_idx][candidate_val_str] = domain_idx

            processed_domain_count += 1
        logging.info(f"  Finished processing domains for {processed_domain_count} variables.")
        logging.info(f"  Calculated current_max_domain from data: {current_max_domain}")
        logging.info(f"  Reuse_max_domain provided: {reuse_max_domain}")

        self.max_domain_size = reuse_max_domain if reuse_max_domain is not None else current_max_domain
        if self.max_domain_size <= 0 and self.num_vars > 0:
             logging.warning("Max domain size determined to be <= 0. Check 'domains' table.")
        logging.info(f"  Final Max domain size being used: {self.max_domain_size}")


        #load features per state - store feature indices directly
        logging.info("  Loading features per state (mapping to indices)...")
        self.state_features = defaultdict(list)
        if self.max_domain_size <= 0 and self.num_vars > 0:
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

                domain_val_idx = self.var_idx_to_val_to_idx[var_idx].get(candidate_val_str)
                if domain_val_idx is None:
                     if skipped_candidate_feature_count < 10:
                         logging.debug(f"    Skipping feature '{feature_name}' for Cell={cell}, Candidate='{candidate_val_str}' because candidate not in mapped domain.")
                     skipped_candidate_feature_count += 1
                     continue

                #get the index for this feature name
                feature_idx = self.feature_to_idx.get(feature_name)
                if feature_idx is None:
                     if unknown_feature_count < 10:
                         logging.debug(f"    Skipping unknown feature '{feature_name}' for Cell={cell}, Candidate='{candidate_val_str}'.")
                     unknown_feature_count += 1
                     continue

                #store the feature INDEX associated with this state
                state_key = (var_idx, domain_val_idx)
                self.state_features[state_key].append(feature_idx)
                chunk_mapped_count += 1
            pbar_features.update(chunk_mapped_count)
            processed_features_count += chunk_mapped_count
        pbar_features.close()

        #unduplicate feature indices per state
        logging.info("  Deduplicating feature indices per state...")
        for key in self.state_features:
            self.state_features[key] = list(set(self.state_features[key]))

        if unknown_feature_count > 0:
             logging.warning(f"  Skipped {unknown_feature_count} feature names not present in the loaded feature map.")
        if skipped_candidate_feature_count > 0:
             logging.warning(f"  Skipped {skipped_candidate_feature_count} features because their candidate value was not found in the mapped domain.")
        logging.info(f"  Built feature index map for {len(self.state_features)} states from {processed_features_count} feature entries.")

        load_end = time.time()
        logging.info(f"[TensorBuilder] Metadata loaded and mappings built in {load_end - load_start:.2f}s.")


    #builds the final tensors (Sparse X, Dense Y/Mask).
    def build_tensors(self):

        if self.X_train_sparse is not None:
            logging.warning("Tensors already built. Skipping.")
            return

        needs_load = False
        if self.num_features == 0 or self.num_vars == 0: needs_load = True
        elif self.max_domain_size <= 0 and self.num_vars > 0: needs_load = True
        if needs_load:
             logging.warning("Metadata not loaded or empty (or max_domain_size <= 0). Attempting to load now...")
             if self.loaded_state:
                  self._load_metadata(reuse_feature_map=self.feature_to_idx, reuse_max_domain=self.max_domain_size)
             else:
                  self._load_metadata()
             if self.num_features == 0 or self.num_vars == 0 or (self.max_domain_size <= 0 and self.num_vars > 0):
                  raise ValueError("Metadata still empty or max_domain_size <= 0 after attempting load. Cannot build tensors.")
        
        logging.info("[TensorBuilder] Building PyTorch tensors (Sparse X)...")
        build_start = time.time()
        num_evidence = len(self.evidence_var_indices)
        num_query = len(self.query_var_indices)

        self.Y_train = torch.full((num_evidence,), -1, dtype=torch.long)
        self.mask_train = torch.full((num_evidence, self.max_domain_size), -1e6, dtype=torch.float32)
        self.mask_pred = torch.full((num_query, self.max_domain_size), -1e6, dtype=torch.float32)
        self.pred_indices = torch.tensor(self.query_var_indices, dtype=torch.long)

        #lists that collect sparse tensor components
        train_indices = []
        train_values = []
        pred_indices_sparse = []
        pred_values = []

        logging.info(f"  Populating training tensors ({num_evidence} variables)...")
        valid_training_samples = 0
        skipped_training_samples = 0
        feature_count_train = 0

        for train_idx, var_idx in enumerate(tqdm(self.evidence_var_indices, desc="  Training tensors", unit=" vars")):
            cell = self.var_idx_to_cell.get(var_idx)
            initial_val_str = self.initial_values_map_str.get(cell)

            if var_idx not in self.var_idx_to_domain or not self.var_idx_to_domain[var_idx]:
                 logging.warning(f"No domain found or empty domain for evidence VarIdx {var_idx} (Cell: {cell}). Skipping training sample.")
                 skipped_training_samples += 1
                 continue

            if initial_val_str is None:
                 logging.warning(f"Initial value string not found for evidence VarIdx {var_idx} (Cell: {cell}). Skipping training sample.")
                 skipped_training_samples += 1
                 continue

            initial_idx_in_domain = self.var_idx_to_val_to_idx[var_idx].get(initial_val_str)

            #populate Y_train and Mask
            if initial_idx_in_domain is not None:
                 self.Y_train[train_idx] = initial_idx_in_domain
                 valid_training_samples += 1

                 for domain_val_idx in self.var_idx_to_domain[var_idx].keys():
                     self.mask_train[train_idx, domain_val_idx] = 0.0
            else:
                 logging.debug(f"Initial value '{initial_val_str}' for evidence VarIdx {var_idx} (Cell: {cell}) not found in its domain. Masking out sample.")
                 skipped_training_samples += 1

            #populate sparse X_train components 
            for domain_val_idx in self.var_idx_to_domain[var_idx].keys():
                state_key = (var_idx, domain_val_idx)
                feature_indices_for_state = self.state_features.get(state_key, [])
                
                for feature_idx in feature_indices_for_state:
                    
                    
                    if 0 <= feature_idx < self.num_features:
                        train_indices.append([train_idx, domain_val_idx, feature_idx])
                        train_values.append(1.0)
                        feature_count_train += 1
                    else:
                         logging.warning(f"Invalid feature index {feature_idx} encountered for state {state_key}. Max features: {self.num_features}")


        logging.info(f"  Finished populating training tensors. Valid samples: {valid_training_samples}, Skipped samples: {skipped_training_samples}. Features added: {feature_count_train}")

        logging.info(f"  Populating prediction tensors ({num_query} variables)...")
        feature_count_pred = 0
        for pred_idx, var_idx in enumerate(tqdm(self.query_var_indices, desc="  Prediction tensors", unit=" vars")):
            if var_idx not in self.var_idx_to_domain or not self.var_idx_to_domain[var_idx]:
                 logging.warning(f"No domain found or empty domain for query VarIdx {var_idx} (Cell: {self.var_idx_to_cell.get(var_idx)}). Predictions will be based on blocked mask.")
                 continue

            #populate sparse X_pred components and unblock mask
            for domain_val_idx in self.var_idx_to_domain[var_idx].keys():
                 self.mask_pred[pred_idx, domain_val_idx] = 0.0
                 state_key = (var_idx, domain_val_idx)
                 feature_indices_for_state = self.state_features.get(state_key, [])

                 for feature_idx in feature_indices_for_state:
                     
                     if 0 <= feature_idx < self.num_features:
                         pred_indices_sparse.append([pred_idx, domain_val_idx, feature_idx])
                         pred_values.append(1.0)
                         feature_count_pred += 1

                     else:
                          logging.warning(f"Invalid feature index {feature_idx} encountered for query state {state_key}. Max features: {self.num_features}")

        logging.info(f"  Finished populating prediction tensors. Features added: {feature_count_pred}")

        logging.info("  Creating sparse tensors...")
        sparse_create_start = time.time()

        #training Tensor
        if train_indices:
            train_indices_tensor = torch.tensor(train_indices, dtype=torch.long).t()
            train_values_tensor = torch.tensor(train_values, dtype=torch.float32)
            train_size = (num_evidence, self.max_domain_size, self.num_features)
            self.X_train_sparse = torch.sparse_coo_tensor(train_indices_tensor, train_values_tensor, train_size)
            logging.info(f"  Created X_train_sparse: size={self.X_train_sparse.size()}, nnz={self.X_train_sparse._nnz()}")
        else:
            logging.warning("  No non-zero features found for training data. X_train_sparse will be empty.")
            train_size = (num_evidence, self.max_domain_size, self.num_features)
            self.X_train_sparse = torch.sparse_coo_tensor(torch.empty((3, 0), dtype=torch.long), torch.empty(0), train_size)

        # prediction
        if pred_indices_sparse:
            pred_indices_tensor = torch.tensor(pred_indices_sparse, dtype=torch.long).t()
            pred_values_tensor = torch.tensor(pred_values, dtype=torch.float32)
            pred_size = (num_query, self.max_domain_size, self.num_features)
            self.X_pred_sparse = torch.sparse_coo_tensor(pred_indices_tensor, pred_values_tensor, pred_size)
            logging.info(f"  Created X_pred_sparse: size={self.X_pred_sparse.size()}, nnz={self.X_pred_sparse._nnz()}")
        else:
            logging.warning("  No non-zero features found for prediction data. X_pred_sparse will be empty.")
            pred_size = (num_query, self.max_domain_size, self.num_features)
            self.X_pred_sparse = torch.sparse_coo_tensor(torch.empty((3, 0), dtype=torch.long), torch.empty(0), pred_size)

        sparse_create_end = time.time()
        logging.info(f"  Sparse tensor creation took {sparse_create_end - sparse_create_start:.2f}s.")

        logging.info("  Coalescing sparse tensors...")
        coalesce_start = time.time()
        if self.X_train_sparse is not None: self.X_train_sparse = self.X_train_sparse.coalesce()
        if self.X_pred_sparse is not None: self.X_pred_sparse = self.X_pred_sparse.coalesce()
        coalesce_end = time.time()
        logging.info(f"  Coalescing took {coalesce_end - coalesce_start:.2f}s.")


        build_end = time.time()
        logging.info(f"[TensorBuilder] Tensors built in {build_end - build_start:.2f}s.")

    #returns sparse X_train, dense Y_train, dense mask_train.
    def get_training_data(self):
        if self.X_train_sparse is None:
            self.build_tensors()
        return self.X_train_sparse, self.Y_train, self.mask_train

    #returns sparse X_pred, dense mask_pred, dense pred_indices.
    def get_prediction_data(self):
        if self.X_pred_sparse is None:
            self.build_tensors()
        return self.X_pred_sparse, self.mask_pred, self.pred_indices

    def get_feature_info(self):
        if self.max_domain_size <= 0:
             logging.warning("get_feature_info called but max_domain_size is not positive. Trying to determine it.")
             if not self.loaded_state and self.X_train_sparse is None:
                  self.build_tensors()
             if self.max_domain_size <= 0:
                  logging.error("Failed to determine positive max_domain_size in get_feature_info.")
                  return self.num_features, 0
        return self.num_features, self.max_domain_size

    def get_minimality_feature_index(self):
        return self.minimality_feature_idx

    def get_var_idx_to_domain_map(self):
        return self.var_idx_to_domain

    def get_var_idx_to_cell_map(self):
         return self.var_idx_to_cell

    #saves state needed for prediction (feature map, domain size type things)
    def save_state(self, filepath="tensor_builder_state.pkl"):
        if self.max_domain_size <= 0:
             logging.warning(f"Attempting to save state but max_domain_size is not positive ({self.max_domain_size}).")

        # saves state thats needed for prediction consistency
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

    #loads state before building tensors for prediction.
    def load_state(self, filepath="tensor_builder_state.pkl"):
        if not os.path.exists(filepath):
             logging.error(f"TensorBuilder state file not found: {filepath}")
             self.loaded_state = False
             return False
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.feature_to_idx = state.get('feature_to_idx', {})
            self.num_features = state.get('num_features', 0)
            self.max_domain_size = state.get('max_domain_size', 0)
            self.minimality_feature_idx = state.get('minimality_feature_idx', -1)

            #rebuild reverse map
            self.idx_to_feature = {v: k for k, v in self.feature_to_idx.items()}

            logging.info(f"[TensorBuilder] State loaded from {filepath}")
            logging.info(f"  Loaded num_features: {self.num_features}")
            logging.info(f"  Loaded max_domain_size: {self.max_domain_size}")
            logging.info(f"  Loaded minimality_feature_idx: {self.minimality_feature_idx}")
            self.loaded_state = True
            if self.max_domain_size <= 0 and self.num_features > 0:
                 logging.warning("Loaded state has max_domain_size <= 0.")
            return True
        except Exception as e:
            logging.error(f"Error loading TensorBuilder state: {e}", exc_info=True)
            self.loaded_state = False
            return False

