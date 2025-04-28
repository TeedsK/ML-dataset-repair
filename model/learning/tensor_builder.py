# File: tensor_builder.py
# Builds PyTorch tensors needed for the HoloClean model from database data.

import pandas as pd
import torch
import time
from collections import defaultdict
import numpy as np
from tqdm import tqdm # For progress bars

# Define NULL representation consistent with HoloClean/other parts if needed
# From utils.py in original HoloClean: NULL_REPR = "_nan_"
# We will handle None directly from DB where possible. Use a placeholder if strings needed.
NULL_REPR_PLACEHOLDER = "__NULL__" # Placeholder for nulls in domain map if needed

class TensorBuilder:
    """
    Queries the database and constructs PyTorch tensors (X, Y, mask)
    for training and inference, similar to FeaturizedDataset from HoloClean.
    """
    def __init__(self, db_conn):
        self.db_conn = db_conn
        print("[TensorBuilder] Initializing...")

        # --- Mappings and Metadata ---
        # Features
        self.feature_to_idx: dict[str, int] = {}
        self.idx_to_feature: dict[int, str] = {}
        self.num_features: int = 0
        # Variables (Cells) & VIDs
        self.cell_to_vid: dict[tuple[int, str], int] = {} # (tid, attr) -> vid
        self.vid_to_cell: dict[int, tuple[int, str]] = {} # vid -> (tid, attr)
        self.total_vars: int = 0
        # Domains
        self.vid_to_domain_map: dict[int, dict[str, int]] = defaultdict(dict) # vid -> {cand_val_str: domain_idx}
        self.vid_to_domain_size: dict[int, int] = defaultdict(int) # vid -> num_states
        self.vid_to_idx_to_val: dict[int, dict[int, str]] = defaultdict(dict) # vid -> {domain_idx: cand_val_str}
        self.max_domain_size: int = 0 # Max number of states across all vars
        # Features per State
        self.vid_domainidx_to_featureidxs: dict[tuple[int, int], list[int]] = defaultdict(list) # (vid, domain_idx) -> [feat_idx1, feat_idx2, ...]
        # Evidence & Query Separation
        self.evidence_vids: list[int] = []
        self.query_vids: list[int] = []
        self.vid_to_evidence_domain_idx: dict[int, int] = {} # For Y_train: vid -> correct_domain_idx
        # Mapping VIDs to Tensor Rows
        self.vid_to_train_idx: dict[int, int] = {} # Evidence VID -> row index in X_train etc.
        self.vid_to_pred_idx: dict[int, int] = {} # Query VID -> row index in X_pred etc.
        self.pred_idx_to_vid: dict[int, int] = {} # Row index in X_pred -> Query VID

        # --- Load data and build all mappings ---
        load_start_time = time.time()
        if not self._load_and_build_metadata():
             raise RuntimeError("TensorBuilder failed to load metadata from database.")
        print(f"[TensorBuilder] Metadata loaded and mappings built in {time.time() - load_start_time:.2f}s.")

        # --- Initialize tensors (will be populated by build_tensors) ---
        self.X_train = None
        self.Y_train = None
        self.mask_train = None
        self.X_pred = None
        self.mask_pred = None


    def _load_and_build_metadata(self) -> bool:
        """Queries DB tables and populates internal mappings."""
        print("[TensorBuilder] Loading metadata from DB (cells, domains, features)...")
        try:
            # 1. Process Features (get unique features and assign indices)
            print("  Loading features...")
            features_df = pd.read_sql("SELECT DISTINCT feature FROM features", self.db_conn)
            if features_df.empty: print("WARNING: No features found in the database!"); return False
            self.feature_to_idx = {name: i for i, name in enumerate(features_df['feature'])}
            self.idx_to_feature = {i: name for name, i in self.feature_to_idx.items()}
            self.num_features = len(self.feature_to_idx)
            print(f"  Found {self.num_features} unique features.")

            # 2. Process Cells (Assign VIDs, identify evidence/query)
            print("  Loading cells info...")
            cells_df = pd.read_sql("SELECT tid, attr, val, is_noisy FROM cells ORDER BY tid, attr", self.db_conn)
            if cells_df.empty: print("ERROR: Cells table is empty."); return False
            self.total_vars = 0
            cell_original_value_map = {} # Store original value for evidence lookup
            for idx, row in cells_df.iterrows():
                vid = self.total_vars # Assign sequential VIDs
                cell = (row['tid'], row['attr'])
                self.cell_to_vid[cell] = vid
                self.vid_to_cell[vid] = cell
                cell_original_value_map[vid] = row['val'] # Store original value string or None

                if row['is_noisy']:
                    self.query_vids.append(vid)
                else:
                    self.evidence_vids.append(vid)
                self.total_vars += 1
            print(f"  Processed {self.total_vars} total cells (variables).")
            print(f"  Identified {len(self.evidence_vids)} evidence variables and {len(self.query_vids)} query variables.")
            if len(self.evidence_vids) == 0: print("WARNING: No evidence variables found for training!")

            # 3. Process Domains (Map candidates to domain indices for each VID)
            print("  Loading domains...")
            domains_df = pd.read_sql("SELECT tid, attr, candidate_val FROM domains ORDER BY tid, attr, candidate_val", self.db_conn)
            if domains_df.empty: print("ERROR: Domains table is empty."); return False

            current_vid = -1
            current_domain_idx = 0
            max_domain_size_found = 0
            for _, row in tqdm(domains_df.iterrows(), total=len(domains_df), desc="  Processing domains"):
                cell = (row['tid'], row['attr'])
                vid = self.cell_to_vid.get(cell)
                if vid is None: continue # Should not happen if domains are valid

                if vid != current_vid:
                    # Finished processing previous variable's domain
                    if current_vid != -1:
                         self.vid_to_domain_size[current_vid] = current_domain_idx
                         max_domain_size_found = max(max_domain_size_found, current_domain_idx)
                    # Start new variable
                    current_vid = vid
                    current_domain_idx = 0

                cand_val_str = str(row['candidate_val']) # Ensure string representation
                # Handle potential None values from DB explicitly if needed
                if cand_val_str == 'None' or cand_val_str is None:
                     cand_val_str = NULL_REPR_PLACEHOLDER # Use consistent placeholder

                self.vid_to_domain_map[vid][cand_val_str] = current_domain_idx
                self.vid_to_idx_to_val[vid][current_domain_idx] = cand_val_str

                # If this is an evidence VID, find the domain index of its original value
                if vid in self.evidence_vids:
                     original_val = cell_original_value_map.get(vid)
                     original_val_str = str(original_val) if original_val is not None else NULL_REPR_PLACEHOLDER
                     if cand_val_str == original_val_str:
                         self.vid_to_evidence_domain_idx[vid] = current_domain_idx

                current_domain_idx += 1

            # Record domain size for the last variable
            if current_vid != -1:
                 self.vid_to_domain_size[current_vid] = current_domain_idx
                 max_domain_size_found = max(max_domain_size_found, current_domain_idx)
            self.max_domain_size = max_domain_size_found
            print(f"  Max domain size across all variables: {self.max_domain_size}")

            # Verify all evidence variables got an evidence index
            for vid in self.evidence_vids:
                 if vid not in self.vid_to_evidence_domain_idx:
                     print(f"WARNING: Evidence VID {vid} {self.vid_to_cell[vid]} did not find its original value ('{cell_original_value_map.get(vid)}') in its domain candidates. Cannot use for training.")
                     # Optionally remove from evidence_vids list here if needed downstream
                     # self.evidence_vids.remove(vid) # Be careful if iterating

            # 4. Process Features per State (Map (vid, domain_idx) -> [feat_idx])
            print("  Loading features per state...")
            # Fetch all features with their cell and candidate value
            all_features_df = pd.read_sql(
                "SELECT tid, attr, candidate_val, feature FROM features",
                self.db_conn
            )
            if all_features_df.empty: print("WARNING: No features found in features table!"); return False

            for _, row in tqdm(all_features_df.iterrows(), total=len(all_features_df), desc="  Mapping features"):
                 cell = (row['tid'], row['attr'])
                 vid = self.cell_to_vid.get(cell)
                 feat_idx = self.feature_to_idx.get(row['feature'])
                 if vid is None or feat_idx is None: continue # Skip if cell or feature not recognized

                 cand_val_str = str(row['candidate_val'])
                 if cand_val_str == 'None' or cand_val_str is None:
                      cand_val_str = NULL_REPR_PLACEHOLDER

                 domain_idx = self.vid_to_domain_map[vid].get(cand_val_str)
                 if domain_idx is not None:
                     self.vid_domainidx_to_featureidxs[(vid, domain_idx)].append(feat_idx)
                 # else: print(f"WARN: Feature references unknown candidate '{cand_val_str}' for VID {vid}") # Can be verbose

            print(f"  Built feature map for states.")

            # 5. Create Mappings for Tensor Indices
            for i, vid in enumerate(self.evidence_vids): self.vid_to_train_idx[vid] = i
            for i, vid in enumerate(self.query_vids):
                 self.vid_to_pred_idx[vid] = i
                 self.pred_idx_to_vid[i] = vid # Reverse map for inference results

            return True

        except Exception as e:
            print(f"ERROR loading metadata: {e}")
            import traceback
            traceback.print_exc()
            return False


    def build_tensors(self):
        """Constructs and populates the PyTorch tensors."""
        if not self.evidence_vids and not self.query_vids:
             print("ERROR: No evidence or query variables identified. Cannot build tensors.")
             return False
        if self.max_domain_size == 0 or self.num_features == 0:
             print("ERROR: Max domain size or number of features is zero. Cannot build tensors.")
             return False

        print("[TensorBuilder] Building PyTorch tensors...")
        build_start_time = time.time()

        # --- Initialize Tensors ---
        num_train = len(self.evidence_vids)
        num_pred = len(self.query_vids)

        self.X_train = torch.zeros((num_train, self.max_domain_size, self.num_features), dtype=torch.float32)
        self.Y_train = torch.full((num_train,), -1, dtype=torch.long) # Initialize with -1
        self.mask_train = torch.full((num_train, self.max_domain_size), -1e6, dtype=torch.float32) # Mask value

        self.X_pred = torch.zeros((num_pred, self.max_domain_size, self.num_features), dtype=torch.float32)
        self.mask_pred = torch.full((num_pred, self.max_domain_size), -1e6, dtype=torch.float32)

        # --- Populate Training Tensors ---
        print(f"  Populating training tensors ({num_train} variables)...")
        for vid in tqdm(self.evidence_vids, desc="  Training tensors"):
             row_idx = self.vid_to_train_idx.get(vid)
             if row_idx is None: continue # Should not happen

             # Populate Y_train
             evidence_domain_idx = self.vid_to_evidence_domain_idx.get(vid)
             if evidence_domain_idx is not None:
                 self.Y_train[row_idx] = evidence_domain_idx
             # else: already warned in metadata loading

             # Populate X_train and mask_train
             domain_size = self.vid_to_domain_size.get(vid, 0)
             self.mask_train[row_idx, :domain_size] = 0.0 # Unmask valid domain indices
             for domain_idx in range(domain_size):
                 feature_indices = self.vid_domainidx_to_featureidxs.get((vid, domain_idx), [])
                 if feature_indices:
                     # Set active features to 1.0 for this state (domain_idx)
                     self.X_train[row_idx, domain_idx, feature_indices] = 1.0

        # --- Populate Prediction Tensors ---
        print(f"  Populating prediction tensors ({num_pred} variables)...")
        for vid in tqdm(self.query_vids, desc="  Prediction tensors"):
             row_idx = self.vid_to_pred_idx.get(vid)
             if row_idx is None: continue # Should not happen

             # Populate X_pred and mask_pred
             domain_size = self.vid_to_domain_size.get(vid, 0)
             self.mask_pred[row_idx, :domain_size] = 0.0 # Unmask valid domain indices
             for domain_idx in range(domain_size):
                 feature_indices = self.vid_domainidx_to_featureidxs.get((vid, domain_idx), [])
                 if feature_indices:
                     self.X_pred[row_idx, domain_idx, feature_indices] = 1.0

        print(f"[TensorBuilder] Tensors built in {time.time() - build_start_time:.2f}s.")
        return True

    def get_training_data(self):
        """Returns the tensors needed for training."""
        if self.X_train is None or self.Y_train is None or self.mask_train is None:
            print("WARN: Training tensors not built yet. Call build_tensors() first.")
            # Attempt to build them now
            if not self.build_tensors():
                 return None, None, None
        return self.X_train, self.Y_train, self.mask_train

    def get_infer_data(self):
        """Returns the tensors needed for inference/prediction."""
        if self.X_pred is None or self.mask_pred is None:
            print("WARN: Prediction tensors not built yet. Call build_tensors() first.")
            if not self.build_tensors():
                return None, None, None
        return self.X_pred, self.mask_pred, self.pred_idx_to_vid

    # --- Helper method to get domain info for the model ---
    def get_tensor_info(self):
        """Returns essential dimensions needed by the PyTorch model."""
        return {
            'num_features': self.num_features,
            'max_domain_size': self.max_domain_size
        }

    # --- Helper method to map inferred indices back to values ---
    def get_domain_value(self, vid, domain_idx):
        """Gets the candidate value string for a given VID and domain index."""
        val = self.vid_to_idx_to_val.get(vid, {}).get(domain_idx)
        # Convert placeholder back if needed, though None might be better representation
        return val if val != NULL_REPR_PLACEHOLDER else None

    def get_domain_size(self, vid):
        """Gets the actual domain size for a specific VID."""
        return self.vid_to_domain_size.get(vid, 0)