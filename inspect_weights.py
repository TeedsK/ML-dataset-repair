import pickle
import logging
import numpy as np
import sys
import os

from model.learning.pytorch_model import RepairModel 

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] {%(levelname)s} %(message)s')

#used to look at the weight values that are produced after training the model
def inspect_weights(model_path, builder_state_path):

    if not os.path.exists(model_path):
        logging.error(f"Model weights file not found: {model_path}")
        return
    if not os.path.exists(builder_state_path):
        logging.error(f"Builder state file not found: {builder_state_path}")
        return
    try:
        with open(builder_state_path, 'rb') as f:
            state = pickle.load(f)
        feature_to_idx = state.get('feature_to_idx', {})
        num_features = state.get('num_features', 0)
        max_domain_size = state.get('max_domain_size', 0)
        minimality_feature_idx_saved = state.get('minimality_feature_idx', -1)
        idx_to_feature = {v: k for k, v in feature_to_idx.items()}
        logging.info(f"Loaded builder state: {num_features} features, max domain {max_domain_size}")
        if minimality_feature_idx_saved != -1:
             logging.info(f"  Minimality feature index from state: {minimality_feature_idx_saved} ('{idx_to_feature.get(minimality_feature_idx_saved)}')")
        else:
             logging.warning("  Minimality feature index not found in state.")
    except Exception as e:
        logging.error(f"Error loading builder state: {e}", exc_info=True)
        return

    if num_features <= 0 or max_domain_size <= 0:
        logging.error("Invalid num_features or max_domain_size loaded from state.")
        return

    try:
        repair_model = RepairModel(num_features, max_domain_size, device='cpu')
    except Exception as e:
        logging.error(f"Error initializing RepairModel structure: {e}", exc_info=True)
        return

    if not repair_model.load_weights(model_path):
        logging.error("Failed to load model weights.")
        return

    try:
        weights = repair_model.model.weight.data.squeeze().cpu().numpy()

        if weights.shape[0] != num_features:
             logging.error(f"Mismatch between loaded weights shape ({weights.shape[0]}) and num_features from state ({num_features})")
             return
        
        logging.info(f"Total number of weights (features): {len(weights)}")
        prior_weight = np.nan
        if minimality_feature_idx_saved != -1 and 0 <= minimality_feature_idx_saved < len(weights):
            prior_weight = weights[minimality_feature_idx_saved]
            logging.info(f"Weight for 'prior_minimality' (Index {minimality_feature_idx_saved}): {prior_weight:.4f}")
        else:
            logging.warning("'prior_minimality' feature index not found or invalid in loaded weights.")

        dc_violation_weights = []
        cooc_weights = []
        freq_weights = []
        other_weights = []

        for idx, weight in enumerate(weights):
            if idx == minimality_feature_idx_saved:
                continue
            feature_name = idx_to_feature.get(idx, f"UNKNOWN_FEATURE_{idx}")

            if feature_name.startswith("DC_VIOLATES_"):
                dc_violation_weights.append(weight)
            elif feature_name.startswith("cooc_"):
                cooc_weights.append(weight)
            elif feature_name.startswith("LOGFREQ_"):
                freq_weights.append(weight)
            else:
                other_weights.append((feature_name, weight))

        def report_stats(name, weight_list):
            if not weight_list:
                logging.info(f"{name}: No features found.")
                return
            weights_np = np.array(weight_list)
            logging.info(f"{name} ({len(weights_np)} features):")
            logging.info(f"  Mean: {np.mean(weights_np):.4f}, StdDev: {np.std(weights_np):.4f}")
            logging.info(f"  Min: {np.min(weights_np):.4f}, Max: {np.max(weights_np):.4f}")
            logging.info(f"  Mean Absolute: {np.mean(np.abs(weights_np)):.4f}")
           
        report_stats("DC Violation Features", dc_violation_weights)
        report_stats("Cooccurrence Features", cooc_weights)
        report_stats("Frequency Features", freq_weights)

        if other_weights:
             logging.warning(f"Found {len(other_weights)} features with unexpected names:")
             for name, w in other_weights[:10]:
                  logging.warning(f"  '{name}': {w:.4f}")

        logging.info("--- End Weight Inspection ---")

    except Exception as e:
        logging.error(f"An error occurred during weight inspection: {e}", exc_info=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inspect_weights.py <path_to_model.pth> <path_to_builder_state.pkl>")
        sys.exit(1)

    model_file = sys.argv[1]
    builder_file = sys.argv[2]

    inspect_weights(model_file, builder_file)
