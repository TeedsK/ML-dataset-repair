# File: run_inference.py
# Main script to run model training and inference.
# ADDED: Modes for train/predict and save/load paths.

import time
import logging
import argparse
import pickle
import os # Added

import psycopg2
import torch

import config # Assuming config.py holds DB settings
from model.learning.tensor_builder import TensorBuilder
from model.learning.pytorch_model import RepairModel # Your model class

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_pipeline(db_conn, args):
    """Runs the training and/or inference pipeline based on mode."""
    pipeline_start_time = time.time()

    builder = TensorBuilder(db_conn)
    repair_model = None # Initialize

    # --- Mode Logic ---
    if args.mode == 'train' or args.mode == 'train_predict':
        logging.info("--- Running in Training Mode ---")
        # 1a. Build Tensors for Training (Discover features)
        builder._load_metadata() # Discover features, domains etc. from current DB
        X_train, Y_train, mask_train = builder.get_training_data()
        num_features, max_domain_size = builder.get_feature_info()
        minimality_idx = builder.get_minimality_feature_index()

        if X_train.shape[0] == 0:
            logging.error("No training data available. Check DB state.")
            return

        # --- ADDED: Tensor inspection ---
        if args.inspect_idx >= 0:
             inspect_tensor(builder, X_train, Y_train, sample_train_idx=args.inspect_idx)
        # --- END ADDED ---

        # 1b. Initialize Model for Training
        repair_model = RepairModel(
            num_features=num_features,
            output_dim=max_domain_size,
            learning_rate=args.lr,
            weight_decay=args.weightdecay,
            optimizer_type=args.optimizer,
            device=args.device
        )
        repair_model.set_minimality_feature_index(minimality_idx)

        # 1c. Train Model
        repair_model.fit(
            X_train, Y_train, mask_train,
            epochs=args.learniter,
            batch_size=args.batchsize
        )

        # 1d. Save Model and Builder State
        if args.save_model_path:
            repair_model.save_weights(args.save_model_path)
        if args.save_builder_path:
            builder.save_state(args.save_builder_path) # Save the map used for training

    if args.mode == 'predict' or args.mode == 'train_predict':
        logging.info("--- Running in Prediction Mode ---")

        # 2a. Load Builder State (Feature Map, Domain Size) from Training Phase
        if args.load_builder_path and os.path.exists(args.load_builder_path):
             if not builder.load_state(args.load_builder_path):
                 logging.error("Failed to load TensorBuilder state. Cannot proceed with prediction.")
                 return
             # Now load metadata *using* the loaded state
             builder._load_metadata(
                 reuse_feature_map=builder.feature_to_idx,
                 reuse_max_domain=builder.max_domain_size
             )
        elif args.mode == 'predict': # Require builder state if *only* predicting
             logging.error(f"--load_builder_path='{args.load_builder_path}' is required for predict mode.")
             return
        else: # train_predict mode, builder already initialized in train part
             logging.info("Continuing with builder state from training phase.")
             pass # builder is already populated from training step


        # 2b. Get Prediction Data Tensors using loaded/current builder state
        # build_tensors will use the potentially loaded feature map & max domain size
        X_pred, mask_pred, pred_indices = builder.get_prediction_data()
        num_features, max_domain_size = builder.get_feature_info() # Get potentially loaded info
        minimality_idx = builder.get_minimality_feature_index() # Get potentially loaded info


        if X_pred.shape[0] == 0:
             logging.warning("No prediction data (query variables) found.")
             # Proceed to save empty marginals or exit? Exit for now.
             return

        # 2c. Initialize Model structure (if not already initialized in train_predict mode)
        if repair_model is None:
             repair_model = RepairModel(
                 num_features=num_features,
                 output_dim=max_domain_size,
                 device=args.device
                 # LR etc. not needed if only predicting
             )
             # Must set minimality index if gradient zeroing needed during predict?
             # Typically not needed, but set it for consistency if loaded.
             repair_model.set_minimality_feature_index(minimality_idx)

        # 2d. Load Trained Weights
        if args.load_model_path and os.path.exists(args.load_model_path):
             if not repair_model.load_weights(args.load_model_path):
                  logging.error("Failed to load model weights. Cannot proceed with prediction.")
                  return
        elif args.mode == 'predict': # Require weights if *only* predicting
             logging.error(f"--load_model_path='{args.load_model_path}' is required for predict mode.")
             return
        else: # train_predict mode, model already trained
             logging.info("Using model weights from current training phase.")
             pass

        # 2e. Predict Probabilities
        pred_probas = repair_model.predict_proba(X_pred, mask_pred, batch_size=args.batchsize)

        # 2f. Format and Save Results
        logging.info("Formatting probabilities into marginals dictionary...")
        format_start = time.time()
        var_idx_to_domain = builder.get_var_idx_to_domain_map()
        var_idx_to_cell = builder.get_var_idx_to_cell_map()
        marginals = {}
        pred_indices_list = pred_indices.tolist()

        for i in range(pred_probas.shape[0]):
            original_var_idx = pred_indices_list[i]
            cell_id = var_idx_to_cell.get(original_var_idx)
            if cell_id is None: continue
            domain_map = var_idx_to_domain.get(original_var_idx, {})
            cell_marginals = {}
            num_actual_candidates = len(domain_map)
            tensor_width = pred_probas.shape[1]
            for domain_idx in range(min(num_actual_candidates, tensor_width)):
                 candidate_val = domain_map.get(domain_idx, f"UNK_IDX_{domain_idx}")
                 prob = pred_probas[i, domain_idx].item()
                 cell_marginals[candidate_val] = prob
            marginals[cell_id] = cell_marginals
        format_end = time.time()
        logging.info(f"Formatted marginals for {len(marginals)} cells in {format_end - format_start:.3f}s.")

        output_file = args.pred_output_file # Use specified output file
        with open(output_file, 'wb') as f:
            pickle.dump(marginals, f)
        logging.info(f"Marginal probabilities saved to {output_file}\n")

        # ... (Optional: print examples) ...

    pipeline_end_time = time.time()
    logging.info(f"\n--- Pipeline ({args.mode} mode) Finished ---")
    logging.info(f"--- Total execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds ---")


# --- ADDED: Function to inspect features (keep from previous version) ---
def inspect_tensor(tensor_builder, X_train, Y_train, sample_train_idx=0):
    """Prints active features for a sample training variable."""
    # ... [Paste the inspect_tensor function from the previous response here] ...
    if sample_train_idx >= X_train.shape[0]:
        logging.warning(f"Cannot inspect sample index {sample_train_idx}, only {X_train.shape[0]} training samples exist.")
        return
    logging.info(f"\n--- Inspecting Features for Training Sample Index: {sample_train_idx} ---")
    original_var_idx = tensor_builder.evidence_var_indices[sample_train_idx]
    cell_id = tensor_builder.get_var_idx_to_cell_map().get(original_var_idx, "Unknown Cell")
    logging.info(f"  Corresponds to original var_idx: {original_var_idx}, Cell: {cell_id}")
    target_domain_idx = Y_train[sample_train_idx].item()
    logging.info(f"  Target domain index (initial value): {target_domain_idx}")
    domain_map = tensor_builder.get_var_idx_to_domain_map().get(original_var_idx, {})
    target_val = domain_map.get(target_domain_idx, "Unknown Target Value")
    logging.info(f"  Target value: '{target_val}'")
    target_features_slice = X_train[sample_train_idx, target_domain_idx, :]
    active_target_indices = target_features_slice.nonzero(as_tuple=True)[0].tolist()
    active_target_features = [tensor_builder.idx_to_feature.get(idx, f"UNK_FEAT_{idx}") for idx in active_target_indices]
    logging.info(f"\n  Features ACTIVE for TARGET index {target_domain_idx} ('{target_val}'):")
    logging.info(f"    Indices: {active_target_indices}")
    logging.info(f"    Names: {active_target_features}")
    other_domain_idx = -1
    other_val = "N/A"
    max_idx = X_train.shape[1] - 1
    for idx_to_check in range(max_idx + 1):
         if idx_to_check != target_domain_idx and idx_to_check in domain_map:
              other_domain_idx = idx_to_check
              other_val = domain_map.get(other_domain_idx, "Unknown Other Value")
              break
    if other_domain_idx != -1:
        logging.info(f"\n  Comparing with OTHER valid index {other_domain_idx} ('{other_val}'):")
        other_features_slice = X_train[sample_train_idx, other_domain_idx, :]
        active_other_indices = other_features_slice.nonzero(as_tuple=True)[0].tolist()
        active_other_features = [tensor_builder.idx_to_feature.get(idx, f"UNK_FEAT_{idx}") for idx in active_other_indices]
        logging.info(f"    Indices: {active_other_indices}")
        logging.info(f"    Names: {active_other_features}")
        target_set = set(active_target_indices)
        other_set = set(active_other_indices)
        only_in_target_indices = list(target_set - other_set)
        only_in_other_indices = list(other_set - target_set)
        only_in_target_names = [tensor_builder.idx_to_feature.get(idx, f"UNK_FEAT_{idx}") for idx in only_in_target_indices]
        only_in_other_names = [tensor_builder.idx_to_feature.get(idx, f"UNK_FEAT_{idx}") for idx in only_in_other_indices]
        logging.info(f"\n  DIFFERENCES:")
        logging.info(f"    Features ONLY active for TARGET index {target_domain_idx}: {only_in_target_names}")
        logging.info(f"      (Indices: {only_in_target_indices})")
        logging.info(f"    Features ONLY active for OTHER index {other_domain_idx}: {only_in_other_names}")
        logging.info(f"      (Indices: {only_in_other_indices})")
    else:
         logging.info("\n  No other valid domain index found to compare with.")
    logging.info(f"--- End Feature Inspection ---\n")
# --- END ADDED Function ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HoloClean Inference Model Training and Prediction")
    # --- ADDED Mode Arguments ---
    parser.add_argument('--mode', type=str, default='train_predict', choices=['train', 'predict', 'train_predict'],
                        help='Operation mode: train only, predict only, or train then predict')
    parser.add_argument('--save_model_path', type=str, default='repair_model_weights.pth',
                        help='Path to save trained model weights (used in train modes)')
    parser.add_argument('--load_model_path', type=str, default='repair_model_weights.pth',
                        help='Path to load trained model weights (used in predict mode)')
    parser.add_argument('--save_builder_path', type=str, default='tensor_builder_state.pkl',
                        help='Path to save TensorBuilder state (used in train modes)')
    parser.add_argument('--load_builder_path', type=str, default='tensor_builder_state.pkl',
                        help='Path to load TensorBuilder state (used in predict mode)')
    parser.add_argument('--pred_output_file', type=str, default='inference_marginals.pkl',
                        help='Output file for prediction marginals (used in predict modes)')
    # --- END ADDED ---

    # --- Keep Original Arguments ---
    parser.add_argument('--learniter', type=int, default=10, help='Number of learning iterations (epochs)')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training and prediction')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weightdecay', type=float, default=1e-6, help='Weight decay (L2 penalty)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to run on (e.g., cpu, cuda, mps)')
    parser.add_argument('--inspect_idx', type=int, default=-1, help='Index of training sample to inspect features for (-1 to disable)') # Default -1

    args = parser.parse_args()

    db_connection = None
    try:
        logging.info("Connecting to database...")
        db_connection = psycopg2.connect(**config.DB_SETTINGS)
        logging.info("Connection successful.")
        run_pipeline(db_connection, args)
    except Exception as e:
        logging.error(f"An error occurred during the pipeline: {e}", exc_info=True)
    finally:
        if db_connection:
            db_connection.close()
            logging.info("Database connection closed.")