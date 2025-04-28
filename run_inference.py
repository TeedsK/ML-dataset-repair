# File: run_inference.py (Modified for PyTorch)
# Orchestrates model building, training (fitting), and inference using PyTorch.

import psycopg2
import psycopg2.extras # Needed for execute_batch if used internally by TensorBuilder
import sys
import argparse
import pickle # To save results
import time
import torch # Need torch for tensor operations
import logging
import pandas as pd # Needed for saving results potentially

from config import DB_SETTINGS
# --- Remove Old Import ---
# from model.factor_graph import HoloCleanFactorGraph

# --- Add New Imports ---
from model.learning.tensor_builder import TensorBuilder
from model.learning.pytorch_model import RepairModel

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')


def format_marginals(vid_to_cell, pred_idx_to_vid, probabilities, tensor_builder):
    """
    Converts the raw probability tensor output from the model into the
    desired marginals dictionary format: {(tid, attr): {cand_val: prob}}.
    """
    logging.info("Formatting probabilities into marginals dictionary...")
    marginals = {}
    num_predictions = probabilities.shape[0]

    for pred_idx in range(num_predictions):
        vid = pred_idx_to_vid.get(pred_idx)
        if vid is None:
            logging.warning(f"Cannot find VID for prediction index {pred_idx}.")
            continue

        cell = vid_to_cell.get(vid)
        if cell is None:
            logging.warning(f"Cannot find cell (tid, attr) for VID {vid}.")
            continue

        tid, attr = cell
        domain_size = tensor_builder.get_domain_size(vid)
        # Get probabilities for this variable's valid domain
        # Probabilities tensor shape: (num_preds, max_domain_size)
        var_probs = probabilities[pred_idx, :domain_size].numpy() # Get valid probs as numpy array

        # Normalize just in case softmax didn't sum perfectly to 1 due to masking/float issues
        prob_sum = var_probs.sum()
        if prob_sum > 1e-6 and abs(prob_sum - 1.0) > 1e-5:
             var_probs /= prob_sum
        elif prob_sum < 1e-6: # Handle case where all probabilities might be zero
             # Assign uniform probability if sum is too small
             logging.warning(f"Probabilities sum to near zero for VID {vid}. Assigning uniform.")
             var_probs = np.ones(domain_size) / domain_size

        # Map domain indices back to candidate values
        cell_marginals = {}
        for domain_idx in range(domain_size):
            candidate_val = tensor_builder.get_domain_value(vid, domain_idx)
            # Use original value if placeholder couldn't be mapped back, or handle as needed
            if candidate_val is None:
                candidate_val = f"__UNKNOWN_IDX_{domain_idx}__" # Placeholder if mapping failed
            cell_marginals[candidate_val] = float(var_probs[domain_idx])

        marginals[(tid, attr)] = cell_marginals

    logging.info(f"Formatted marginals for {len(marginals)} cells.")
    return marginals


def main():
    parser = argparse.ArgumentParser(description="Run HoloClean Learning and Inference (PyTorch version)")
    # Keep relevant arguments for PyTorch model
    parser.add_argument('--learniter', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for Adam optimizer.')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--weightdecay', type=float, default=1e-6, help='Weight decay (L2 penalty) for Adam optimizer.')
    # Remove args specific to old Gibbs sampling
    # parser.add_argument('--samples', type=int, default=1000, help='Number of Gibbs samples.')
    # parser.add_argument('--burnin', type=int, default=200, help='Number of Gibbs burn-in samples.')
    parser.add_argument('--outfile', type=str, default='inference_marginals.pkl', help='File to save the resulting marginals.')
    args = parser.parse_args()

    if args.learniter <= 0 or args.lr <= 0 or args.batchsize <= 0 or args.weightdecay < 0:
        logging.error("Error: learniter, lr, batchsize must be > 0; weightdecay >= 0.")
        sys.exit(1)

    conn = None
    try:
        start_pipeline_time = time.time()
        logging.info("Connecting to database...")
        conn = psycopg2.connect(**DB_SETTINGS)
        # psycopg2.extras.register_uuid() # May not be needed if TensorBuilder doesn't use execute_batch
        logging.info("Connection successful.")

        # --- 1. Build Tensors ---
        tensor_builder = TensorBuilder(db_conn=conn)
        # build_tensors is implicitly called by get_ methods if tensors are None
        X_train, Y_train, mask_train = tensor_builder.get_training_data()
        X_pred, mask_pred, pred_idx_to_vid = tensor_builder.get_infer_data()
        tensor_info = tensor_builder.get_tensor_info()

        if X_train is None or Y_train is None or X_pred is None:
             logging.error("Failed to build necessary tensors. Exiting.")
             sys.exit(1)

        # --- 2. Instantiate and Train Model ---
        repair_model = RepairModel(
            num_features=tensor_info['num_features'],
            max_domain_size=tensor_info['max_domain_size'],
            lr=args.lr,
            epochs=args.learniter,
            batch_size=args.batchsize,
            weight_decay=args.weightdecay,
            use_bias=False # Keep bias off for now, consistent with earlier attempts
        )

        repair_model.fit(X_train, Y_train, mask_train)

        # --- 3. Predict Probabilities ---
        # predict_proba returns torch tensor (on CPU)
        Y_pred_proba = repair_model.predict_proba(X_pred, mask_pred)

        if Y_pred_proba is not None:
            logging.info(f"Prediction tensor shape: {Y_pred_proba.shape}")

            # --- 4. Format Marginals ---
            marginals = format_marginals(
                vid_to_cell=tensor_builder.vid_to_cell,
                pred_idx_to_vid=pred_idx_to_vid,
                probabilities=Y_pred_proba,
                tensor_builder=tensor_builder # Pass for helper methods
            )

            # --- 5. Save and Print Results ---
            print(f"\n--- Inference Complete ---")
            try:
                with open(args.outfile, 'wb') as f:
                    pickle.dump(marginals, f)
                logging.info(f"Marginal probabilities saved to {args.outfile}")

                # Print some example marginals
                print("\nExample marginals:")
                count = 0
                for (tid, attr), probs in marginals.items():
                     print(f"Cell ({tid}, '{attr}'):")
                     # Sort probabilities descending for display
                     sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                     for val, prob in sorted_probs[:5]: # Show top 5
                         # Ensure proper display of None/Null values
                         display_val = "'None'" if val is None else f"'{val}'"
                         print(f"  {display_val}: {prob:.4f}")
                     if len(sorted_probs) > 5:
                         print("  ...")
                     count += 1
                     if count >= 5: # Print for first 5 noisy cells
                         break
            except Exception as e:
                logging.error(f"Error saving or printing marginals: {e}")
        else:
             logging.error("Inference prediction failed.")

        pipeline_time = time.time() - start_pipeline_time
        logging.info(f"--- Pipeline execution finished in {pipeline_time:.2f} seconds ---")

    except psycopg2.Error as db_err:
        logging.error(f"Database error during inference pipeline: {db_err}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during inference pipeline: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()