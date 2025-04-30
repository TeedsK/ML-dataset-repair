import time
import logging
import argparse
import pickle
import os
import psycopg2
import config
from model.learning.tensor_builder import TensorBuilder
from model.learning.pytorch_model import RepairModel

# COMMENT OR UNCOMMENT THIS IF YOU WANT TO RECEIVE LOGS LIKE DEBUGS OR INFO
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s: %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

#runs the training and/or inference pipeline
def run_pipeline(db_conn, args):
   
    pipeline_start_time = time.time()

    builder = TensorBuilder(db_conn)
    repair_model = None

    if args.mode == 'train' or args.mode == 'train_predict':
        logging.info("--- Running in Training Mode ---")
       
        builder._load_metadata()
        X_train, Y_train, mask_train = builder.get_training_data()
        num_features, max_domain_size = builder.get_feature_info()
        minimality_idx = builder.get_minimality_feature_index()

        if X_train.shape[0] == 0:
            logging.error("No training data available. Check DB state.")
            return

        if args.inspect_idx >= 0:
             inspect_tensor(builder, X_train, Y_train, sample_train_idx=args.inspect_idx)
       
        repair_model = RepairModel(
            num_features=num_features,
            output_dim=max_domain_size,
            learning_rate=args.lr,
            weight_decay=args.weightdecay,
            optimizer_type=args.optimizer,
            device=args.device
        )
        repair_model.set_minimality_feature_index(minimality_idx)

        repair_model.fit(
            X_train, Y_train, mask_train,
            epochs=args.learniter,
            batch_size=args.batchsize
        )

        if args.save_model_path:
            repair_model.save_weights(args.save_model_path)
        if args.save_builder_path:
            builder.save_state(args.save_builder_path)

    if args.mode == 'predict' or args.mode == 'train_predict':
        logging.info("--- Running in Prediction Mode ---")

        if args.load_builder_path and os.path.exists(args.load_builder_path):
             if not builder.load_state(args.load_builder_path):
                 logging.error("Failed to load TensorBuilder state.")
                 return
             
             builder._load_metadata(
                 reuse_feature_map=builder.feature_to_idx,
                 reuse_max_domain=builder.max_domain_size
             )
        elif args.mode == 'predict':
             logging.error(f"needs builder path.")
             return
        else:
             logging.info("Continuing with builder state from training phase.")
             pass

        X_pred, mask_pred, pred_indices = builder.get_prediction_data()
        num_features, max_domain_size = builder.get_feature_info()
        minimality_idx = builder.get_minimality_feature_index()


        if X_pred.shape[0] == 0:
             logging.warning("No prediction data (query variables) found.")
             return

        if repair_model is None:
             repair_model = RepairModel(
                 num_features=num_features,
                 output_dim=max_domain_size,
                 device=args.device
             )
             repair_model.set_minimality_feature_index(minimality_idx)

        if args.load_model_path and os.path.exists(args.load_model_path):
             if not repair_model.load_weights(args.load_model_path):
                  logging.error("Failed to load model weights")
                  return
        elif args.mode == 'predict':
             logging.error(f"load model path is required for predict mode.")
             return
        else:
             logging.info("Using model weights from current training phase.")
             pass

        pred_probas = repair_model.predict_proba(X_pred, mask_pred, batch_size=args.batchsize)

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

        output_file = args.pred_output_file
        with open(output_file, 'wb') as f:
            pickle.dump(marginals, f)
        logging.info(f"Marginal probabilities saved to {output_file}\n")

    pipeline_end_time = time.time()
    logging.info(f"\n--- Pipeline ({args.mode} mode) Finished ---")
    logging.info(f"--- Total execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds ---")


#prints active features for a sample training variable.
def inspect_tensor(tensor_builder, X_train, Y_train, sample_train_idx=0):
    if sample_train_idx >= X_train.shape[0]:
        logging.warning(f"Cannot inspect sample index {sample_train_idx}")
        logging.warning(f"Only {X_train.shape[0]} training samples exist.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HoloClean Inference Model Training and Prediction")
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
    parser.add_argument('--learniter', type=int, default=10, help='Number of learning iterations (epochs)')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training and prediction')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weightdecay', type=float, default=1e-6, help='Weight decay (L2 penalty)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to run on (e.g., cpu, cuda, mps)')
    parser.add_argument('--inspect_idx', type=int, default=-1, help='Index of training sample to inspect features for (-1 to disable)')

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