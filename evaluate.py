import pandas as pd
import pickle
import psycopg2
import logging
import argparse
import os
import sys

try:
    import config
    NULL_REPR_PLACEHOLDER = config.NULL_REPR_PLACEHOLDER
except ImportError:
    logging.error("config.py not found or NULL_REPR_PLACEHOLDER not defined.")
    NULL_REPR_PLACEHOLDER = "__NULL__"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


# loads the ground truth from the CSV file
def load_ground_truth(filepath="hospital_100_clean.csv"):
    logging.info(f"Loading ground truth from: {filepath}")
    try:
        df_clean = pd.read_csv(filepath, names=['tid_csv', 'attribute', 'correct_val'], header=0, keep_default_na=False, dtype={'tid_csv': int, 'attribute': str, 'correct_val': str})
        ground_truth = {}
        processed_rows = 0
        for _, row in df_clean.iterrows():
            tid = row['tid_csv'] + 1
            attr = row['attribute']
            true_val = row['correct_val'] if row['correct_val'] != '' else NULL_REPR_PLACEHOLDER
            ground_truth[(tid, attr)] = str(true_val)
            processed_rows += 1
        logging.info(f"Loaded ground truth for {processed_rows} cells from {filepath}")
        if not ground_truth:
             logging.warning(f"Ground truth dictionary is empty after loading {filepath}. Check file format and content.")
        return ground_truth
    except FileNotFoundError:
        logging.error(f"Ground truth file not found at {filepath}")
        return None
    except KeyError as e:
         logging.error(f"Missing expected column in {filepath}: {e}. Expecting 'tid_csv', 'attribute', 'correct_val'.")
         return None
    except Exception as e:
        logging.error(f"Error loading ground truth: {e}", exc_info=True)
        return None

#loads the predictions from the marginals file
def load_predictions(filepath="inference_marginals.pkl"):
    logging.info(f"Loading predictions from: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            marginals = pickle.load(f)
        predictions = {}
        for cell_key, probs in marginals.items():
            if isinstance(cell_key, tuple) and len(cell_key) == 2:
                 if probs:
                     predicted_val = max(probs, key=probs.get)
                     predictions[cell_key] = str(predicted_val)
                 else:
                     logging.warning(f"No probabilities found for cell {cell_key} in marginals file.")
            else:
                 logging.warning(f"Skipping invalid key format in marginals file: {cell_key}")
        logging.info(f"Loaded predictions for {len(predictions)} cells.")
        return predictions
    except FileNotFoundError:
        logging.error(f"Predictions file not found at {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading predictions: {e}", exc_info=True)
        return None

#fetches original values and noisy status from the database
def get_original_data(db_conn):
    logging.info("Fetching original data and noisy status from database...")
    originals = {}
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT tid, attr, val, is_noisy FROM cells;")
            fetched_count = 0
            for tid, attr, val, is_noisy in cur:
                original_val_str = NULL_REPR_PLACEHOLDER if val is None else str(val)
                originals[(tid, attr)] = {"original": original_val_str, "is_noisy": is_noisy}
                fetched_count += 1
        logging.info(f"Fetched original data for {fetched_count} cells.")
        return originals
    except Exception as e:
        logging.error(f"Error fetching original data: {e}", exc_info=True)
        return None

#calculates precision, recall, f1-Score
def evaluate_repairs(predictions, ground_truth, originals):
    if predictions is None or ground_truth is None or originals is None:
        logging.error("Cannot evaluate due to missing data.")
        return None

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    fp_examples = []
    fn_examples = []
    tp_examples = []

    actual_errors = set()
    for cell, data in originals.items():
        true_val = ground_truth.get(cell)
        if true_val is not None and data["original"] != true_val:
            actual_errors.add(cell)
    actual_errors_count = len(actual_errors)
    logging.info(f"Total actual errors identified in original data (compared to ground truth): {actual_errors_count}")

    noisy_cells = {cell for cell, data in originals.items() if data["is_noisy"]}
    logging.info(f"Total noisy cells targeted for prediction: {len(noisy_cells)}")

    predictions_made_count = 0
    correctly_predicted_errors = set()
    incorrect_predictions = set()

    for cell in noisy_cells:
        predicted_val = predictions.get(cell)
        true_val = ground_truth.get(cell)
        original_val = originals[cell]["original"] if cell in originals else None

        if predicted_val is None or true_val is None or original_val is None:
            if cell in actual_errors:
                if predicted_val is None:
                     fn_examples.append({
                         "cell": cell, "original": original_val, "truth": true_val,
                         "predicted": "NO_PREDICTION", "reason": "No prediction available"
                     })
            continue

        predictions_made_count += 1

        is_correct_prediction = (predicted_val == true_val)
        is_actual_error = (cell in actual_errors)

        if is_correct_prediction and is_actual_error:
            true_positives += 1
            correctly_predicted_errors.add(cell)
            tp_examples.append({
                "cell": cell, "original": original_val, "truth": true_val, "predicted": predicted_val
            })
        elif not is_correct_prediction:
            false_positives += 1
            incorrect_predictions.add(cell)
            fp_examples.append({
                "cell": cell, "original": original_val, "truth": true_val, "predicted": predicted_val
            })
            
    actual_errors_in_noisy_set = actual_errors.intersection(noisy_cells)
    missed_or_wrongly_repaired_errors = actual_errors_in_noisy_set - correctly_predicted_errors
    false_negatives = len(missed_or_wrongly_repaired_errors)
    for cell in missed_or_wrongly_repaired_errors:
         original_val = originals[cell]["original"]
         true_val = ground_truth[cell]
         predicted_val = predictions.get(cell)
         if predicted_val is None:
              if not any(fn['cell'] == cell for fn in fn_examples):
                   fn_examples.append({
                       "cell": cell, "original": original_val, "truth": true_val,
                       "predicted": "NO_PREDICTION", "reason": "Error in noisy set, but no prediction made"
                   })
         elif predicted_val != true_val:
             if not any(fn['cell'] == cell for fn in fn_examples):
                 fn_examples.append({
                       "cell": cell, "original": original_val, "truth": true_val,
                       "predicted": predicted_val, "reason": "Error in noisy set, incorrect prediction made"
                   })

    logging.info(f"Evaluation Summary:")
    logging.info(f"  Noisy cells evaluated (where prediction exists): {predictions_made_count}")
    logging.info(f"  True Positives (Correct Repairs of Errors): {true_positives}")
    logging.info(f"  False Positives (Incorrect Predictions): {false_positives}")
    logging.info(f"  False Negatives (Actual Errors in Noisy Set Not Correctly Repaired): {false_negatives}")
    logging.info(f"  Total Actual Errors in Noisy Cells: {len(actual_errors_in_noisy_set)}")

    if tp_examples:
        logging.info("\n--- Examples: True Positives (Correct Repairs) ---")
        for i, ex in enumerate(tp_examples):
            logging.info(f"  {ex['cell']}: Original='{ex['original']}', Predicted='{ex['predicted']}' (Correct)")
            if i >= 4: break
    if fp_examples:
        logging.info("\n--- Examples: False Positives (Incorrect Repairs/Changes) ---")
        for i, ex in enumerate(fp_examples):
            logging.info(f"  {ex['cell']}: Original='{ex['original']}', Truth='{ex['truth']}', Predicted='{ex['predicted']}' (Incorrect)")
            if i >= 9: break
    if fn_examples:
        logging.info("\n--- Examples: False Negatives (Missed/Incorrect Repairs of Errors) ---")
        for i, ex in enumerate(fn_examples):
             logging.info(f"  {ex['cell']}: Original='{ex['original']}', Truth='{ex['truth']}', Predicted='{ex['predicted']}' ({ex.get('reason', 'Missed')})")
             if i >= 9: break

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall_noisy = true_positives / len(actual_errors_in_noisy_set) if len(actual_errors_in_noisy_set) > 0 else 0
    f1_noisy = 2 * (precision * recall_noisy) / (precision + recall_noisy) if (precision + recall_noisy) > 0 else 0

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (vs noisy errors):  {recall_noisy:.4f}")
    print(f"F1-Score (vs noisy errors):{f1_noisy:.4f}")

    return {"precision": precision, "recall_noisy": recall_noisy, "f1_score_noisy": f1_noisy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HoloClean Repair Results")
    parser.add_argument('--pred_file', type=str, default='inference_marginals.pkl', help='Path to the prediction marginals pickle file')
    parser.add_argument('--truth_file', type=str, default='hospital_100_clean.csv', help='Path to the ground truth CSV file')

    args = parser.parse_args()
    if 'NULL_REPR_PLACEHOLDER' not in dir(config):
         logging.error("NULL_REPR_PLACEHOLDER not found in config.py. Please define it.")
         sys.exit(1)
    if not os.path.exists(args.pred_file):
         logging.error(f"Prediction file not found: {args.pred_file}")
         sys.exit(1)
    if not os.path.exists(args.truth_file):
         logging.error(f"Ground truth file not found: {args.truth_file}")
         sys.exit(1)

    db_connection = None
    try:
        logging.info("Connecting to database...")
        db_connection = psycopg2.connect(**config.DB_SETTINGS)
        logging.info("Connection successful.")

        ground_truth_data = load_ground_truth(args.truth_file)
        prediction_data = load_predictions(args.pred_file)
        original_data = get_original_data(db_connection)

        if ground_truth_data and prediction_data and original_data:
            evaluate_repairs(prediction_data, ground_truth_data, original_data)
        else:
            logging.error("Evaluation aborted due to errors loading data.")

    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}", exc_info=True)
    finally:
        if db_connection:
            db_connection.close()
            logging.info("Database connection closed.")