# File: run_inference.py
# Orchestrates model building, training (fitting), and inference.

import psycopg2
import sys
import argparse
import pickle # To save results
from config import DB_SETTINGS
from model.factor_graph import HoloCleanFactorGraph
import psycopg2.extras # Needed for execute_batch if used internally

def main():
    parser = argparse.ArgumentParser(description="Run HoloClean Learning and Inference")
    parser.add_argument('--samples', type=int, default=1000, help='Number of Gibbs samples.')
    parser.add_argument('--burnin', type=int, default=200, help='Number of Gibbs burn-in samples.')
    parser.add_argument('--learniter', type=int, default=100, help='Max iterations for weight learning.')
    parser.add_argument('--outfile', type=str, default='inference_marginals.pkl', help='File to save the resulting marginals.')
    args = parser.parse_args()

    if args.samples <= 0 or args.burnin < 0 or args.learniter <= 0:
        print("Error: samples, burnin, and learniter must be positive integers.", file=sys.stderr)
        sys.exit(1)

    conn = None
    try:
        print("Connecting to database for inference...")
        conn = psycopg2.connect(**DB_SETTINGS)
        psycopg2.extras.register_uuid()
        print("Connection successful.")

        # --- Instantiate and Run Model ---
        model = HoloCleanFactorGraph(conn)
        marginals = model.run_pipeline(
            n_samples=args.samples,
            n_burn_in=args.burnin,
            learn_iter=args.learniter
        )

        if marginals is not None:
            print(f"\n--- Inference Complete ---")
            # Save marginals to a file
            try:
                with open(args.outfile, 'wb') as f:
                    pickle.dump(marginals, f)
                print(f"Marginal probabilities saved to {args.outfile}")

                # Print some example marginals
                print("\nExample marginals:")
                count = 0
                for (tid, attr), probs in marginals.items():
                     print(f"Cell ({tid}, {attr}):")
                     sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                     for val, prob in sorted_probs:
                         print(f"  '{val}': {prob:.4f}")
                     count += 1
                     if count >= 5: # Print for first 5 noisy cells
                         break
            except Exception as e:
                print(f"Error saving or printing marginals: {e}")
        else:
             print("Inference pipeline failed.")


    except psycopg2.Error as db_err:
        print(f"Database error during inference pipeline: {db_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during inference pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()