# File: run_pruning.py
# Orchestrates the execution of the domain pruning step.

import psycopg2
import psycopg2.extras
import sys
import argparse
from config import DB_SETTINGS
from compiler.pruning import DomainPruner

def main():
    parser = argparse.ArgumentParser(description="Run HoloClean Domain Pruning (Algorithm 2)")
    parser.add_argument(
        '--tau',
        type=float,
        default=0.5, # Default threshold suggested by blueprint/paper
        help='Conditional probability threshold (tau) for co-occurrence pruning.'
    )
    args = parser.parse_args()

    if not 0.0 <= args.tau <= 1.0:
        print("Error: Tau threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    conn = None
    try:
        print(f"Connecting to database for domain pruning (tau={args.tau})...")
        conn = psycopg2.connect(**DB_SETTINGS)
        # Import extras for execute_batch
        psycopg2.extras.register_uuid() # Needed for execute_batch potentially

        print("Connection successful.")

        # --- Instantiate and Run Pruner ---
        pruner = DomainPruner(conn)
        pruner.run()

        print("\n--- Domain Pruning Phase Complete ---")

    except psycopg2.Error as db_err:
        print(f"Database error during pruning: {db_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during pruning: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()