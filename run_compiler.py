# File: run_compiler.py
# Orchestrates the execution of the feature compilation step.

import psycopg2
import sys
import argparse
from config import DB_SETTINGS
from compiler.compile import FeatureCompiler
import psycopg2.extras # Make sure extras is imported if using execute_batch

def main():
    parser = argparse.ArgumentParser(description="Run HoloClean Feature Compiler")
    parser.add_argument(
        '--relax-constraints',
        action='store_true', # Flag defaults to False if not present
        help='Use relaxed denial constraints (generate features) instead of hard factors.'
    )
    parser.add_argument(
        '--no-relax-constraints',
        action='store_false',
        dest='relax_constraints', # Explicitly set to false
        help='Use hard denial constraints (requires factor table implementation).'
    )
    # Set default behavior for the flag
    parser.set_defaults(relax_constraints=True)

    args = parser.parse_args()

    conn = None
    try:
        print(f"Connecting to database for feature compilation (Relax Constraints = {args.relax_constraints})...")
        conn = psycopg2.connect(**DB_SETTINGS)
        psycopg2.extras.register_uuid() # Needed for execute_batch
        print("Connection successful.")

        # --- Instantiate and Run Compiler ---
        compiler = FeatureCompiler(conn, relax_constraints=args.relax_constraints)
        compiler.compile_all()

        print("\n--- Feature Compilation Phase Complete ---")

    except psycopg2.Error as db_err:
        print(f"Database error during compilation: {db_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during compilation: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()