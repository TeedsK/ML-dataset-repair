# File: run_detectors.py
# Orchestrates the execution of error detection modules.

import logging
import psycopg2
from config import DB_SETTINGS
from detectors.constraints import ConstraintViolationDetector
from detectors.duplicates import DuplicateDetector
from detectors.outliers import StatisticalOutlierDetector
import sys

logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] %(levelname)s: %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_all_detectors():
    """Connects to the DB and runs all configured detectors."""
    conn = None
    try:
        print("Connecting to database for error detection...")
        conn = psycopg2.connect(**DB_SETTINGS)
        print("Connection successful.")

        # --- Instantiate Detectors ---
        # 1. Constraint Detector
        constraint_detector = ConstraintViolationDetector(conn, "hospital_constraints.txt")

        # 2. Duplicate Detector (Optional, using placeholder)
        # duplicate_detector = DuplicateDetector(conn, attribute='HospitalName')

        # 3. Outlier Detector (Optional, basic Z-score implemented)
        # Only run on attributes likely to be numeric and where outliers matter.
        # Need to be careful with 'Score' and 'Sample' as they have '%' and ' patients'
        # Requires cleaning *before* outlier detection. For now, let's skip them or
        # implement cleaning within the detector if needed.
        # Example: outlier_detector = StatisticalOutlierDetector(conn, attributes=['SomeNumericColumn'])
        # For demonstration, let's try Score but expect issues without cleaning:
        outlier_detector = StatisticalOutlierDetector(conn, attributes=['Score'], method='zscore', threshold=3.0)


        # --- Run Detectors ---
        print("\n--- Running Constraint Violation Detector ---")
        constraint_detector.run()

        # print("\n--- Running Duplicate Detector ---")
        # duplicate_detector.detect_errors() # Uncomment when implemented

        print("\n--- Running Statistical Outlier Detector ---")
        outlier_detector.detect_errors() # Uncomment and configure when ready

        print("\n--- Error Detection Phase Complete ---")

    except psycopg2.Error as db_err:
        print(f"Database error during detection: {db_err}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during detection: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    # Ensure hospital_constraints.txt is available
    run_all_detectors()