import logging
import psycopg2
from config import DB_SETTINGS
from detectors.constraints import ConstraintViolationDetector
from detectors.outliers import StatisticalOutlierDetector
import sys

# COMMENT OR UNCOMMENT THIS IF YOU WANT TO RECEIVE LOGS LIKE DEBUGS OR INFO
logging.basicConfig(
    level=logging.INFO, 
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

        constraint_detector = ConstraintViolationDetector(conn, "hospital_constraints.txt")

        outlier_detector = StatisticalOutlierDetector(conn, attributes=['Score'], method='zscore', threshold=3.0)

        print("\n--- Running Constraint Violation Detector ---")
        constraint_detector.run()

        print("\n--- Running Statistical Outlier Detector ---")
        outlier_detector.detect_errors() 

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
    run_all_detectors()