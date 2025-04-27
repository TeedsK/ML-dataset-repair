# File: ingest.py
# Reads the input CSV data and populates the 'cells' table in PostgreSQL.

import pandas as pd
import psycopg2
import sys
import io
from config import DB_SETTINGS # Assuming configuration is stored separately

def ingest_data(csv_filepath, db_conn):
    """
    Reads data from a CSV file and ingests it into the 'cells' table.

    Args:
        csv_filepath (str): Path to the input CSV file (e.g., 'hospital.csv').
        db_conn: Active psycopg2 database connection object.
    """
    print(f"Starting ingestion from: {csv_filepath}")
    try:
        # Read CSV - Be mindful of potential encoding issues and large files
        # Use low_memory=False to prevent dtype guessing issues with mixed types
        df = pd.read_csv(csv_filepath, dtype=str, keep_default_na=False, low_memory=False)
        print(f"Read {len(df)} rows and {len(df.columns)} columns.")

        # --- Data Preprocessing & Transformation ---
        # 1. Assign Stable Tuple ID (tid)
        # We use the 1-based index of the row as the tid.
        df['tid'] = range(1, len(df) + 1)

        # 2. Handle missing/empty values - replace 'empty' string with None (NULL in DB)
        # Also strip leading/trailing whitespace which can cause subtle errors
        for col in df.columns:
            if df[col].dtype == 'object': # Apply only to string-like columns
                df[col] = df[col].str.strip()
        df.replace('empty', None, inplace=True)
        df.replace('', None, inplace=True) # Handle empty strings as well if desired

        # 3. Transform DataFrame from wide to long format for 'cells' table
        # Melt the DataFrame: tid | attribute | value
        cells_df = pd.melt(df,
                           id_vars=['tid'],
                           var_name='attr',
                           value_name='val')

        # Remove rows where 'val' became NaN/None *after* melt if original wasn't truly missing
        # (though keep_default_na=False should prevent this unless NaNs were already present)
        # Ensure correct dtypes before insertion
        cells_df['tid'] = cells_df['tid'].astype(int)
        cells_df['attr'] = cells_df['attr'].astype(str)
        cells_df['val'] = cells_df['val'].astype(str).replace('nan', None) # Replace pandas 'nan' string if it occurs
        cells_df['is_noisy'] = False # Default value

        # Reorder columns to match table definition
        cells_df = cells_df[['tid', 'attr', 'val', 'is_noisy']]
        print(f"Transformed data into {len(cells_df)} cell entries.")

        # --- Database Insertion ---
        cursor = db_conn.cursor()

        # Clear existing data from cells (and dependent tables due to CASCADE)
        print("Clearing existing data from 'cells' table...")
        cursor.execute("DELETE FROM cells;")
        print("'cells' table cleared.")

        # Use psycopg2's copy_from for efficient bulk insertion
        print(f"Starting bulk insert into 'cells' table...")
        buffer = io.StringIO()
        cells_df.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N') # Use tab separation, handle NULLs
        buffer.seek(0)

        try:
            # COPY command expects columns in order
            cursor.copy_expert(f"COPY cells (tid, attr, val, is_noisy) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '\\N')", buffer)
            db_conn.commit()
            print(f"Successfully inserted {len(cells_df)} rows into 'cells' table.")
        except Exception as e:
            db_conn.rollback()
            print(f"Error during bulk insert: {e}", file=sys.stderr)
            raise # Re-raise the exception

        cursor.close()

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {csv_filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during ingestion: {e}", file=sys.stderr)
        # Rollback in case of error during processing before DB commit
        if 'db_conn' in locals() and not db_conn.closed:
             db_conn.rollback()
        raise # Re-raise the exception


def main(csv_path='hospital.csv'):
    """Main function to connect to DB and run ingestion."""
    conn = None
    try:
        print("Connecting to the database...")
        conn = psycopg2.connect(**DB_SETTINGS)
        print("Database connection successful.")
        ingest_data(csv_path, conn)
    except psycopg2.Error as db_err:
        print(f"Database error: {db_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    # Assuming hospital.csv is in the same directory or a known path
    # You might want to use argparse for command-line arguments
    input_csv = 'hospital.csv' # Make sure this path is correct
    main(input_csv)