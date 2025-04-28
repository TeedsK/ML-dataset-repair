# File: ingest.py
# Reads the input CSV data and populates the 'cells' table in PostgreSQL.

import pandas as pd
import psycopg2
import sys
import io
from config import DB_SETTINGS # Assuming configuration is stored separately

def ingest_data(csv_filepath, db_conn, row_limit=None): # Add row_limit parameter
    """
    Reads data from a CSV file and ingests it into the 'cells' table.

    Args:
        csv_filepath (str): Path to the input CSV file (e.g., 'hospital.csv').
        db_conn: Active psycopg2 database connection object.
        row_limit (int, optional): Maximum number of rows to read from CSV. Defaults to None (read all).
    """
    print(f"Starting ingestion from: {csv_filepath}")
    if row_limit:
        print(f"Limiting ingestion to the first {row_limit} rows.")
    try:
        # Read CSV - Add nrows parameter here
        df = pd.read_csv(
            csv_filepath,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
            nrows=row_limit # <--- ADD THIS LINE
        )
        print(f"Read {len(df)} rows and {len(df.columns)} columns.")

        # --- Data Preprocessing & Transformation ---
        # 1. Assign Stable Tuple ID (tid)
        df['tid'] = range(1, len(df) + 1)

        # 2. Handle missing/empty values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        df.replace('empty', None, inplace=True)
        df.replace('', None, inplace=True)

        # 3. Transform DataFrame from wide to long format
        cells_df = pd.melt(df,
                           id_vars=['tid'],
                           var_name='attr',
                           value_name='val')

        # Ensure correct dtypes and handle potential 'nan' strings
        cells_df['tid'] = cells_df['tid'].astype(int)
        cells_df['attr'] = cells_df['attr'].astype(str)
        # Ensure 'val' is string, replace 'nan' string resulting from melt/type changes
        cells_df['val'] = cells_df['val'].astype(str)
        cells_df['val'] = cells_df['val'].replace({'nan': None, 'None': None}) # Handle 'nan' and 'None' strings
        cells_df['is_noisy'] = False # Default value

        # Reorder columns
        cells_df = cells_df[['tid', 'attr', 'val', 'is_noisy']]
        print(f"Transformed data into {len(cells_df)} cell entries.")

        # --- Database Insertion ---
        cursor = db_conn.cursor()

        # Clear existing data - IMPORTANT!
        # Need to clear all relevant tables because subsequent steps depend on 'cells'
        print("Clearing existing data from tables (cells, violations, noisy_cells, domains, features)...")
        # Clear in reverse order of dependency or rely on CASCADE if set up in schema.sql
        # Assuming CASCADE is set up correctly from 'cells' deletion in schema.sql
        cursor.execute("TRUNCATE TABLE violations, noisy_cells, domains, features, cells RESTART IDENTITY CASCADE;")
        # Alternatively, if no CASCADE:
        # cursor.execute("TRUNCATE TABLE violations RESTART IDENTITY;")
        # cursor.execute("TRUNCATE TABLE noisy_cells RESTART IDENTITY;")
        # cursor.execute("TRUNCATE TABLE domains RESTART IDENTITY;")
        # cursor.execute("TRUNCATE TABLE features RESTART IDENTITY;")
        # cursor.execute("TRUNCATE TABLE cells RESTART IDENTITY;")
        print("Tables cleared.")

        # Use psycopg2's copy_from for efficient bulk insertion
        print(f"Starting bulk insert into 'cells' table...")
        buffer = io.StringIO()
        # Use quote char to handle potential tabs within data if using tab delimiter
        cells_df.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N', quoting=3) # quoting=3 means csv.QUOTE_NONE, risky if delimiter in data
        buffer.seek(0)

        try:
            # COPY command expects columns in order
            cursor.copy_expert(f"COPY cells (tid, attr, val, is_noisy) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '\\N', QUOTE E'\\x01')", buffer) # Use an unlikely quote char
            db_conn.commit()
            print(f"Successfully inserted {len(cells_df)} rows into 'cells' table.")
        except Exception as e:
            db_conn.rollback()
            print(f"Error during bulk insert: {e}", file=sys.stderr)
            print("--- Start of failing data sample (first 5 rows) ---")
            buffer.seek(0)
            print(buffer.read(500)) # Print first 500 chars of buffer
            print("--- End of failing data sample ---")
            raise

        cursor.close()

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {csv_filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during ingestion: {e}", file=sys.stderr)
        if 'db_conn' in locals() and db_conn and not db_conn.closed:
             try:
                 db_conn.rollback()
             except psycopg2.InterfaceError: # Handle case where connection might be closed already
                 pass
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit on error


def main(csv_path='hospital.csv', limit=None): # Add limit parameter
    """Main function to connect to DB and run ingestion."""
    conn = None
    try:
        print("Connecting to the database...")
        conn = psycopg2.connect(**DB_SETTINGS)
        print("Database connection successful.")
        # Pass the limit to ingest_data
        ingest_data(csv_path, conn, row_limit=limit)
    except psycopg2.Error as db_err:
        print(f"Database error: {db_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Error already printed in ingest_data, just ensure exit
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest hospital data into HoloClean DB.")
    parser.add_argument('--file', type=str, default='hospital.csv', help='Path to the input CSV file.')
    parser.add_argument('--limit', type=int, default=None, help='Limit processing to the first N rows.')
    args = parser.parse_args()

    main(csv_path=args.file, limit=args.limit)