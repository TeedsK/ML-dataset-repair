import pandas as pd
import psycopg2
import sys
import io
from config import DB_SETTINGS

#Reads data from a CSV file and ingests it into the 'cells' table.
def ingest_data(csv_filepath, db_conn, row_limit=None):
    
    print(f"starting ingestion from: {csv_filepath}")
    if row_limit:
        print(f"limiting set: {row_limit} rows.")

    try:
        df = pd.read_csv(
            csv_filepath,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
            nrows=row_limit
        )
        print(f"read {len(df)} rows and {len(df.columns)} columns.")

        df['tid'] = range(1, len(df) + 1)

        #handles missing/empty values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        df.replace('empty', None, inplace=True)
        df.replace('', None, inplace=True)

        cells_df = pd.melt(df,
                           id_vars=['tid'],
                           var_name='attr',
                           value_name='val')

        cells_df['tid'] = cells_df['tid'].astype(int)
        cells_df['attr'] = cells_df['attr'].astype(str)
        cells_df['val'] = cells_df['val'].astype(str)
        cells_df['val'] = cells_df['val'].replace({'nan': None, 'None': None})
        cells_df['is_noisy'] = False
        cells_df = cells_df[['tid', 'attr', 'val', 'is_noisy']]
        
        cursor = db_conn.cursor()

        print("clearing any existing data from tables...")
        cursor.execute("TRUNCATE TABLE violations, noisy_cells, domains, features, cells RESTART IDENTITY CASCADE;")
        print("cleared.")

        buffer = io.StringIO()
        cells_df.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N', quoting=3) 
        buffer.seek(0)
        try:
            cursor.copy_expert(f"COPY cells (tid, attr, val, is_noisy) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '\\N', QUOTE E'\\x01')", buffer)
            db_conn.commit()
            print(f"successfully inserted {len(cells_df)} rows into the 'cells' table.")
        except Exception as e:
            db_conn.rollback()
            print("Error while copying")
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
             except psycopg2.InterfaceError:
                 pass
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main(csv_path='hospital_100.csv', limit=None):
    """Main function to connect to DB and run ingestion."""
    conn = None
    try:
        print("connecting to the Database...")
        conn = psycopg2.connect(**DB_SETTINGS)
        print("Database connection successful.")
        print("Using CSV: ", csv_path)
        ingest_data(csv_path, conn, row_limit=limit)
    except psycopg2.Error as db_err:
        print(f"Database error: {db_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Functions to put hospital data into HoloClean DB")
    parser.add_argument('--file', type=str, default='hospital_100.csv', help='the path to the input CSV file')
    parser.add_argument('--limit', type=int, default=None, help='limits the amount of rows inputted')
    args = parser.parse_args()
    main(csv_path=args.file, limit=args.limit)