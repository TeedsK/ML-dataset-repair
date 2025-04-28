# File: detectors/base_detector.py (Modified)
# Defines a base class for error detectors.

from abc import ABC, abstractmethod
import psycopg2 # Import psycopg2 if not already imported

class BaseDetector(ABC):
    """Abstract base class for error detection methods."""

    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.detector_name = self.__class__.__name__

    @abstractmethod
    def detect_errors(self):
        """
        Abstract method to run the detection logic.
        Should populate the 'violations' and/or 'noisy_cells' tables,
        and update the 'cells.is_noisy' flag.
        Returns the number of errors/violations found.
        """
        pass

    def _add_noisy_cells(self, noisy_cells_list):
        """
        Helper method to insert identified noisy cells into the noisy_cells table
        AND update the is_noisy flag in the main cells table.

        Args:
            noisy_cells_list (list): A list of tuples [(tid, attr), ...].
        """
        if not noisy_cells_list:
            return 0

        added_count = 0
        updated_flag_count = 0

        sql_insert_noisy = """
            INSERT INTO noisy_cells (tid, attr, detection_method)
            VALUES (%s, %s, %s)
            ON CONFLICT (tid, attr) DO NOTHING;
        """
        # *** ADDED: SQL to update the flag in the cells table ***
        sql_update_flag = """
            UPDATE cells SET is_noisy = TRUE WHERE tid = %s AND attr = %s;
        """

        # Process in chunks to avoid holding locks for too long? For now, loop.
        processed_cells = set() # Keep track of processed cells to avoid duplicate updates

        try:
            with self.db_conn.cursor() as cur:
                for tid, attr in noisy_cells_list:
                    cell_key = (tid, attr)
                    if cell_key in processed_cells:
                        continue # Avoid redundant operations for the same cell

                    # 1. Insert into noisy_cells table
                    try:
                        cur.execute(sql_insert_noisy, (tid, attr, self.detector_name))
                        added_count += cur.rowcount # Counts newly inserted rows
                    except Exception as e:
                        print(f"Error inserting noisy cell ({tid}, {attr}): {e}")
                        self.db_conn.rollback() # Rollback specific error if needed
                        continue # Skip update if insert failed

                    # 2. Update the is_noisy flag in cells table
                    try:
                        cur.execute(sql_update_flag, (tid, attr))
                        # Don't rely solely on add_count, track separately if needed
                        # updated_flag_count += cur.rowcount # rows affected by update
                    except Exception as e:
                        print(f"Error updating is_noisy flag for cell ({tid}, {attr}): {e}")
                        self.db_conn.rollback()
                        # Consider how to handle: maybe remove from noisy_cells if flag update fails?

                    processed_cells.add(cell_key)

                self.db_conn.commit() # Commit all changes together

            # Get the actual count of updated flags after commit (optional)
            with self.db_conn.cursor() as cur:
                 cur.execute("SELECT COUNT(*) FROM cells WHERE is_noisy = TRUE;")
                 total_noisy_flags = cur.fetchone()[0]

            print(f"[{self.detector_name}] Added {added_count} unique entries to noisy_cells table.")
            print(f"[{self.detector_name}] Total cells marked as is_noisy=TRUE in 'cells' table: {total_noisy_flags}") # Reflects total including previous runs if not reset

            # Return the number of newly identified noisy cells
            return added_count # Or perhaps len(processed_cells)?

        except Exception as e:
            self.db_conn.rollback()
            print(f"[{self.detector_name}] Error processing noisy cells batch: {e}")
            raise