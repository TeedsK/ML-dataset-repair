# File: detectors/base_detector.py
# Defines a base class for error detectors.

from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """Abstract base class for error detection methods."""

    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.detector_name = self.__class__.__name__

    @abstractmethod
    def detect_errors(self):
        """
        Abstract method to run the detection logic.
        Should populate the 'violations' and/or 'noisy_cells' tables.
        Returns the number of errors/violations found.
        """
        pass

    def _add_noisy_cells(self, noisy_cells_list):
        """
        Helper method to insert identified noisy cells into the noisy_cells table.

        Args:
            noisy_cells_list (list): A list of tuples [(tid, attr), ...].
        """
        if not noisy_cells_list:
            return 0

        added_count = 0
        sql = """
            INSERT INTO noisy_cells (tid, attr, detection_method)
            VALUES (%s, %s, %s)
            ON CONFLICT (tid, attr) DO NOTHING;
        """
        try:
            with self.db_conn.cursor() as cur:
                # Use execute_batch for potential efficiency, or loop for simplicity/error handling
                for tid, attr in noisy_cells_list:
                    try:
                        cur.execute(sql, (tid, attr, self.detector_name))
                        added_count += cur.rowcount # Adds 1 if inserted, 0 if conflict
                    except Exception as e:
                        print(f"Error inserting noisy cell ({tid}, {attr}): {e}")
                        self.db_conn.rollback() # Rollback individual error if needed
                        # Decide whether to continue or raise
                self.db_conn.commit()
            print(f"[{self.detector_name}] Added {added_count} unique noisy cells.")
            return added_count
        except Exception as e:
            self.db_conn.rollback()
            print(f"[{self.detector_name}] Error batch inserting noisy cells: {e}")
            raise