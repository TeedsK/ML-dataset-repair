# File: compiler/pruning.py (Corrected)
# Implements candidate domain pruning (Algorithm 2).
# Ensures original value (including None representation) is always included.

import pandas as pd
import time
import psycopg2
import psycopg2.extras # Import extras
from collections import defaultdict
import itertools
import sys # For progress bar writing

# Define a consistent placeholder for None/NULL values
# Ensure this is used consistently in TensorBuilder as well
NULL_REPR_PLACEHOLDER = "__NULL__"

class DomainPruner:
    """Calculates candidate domains for cells based on co-occurrence."""

    def __init__(self, db_conn, tau=0.01):
        """
        Initializes the DomainPruner.

        Args:
            db_conn: Active psycopg2 database connection.
            tau (float): The minimum conditional probability threshold for co-occurrence.
        """
        self.db_conn = db_conn
        self.tau = tau
        self.pair_counts = defaultdict(int)
        self.single_counts = defaultdict(int)
        self.all_attr_domains = {}
        # Fetch all cells including NULLs for original value reference, but only non-null for counts/domains
        self.all_cells_df_non_null = None # For counts and global domains
        self.all_cells_df_with_null = None # For iteration and original value lookup
        self.noisy_cells = set() # Set of (tid, attr) tuples

    def _fetch_data(self):
        """Fetches all cells and noisy cells information from the database."""
        print("[Pruner] Fetching data from database...")
        start_time = time.time()
        try:
            # Fetch ALL cells, including NULLs, keeping tid/attr index for original val lookup
            # Order by tid, attr to ensure consistency if needed later
            self.all_cells_df_with_null = pd.read_sql("SELECT tid, attr, val FROM cells ORDER BY tid, attr", self.db_conn, index_col=['tid', 'attr'])
            print(f"Fetched {len(self.all_cells_df_with_null)} total cell entries (including NULLs).")

            # Fetch only non-NULL cells for co-occurrence counts and global domain determination
            self.all_cells_df_non_null = pd.read_sql("SELECT tid, attr, val FROM cells WHERE val IS NOT NULL ORDER BY tid, attr", self.db_conn, index_col=['tid', 'attr'])
            print(f"Fetched {len(self.all_cells_df_non_null)} non-null cell entries (for counts/domains).")

            # Fetch noisy cells identifiers
            noisy_df = pd.read_sql("SELECT tid, attr FROM noisy_cells", self.db_conn)
            self.noisy_cells = set(tuple(row) for row in noisy_df.to_numpy())
            print(f"Fetched {len(self.noisy_cells)} unique noisy cell identifiers.")

            # Determine global domain for each attribute using ONLY non-null values
            if not self.all_cells_df_non_null.empty:
                 # Reset index temporarily for groupby
                 non_null_reset = self.all_cells_df_non_null.reset_index()
                 self.all_attr_domains = non_null_reset.groupby('attr')['val'].unique().apply(set).to_dict()
                 # Ensure global domain includes original non-null values even if rare
                 # (This loop might be redundant if groupby already captures all unique non-nulls)
                 # for idx, row in self.all_cells_df_non_null.iterrows():
                 #     tid, attr = idx
                 #     val = row['val']
                 #     if val is not None: # Should always be true here
                 #         if attr not in self.all_attr_domains: self.all_attr_domains[attr] = set()
                 #         self.all_attr_domains[attr].add(str(val)) # Store as string
            else:
                 self.all_attr_domains = {}
            print(f"Determined global domains (from non-null values) for {len(self.all_attr_domains)} attributes.")

            fetch_time = time.time() - start_time
            print(f"[Pruner] Data fetching completed in {fetch_time:.2f} seconds.")
            return True
        except Exception as e:
            print(f"[Pruner] Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _calculate_counts(self):
        """Calculates single and pairwise co-occurrence counts using only non-null cells."""
        if self.all_cells_df_non_null is None:
            print("[Pruner] Error: Non-null cells data not fetched.")
            return

        print("[Pruner] Calculating co-occurrence counts (using non-null values)...")
        start_time = time.time()
        self.single_counts.clear()
        self.pair_counts.clear()

        # Reset index to iterate over tuples easily
        cells_to_iterate = self.all_cells_df_non_null.reset_index()
        # Group by tid first for efficient iteration over tuples
        tuple_groups = cells_to_iterate.groupby('tid')

        for tid, group in tuple_groups:
            # Create pairs of (attr, val_str) for the tuple, ensuring strings
            tuple_cells = list(zip(group['attr'].tolist(), map(str, group['val'].tolist())))

            # Increment single counts
            for cell in tuple_cells:
                self.single_counts[cell] += 1

            # Increment pair counts for all unique pairs within the tuple
            for cell1, cell2 in itertools.combinations(tuple_cells, 2):
                # Store pairs in a canonical order (e.g., attr1 < attr2)
                if cell1[0] < cell2[0]:
                    self.pair_counts[(cell1[0], cell1[1], cell2[0], cell2[1])] += 1
                else:
                    self.pair_counts[(cell2[0], cell2[1], cell1[0], cell1[1])] += 1

        count_time = time.time() - start_time
        print(f"[Pruner] Co-occurrence counts calculated in {count_time:.2f} seconds.")
        print(f"Total single counts: {len(self.single_counts)}, Total pair counts: {len(self.pair_counts)}")


    def prune_domains(self):
        """
        Calculates candidate domains for ALL cells and populates the 'domains' table.
        Ensures the original value (including None representation) is always included.
        """
        if not self._fetch_data():
            return
        self._calculate_counts() # Calculate counts based on non-null data

        # Use the df *with* nulls for iteration to ensure all cells are processed
        if self.all_cells_df_with_null is None:
            print("[Pruner] Error: Full cell data (with NULLs) not available.")
            return

        print(f"[Pruner] Starting domain calculation for ALL {len(self.all_cells_df_with_null)} cells (Tau={self.tau} for noisy)...")
        start_time = time.time()
        domains_to_insert = [] # List to store (tid, attr, candidate_val_str) tuples

        processed_count = 0
        total_cells_to_process = len(self.all_cells_df_with_null)

        # Build tuple cache from *non-null* df for co-occurrence lookups
        cells_for_tuples = self.all_cells_df_non_null.reset_index()
        tuple_groups = cells_for_tuples.groupby('tid')
        tuple_data_cache = {tid: group.set_index('attr')['val'].apply(str).to_dict() for tid, group in tuple_groups}
        print(f"Built cache for {len(tuple_data_cache)} tuples (using non-null values for co-occurrence checks).")

        # Iterate through all cells using the df that includes nulls
        for cell_index, cell_row in self.all_cells_df_with_null.iterrows():
            tid, attr = cell_index
            original_value = cell_row['val'] # This can be None

            processed_count += 1
            if processed_count % 5000 == 0 or processed_count == total_cells_to_process :
                 # Use sys.stdout.write for cleaner progress updates
                 sys.stdout.write(f"\rProcessed {processed_count}/{total_cells_to_process} cells...")
                 sys.stdout.flush()

            # --- Step 1: Initialize domain with the original value representation ---
            candidate_domain = set()
            # Use the consistent placeholder if original_value is None
            original_val_repr = NULL_REPR_PLACEHOLDER if original_value is None else str(original_value)
            candidate_domain.add(original_val_repr)
            # --- End Step 1 ---

            # --- Step 2: If noisy, apply co-occurrence pruning ---
            if cell_index in self.noisy_cells: # Check using tuple (tid, attr)
                # Get the global domain (candidates derived from non-null values)
                # Represent candidates as strings, handle potential missing attrs
                global_domain_for_attr = set(map(str, self.all_attr_domains.get(attr, [])))

                # Get the non-null data for the current tuple from the cache for co-occurrence check
                current_tuple_non_null_data = tuple_data_cache.get(tid)
                if current_tuple_non_null_data: # Check if tuple has any non-null values
                    pruned_candidates = set()
                    for other_attr, other_val_str in current_tuple_non_null_data.items():
                        if other_attr == attr: continue # Skip self co-occurrence

                        # Denominator: P(c') based on counts from non-null data
                        denominator = self.single_counts.get((other_attr, other_val_str), 0)
                        if denominator == 0: continue

                        # Check each potential *string* candidate value (v) from the global domain
                        for cand_val_str in global_domain_for_attr:
                            # Skip adding the original value again if it comes up during pruning
                            if cand_val_str == original_val_repr: continue

                            # Numerator: P(v, c') based on counts from non-null data
                            # Ensure canonical order for lookup (using string representations)
                            if attr < other_attr:
                                numerator = self.pair_counts.get((attr, cand_val_str, other_attr, other_val_str), 0)
                            else:
                                numerator = self.pair_counts.get((other_attr, other_val_str, attr, cand_val_str), 0)

                            cond_prob = numerator / denominator # Denominator > 0 checked above
                            if cond_prob >= self.tau:
                                pruned_candidates.add(cand_val_str)

                    # Add the pruned candidates to the main set
                    candidate_domain.update(pruned_candidates)

            # --- Step 3: Add results to the insertion list ---
            # `candidate_domain` now contains original_val_repr + potentially pruned candidates (all as strings)
            for cand_val_str in candidate_domain:
                 # cand_val_str is already the correct string representation (or placeholder)
                 domains_to_insert.append((tid, attr, cand_val_str))

        print() # Newline after progress indicator
        pruning_time = time.time() - start_time
        print(f"[Pruner] Domain calculation finished in {pruning_time:.2f} seconds.")

        # Calculate unique domains before reporting count
        unique_domains = set(domains_to_insert)
        print(f"Generated {len(unique_domains)} unique candidate domain entries (original values always included).")

        # --- Database Insertion (remains the same) ---
        if unique_domains:
            print("[Pruner] Inserting candidate domains into database...")
            insert_start_time = time.time()
            unique_domains_list = list(unique_domains)
            sql_insert = "INSERT INTO domains (tid, attr, candidate_val) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
            try:
                with self.db_conn.cursor() as cur:
                    print("Clearing previous domains...")
                    cur.execute("DELETE FROM domains;")
                    psycopg2.extras.execute_batch(cur, sql_insert, unique_domains_list, page_size=1000)
                self.db_conn.commit()
                insert_time = time.time() - insert_start_time
                print(f"[Pruner] Successfully attempted insertion of {len(unique_domains_list)} domains in {insert_time:.2f} seconds.")
            except Exception as e:
                self.db_conn.rollback()
                print(f"[Pruner] Error inserting domains into database: {e}")
                raise
        else:
             print("[Pruner] No candidate domains generated to insert.")