# File: compiler/pruning.py
# Implements candidate domain pruning (Algorithm 2).

import pandas as pd
import time
import psycopg2
import psycopg2.extras # Import extras
from collections import defaultdict
import itertools

class DomainPruner:
    """Calculates candidate domains for cells based on co-occurrence."""

    def __init__(self, db_conn, tau=0.5):
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
        self.all_cells_df = None
        self.noisy_cells = set() # Set of (tid, attr) tuples

    def _fetch_data(self):
        """Fetches all cells and noisy cells information from the database."""
        print("[Pruner] Fetching data from database...")
        start_time = time.time()
        try:
            # Fetch all cells data - Important: Keep index for later lookup
            self.all_cells_df = pd.read_sql("SELECT tid, attr, val FROM cells WHERE val IS NOT NULL", self.db_conn, index_col=['tid', 'attr'])
            print(f"Fetched {len(self.all_cells_df)} non-null cell entries.")

            # Fetch noisy cells identifiers
            noisy_df = pd.read_sql("SELECT tid, attr FROM noisy_cells", self.db_conn)
            self.noisy_cells = set(tuple(row) for row in noisy_df.to_numpy())
            print(f"Fetched {len(self.noisy_cells)} unique noisy cell identifiers.")

            # Determine global domain for each attribute using the original DataFrame before setting index
            all_cells_for_domains = pd.read_sql("SELECT attr, val FROM cells WHERE val IS NOT NULL", self.db_conn)
            self.all_attr_domains = all_cells_for_domains.groupby('attr')['val'].unique().apply(set).to_dict()
            # Ensure global domain includes original values even if rare
            for idx, row in self.all_cells_df.iterrows():
                tid, attr = idx
                val = row['val']
                if val is not None:
                    if attr not in self.all_attr_domains:
                        self.all_attr_domains[attr] = set()
                    self.all_attr_domains[attr].add(val)

            print(f"Determined global domains for {len(self.all_attr_domains)} attributes.")

            fetch_time = time.time() - start_time
            print(f"[Pruner] Data fetching completed in {fetch_time:.2f} seconds.")
            return True
        except Exception as e:
            print(f"[Pruner] Error fetching data: {e}")
            return False

    def _calculate_counts(self):
        """Calculates single and pairwise co-occurrence counts from all_cells_df."""
        if self.all_cells_df is None:
            print("[Pruner] Error: Cells data not fetched.")
            return

        print("[Pruner] Calculating co-occurrence counts...")
        start_time = time.time()
        # Reset index to iterate over tuples easily
        cells_to_iterate = self.all_cells_df.reset_index()
        # Group by tid first for efficient iteration over tuples
        tuple_groups = cells_to_iterate.groupby('tid')

        for tid, group in tuple_groups:
            # Get attributes and values for the current tuple
            attrs = group['attr'].tolist()
            vals = group['val'].tolist()
            # Create pairs of (attr, val) for the tuple
            tuple_cells = list(zip(attrs, vals))

            # Increment single counts
            for cell in tuple_cells:
                self.single_counts[cell] += 1

            # Increment pair counts for all unique pairs within the tuple
            for cell1, cell2 in itertools.combinations(tuple_cells, 2):
                # Store pairs in a canonical order (e.g., attr1 < attr2) to avoid duplication
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
        For noisy cells, uses co-occurrence pruning (Alg 2).
        For non-noisy cells, the domain is just the original value.
        """
        if not self._fetch_data():
            return
        self._calculate_counts()

        print(f"[Pruner] Starting domain calculation for ALL {len(self.all_cells_df)} cells (Tau={self.tau} for noisy)...")
        start_time = time.time()
        domains_to_insert = [] # List to store (tid, attr, candidate_val) tuples

        # Keep track of processed cells for progress update
        processed_count = 0
        total_cells_to_process = len(self.all_cells_df)

        # Create a temporary df with index reset for tuple grouping
        cells_for_tuples = self.all_cells_df.reset_index()
        # Group by TID to get data for each tuple efficiently
        tuple_groups = cells_for_tuples.groupby('tid')
        tuple_data_cache = {tid: group.set_index('attr')['val'].to_dict() for tid, group in tuple_groups}
        print(f"Built cache for {len(tuple_data_cache)} tuples.")


        # Iterate through all unique cells (tid, attr) from the fetched data
        for cell_index, cell_row in self.all_cells_df.iterrows():
            tid, attr = cell_index
            original_value = cell_row['val']

            processed_count += 1
            if processed_count % 5000 == 0 or processed_count == total_cells_to_process :
                 print(f"Processed {processed_count}/{total_cells_to_process} cells...")

            candidate_domain = set()

            # --- Always add the original value if it exists ---
            if original_value is not None:
                candidate_domain.add(original_value)

            # --- If the cell is noisy, apply pruning based on co-occurrence ---
            if (tid, attr) in self.noisy_cells:
                # Get the global domain for the attribute being pruned
                global_domain_for_attr = self.all_attr_domains.get(attr, set())

                # Get the data for the current tuple from the cache
                current_tuple_data = tuple_data_cache.get(tid)
                if not current_tuple_data:
                    continue # Should not happen if cell_index came from all_cells_df

                # Iterate through other cells (c') in the same tuple
                for other_attr, other_val in current_tuple_data.items():
                    if other_attr == attr or other_val is None: # Skip self and nulls
                        continue

                    # Single count for the evidence P(c')
                    denominator = self.single_counts.get((other_attr, other_val), 0)
                    if denominator == 0:
                        continue

                    # Check each potential candidate value (v) from the global domain
                    for candidate_val in global_domain_for_attr:
                         if candidate_val is None: continue # Skip None candidates
                         # Pair count for the co-occurrence P(v, c')
                         # Ensure canonical order for lookup
                         if attr < other_attr:
                             numerator = self.pair_counts.get((attr, candidate_val, other_attr, other_val), 0)
                         else:
                             numerator = self.pair_counts.get((other_attr, other_val, attr, candidate_val), 0)

                         # Calculate conditional probability P(v | c')
                         cond_prob = numerator / denominator
                         if cond_prob >= self.tau:
                             candidate_domain.add(candidate_val)

            # --- Add results to the insertion list ---
            # The candidate domain now contains original_value + pruned candidates (if noisy)
            # or just original_value (if not noisy)
            for cand_val in candidate_domain:
                 # Ensure candidate value is not None before adding
                 if cand_val is not None:
                     domains_to_insert.append((tid, attr, str(cand_val))) # Ensure value is string

        pruning_time = time.time() - start_time
        print(f"[Pruner] Domain calculation finished in {pruning_time:.2f} seconds.")
        # Calculate unique domains to report accurate count
        unique_domains = set(domains_to_insert)
        print(f"Generated {len(unique_domains)} unique candidate domain entries.")

        # --- Database Insertion ---
        if unique_domains: # Insert unique domains
            print("[Pruner] Inserting candidate domains into database...")
            insert_start_time = time.time()
            # Use list for insertion as executemany expects a sequence
            unique_domains_list = list(unique_domains)
            sql_insert = "INSERT INTO domains (tid, attr, candidate_val) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
            try:
                with self.db_conn.cursor() as cur:
                    # Clear previous domains first
                    print("Clearing previous domains...")
                    cur.execute("DELETE FROM domains;")
                    # Use execute_batch for potentially faster batch insertion
                    psycopg2.extras.execute_batch(cur, sql_insert, unique_domains_list, page_size=1000)
                self.db_conn.commit()
                insert_time = time.time() - insert_start_time
                # Report based on the number of unique domains attempted insertion
                print(f"[Pruner] Successfully attempted insertion of {len(unique_domains_list)} domains in {insert_time:.2f} seconds.")
            except Exception as e:
                self.db_conn.rollback()
                print(f"[Pruner] Error inserting domains into database: {e}")
                raise # Re-raise after logging and rollback
        else:
             print("[Pruner] No candidate domains generated to insert.")