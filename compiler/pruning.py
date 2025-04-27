# File: compiler/pruning.py
# Implements candidate domain pruning (Algorithm 2).

import pandas as pd
import time
import psycopg2
from collections import defaultdict
import itertools

class DomainPruner:
    """Calculates candidate domains for noisy cells based on co-occurrence."""

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
            # Fetch all cells data
            self.all_cells_df = pd.read_sql("SELECT tid, attr, val FROM cells WHERE val IS NOT NULL", self.db_conn)
            print(f"Fetched {len(self.all_cells_df)} non-null cell entries.")

            # Fetch noisy cells identifiers
            noisy_df = pd.read_sql("SELECT tid, attr FROM noisy_cells", self.db_conn)
            self.noisy_cells = set(tuple(row) for row in noisy_df.to_numpy())
            print(f"Fetched {len(self.noisy_cells)} unique noisy cell identifiers.")

            # Determine global domain for each attribute
            self.all_attr_domains = self.all_cells_df.groupby('attr')['val'].unique().apply(set).to_dict()
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
        # Group by tid first for efficient iteration over tuples
        tuple_groups = self.all_cells_df.groupby('tid')

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
        Performs domain pruning for all noisy cells and populates the 'domains' table.
        """
        if not self._fetch_data():
            return
        self._calculate_counts()

        print(f"[Pruner] Starting domain pruning for {len(self.noisy_cells)} noisy cells with tau={self.tau}...")
        start_time = time.time()
        domains_to_insert = [] # List to store (tid, attr, candidate_val) tuples

        # Get all data once, indexed by tid for quick tuple lookup
        self.all_cells_df.set_index('tid', inplace=True)

        processed_count = 0
        for tid, attr in self.noisy_cells:
            processed_count += 1
            if processed_count % 1000 == 0:
                 print(f"Processed {processed_count}/{len(self.noisy_cells)} noisy cells...")

            try:
                # Get the full tuple data for the current noisy cell's tid
                # Need to handle potential KeyError if tid isn't in the index (shouldn't happen)
                current_tuple_df = self.all_cells_df.loc[[tid]] # Use double brackets to keep DataFrame format
            except KeyError:
                # This might happen if a noisy cell tid doesn't exist in cells (data inconsistency)
                print(f"Warning: TID {tid} found in noisy_cells but not in cells table. Skipping.")
                continue

            # Get the original value of the noisy cell
            original_value_series = current_tuple_df[current_tuple_df['attr'] == attr]['val']
            original_value = original_value_series.iloc[0] if not original_value_series.empty else None

            # Initialize candidate domain - always include the original value if not null
            candidate_domain = set()
            if original_value is not None:
                candidate_domain.add(original_value)

            # Get the global domain for the attribute being pruned
            global_domain_for_attr = self.all_attr_domains.get(attr, set())
            if not global_domain_for_attr:
                # If attribute has no values in the dataset, skip (or handle differently)
                continue

            # Iterate through other cells (c') in the same tuple
            other_cells = current_tuple_df[current_tuple_df['attr'] != attr]
            for _, other_cell in other_cells.iterrows():
                other_attr = other_cell['attr']
                other_val = other_cell['val']
                # Single count for the evidence P(c')
                denominator = self.single_counts.get((other_attr, other_val), 0)
                if denominator == 0:
                    continue

                # Check each potential candidate value (v) from the global domain
                for candidate_val in global_domain_for_attr:
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

            # Add results to the insertion list
            for cand_val in candidate_domain:
                 # Ensure candidate value is not None before adding
                 if cand_val is not None:
                     domains_to_insert.append((tid, attr, str(cand_val))) # Ensure value is string

        pruning_time = time.time() - start_time
        print(f"[Pruner] Domain calculation finished in {pruning_time:.2f} seconds.")
        print(f"Generated {len(domains_to_insert)} candidate domain entries.")

        # --- Database Insertion ---
        if domains_to_insert:
            print("[Pruner] Inserting candidate domains into database...")
            insert_start_time = time.time()
            sql_insert = "INSERT INTO domains (tid, attr, candidate_val) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
            try:
                with self.db_conn.cursor() as cur:
                    # Clear previous domains first
                    print("Clearing previous domains...")
                    cur.execute("DELETE FROM domains;")
                    # Use executemany for potentially faster batch insertion
                    psycopg2.extras.execute_batch(cur, sql_insert, domains_to_insert, page_size=1000)
                self.db_conn.commit()
                insert_time = time.time() - insert_start_time
                print(f"[Pruner] Successfully inserted domains in {insert_time:.2f} seconds.")
            except Exception as e:
                self.db_conn.rollback()
                print(f"[Pruner] Error inserting domains into database: {e}")
        else:
             print("[Pruner] No candidate domains generated to insert.")