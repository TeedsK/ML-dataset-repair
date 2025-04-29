# File: compiler/pruning.py
# Calculates candidate domains for cells, potentially applying pruning based on co-occurrence.

import pandas as pd
import time
from math import log
from collections import Counter, defaultdict
import logging

import psycopg2

# Define a placeholder for NULL/None to be used in domains/features if needed
# Ensure this is consistent across compile.py and tensor_builder.py
NULL_REPR_PLACEHOLDER = "__NULL__"


class DomainPruner:
    """Calculates and prunes candidate domains for cells."""

    def __init__(self, db_conn, tau=0.01):
        self.db_conn = db_conn
        self.tau = tau
        # DataFrames for efficient access
        self.all_cells_df_with_null = None # Includes NULLs, indexed by (tid, attr)
        self.all_cells_df_non_null = None # Excludes NULLs, indexed by (tid, attr), for counts
        self.noisy_cells_set = set() # Set of (tid, attr) tuples marked as noisy
        self.global_attr_domains = defaultdict(Counter) # {attr: Counter(val: count)}
        self.cooccurrence_counts = defaultdict(lambda: defaultdict(Counter)) # {attr1: {attr2: Counter((val1, val2): count)}}
        self.single_counts = defaultdict(Counter) # {attr: Counter(val: count)}
        self.tuple_context_cache = {} # {tid: {attr: val_str}}

    def _fetch_data(self):
        """Fetches required data from the database."""
        logging.info("[Pruner] Fetching data from database...")
        fetch_start = time.time()
        try:
            # Fetch all cells, including those with NULL values, index by tid, attr
            self.all_cells_df_with_null = pd.read_sql("SELECT tid, attr, val FROM cells ORDER BY tid, attr", self.db_conn, index_col=['tid', 'attr'])
            logging.info(f"Fetched {len(self.all_cells_df_with_null)} total cell entries (including NULLs).")

            # Fetch non-null cells separately for calculating counts/domains
            self.all_cells_df_non_null = pd.read_sql("SELECT tid, attr, val FROM cells WHERE val IS NOT NULL ORDER BY tid, attr", self.db_conn, index_col=['tid', 'attr'])
            logging.info(f"Fetched {len(self.all_cells_df_non_null)} non-null cell entries (for counts/domains).")

            # Fetch noisy cells into a set for quick lookup
            noisy_df = pd.read_sql("SELECT tid, attr FROM noisy_cells", self.db_conn)
            self.noisy_cells_set = set(tuple(x) for x in noisy_df.to_numpy())
            logging.info(f"Fetched {len(self.noisy_cells_set)} unique noisy cell identifiers.")

            # Calculate global domains and single counts from non-null data
            for (tid, attr), row in self.all_cells_df_non_null.iterrows():
                val = str(row['val']) # Work with string representation
                self.global_attr_domains[attr][val] += 1
                self.single_counts[attr][val] += 1
            logging.info(f"Determined global domains (from non-null values) for {len(self.global_attr_domains)} attributes.")

            fetch_end = time.time()
            logging.info(f"[Pruner] Data fetching completed in {fetch_end - fetch_start:.2f} seconds.")
            return True
        except Exception as e:
            logging.error(f"Error fetching data: {e}", exc_info=True)
            return False

    def _calculate_cooccurrences(self):
        """Calculates co-occurrence counts between attribute values within tuples."""
        if self.all_cells_df_non_null is None:
            logging.error("Non-null cell data not loaded. Cannot calculate co-occurrences.")
            return

        logging.info("[Pruner] Calculating co-occurrence counts (using non-null values)...")
        cooc_start = time.time()
        # Group by tid (level 0 of the index)
        grouped_by_tid = self.all_cells_df_non_null.groupby(level=0)

        total_pairs = 0
        for tid, group in grouped_by_tid:
            attributes = list(group.index.get_level_values('attr'))
            # Create combinations of attributes within the tuple
            for i in range(len(attributes)):
                for j in range(i + 1, len(attributes)):
                    attr1, attr2 = attributes[i], attributes[j]
                    val1 = str(group.loc[(tid, attr1), 'val'])
                    val2 = str(group.loc[(tid, attr2), 'val'])

                    # Store counts symmetrically for easier lookup later
                    self.cooccurrence_counts[attr1][attr2][(val1, val2)] += 1
                    self.cooccurrence_counts[attr2][attr1][(val2, val1)] += 1
                    total_pairs += 1

        cooc_end = time.time()
        single_count_total = sum(len(v) for v in self.single_counts.values())
        logging.info(f"[Pruner] Co-occurrence counts calculated in {cooc_end - cooc_start:.2f} seconds.")
        logging.info(f"Total single counts: {single_count_total}, Total pair counts: {total_pairs}")


    def _build_tuple_context_cache(self):
        """Builds a cache {tid: {attr: val_str}} using non-null values for context."""
        logging.info("Building tuple context cache (using non-null values for co-occurrence checks)...")
        grouped_by_tid = self.all_cells_df_non_null.groupby(level=0)
        for tid, group in grouped_by_tid:
            self.tuple_context_cache[tid] = {attr: str(val) for attr, val in group['val'].items()}
        logging.info(f"Built cache for {len(self.tuple_context_cache)} tuples.")


    def _calculate_domain_for_cell(self, tid, attr):
        """Calculates the candidate domain for a single cell (tid, attr)."""
        # Get the original value (can be None)
        try:
             original_value = self.all_cells_df_with_null.loc[(tid, attr), 'val']
        except KeyError:
             logging.warning(f"Cell ({tid}, {attr}) not found in all_cells_df_with_null. Skipping domain calculation.")
             return set()

        # Use the placeholder if the original value is None
        original_value_str = NULL_REPR_PLACEHOLDER if original_value is None else str(original_value)
        # The domain always includes the original value
        final_domain = {original_value_str}

        # Check if cell is noisy - apply pruning only to noisy cells
        is_noisy = (tid, attr) in self.noisy_cells_set

        if is_noisy and self.tau > 0: # Apply co-occurrence pruning only if noisy and tau > 0
            # Use the context cache based on non-null values
            tuple_context = self.tuple_context_cache.get(tid, {})
            if not tuple_context: # No context if tuple had only nulls or one value
                return final_domain

            # Calculate min frequency threshold based on tau (using log for stability)
            # This interpretation might need refinement based on the exact HoloClean paper formula.
            # Simplified: Keep if P(candidate | any other value in tuple) >= tau
            # P(v_c | v_o) = count(v_c, v_o) / count(v_o)
            # We check against tau directly here.

            global_candidates = self.global_attr_domains[attr].keys()

            for candidate_val_str in global_candidates:
                 if candidate_val_str == original_value_str: continue # Already added

                 # Check co-occurrence with other attributes in the tuple
                 should_add = False
                 for other_attr, other_val_str in tuple_context.items():
                     if other_attr == attr: continue # Skip self

                     count_other = self.single_counts[other_attr].get(other_val_str, 0)
                     if count_other == 0: continue

                     # Look up co-occurrence count
                     # Assumes keys are (candidate_val_str, other_val_str)
                     cooc_count = self.cooccurrence_counts[attr][other_attr].get((candidate_val_str, other_val_str), 0)

                     # Simple conditional probability check P(candidate | other)
                     prob_cond = cooc_count / count_other if count_other > 0 else 0

                     if prob_cond >= self.tau:
                         should_add = True
                         break # Add if condition met for at least one context value

                 if should_add:
                     final_domain.add(candidate_val_str)

        # --- ADDED: Ensure non-noisy cells have >1 candidate if possible ---
        if not is_noisy and len(final_domain) == 1:
            # This is a training cell with only its original value in the domain.
            # Try to add the globally most frequent *other* value for this attribute.
            most_common_alt = None
            # Get top 2 most common values globally for this attribute
            top_two = self.global_attr_domains[attr].most_common(2)
            if len(top_two) > 0:
                # If the most common is the original value, try the second most common
                if top_two[0][0] == original_value_str:
                     if len(top_two) > 1:
                         most_common_alt = top_two[1][0]
                else: # Otherwise, the most common is already an alternative
                     most_common_alt = top_two[0][0]

            if most_common_alt is not None:
                 logging.debug(f"Adding most common alt '{most_common_alt}' to domain for non-noisy cell ({tid}, {attr})")
                 final_domain.add(most_common_alt)
        # --- END ADDED ---

        return final_domain


    def calculate_all_domains(self):
        """Calculates domains for all cells and returns them as a list of tuples."""
        if not self._fetch_data():
            return None # Error during data fetching
        self._calculate_cooccurrences()
        self._build_tuple_context_cache()

        all_domain_entries = []
        logging.info(f"Starting domain calculation for ALL {len(self.all_cells_df_with_null)} cells (Tau={self.tau} for noisy)...")
        domain_start_time = time.time()

        # Iterate through all cells defined in the index of all_cells_df_with_null
        total_cells = len(self.all_cells_df_with_null)
        processed_count = 0
        progress_interval = max(1, total_cells // 20) # Update progress roughly 20 times

        for (tid, attr), _ in self.all_cells_df_with_null.iterrows():
            domain_set = self._calculate_domain_for_cell(tid, attr)
            for candidate_val in domain_set:
                all_domain_entries.append((tid, attr, candidate_val))

            processed_count += 1
            if processed_count % progress_interval == 0 or processed_count == total_cells:
                 logging.info(f"Processed {processed_count}/{total_cells} cells...")


        domain_end_time = time.time()
        logging.info(f"[Pruner] Domain calculation finished in {domain_end_time - domain_start_time:.2f} seconds.")
        logging.info(f"Generated {len(all_domain_entries)} unique candidate domain entries (original values always included).")
        return all_domain_entries

    def insert_domains_to_db(self, domain_entries):
        """Inserts calculated domain entries into the database."""
        if domain_entries is None:
            logging.error("Cannot insert domains, calculation failed.")
            return False

        logging.info("[Pruner] Inserting candidate domains into database...")
        insert_start = time.time()
        try:
            with self.db_conn.cursor() as cur:
                logging.info("Clearing previous domains...")
                cur.execute("DELETE FROM domains;")
                # Use execute_batch for efficient insertion
                insert_sql = "INSERT INTO domains (tid, attr, candidate_val) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
                psycopg2.extras.execute_batch(cur, insert_sql, domain_entries, page_size=5000)
            self.db_conn.commit()
            insert_end = time.time()
            logging.info(f"[Pruner] Successfully attempted insertion of {len(domain_entries)} domains in {insert_end - insert_start:.2f} seconds.")
            return True
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"Error inserting domains: {e}", exc_info=True)
            return False


    def run(self):
        """Executes the full pruning and insertion process."""
        domain_entries = self.calculate_all_domains()
        success = self.insert_domains_to_db(domain_entries)
        return success

# Example usage (if run directly)
if __name__ == '__main__':
    import config
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Example: Get tau from command line or use default
    tau_value = 0.3 # Default or read from args
    if len(sys.argv) > 1:
        try:
             tau_value = float(sys.argv[1])
        except ValueError:
             logging.warning(f"Invalid tau value provided '{sys.argv[1]}'. Using default {tau_value}")

    try:
        db_conn = psycopg2.connect(**config.DB_SETTINGS)
        logging.info("Database connection successful.")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)

    pruner = DomainPruner(db_conn, tau=tau_value)
    pruner.run()

    db_conn.close()
    logging.info("Database connection closed.")