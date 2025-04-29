# File: compiler/pruning.py
# Calculates candidate domains for cells using correlations and co-occurrence.
# VERSION: Revised based on HoloClean principles

import pandas as pd
import time
from math import log, log2
from collections import Counter, defaultdict
import logging
import itertools
import numpy as np
from scipy.stats import entropy # Using scipy for entropy calculation

import psycopg2
import psycopg2.extras
from tqdm import tqdm # For execute_batch

# Assuming config.py exists and defines these:
try:
    import config
    NULL_REPR_PLACEHOLDER = config.NULL_REPR_PLACEHOLDER
    # --- Add Thresholds to your config.py ---
    # Threshold for correlation (e.g., normalized conditional entropy > threshold)
    DEFAULT_CORRELATION_THRESHOLD = config.PRUNING_CORRELATION_THRESHOLD # e.g., 0.1 or higher
    # Threshold for co-occurrence P(candidate | cond_val)
    DEFAULT_DOMAIN_COOCCURRENCE_THRESHOLD = config.PRUNING_DOMAIN_COOCCURRENCE_THRESHOLD # e.g., 0.01 or higher
    # Max candidates from one correlated attribute
    DEFAULT_MAX_CANDIDATES_PER_CORR_ATTR = config.PRUNING_MAX_CANDIDATES_PER_CORR_ATTR # e.g., 10
    # Overall max domain size per cell
    DEFAULT_MAX_DOMAIN_SIZE_PER_CELL = config.PRUNING_MAX_DOMAIN_SIZE_PER_CELL # e.g., 50 or 100
    # Min domain size (trigger adding global frequent)
    DEFAULT_MIN_DOMAIN_SIZE = config.PRUNING_MIN_DOMAIN_SIZE # e.g., 2

except (ImportError, AttributeError) as e:
    logging.warning(f"Could not import settings from config.py ({e}). Using defaults.")
    NULL_REPR_PLACEHOLDER = "__NULL__"
    DEFAULT_CORRELATION_THRESHOLD = 0.1
    DEFAULT_DOMAIN_COOCCURRENCE_THRESHOLD = 0.01
    DEFAULT_MAX_CANDIDATES_PER_CORR_ATTR = 10
    DEFAULT_MAX_DOMAIN_SIZE_PER_CELL = 50
    DEFAULT_MIN_DOMAIN_SIZE = 2


class DomainPruner:
    """
    Calculates candidate domains using correlations and co-occurrence statistics,
    closer to the HoloClean reference implementation.
    """

    def __init__(self, db_conn,
                 correlation_threshold=DEFAULT_CORRELATION_THRESHOLD,
                 domain_cooccurrence_threshold=DEFAULT_DOMAIN_COOCCURRENCE_THRESHOLD,
                 max_candidates_per_corr_attr=DEFAULT_MAX_CANDIDATES_PER_CORR_ATTR,
                 max_domain_size_per_cell=DEFAULT_MAX_DOMAIN_SIZE_PER_CELL,
                 min_domain_size=DEFAULT_MIN_DOMAIN_SIZE
                 ):
        self.db_conn = db_conn
        self.correlation_threshold = correlation_threshold
        self.domain_cooccurrence_threshold = domain_cooccurrence_threshold
        self.max_candidates_per_corr_attr = max_candidates_per_corr_attr
        self.max_domain_size_per_cell = max_domain_size_per_cell
        self.min_domain_size = min_domain_size

        # Data storage
        self.raw_df = None # Raw data including NULLs
        self.attributes = []
        self.raw_records_by_tid = {} # Cache {tid: {attr: val_str}}

        # Statistics
        self.single_stats = defaultdict(Counter) # {attr: Counter(val_str: count)}
        self.pair_stats_raw = defaultdict(lambda: defaultdict(Counter)) # {attr1: {attr2: Counter((val1_str, val2_str): count)}}
        self.total_tuples = 0

        # Correlations
        self.correlations = defaultdict(dict) # {attr1: {attr2: correlation_score}}
        self._corr_attrs_cache = {} # Cache for get_correlated_attributes

        # Candidate generation cache (optional, can be large)
        # {cond_attr: {cond_val_str: {target_attr: [candidate_val_str]}}}
        # self.precomputed_candidates = defaultdict(lambda: defaultdict(dict))

        # Misc
        self.noisy_cells_set = set() # Set of (tid, attr) tuples marked as noisy


    def _fetch_data(self):
        """Fetches raw data and noisy cells."""
        logging.info("[PrunerV2] Fetching data from database...")
        fetch_start = time.time()
        try:
            # Fetch all cells into pandas DataFrame
            # Important: Ensure tid is treated as int if it comes as float/object
            self.raw_df = pd.read_sql("SELECT tid, attr, val FROM cells ORDER BY tid, attr", self.db_conn)
            logging.info(f"Fetched {len(self.raw_df)} total cell entries.")
            if self.raw_df.empty:
                 logging.error("Fetched DataFrame is empty. Cannot proceed.")
                 return False

            # Convert tid to int if needed and handle potential errors
            try:
                 self.raw_df['tid'] = self.raw_df['tid'].astype(int)
            except ValueError as e:
                 logging.error(f"Could not convert 'tid' column to integer: {e}. Check data.")
                 # Optionally try handling non-numeric tids if expected
                 return False

            self.attributes = sorted(self.raw_df['attr'].unique())
            logging.info(f"Found {len(self.attributes)} unique attributes.")

            # Create tuple context cache {tid: {attr: val_str}}
            logging.info("Building tuple context cache...")
            self.raw_records_by_tid = {}
            for tid, group in self.raw_df.groupby('tid'):
                 self.raw_records_by_tid[tid] = {
                     row['attr']: NULL_REPR_PLACEHOLDER if pd.isna(row['val']) else str(row['val'])
                     for _, row in group.iterrows()
                 }
            self.total_tuples = len(self.raw_records_by_tid)
            logging.info(f"Built cache for {self.total_tuples} tuples.")


            # Fetch noisy cells into a set for quick lookup
            noisy_df = pd.read_sql("SELECT tid, attr FROM noisy_cells", self.db_conn)
            # Ensure tid is int here too for consistency
            try:
                 noisy_df['tid'] = noisy_df['tid'].astype(int)
            except ValueError as e:
                 logging.warning(f"Could not convert 'tid' in noisy_cells to integer: {e}. Matching might fail.")

            self.noisy_cells_set = set(tuple(x) for x in noisy_df[['tid', 'attr']].to_numpy())
            logging.info(f"Fetched {len(self.noisy_cells_set)} unique noisy cell identifiers.")

            fetch_end = time.time()
            logging.info(f"[PrunerV2] Data fetching completed in {fetch_end - fetch_start:.2f} seconds.")
            return True

        except Exception as e:
            logging.error(f"Error fetching data: {e}", exc_info=True)
            return False

    def _compute_statistics(self):
        """Computes single and pairwise statistics from the raw data."""
        if self.raw_df is None or self.raw_df.empty:
            logging.error("Raw data not loaded. Cannot compute statistics.")
            return

        logging.info("[PrunerV2] Calculating single and pairwise statistics...")
        stats_start = time.time()

        # Single counts (including NULL placeholder)
        self.single_stats = defaultdict(Counter)
        for _, row in self.raw_df.iterrows():
            attr = row['attr']
            val_str = NULL_REPR_PLACEHOLDER if pd.isna(row['val']) else str(row['val'])
            self.single_stats[attr][val_str] += 1

        # Pairwise counts (only non-null pairs for co-occurrence candidates)
        self.pair_stats_raw = defaultdict(lambda: defaultdict(Counter))
        # Group by tid
        grouped_by_tid = self.raw_df.groupby('tid')

        for tid, group in tqdm(grouped_by_tid, desc="Calculating Pair Stats", unit=" tuple"):
            # Get non-null cells for this tuple
            non_null_cells = group.dropna(subset=['val'])
            attributes_in_tuple = list(non_null_cells['attr'])

            # Create combinations of attributes within the tuple
            for i in range(len(attributes_in_tuple)):
                for j in range(i + 1, len(attributes_in_tuple)):
                    attr1 = attributes_in_tuple[i]
                    attr2 = attributes_in_tuple[j]
                    # Get values (already checked for non-null)
                    val1_str = str(non_null_cells[non_null_cells['attr'] == attr1]['val'].iloc[0])
                    val2_str = str(non_null_cells[non_null_cells['attr'] == attr2]['val'].iloc[0])

                    # Store counts symmetrically
                    self.pair_stats_raw[attr1][attr2][(val1_str, val2_str)] += 1
                    self.pair_stats_raw[attr2][attr1][(val2_str, val1_str)] += 1

        stats_end = time.time()
        single_count_total = sum(sum(c.values()) for c in self.single_stats.values())
        pair_count_total = sum(sum(sum(c.values()) for c in inner.values()) for inner in self.pair_stats_raw.values()) // 2 # Divide by 2 as stored symmetrically
        logging.info(f"[PrunerV2] Statistics calculated in {stats_end - stats_start:.2f} seconds.")
        logging.info(f"Total single counts: {single_count_total}, Total unique pair counts: {pair_count_total}")


    def _normalize_entropy(self, h_cond, h_base):
        """ Normalizes conditional entropy. Returns 1.0 - H(X|Y)/H(X)."""
        if h_base == 0: # If base entropy is 0, variable is constant
             return 1.0 # Fully predictable
        if h_cond < 1e-9: # Avoid floating point issues close to 0
             h_cond = 0.0
        # Normalize: 0 means independent, 1 means Y determines X
        norm_h_cond = h_cond / h_base
        # Return correlation: 1 - normalized_conditional_entropy
        # 1 means Y determines X, 0 means independent
        return 1.0 - norm_h_cond

    def _compute_correlations(self):
        """Computes pairwise attribute correlations using normalized conditional entropy."""
        if self.raw_df is None or self.raw_df.empty:
             logging.error("Raw data not loaded. Cannot compute correlations.")
             return

        logging.info("[PrunerV2] Calculating attribute correlations (using conditional entropy)...")
        corr_start = time.time()
        self.correlations = defaultdict(dict)

        # Pivot table for easier entropy calculation
        # Ensure tid is the index, columns are attributes, values are cell values (as strings)
        try:
            pivot_df = self.raw_df.pivot(index='tid', columns='attr', values='val').fillna(NULL_REPR_PLACEHOLDER).astype(str)
        except Exception as e:
            logging.error(f"Failed to pivot DataFrame for correlation calculation: {e}. Check for duplicate (tid, attr) pairs.")
            # Fallback or re-fetch data ensuring uniqueness? For now, return.
            return

        # Pre-calculate base entropies H(X)
        base_entropies = {}
        for attr in self.attributes:
            if attr not in pivot_df.columns: continue # Skip if attribute somehow missing
            counts = pivot_df[attr].value_counts(normalize=True)
            base_entropies[attr] = entropy(counts, base=2) # Using base 2

        logging.info(f"Calculated base entropies for {len(base_entropies)} attributes.")

        # Compute pairwise conditional entropies H(X|Y)
        for attr_x in tqdm(self.attributes, desc="Calculating Correlations", unit=" attr"):
            if attr_x not in pivot_df.columns: continue
            h_x = base_entropies.get(attr_x, 0)
            self.correlations[attr_x][attr_x] = 1.0 # Correlation with self is 1

            for attr_y in self.attributes:
                if attr_x == attr_y or attr_y not in pivot_df.columns:
                    continue

                try:
                    # Calculate H(X, Y) - Joint Entropy
                    joint_counts = pivot_df.groupby([attr_x, attr_y]).size()
                    h_xy = entropy(joint_counts, base=2)

                    # Calculate H(Y)
                    h_y = base_entropies.get(attr_y, 0)

                    # Calculate H(X|Y) = H(X, Y) - H(Y)
                    h_x_given_y = h_xy - h_y
                    if h_x_given_y < 0: h_x_given_y = 0 # Handle potential float precision issues

                    # Normalize and store correlation
                    corr_xy = self._normalize_entropy(h_x_given_y, h_x)
                    self.correlations[attr_x][attr_y] = corr_xy

                except Exception as e:
                    logging.warning(f"Could not calculate correlation between '{attr_x}' and '{attr_y}': {e}")
                    self.correlations[attr_x][attr_y] = 0.0 # Default to 0 on error

        corr_end = time.time()
        logging.info(f"[PrunerV2] Correlations calculated in {corr_end - corr_start:.2f} seconds.")


    def get_correlated_attributes(self, target_attr):
        """Gets attributes correlated with target_attr above the threshold."""
        cache_key = (target_attr, self.correlation_threshold)
        if cache_key not in self._corr_attrs_cache:
            correlated = []
            if target_attr in self.correlations:
                for other_attr, score in self.correlations[target_attr].items():
                    if other_attr != target_attr and score >= self.correlation_threshold:
                        correlated.append(other_attr)
            self._corr_attrs_cache[cache_key] = sorted(correlated)
        return self._corr_attrs_cache[cache_key]

    def _get_cooccurrence_candidates(self, target_attr, cond_attr, cond_val_str):
        """
        Gets candidate values for target_attr that co-occur with
        cond_val_str in cond_attr, pruned by threshold P(target_val | cond_val).
        """
        candidates = []
        cond_val_count = self.single_stats[cond_attr].get(cond_val_str, 0)
        if cond_val_count == 0:
            return candidates # Cannot calculate probability

        # Iterate through potential target values that co-occurred with cond_val
        if cond_attr in self.pair_stats_raw and target_attr in self.pair_stats_raw[cond_attr]:
             # pair_stats_raw stores (cond_val, target_val) -> count keyed by cond_attr, target_attr
             relevant_pairs = {pair: count for pair, count in self.pair_stats_raw[cond_attr][target_attr].items() if pair[0] == cond_val_str}

             cand_probs = []
             for (cv, target_val), pair_count in relevant_pairs.items():
                 prob_target_given_cond = pair_count / cond_val_count
                 if prob_target_given_cond >= self.domain_cooccurrence_threshold:
                      cand_probs.append((target_val, prob_target_given_cond))

             # Sort by probability and take top N
             cand_probs.sort(key=lambda x: x[1], reverse=True)
             candidates = [cand for cand, prob in cand_probs[:self.max_candidates_per_corr_attr]]

        return candidates


    def _generate_domain_for_cell(self, tid, attr):
        """Generates the candidate domain for a single cell (tid, attr)."""
        if tid not in self.raw_records_by_tid:
            logging.warning(f"TID {tid} not found in tuple cache. Skipping domain for ({tid}, {attr}).")
            return set()

        row_dict = self.raw_records_by_tid[tid]
        init_value_str = row_dict.get(attr, NULL_REPR_PLACEHOLDER) # Get initial value as string

        # Start domain with initial value if not null
        domain_set = set()
        if init_value_str != NULL_REPR_PLACEHOLDER:
            domain_set.add(init_value_str)

        # 1. Candidates from Correlated Attributes Co-occurrence
        correlated_attributes = self.get_correlated_attributes(attr)
        logging.debug(f"Cell ({tid}, {attr}): Correlated Attrs (>{self.correlation_threshold}): {correlated_attributes}")

        for cond_attr in correlated_attributes:
            cond_val_str = row_dict.get(cond_attr)
            if cond_val_str is None or cond_val_str == NULL_REPR_PLACEHOLDER:
                continue # Skip if context value is null

            # Get candidates based on P(target_val | cond_val) > threshold
            cooc_candidates = self._get_cooccurrence_candidates(attr, cond_attr, cond_val_str)
            if cooc_candidates:
                logging.debug(f"  Cell ({tid}, {attr}): Adding {len(cooc_candidates)} candidates from P(target|{cond_attr}='{cond_val_str}') >= {self.domain_cooccurrence_threshold}: {cooc_candidates[:5]}...")
                domain_set.update(cooc_candidates)

        # 2. Candidates from Constraint Violations (Placeholder)
        # TODO: Implement logic to check 'violations' table for (tid, attr)
        #       Parse relevant constraints, find alternative values from non-violating tuples
        #       or from the constraint definition itself. Add valid candidates to domain_set.
        # Example: If FD Zip->City is violated, add the City from another tuple with the same Zip.
        # constraint_candidates = self._get_constraint_candidates(tid, attr)
        # domain_set.update(constraint_candidates)


        # 3. Ensure Minimum Domain Size (for noisy cells or all cells?)
        #    Using globally frequent values if domain is too small.
        #    Apply this check *before* final size capping.
        is_noisy = (tid, attr) in self.noisy_cells_set
        if is_noisy and len(domain_set) < self.min_domain_size:
            logging.debug(f"  Cell ({tid}, {attr}): Domain size {len(domain_set)} < min {self.min_domain_size}. Adding frequent global values.")
            # Get globally frequent values for 'attr', excluding NULL and existing domain values
            num_to_add = self.min_domain_size - len(domain_set)
            added_count = 0
            # Sort global candidates by frequency
            global_candidates_sorted = sorted(
                [(val, count) for val, count in self.single_stats[attr].items() if val != NULL_REPR_PLACEHOLDER],
                key=lambda x: x[1], reverse=True
            )
            for val, count in global_candidates_sorted:
                if val not in domain_set:
                    domain_set.add(val)
                    added_count += 1
                    logging.debug(f"    Added global candidate '{val}' (Count: {count})")
                    if added_count >= num_to_add:
                        break

        # 4. Cap Maximum Domain Size
        if len(domain_set) > self.max_domain_size_per_cell:
             logging.debug(f"  Cell ({tid}, {attr}): Domain size {len(domain_set)} > max {self.max_domain_size_per_cell}. Capping not implemented (consider probability ranking if needed).")
             # Simple truncation for now, but ideally rank candidates (e.g., by global freq, avg co-occurrence prob)
             # For simplicity, we might just let it be larger for now, TensorBuilder handles variable sizes.
             # Or convert back to list, sort (e.g., alphabetically?), and take top N. Let's let it be for now.
             pass # domain_set = set(sorted(list(domain_set))[:self.max_domain_size_per_cell])


        # Ensure NULL placeholder is never in the final domain
        domain_set.discard(NULL_REPR_PLACEHOLDER)

        logging.debug(f"  Cell ({tid}, {attr}): Final domain size = {len(domain_set)}, Domain = {list(domain_set)[:10]}...")
        return domain_set


    def calculate_and_insert_domains(self):
        """Calculates domains for all cells and inserts them into the database."""
        if not self._fetch_data(): return False
        self._compute_statistics()
        self._compute_correlations()
        # self._precompute_candidates() # Optional precomputation step

        all_domain_entries = []
        logging.info(f"[PrunerV2] Starting domain calculation for ALL {self.total_tuples * len(self.attributes)} potential cells...")
        domain_start_time = time.time()

        # Iterate through all known tuples and all attributes
        processed_cells = 0
        # Use list(self.raw_records_by_tid.keys()) to avoid issues if dict changes during iteration
        all_tids = list(self.raw_records_by_tid.keys())

        for tid in tqdm(all_tids, desc="Generating Domains", unit=" tuple"):
            for attr in self.attributes:
                 domain_set = self._generate_domain_for_cell(tid, attr)
                 for candidate_val in domain_set:
                     # Ensure candidate_val is string, handle potential type issues if necessary
                     all_domain_entries.append((tid, attr, str(candidate_val)))
                 processed_cells += 1

        domain_end_time = time.time()
        logging.info(f"[PrunerV2] Domain calculation finished in {domain_end_time - domain_start_time:.2f} seconds.")
        logging.info(f"Generated {len(all_domain_entries)} domain entries for {processed_cells} cells.")

        # Insertion into DB
        return self.insert_domains_to_db(all_domain_entries)


    def insert_domains_to_db(self, domain_entries):
        """Inserts calculated domain entries into the database."""
        if not domain_entries: # Check if list is empty
            logging.warning("No domain entries generated. Skipping database insertion.")
            # Consider if this should be an error or just a warning
            return True # Return True as no insertion error occurred

        logging.info("[PrunerV2] Inserting candidate domains into database...")
        insert_start = time.time()
        try:
            with self.db_conn.cursor() as cur:
                logging.info("Clearing previous domains...")
                cur.execute("DELETE FROM domains;")
                logging.info(f"Attempting to insert {len(domain_entries)} domain entries...")
                # Use execute_batch for efficient insertion
                insert_sql = "INSERT INTO domains (tid, attr, candidate_val) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
                psycopg2.extras.execute_batch(cur, insert_sql, domain_entries, page_size=10000) # Adjusted page size
            self.db_conn.commit()
            insert_end = time.time()
            # Check actual inserted count? Might be complex with ON CONFLICT
            logging.info(f"[PrunerV2] Database commit successful for domains in {insert_end - insert_start:.2f} seconds.")
            return True
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"Error inserting domains: {e}", exc_info=True)
            return False

    def run(self):
        """Executes the full pruning and insertion process."""
        success = self.calculate_and_insert_domains()
        return success

# --- Main execution block ---
if __name__ == '__main__':
    import sys
    # Setup logging
    logging.basicConfig(level=logging.INFO, # Change to DEBUG for detailed domain generation logs
                        format='[%(asctime)s] {%(levelname)s} %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info("Running Revised Domain Pruner (pruning.py)...")

    # --- Configuration (Read from config.py or use defaults) ---
    # These should ideally be passed via argparse or read from config
    corr_thresh = DEFAULT_CORRELATION_THRESHOLD
    cooccur_thresh = DEFAULT_DOMAIN_COOCCURRENCE_THRESHOLD
    max_cand_per_attr = DEFAULT_MAX_CANDIDATES_PER_CORR_ATTR
    max_domain_cell = DEFAULT_MAX_DOMAIN_SIZE_PER_CELL
    min_domain = DEFAULT_MIN_DOMAIN_SIZE

    logging.info(f"Using Parameters: Corr>={corr_thresh}, Cooccur>={cooccur_thresh}, MaxCandPerAttr={max_cand_per_attr}, MaxDomain={max_domain_cell}, MinDomain={min_domain}")

    # --- Database Connection ---
    try:
        # Ensure config.py DB_SETTINGS are loaded correctly
        from config import DB_SETTINGS
        db_conn = psycopg2.connect(**DB_SETTINGS)
        logging.info("Database connection successful.")
    except ImportError:
        logging.error("Could not import DB_SETTINGS from config.py. Please ensure config.py exists and is configured.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)

    # --- Run Pruner ---
    try:
        pruner = DomainPruner(
            db_conn,
            correlation_threshold=corr_thresh,
            domain_cooccurrence_threshold=cooccur_thresh,
            max_candidates_per_corr_attr=max_cand_per_attr,
            max_domain_size_per_cell=max_domain_cell,
            min_domain_size=min_domain
        )
        success = pruner.run()
        if success:
            logging.info("Domain pruning and insertion process completed successfully.")
        else:
            logging.error("Domain pruning and insertion process failed.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during pruning: {e}", exc_info=True)
    finally:
        # --- Close Connection ---
        if db_conn:
            db_conn.close()
            logging.info("Database connection closed.")