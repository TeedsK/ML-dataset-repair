# File: compiler/compile.py
# Compiles signals into features, RE-INTRODUCING HASHING with a LARGER size.

import pandas as pd
import time
import psycopg2
import psycopg2.extras
from collections import defaultdict
import itertools
import logging
import hashlib # Using hashlib for potentially better distribution than built-in hash

from detectors.constraints import ConstraintViolationDetector

NULL_REPR_PLACEHOLDER = "__NULL__"
# --- Define HASH Size ---
# Choose a size based on available memory and desired granularity.
# Increase significantly from 5000. Try 100k, 500k, or 1M as starting points.
# Let's try 200,000. Adjust if needed based on memory/performance.
FEATURE_HASH_SIZE = 200000 # ADJUST THIS VALUE AS NEEDED
# --- END Define HASH Size ---

class FeatureCompiler:
    """Generates features for the HoloClean probabilistic model using hashing."""

    def __init__(self, db_conn, relax_constraints=True, constraints_filepath="hospital_constraints.txt"):
        self.db_conn = db_conn
        self.relax_constraints = relax_constraints
        self.constraints_filepath = constraints_filepath
        self.constraints = []
        self.detector_helper = None
        try:
            self.detector_helper = ConstraintViolationDetector(db_conn, constraints_filepath)
            self.constraints = self.detector_helper.constraints
            logging.info(f"[Compiler] Loaded {len(self.constraints)} constraints for checking.")
        except Exception as e:
            logging.error(f"[Compiler] Failed to initialize ConstraintViolationDetector: {e}", exc_info=True)
            self.constraints = []

    def _clear_features(self):
        logging.info("[Compiler] Clearing existing features...")
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("DELETE FROM features;")
            self.db_conn.commit()
            logging.info("Features table cleared.")
            return True
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"[Compiler] Error clearing features table: {e}", exc_info=True)
            return False

    # --- RE-ADDED Hashing Function ---
    def _hash_feature(self, feature_name_str):
        """Hashes a feature string to an index within FEATURE_HASH_SIZE."""
        hasher = hashlib.sha1(feature_name_str.encode('utf-8'))
        hash_bytes = hasher.digest()[:4] # Use first 4 bytes for ~4 billion range before modulo
        hash_int = int.from_bytes(hash_bytes, 'big', signed=False)
        # Add 1 to reserve index 0? Or handle potential 0 hash if needed.
        # Let's map to 0..HASH_SIZE-1 for now.
        hash_index = hash_int % FEATURE_HASH_SIZE
        # Return as string 'h_INDEX' to distinguish from prior feature?
        # Or rely on prior having a non-numeric name? Let's return string hash id.
        return f'h_{hash_index}' # Prefix to make it clearly a hashed feature string
    # --- END RE-ADDED Hashing Function ---


    def _insert_features(self, features_list):
        """
        Inserts features. Expects list of tuples:
        (tid, attr, candidate_val, feature_identifier_string)
        """
        if not features_list:
            logging.warning("[Compiler] No features generated to insert.")
            return 0

        unique_features_set = set(features_list)
        unique_features_list = list(unique_features_set)

        inserted_count = 0
        # Feature column is TEXT, so strings ('prior_minimality', 'h_12345') are fine.
        sql_insert = """
            INSERT INTO features (tid, attr, candidate_val, feature)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (tid, attr, candidate_val, feature) DO NOTHING;
        """
        logging.info(f"[Compiler] Attempting to insert {len(unique_features_list)} unique feature identifiers...")
        insert_start_time = time.time()
        try:
            with self.db_conn.cursor() as cur:
                 psycopg2.extras.execute_batch(cur, sql_insert, unique_features_list, page_size=10000)
                 inserted_count = len(unique_features_list)
            self.db_conn.commit()
            insert_time = time.time() - insert_start_time
            logging.info(f"[Compiler] Feature insertion complete in {insert_time:.2f} seconds ({inserted_count} unique features targeted).")
            return inserted_count
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"[Compiler] Error inserting features: {e}", exc_info=True)
            return 0


    def generate_cooccurrence_features(self):
        """
        Generates HASHED co-occurrence features using a larger hash space.
        """
        logging.info("[Compiler] Generating HASHED co-occurrence features...")
        start_time = time.time()
        sql = f"""
            SELECT
                d.tid,
                d.attr,
                d.candidate_val,
                c.attr AS other_attr,
                c.val AS other_val
            FROM domains d
            JOIN cells c ON d.tid = c.tid AND d.attr <> c.attr
            WHERE c.val IS NOT NULL;
        """
        features_list = []
        try:
            with self.db_conn.cursor(name='hashed_cooc_feature_cursor') as cur:
                 cur.itersize = 50000 # Adjust buffer size
                 cur.execute(sql)
                 processed_rows = 0
                 for row in cur:
                     tid, attr, candidate_val_str, other_attr, other_val_str = row[0], row[1], str(row[2]), row[3], str(row[4])
                     # Construct the specific feature name before hashing
                     feature_name = f'cooc_{attr}={candidate_val_str}_{other_attr}={other_val_str}'
                     # --- Hash the feature name ---
                     feature_hash_id = self._hash_feature(feature_name)
                     # --- END Hash the feature name ---
                     features_list.append((tid, attr, candidate_val_str, feature_hash_id)) # Store hash ID string
                     processed_rows += 1

            inserted = self._insert_features(features_list)
            duration = time.time() - start_time
            logging.info(f"[Compiler] Generated {len(features_list)} hashed co-occurrence feature instances from {processed_rows} DB rows in {duration:.2f}s.")
            logging.info(f"[Compiler] Inserted approx {inserted} unique hashed co-occurrence features.")
            return inserted
        except Exception as e:
            logging.error(f"Error generating hashed co-occurrence features: {e}", exc_info=True)
            return 0


    def generate_minimality_features(self):
        """Generates the minimality prior feature (still unique name)."""
        logging.info("[Compiler] Generating minimality prior features (non-hashed)...")
        start_time = time.time()
        prior_feature_name = 'prior_minimality' # Keep original unique name
        sql = f"""
            SELECT
                d.tid,
                d.attr,
                d.candidate_val
            FROM domains d
            JOIN cells c ON d.tid = c.tid AND d.attr = c.attr
            WHERE
                (c.val IS NULL AND d.candidate_val = '{NULL_REPR_PLACEHOLDER}')
                OR
                (c.val IS NOT NULL AND d.candidate_val = CAST(c.val AS TEXT));
        """
        features_list = []
        try:
            with self.db_conn.cursor(name='minimality_feature_cursor') as cur:
                 cur.execute(sql)
                 processed_rows = 0
                 for row in cur:
                      features_list.append((row[0], row[1], str(row[2]), prior_feature_name))
                      processed_rows += 1

            inserted = self._insert_features(features_list)
            duration = time.time() - start_time
            logging.info(f"[Compiler] Generated {len(features_list)} minimality feature instances from {processed_rows} DB rows in {duration:.2f}s.")
            logging.info(f"[Compiler] Inserted approx {inserted} minimality features.")
            return inserted
        except Exception as e:
            logging.error(f"Error generating minimality features: {e}", exc_info=True)
            return 0


    def generate_relaxed_dc_features(self):
        """
        Generates HASHED features based on potential DC violations.
        """
        if not self.relax_constraints:
            logging.info("[Compiler] Skipping relaxed DC features (relax_constraints=False).")
            return 0
        if not self.constraints or self.detector_helper is None:
             logging.warning("[Compiler] No constraints loaded/parsed or detector helper missing. Cannot generate relaxed DC features.")
             return 0

        logging.info("[Compiler] Generating HASHED relaxed denial constraint features...")
        total_dc_features_generated = 0
        start_time = time.time()

        logging.info("[Compiler-RelaxDC] Fetching initial cell values...")
        tuple_data_cache = defaultdict(dict)
        try:
            with self.db_conn.cursor(name='cell_data_cursor_dc') as cur:
                cur.itersize = 50000
                cur.execute("SELECT tid, attr, val FROM cells")
                for tid, attr, val in cur:
                     tuple_data_cache[int(tid)][attr] = NULL_REPR_PLACEHOLDER if pd.isna(val) else val
            logging.info(f"[Compiler-RelaxDC] Built initial value cache for {len(tuple_data_cache)} tuples.")
        except Exception as e:
            logging.error(f"Error fetching initial cell data for DC checks: {e}", exc_info=True)
            return 0

        logging.info("[Compiler-RelaxDC] Fetching candidate domains...")
        cell_domains = defaultdict(set)
        try:
            with self.db_conn.cursor(name='domain_data_cursor_dc') as cur:
                cur.itersize = 50000
                cur.execute("SELECT tid, attr, candidate_val FROM domains")
                for tid, attr, candidate_val in cur:
                     cell_domains[(int(tid), attr)].add(str(candidate_val))
            logging.info(f"[Compiler-RelaxDC] Fetched domains for {len(cell_domains)} cells.")
        except Exception as e:
            logging.error(f"Error fetching domains for DC checks: {e}", exc_info=True)
            return 0

        dc_features_list = []
        processed_constraints = 0

        for constraint in self.constraints:
            constraint_id_str = str(constraint['id'])
            logging.debug(f"[Compiler-RelaxDC] Processing Constraint {constraint_id_str}")

            # Find pairs initially violating the constraint
            # (Reusing original logic, potentially slow)
            eq_preds = [p for p in constraint['predicates'] if p['type'] == 'EQ']
            if not eq_preds: continue
            key_attr1 = eq_preds[0]['a1']
            sql_fetch_keys = f"SELECT tid, val FROM cells WHERE attr = %s AND val IS NOT NULL;"
            try:
                 key_df = pd.read_sql(sql_fetch_keys, self.db_conn, params=(key_attr1,))
                 key_df['tid'] = key_df['tid'].astype(int)
            except Exception as e:
                 logging.warning(f"Error fetching keys for constraint {constraint_id_str}, skipping: {e}")
                 continue

            grouped_tids = key_df.groupby('val')['tid'].apply(list)
            initial_violating_pairs = set()
            checked_pairs = set()

            for _, tids in grouped_tids.items():
                if len(tids) > 1:
                    for tid1, tid2 in itertools.combinations(tids, 2):
                         pair_key = tuple(sorted((tid1, tid2)))
                         if pair_key in checked_pairs: continue
                         checked_pairs.add(pair_key)
                         tuple1_data = tuple_data_cache.get(tid1)
                         tuple2_data = tuple_data_cache.get(tid2)
                         if not tuple1_data or not tuple2_data: continue
                         if self.detector_helper._check_violation(tuple1_data, tuple2_data, constraint):
                             initial_violating_pairs.add(pair_key)

            logging.debug(f"[Compiler-RelaxDC] Found {len(initial_violating_pairs)} initially violating pairs for constraint {constraint_id_str}.")

            for tid1, tid2 in initial_violating_pairs:
                tuple1_orig_data = tuple_data_cache.get(tid1)
                tuple2_orig_data = tuple_data_cache.get(tid2)
                if not tuple1_orig_data or not tuple2_orig_data: continue

                # Determine involved cells
                involved_cells_t1 = set((tid1, p['a1']) for p in constraint['predicates'] if p['a1'] in tuple1_orig_data)
                involved_cells_t2 = set((tid2, p['a2']) for p in constraint['predicates'] if p['a2'] in tuple2_orig_data)
                for p in eq_preds:
                    if p['a1'] in tuple1_orig_data: involved_cells_t1.add((tid1, p['a1']))
                    if p['a2'] in tuple2_orig_data: involved_cells_t2.add((tid2, p['a2']))
                all_involved_cells = involved_cells_t1.union(involved_cells_t2)

                for cell_tid, cell_attr in all_involved_cells:
                    if (cell_tid, cell_attr) not in cell_domains: continue
                    candidate_vals = cell_domains[(cell_tid, cell_attr)]
                    original_val_tuple = tuple1_orig_data if cell_tid == tid1 else tuple2_orig_data

                    for candidate_val_str in candidate_vals:
                        temp_tuple_data = original_val_tuple.copy()
                        temp_tuple_data[cell_attr] = candidate_val_str # Assuming _check_violation handles placeholder

                        violation_persists = False
                        if cell_tid == tid1:
                             violation_persists = self.detector_helper._check_violation(temp_tuple_data, tuple2_orig_data, constraint)
                        else:
                             violation_persists = self.detector_helper._check_violation(tuple1_orig_data, temp_tuple_data, constraint)

                        if violation_persists:
                             # --- Use HASHED feature name ---
                             # Feature indicates violation of constraint k if this candidate is chosen for this cell
                             # We hash the constraint ID itself to reduce dimensionality, accepting collisions.
                             # More granular would be hashing constraint+cell+candidate, but might be too many.
                             feature_name = f"DCViolates:{constraint_id_str}" # Base name before hashing
                             feature_hash_id = self._hash_feature(feature_name)
                             # --- END Use HASHED feature name ---
                             dc_features_list.append((cell_tid, cell_attr, candidate_val_str, feature_hash_id))

            processed_constraints += 1

        logging.info(f"[Compiler-RelaxDC] Finished checking {processed_constraints} constraints.")
        duration = time.time() - start_time
        logging.info(f"Generated {len(dc_features_list)} raw hashed relaxed DC feature instances in {duration:.2f}s (before deduplication).")
        inserted = self._insert_features(dc_features_list)
        logging.info(f"[Compiler] Inserted approx {inserted} unique hashed relaxed DC features.")
        return inserted


    def compile_all(self):
        """Runs all feature generation steps."""
        if not self._clear_features():
            logging.error("Aborting compilation due to error clearing features.")
            return
        total_features_est = 0
        compile_start_time = time.time()
        try:
            cooc_feat_count = self.generate_cooccurrence_features()
            min_feat_count = self.generate_minimality_features()
            dc_feat_count = 0
            if self.relax_constraints:
                 dc_feat_count = self.generate_relaxed_dc_features()
            else:
                 logging.warning("[Compiler] Hard constraint factor generation not implemented in this version.")

            # Get final count from DB
            final_feature_count = 0
            try:
                 with self.db_conn.cursor() as cur:
                     cur.execute("SELECT COUNT(DISTINCT feature) FROM features;")
                     final_feature_count = cur.fetchone()[0]
                 logging.info(f"[Compiler] Final total unique features in table: {final_feature_count}")
            except Exception as e:
                 logging.error(f"Could not query final feature count: {e}")
                 total_features_est = cooc_feat_count + min_feat_count + dc_feat_count
                 logging.info(f"[Compiler] Estimated total unique features inserted across steps: {total_features_est}")
            else:
                 total_features_est = final_feature_count

        except Exception as e:
             logging.error(f"[Compiler] Error during feature generation: {e}", exc_info=True)
        finally:
            compile_end_time = time.time()
            logging.info(f"\n[Compiler] Feature compilation finished in {compile_end_time - compile_start_time:.2f} seconds.")


# --- Main execution block ---
if __name__ == '__main__':
    import config
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    relax = getattr(config, 'RELAX_CONSTRAINTS', True)
    constraints_file = getattr(config, 'CONSTRAINTS_FILE', 'hospital_constraints.txt')

    logging.info(f"Running Feature Compiler (compile.py) with HASHING (Size={FEATURE_HASH_SIZE}), Relax Constraints = {relax}")

    try:
        db_conn = psycopg2.connect(**config.DB_SETTINGS)
        logging.info("Database connection successful.")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        compiler = FeatureCompiler(db_conn, relax_constraints=relax, constraints_filepath=constraints_file)
        compiler.compile_all()
    except Exception as e:
        logging.error(f"An unexpected error occurred during compilation: {e}", exc_info=True)
    finally:
        if db_conn:
            db_conn.close()
            logging.info("Database connection closed.")