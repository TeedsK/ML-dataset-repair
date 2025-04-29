# File: compiler/compile.py
# Compiles signals into features, using FEATURE HASHING for co-occurrence and DCs.

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
# Too small = too many collisions. Too large = defeats the purpose.
# Let's start with 5000. Adjust if needed.
FEATURE_HASH_SIZE = 5000
# --- END Define HASH Size ---

class FeatureCompiler:
    """Generates features for the HoloClean probabilistic model using hashing."""

    def __init__(self, db_conn, relax_constraints=True, constraints_filepath="hospital_constraints.txt"):
        self.db_conn = db_conn
        self.relax_constraints = relax_constraints
        self.constraints_filepath = constraints_filepath
        self.constraints = []
        self.detector_helper = None
        # Ensure detector_helper is initialized to access _check_violation if needed
        try:
            self.detector_helper = ConstraintViolationDetector(db_conn, constraints_filepath)
            self.constraints = self.detector_helper.constraints
            logging.info(f"[Compiler] Loaded {len(self.constraints)} constraints for checking.")
        except Exception as e:
            logging.error(f"[Compiler] Failed to initialize ConstraintViolationDetector: {e}", exc_info=True)
            self.constraints = []

    def _clear_features(self):
        # (Keep this method as is)
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

    def _hash_feature(self, feature_name_str):
        """Hashes a feature string to an index within FEATURE_HASH_SIZE."""
        # Use a stable hash function like SHA-1, take some bytes, convert to int
        hasher = hashlib.sha1(feature_name_str.encode('utf-8'))
        # Take first 4 bytes (adjust if needed), convert to integer
        hash_bytes = hasher.digest()[:4]
        hash_int = int.from_bytes(hash_bytes, 'big', signed=False)
        return hash_int % FEATURE_HASH_SIZE

    def _insert_features(self, features_list):
        """
        Inserts features. Expects list of tuples:
        (tid, attr, candidate_val, feature_hash_index)
        """
        if not features_list:
            logging.warning("[Compiler] No features generated to insert.")
            return 0

        unique_features_set = set(features_list)
        unique_features_list = list(unique_features_set)

        inserted_count = 0
        # --- MODIFIED SQL: Store hash index as feature identifier ---
        # Note: Assumes 'feature' column in DB is TEXT or compatible with string representation of the hash index.
        # If 'feature' must be TEXT, convert hash index to string. If it can be INT, modify schema & cast here.
        # Let's assume TEXT for now for compatibility.
        sql_insert = """
            INSERT INTO features (tid, attr, candidate_val, feature)
            VALUES (%s, %s, %s, %s::text)
            ON CONFLICT (tid, attr, candidate_val, feature) DO NOTHING;
        """
        # --- END MODIFIED SQL ---
        logging.info(f"[Compiler] Attempting to insert {len(unique_features_list)} unique hashed features...")
        insert_start_time = time.time()
        try:
            with self.db_conn.cursor() as cur:
                 # Prepare data with hash index as string
                 data_to_insert = [(f[0], f[1], f[2], str(f[3])) for f in unique_features_list]
                 psycopg2.extras.execute_batch(cur, sql_insert, data_to_insert, page_size=5000)
                 inserted_count = len(unique_features_list)
            self.db_conn.commit()
            insert_time = time.time() - insert_start_time
            logging.info(f"[Compiler] Hashed feature insertion complete in {insert_time:.2f} seconds (estimated {inserted_count} unique features targeted).")
            return inserted_count
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"[Compiler] Error inserting hashed features: {e}", exc_info=True)
            return 0


    def generate_cooccurrence_features(self):
        """
        Generates hashed co-occurrence features.
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
                 cur.itersize = 20000 # Adjust buffer size
                 cur.execute(sql)
                 for row in cur:
                     tid, attr, candidate_val_str, other_attr, other_val = row[0], row[1], str(row[2]), row[3], row[4]
                     # Construct the specific feature name before hashing
                     feature_name = f'cooc_IF_{attr}={candidate_val_str}_THEN_{other_attr}={other_val}'
                     # --- ADDED: Hash the feature name ---
                     feature_hash_index = self._hash_feature(feature_name)
                     # --- END ADDED ---
                     features_list.append((tid, attr, candidate_val_str, feature_hash_index)) # Store hash index

            duration = time.time() - start_time
            logging.info(f"[Compiler-Debug] Generated {len(features_list)} hashed co-occurrence feature candidates in {duration:.2f}s.")
            inserted = self._insert_features(features_list)
            logging.info(f"[Compiler-Debug] Inserted approx {inserted} unique hashed co-occurrence features.")
            return inserted
        except Exception as e:
            logging.error(f"Error generating hashed co-occurrence features: {e}", exc_info=True)
            return 0


    def generate_minimality_features(self):
        """Generates the minimality prior feature (NO HASHING)."""
        # We typically DON'T hash the prior feature, as we want it to have its own dedicated weight.
        logging.info("[Compiler] Generating minimality prior features (non-hashed)...")
        start_time = time.time()
        prior_feature_name = 'prior_minimality' # Keep original name
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
                (c.val IS NOT NULL AND d.candidate_val = c.val);
        """
        features_list = []
        try:
            with self.db_conn.cursor(name='minimality_feature_cursor') as cur:
                 cur.execute(sql)
                 for row in cur:
                      # Store the original feature name, not a hash
                      features_list.append((row[0], row[1], str(row[2]), prior_feature_name))

            duration = time.time() - start_time
            logging.info(f"[Compiler-Debug] Generated {len(features_list)} minimality feature candidates in {duration:.2f}s.")
            # Use specific insert for non-hashed text feature
            inserted = self._insert_text_features(features_list)
            logging.info(f"[Compiler-Debug] Inserted approx {inserted} minimality features.")
            return inserted
        except Exception as e:
            logging.error(f"Error generating minimality features: {e}", exc_info=True)
            return 0

    # Helper for inserting text features like the prior
    def _insert_text_features(self, features_list):
        if not features_list: return 0
        unique_features_set = set(features_list)
        unique_features_list = list(unique_features_set)
        sql_insert = "INSERT INTO features (tid, attr, candidate_val, feature) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;"
        try:
            with self.db_conn.cursor() as cur:
                 psycopg2.extras.execute_batch(cur, sql_insert, unique_features_list, page_size=5000)
            self.db_conn.commit()
            return len(unique_features_list)
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"Error inserting text features: {e}", exc_info=True)
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
        total_dc_features = 0
        start_time = time.time()

        logging.info("[Compiler-RelaxDC] Fetching initial cell values...")
        tuple_data_cache = defaultdict(dict)
        try:
            # Reuse cached data if already fetched? No, keep it simple for now.
            with self.db_conn.cursor(name='cell_data_cursor_dc') as cur:
                cur.itersize = 10000
                cur.execute("SELECT tid, attr, val FROM cells")
                for tid, attr, val in cur:
                     tuple_data_cache[tid][attr] = val
            logging.info(f"[Compiler-RelaxDC] Built initial value cache for {len(tuple_data_cache)} tuples.")
        except Exception as e:
            logging.error(f"Error fetching initial cell data for DC checks: {e}", exc_info=True)
            return 0

        logging.info("[Compiler-RelaxDC] Fetching candidate domains...")
        cell_domains = defaultdict(set)
        try:
            with self.db_conn.cursor(name='domain_data_cursor_dc') as cur:
                cur.itersize = 10000
                cur.execute("SELECT tid, attr, candidate_val FROM domains")
                for tid, attr, candidate_val in cur:
                     cell_domains[(tid, attr)].add(str(candidate_val))
            logging.info(f"[Compiler-RelaxDC] Fetched domains for {len(cell_domains)} cells.")
        except Exception as e:
            logging.error(f"Error fetching domains for DC checks: {e}", exc_info=True)
            return 0

        dc_features_list = []
        processed_constraints = 0

        for constraint in self.constraints:
            logging.debug(f"[Compiler-RelaxDC] Processing Constraint {constraint['id']}")
            eq_preds = [p for p in constraint['predicates'] if p['type'] == 'EQ']
            if not eq_preds: continue

            key_attr1 = eq_preds[0]['a1']
            sql_fetch_keys = f"SELECT tid, val FROM cells WHERE attr = %s AND val IS NOT NULL;"
            try:
                key_df = pd.read_sql(sql_fetch_keys, self.db_conn, params=(key_attr1,))
            except Exception as e:
                logging.warning(f"Error fetching keys for constraint {constraint['id']}, skipping: {e}")
                continue

            grouped_tids = key_df.groupby('val')['tid'].apply(list)
            initial_violating_pairs = set()

            for _, tids in grouped_tids.items():
                if len(tids) > 1:
                    for tid1, tid2 in itertools.combinations(tids, 2):
                         pair_key = tuple(sorted((tid1, tid2)))
                         if pair_key in initial_violating_pairs: continue
                         tuple1_data = tuple_data_cache.get(tid1)
                         tuple2_data = tuple_data_cache.get(tid2)
                         if not tuple1_data or not tuple2_data: continue
                         if self.detector_helper._check_violation(tuple1_data, tuple2_data, constraint):
                             initial_violating_pairs.add(pair_key)

            logging.debug(f"[Compiler-RelaxDC] Found {len(initial_violating_pairs)} initially violating pairs for constraint {constraint['id']}.")

            # Use a consistent feature name string *before* hashing
            feature_name_base = f"RelaxDC_Violates:{constraint['id']}"

            for tid1, tid2 in initial_violating_pairs:
                tuple1_orig_data = tuple_data_cache.get(tid1)
                tuple2_orig_data = tuple_data_cache.get(tid2)
                if not tuple1_orig_data or not tuple2_orig_data: continue

                involved_cells_t1 = set((tid1, p['a1']) for p in constraint['predicates'] if p['a1'] in tuple1_orig_data)
                involved_cells_t2 = set((tid2, p['a2']) for p in constraint['predicates'] if p['a2'] in tuple2_orig_data)
                all_involved_cells = involved_cells_t1.union(involved_cells_t2)

                for cell_tid, cell_attr in all_involved_cells:
                    if (cell_tid, cell_attr) not in cell_domains: continue
                    candidate_vals = cell_domains[(cell_tid, cell_attr)]
                    original_val_tuple = tuple1_orig_data if cell_tid == tid1 else tuple2_orig_data

                    for candidate_val_str in candidate_vals:
                        temp_tuple_data = original_val_tuple.copy()
                        temp_tuple_data[cell_attr] = None if candidate_val_str == NULL_REPR_PLACEHOLDER else candidate_val_str

                        # Check if *this specific candidate* still causes violation
                        violation_persists = False
                        if cell_tid == tid1:
                             violation_persists = self.detector_helper._check_violation(temp_tuple_data, tuple2_orig_data, constraint)
                        else:
                             violation_persists = self.detector_helper._check_violation(tuple1_orig_data, temp_tuple_data, constraint)

                        if violation_persists:
                             # Construct feature name including candidate value for uniqueness before hashing
                             # Note: Including candidate_val here might be redundant if hash distribution is good,
                             # but let's keep it simple: hash the base violation name.
                             # feature_name_specific = f"{feature_name_base}_IF_{cell_attr}={candidate_val_str}"
                             feature_hash_index = self._hash_feature(feature_name_base) # Hash the generic violation feature
                             dc_features_list.append((cell_tid, cell_attr, candidate_val_str, feature_hash_index))

            processed_constraints += 1

        logging.info(f"[Compiler-RelaxDC] Finished checking {processed_constraints} constraints.")
        duration = time.time() - start_time
        logging.info(f"Generated {len(dc_features_list)} raw hashed relaxed DC feature instances in {duration:.2f}s (before deduplication).")
        inserted = self._insert_features(dc_features_list)
        logging.info(f"[Compiler-Debug] Inserted approx {inserted} unique hashed relaxed DC features.")
        return inserted


    def compile_all(self):
        """Runs all feature generation steps."""
        # (Keep this method mostly as is, relies on updated generators)
        if not self._clear_features():
            logging.error("Aborting compilation due to error clearing features.")
            return
        total_features = 0
        compile_start_time = time.time()
        try:
            cooc_feat_count = self.generate_cooccurrence_features()
            min_feat_count = self.generate_minimality_features() # Keep this non-hashed
            dc_feat_count = 0
            if self.relax_constraints:
                 dc_feat_count = self.generate_relaxed_dc_features()
            else:
                 logging.warning("[Compiler] Hard constraint factor generation not implemented in this version.")

            # Get final count from DB, as inserts estimate unique features inserted per step,
            # but _insert_features uses ON CONFLICT DO NOTHING
            final_feature_count = 0
            try:
                 with self.db_conn.cursor() as cur:
                     cur.execute("SELECT COUNT(DISTINCT feature) FROM features;")
                     final_feature_count = cur.fetchone()[0]
                 logging.info(f"[Compiler] Final total unique features in table: {final_feature_count}")
            except Exception as e:
                 logging.error(f"Could not query final feature count: {e}")
                 # Sum estimates as fallback
                 total_features = cooc_feat_count + min_feat_count + dc_feat_count
                 logging.info(f"[Compiler] Estimated total unique features inserted across steps: {total_features}")
            else:
                 total_features = final_feature_count # Use actual count if query succeeded


        except Exception as e:
             logging.error(f"[Compiler] Error during feature generation: {e}", exc_info=True)
        finally:
            compile_end_time = time.time()
            logging.info(f"\n[Compiler] Feature compilation finished in {compile_end_time - compile_start_time:.2f} seconds.")


# Example usage (if run directly)
if __name__ == '__main__':
    import config
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    try:
        db_conn = psycopg2.connect(**config.DB_SETTINGS)
        logging.info("Database connection successful.")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)

    compiler = FeatureCompiler(db_conn, relax_constraints=True)
    compiler.compile_all()

    db_conn.close()
    logging.info("Database connection closed.")