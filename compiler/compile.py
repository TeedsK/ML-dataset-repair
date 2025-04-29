# File: compiler/compile.py
# Compiles signals into features using UNIQUE FEATURE NAMES (removed hashing).

import pandas as pd
import time
import psycopg2
import psycopg2.extras
from collections import defaultdict
import itertools
import logging
# import hashlib # No longer needed for primary feature hashing

from detectors.constraints import ConstraintViolationDetector

NULL_REPR_PLACEHOLDER = "__NULL__"
# FEATURE_HASH_SIZE = 5000 # No longer used for feature names


class FeatureCompiler:
    """Generates features for the HoloClean probabilistic model using unique names."""

    def __init__(self, db_conn, relax_constraints=True, constraints_filepath="hospital_constraints.txt"):
        self.db_conn = db_conn
        self.relax_constraints = relax_constraints # Keep flag, but logic might need more work
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

    # No longer needed: _hash_feature

    def _insert_features(self, features_list):
        """
        Inserts features. Expects list of tuples:
        (tid, attr, candidate_val, feature_name_string)
        """
        if not features_list:
            logging.warning("[Compiler] No features generated to insert.")
            return 0

        # Deduplicate features before insertion
        unique_features_set = set(features_list)
        unique_features_list = list(unique_features_set)

        inserted_count = 0
        # --- MODIFIED SQL: Store unique feature name string ---
        sql_insert = """
            INSERT INTO features (tid, attr, candidate_val, feature)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (tid, attr, candidate_val, feature) DO NOTHING;
        """
        # --- END MODIFIED SQL ---
        logging.info(f"[Compiler] Attempting to insert {len(unique_features_list)} unique named features...")
        insert_start_time = time.time()
        try:
            with self.db_conn.cursor() as cur:
                 # Data should already be (tid, attr, cand_val, feature_name_str)
                 psycopg2.extras.execute_batch(cur, sql_insert, unique_features_list, page_size=10000) # Increased page size
                 inserted_count = len(unique_features_list) # This is approximate due to ON CONFLICT
            self.db_conn.commit()
            insert_time = time.time() - insert_start_time
            logging.info(f"[Compiler] Named feature insertion complete in {insert_time:.2f} seconds ({inserted_count} unique features targeted).")
            # We can't easily get exact inserted count with ON CONFLICT, but this is the target count
            return inserted_count
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"[Compiler] Error inserting named features: {e}", exc_info=True)
            return 0


    def generate_cooccurrence_features(self):
        """
        Generates named co-occurrence features.
        Feature Name: cooc_{attr}={candidate_val}_{other_attr}={other_val}
        """
        logging.info("[Compiler] Generating NAMED co-occurrence features...")
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
            # Use a server-side cursor for potentially large results
            with self.db_conn.cursor(name='named_cooc_feature_cursor') as cur:
                 cur.itersize = 50000 # Adjust buffer size
                 cur.execute(sql)
                 processed_rows = 0
                 for row in cur:
                     tid, attr, candidate_val_str, other_attr, other_val_str = row[0], row[1], str(row[2]), row[3], str(row[4])
                     # --- Use unique feature name ---
                     # Sanitize values for feature names if they contain special characters? For now, assume simple strings.
                     # Consider length limits if names become too long.
                     feature_name = f'cooc_{attr}={candidate_val_str}_{other_attr}={other_val_str}'
                     # --- END Use unique feature name ---
                     features_list.append((tid, attr, candidate_val_str, feature_name))
                     processed_rows += 1
                     # Optional: Insert in batches within the loop if memory becomes an issue
                     # if len(features_list) >= BATCH_SIZE:
                     #     self._insert_features(features_list)
                     #     features_list = []

            # Insert any remaining features
            inserted = self._insert_features(features_list)

            duration = time.time() - start_time
            logging.info(f"[Compiler] Generated {len(features_list)} named co-occurrence feature instances from {processed_rows} DB rows in {duration:.2f}s.")
            logging.info(f"[Compiler] Inserted approx {inserted} unique named co-occurrence features.")
            return inserted
        except Exception as e:
            logging.error(f"Error generating named co-occurrence features: {e}", exc_info=True)
            return 0


    def generate_minimality_features(self):
        """Generates the minimality prior feature (unique name)."""
        # This feature typically has a single, non-hashed name.
        logging.info("[Compiler] Generating minimality prior features...")
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
                -- Check if candidate matches original value (handling NULLs)
                (c.val IS NULL AND d.candidate_val = '{NULL_REPR_PLACEHOLDER}')
                OR
                (c.val IS NOT NULL AND d.candidate_val = CAST(c.val AS TEXT)); -- Ensure comparison as text
        """
        features_list = []
        try:
            with self.db_conn.cursor(name='minimality_feature_cursor') as cur:
                 cur.execute(sql)
                 processed_rows = 0
                 for row in cur:
                      # Store the original feature name
                      features_list.append((row[0], row[1], str(row[2]), prior_feature_name))
                      processed_rows += 1

            # Insert using the general insert method now
            inserted = self._insert_features(features_list)

            duration = time.time() - start_time
            logging.info(f"[Compiler] Generated {len(features_list)} minimality feature instances from {processed_rows} DB rows in {duration:.2f}s.")
            logging.info(f"[Compiler] Inserted approx {inserted} minimality features (should be 1 unique name).")
            return inserted
        except Exception as e:
            logging.error(f"Error generating minimality features: {e}", exc_info=True)
            return 0


    def generate_relaxed_dc_features(self):
        """
        Generates NAMED features based on potential DC violations (using original relaxed logic).
        Feature Name: RelaxDCViolates:{constraint_id}:Cell={tid},{attr}:Cand={candidate_val}
        """
        if not self.relax_constraints:
            logging.info("[Compiler] Skipping relaxed DC features (relax_constraints=False).")
            return 0
        if not self.constraints or self.detector_helper is None:
             logging.warning("[Compiler] No constraints loaded/parsed or detector helper missing. Cannot generate relaxed DC features.")
             return 0

        logging.info("[Compiler] Generating NAMED relaxed denial constraint features...")
        total_dc_features_generated = 0
        start_time = time.time()

        logging.info("[Compiler-RelaxDC] Fetching initial cell values...")
        tuple_data_cache = defaultdict(dict)
        try:
            # Fetch data including NULLs
            with self.db_conn.cursor(name='cell_data_cursor_dc') as cur:
                cur.itersize = 50000
                cur.execute("SELECT tid, attr, val FROM cells")
                for tid, attr, val in cur:
                     # Use placeholder for NULLs in the cache
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
            constraint_id_str = str(constraint['id']) # Use for feature name
            logging.debug(f"[Compiler-RelaxDC] Processing Constraint {constraint_id_str}")

            # Find pairs initially violating the constraint (reusing detector logic)
            # Note: This part can be slow if not optimized. Consider pre-calculating violations.
            # Let's reuse the pair finding logic from your original code for now.
            eq_preds = [p for p in constraint['predicates'] if p['type'] == 'EQ']
            if not eq_preds: continue
            key_attr1 = eq_preds[0]['a1']
            sql_fetch_keys = f"SELECT tid, val FROM cells WHERE attr = %s AND val IS NOT NULL;"
            try:
                 # Use pandas read_sql directly, ensure correct type handling
                 key_df = pd.read_sql(sql_fetch_keys, self.db_conn, params=(key_attr1,))
                 key_df['tid'] = key_df['tid'].astype(int) # Ensure tid is int
            except Exception as e:
                 logging.warning(f"Error fetching keys for constraint {constraint_id_str}, skipping: {e}")
                 continue

            grouped_tids = key_df.groupby('val')['tid'].apply(list)
            initial_violating_pairs = set()
            checked_pairs = set() # Avoid redundant checks

            for _, tids in grouped_tids.items():
                if len(tids) > 1:
                    for tid1, tid2 in itertools.combinations(tids, 2):
                         pair_key = tuple(sorted((tid1, tid2)))
                         if pair_key in checked_pairs: continue
                         checked_pairs.add(pair_key)

                         tuple1_data = tuple_data_cache.get(tid1)
                         tuple2_data = tuple_data_cache.get(tid2)
                         if not tuple1_data or not tuple2_data: continue

                         # Need to convert tuple_data_cache values (which might include NULL_REPR)
                         # back to None for _check_violation if it expects None.
                         # Assuming _check_violation handles string placeholder for now.
                         if self.detector_helper._check_violation(tuple1_data, tuple2_data, constraint):
                             initial_violating_pairs.add(pair_key)

            logging.debug(f"[Compiler-RelaxDC] Found {len(initial_violating_pairs)} initially violating pairs for constraint {constraint_id_str}.")


            for tid1, tid2 in initial_violating_pairs:
                tuple1_orig_data = tuple_data_cache.get(tid1)
                tuple2_orig_data = tuple_data_cache.get(tid2)
                if not tuple1_orig_data or not tuple2_orig_data: continue

                # Determine which cells are involved in this specific constraint
                involved_cells_t1 = set((tid1, p['a1']) for p in constraint['predicates'] if p['a1'] in tuple1_orig_data)
                involved_cells_t2 = set((tid2, p['a2']) for p in constraint['predicates'] if p['a2'] in tuple2_orig_data)
                # Also include potential key attributes if not already listed
                for p in eq_preds:
                    if p['a1'] in tuple1_orig_data: involved_cells_t1.add((tid1, p['a1']))
                    if p['a2'] in tuple2_orig_data: involved_cells_t2.add((tid2, p['a2']))

                all_involved_cells = involved_cells_t1.union(involved_cells_t2)

                for cell_tid, cell_attr in all_involved_cells:
                    if (cell_tid, cell_attr) not in cell_domains: continue
                    candidate_vals = cell_domains[(cell_tid, cell_attr)]
                    original_val_tuple = tuple1_orig_data if cell_tid == tid1 else tuple2_orig_data

                    for candidate_val_str in candidate_vals:
                        # Create a temporary tuple state with the candidate value
                        temp_tuple_data = original_val_tuple.copy()
                        # Important: Convert back to actual Python None if needed by _check_violation
                        # Assuming _check_violation can handle the string placeholder for now
                        temp_tuple_data[cell_attr] = candidate_val_str # Keep as string

                        # Check if *this specific candidate* still causes violation between the original pair
                        violation_persists = False
                        if cell_tid == tid1:
                             violation_persists = self.detector_helper._check_violation(temp_tuple_data, tuple2_orig_data, constraint)
                        else: # cell_tid == tid2
                             violation_persists = self.detector_helper._check_violation(tuple1_orig_data, temp_tuple_data, constraint)

                        if violation_persists:
                             # --- Use unique feature name ---
                             # Create a distinct feature name for this specific violation scenario
                             # (Constraint ID + Cell + Candidate Value causing continued violation)
                             # Sanitize candidate_val_str if needed for feature name validity
                             feature_name = f"RelaxDCViolates:{constraint_id_str}:Cell={cell_tid},{cell_attr}:Cand={candidate_val_str}"
                             # --- END Use unique feature name ---
                             dc_features_list.append((cell_tid, cell_attr, candidate_val_str, feature_name))

            processed_constraints += 1

        logging.info(f"[Compiler-RelaxDC] Finished checking {processed_constraints} constraints.")
        duration = time.time() - start_time
        logging.info(f"Generated {len(dc_features_list)} raw named relaxed DC feature instances in {duration:.2f}s (before deduplication).")
        inserted = self._insert_features(dc_features_list)
        logging.info(f"[Compiler] Inserted approx {inserted} unique named relaxed DC features.")
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
            min_feat_count = self.generate_minimality_features() # Still unique name
            dc_feat_count = 0
            if self.relax_constraints:
                 # Note: This uses the simplified "relaxed" logic.
                 # Implementing full violation checks against all tuples is more complex.
                 dc_feat_count = self.generate_relaxed_dc_features()
            else:
                 logging.warning("[Compiler] Hard constraint factor generation not implemented in this version.")

            # Get final count from DB for a more accurate picture
            final_feature_count = 0
            try:
                 with self.db_conn.cursor() as cur:
                     # Count distinct text features in the table
                     cur.execute("SELECT COUNT(DISTINCT feature) FROM features;")
                     final_feature_count = cur.fetchone()[0]
                 logging.info(f"[Compiler] Final total unique features in table: {final_feature_count}")
            except Exception as e:
                 logging.error(f"Could not query final feature count: {e}")
                 # Use sum of estimates as fallback, though less accurate now
                 total_features_est = cooc_feat_count + min_feat_count + dc_feat_count
                 logging.info(f"[Compiler] Estimated total unique features inserted across steps: {total_features_est}")
            else:
                 total_features_est = final_feature_count # Use actual count

        except Exception as e:
             logging.error(f"[Compiler] Error during feature generation: {e}", exc_info=True)
        finally:
            compile_end_time = time.time()
            logging.info(f"\n[Compiler] Feature compilation finished in {compile_end_time - compile_start_time:.2f} seconds.")


# --- Main execution block ---
if __name__ == '__main__':
    import config
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Example: Read relax_constraints from config or default to True
    relax = getattr(config, 'RELAX_CONSTRAINTS', True)
    constraints_file = getattr(config, 'CONSTRAINTS_FILE', 'hospital_constraints.txt')

    logging.info(f"Running Feature Compiler (compile.py) with Relax Constraints = {relax}")

    # --- Database Connection ---
    try:
        db_conn = psycopg2.connect(**config.DB_SETTINGS)
        logging.info("Database connection successful.")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)

    # --- Run Compiler ---
    try:
        # Pass constraints file path
        compiler = FeatureCompiler(db_conn, relax_constraints=relax, constraints_filepath=constraints_file)
        compiler.compile_all()
    except Exception as e:
        logging.error(f"An unexpected error occurred during compilation: {e}", exc_info=True)
    finally:
        # --- Close Connection ---
        if db_conn:
            db_conn.close()
            logging.info("Database connection closed.")