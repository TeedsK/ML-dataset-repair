# File: compiler/compile.py
# Compiles signals (co-occurrence, DCs, priors) into features for the factor graph.

import pandas as pd
import time
import psycopg2
import psycopg2.extras # Needed for execute_batch
from detectors.constraints import ConstraintViolationDetector # Reuse parsing logic

class FeatureCompiler:
    """Generates features for the HoloClean probabilistic model."""

    def __init__(self, db_conn, relax_constraints=True, constraints_filepath="hospital_constraints.txt"):
        self.db_conn = db_conn
        self.relax_constraints = relax_constraints
        self.constraints_filepath = constraints_filepath
        # Load and parse constraints immediately if needed for relaxed mode
        if self.relax_constraints:
             # Use the parser from the detector module
             parser = ConstraintViolationDetector(db_conn, constraints_filepath)
             self.constraints = parser.constraints # Reuse parsed constraints
        else:
             self.constraints = []


    def _clear_features(self):
        """Clears existing features from the table."""
        print("[Compiler] Clearing existing features...")
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("DELETE FROM features;")
            self.db_conn.commit()
            print("Features table cleared.")
            return True
        except Exception as e:
            self.db_conn.rollback()
            print(f"[Compiler] Error clearing features table: {e}")
            return False

    def _insert_features(self, features_list):
        """Inserts a list of features into the database."""
        if not features_list:
            print("[Compiler] No features to insert.")
            return 0

        inserted_count = 0
        sql_insert = """
            INSERT INTO features (tid, attr, candidate_val, feature)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (tid, attr, candidate_val, feature) DO NOTHING;
        """
        print(f"[Compiler] Inserting {len(features_list)} generated features...")
        insert_start_time = time.time()
        try:
            with self.db_conn.cursor() as cur:
                 # Using execute_batch for efficiency
                 psycopg2.extras.execute_batch(cur, sql_insert, features_list, page_size=1000)
                 # Cannot easily get rowcount from execute_batch, estimate based on list size
                 inserted_count = len(features_list) # Approximation, actual may be less due to conflicts
            self.db_conn.commit()
            insert_time = time.time() - insert_start_time
            print(f"[Compiler] Feature insertion complete in {insert_time:.2f} seconds (estimated {inserted_count} insertions).")
            return inserted_count # Return estimate
        except Exception as e:
            self.db_conn.rollback()
            print(f"[Compiler] Error inserting features: {e}")
            raise # Re-raise error after rollback


    def generate_cooccurrence_features(self):
        """Generates features based on co-occurrence of candidate values with other cell values in the tuple."""
        print("[Compiler] Generating co-occurrence features...")
        start_time = time.time()
        # This feature connects a candidate 'd' for cell (t, a) with the value 'v_prime'
        # of another cell (t, a_prime) in the same tuple.
        # Feature name format: "cooc_{a_prime}={v_prime}"
        # Requires joining 'domains' with 'cells'
        sql = """
            SELECT
                d.tid,
                d.attr,
                d.candidate_val,
                'cooc_' || c.attr || '=' || c.val AS feature_name
            FROM domains d
            JOIN cells c ON d.tid = c.tid AND d.attr <> c.attr
            WHERE c.val IS NOT NULL; -- Only create features for non-null co-occurring values
        """
        try:
            features_df = pd.read_sql(sql, self.db_conn)
            features_list = [tuple(x) for x in features_df.to_numpy()]
            duration = time.time() - start_time
            print(f"Generated {len(features_list)} co-occurrence feature candidates in {duration:.2f}s.")
            return self._insert_features(features_list)
        except Exception as e:
            print(f"Error generating co-occurrence features: {e}")
            return 0


    def generate_minimality_features(self):
        """Generates features favoring the original value (minimality prior)."""
        print("[Compiler] Generating minimality prior features...")
        start_time = time.time()
        # This feature connects a candidate value 'd' for cell (t, a) to the fact
        # that 'd' was the original value. Feature name: "prior_minimality"
        # Requires joining 'domains' with 'cells' where candidate matches original value.
        sql = """
            SELECT
                d.tid,
                d.attr,
                d.candidate_val,
                'prior_minimality' AS feature_name
            FROM domains d
            JOIN cells c ON d.tid = c.tid AND d.attr = c.attr
            WHERE d.candidate_val = c.val; -- Only add feature if the candidate IS the original value
        """
        try:
            features_df = pd.read_sql(sql, self.db_conn)
            features_list = [tuple(x) for x in features_df.to_numpy()]
            duration = time.time() - start_time
            print(f"Generated {len(features_list)} minimality feature candidates in {duration:.2f}s.")
            return self._insert_features(features_list)
        except Exception as e:
            print(f"Error generating minimality features: {e}")
            return 0


    def generate_relaxed_dc_features(self):
        """Generates features based on potential DC violations using initial values (relaxed approach)."""
        if not self.relax_constraints:
            print("[Compiler] Skipping relaxed DC features (relax_constraints=False).")
            return 0
        if not self.constraints:
             print("[Compiler] No constraints parsed. Cannot generate relaxed DC features.")
             return 0

        print("[Compiler] Generating relaxed denial constraint features...")
        total_dc_features = 0
        start_time = time.time()

        # Reuse the checking logic from the detector, but apply it to *initial* values
        # For each pair (t1, t2) that *would* violate a constraint based on initial values,
        # add features for the cells involved in that violation.

        # Fetch all cell data once
        try:
            all_cells_df = pd.read_sql("SELECT tid, attr, val FROM cells WHERE val IS NOT NULL", self.db_conn)
            all_cells_df.set_index('tid', inplace=True)
             # Convert to a dictionary for faster lookups: {tid: {attr: val}}
            tuple_data_cache = {}
            for tid, group in all_cells_df.groupby(level=0):
                 tuple_data_cache[tid] = group.set_index('attr')['val'].to_dict()
            print("Created tuple data cache for DC checks.")
        except Exception as e:
            print(f"Error fetching data for DC checks: {e}")
            return 0

        dc_features_list = []
        for constraint in self.constraints:
             print(f"Processing relaxed features for constraint {constraint['id']}...")
             # --- Find violating pairs based on initial values ---
             # (This logic mirrors the detector's check violation part)
             eq_preds = [p for p in constraint['predicates'] if p['type'] == 'EQ']
             if not eq_preds: continue # Skip if no EQ predicate for simple filtering

             key_attr1 = eq_preds[0]['a1']
             sql_fetch_keys = f"SELECT tid, val FROM cells WHERE attr = %s AND val IS NOT NULL;"
             try:
                 key_df = pd.read_sql(sql_fetch_keys, self.db_conn, params=(key_attr1,))
             except Exception as e: continue # Skip constraint on error

             grouped_tids = key_df.groupby('val')['tid'].apply(list)

             violating_pairs = set()
             for key_val, tids in grouped_tids.items():
                 if len(tids) > 1:
                     for i in range(len(tids)):
                         for j in range(i + 1, len(tids)):
                             tid1, tid2 = tids[i], tids[j]
                             pair_key = tuple(sorted((tid1, tid2)))
                             if pair_key in violating_pairs: continue

                             tuple1_data = tuple_data_cache.get(tid1, {})
                             tuple2_data = tuple_data_cache.get(tid2, {})
                             if not tuple1_data or not tuple2_data: continue

                             # Use the _check_violation logic from the detector class (or reimplement)
                             # We need an instance or make it static/standalone
                             # For simplicity, let's assume ConstraintViolationDetector has check_violation
                             temp_parser = ConstraintViolationDetector(self.db_conn, self.constraints_filepath)
                             if temp_parser._check_violation(tuple1_data, tuple2_data, constraint):
                                violating_pairs.add(pair_key)
                                # Add features for the cells involved in this specific violation
                                # Feature indicates "this cell was involved in a potential violation of DC X"
                                feature_name = f"dc_relax_{constraint['id']}"
                                for pred in constraint['predicates']:
                                     # Add feature for t1's cell in the predicate
                                     if pred['a1'] in tuple1_data:
                                         original_val1 = tuple1_data[pred['a1']]
                                         # Crucially, link feature to the *original* value's entry in domains
                                         dc_features_list.append((tid1, pred['a1'], original_val1, feature_name))
                                     # Add feature for t2's cell in the predicate
                                     if pred['a2'] in tuple2_data:
                                         original_val2 = tuple2_data[pred['a2']]
                                         dc_features_list.append((tid2, pred['a2'], original_val2, feature_name))

        # Remove duplicates before inserting
        unique_dc_features = list(set(dc_features_list))
        duration = time.time() - start_time
        print(f"Generated {len(unique_dc_features)} unique relaxed DC feature candidates in {duration:.2f}s.")
        total_dc_features = self._insert_features(unique_dc_features)
        return total_dc_features


    def compile_all(self):
        """Runs all feature generation steps."""
        if not self._clear_features():
            print("Aborting compilation due to error clearing features.")
            return

        total_features = 0
        total_features += self.generate_cooccurrence_features()
        total_features += self.generate_minimality_features()
        if self.relax_constraints:
             total_features += self.generate_relaxed_dc_features()
        else:
             print("[Compiler] Hard constraint factor generation not implemented in this version.")
             # Add call to generate hard factors here if implemented

        print(f"\n[Compiler] Total features inserted (estimated): {total_features}")