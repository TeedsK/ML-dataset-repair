# File: compiler/compile.py (Corrected for None Handling)
# Compiles signals (co-occurrence, DCs, priors) into features for the factor graph.

import pandas as pd
import time
import psycopg2
import psycopg2.extras # Needed for execute_batch
from detectors.constraints import ConstraintViolationDetector # Reuse parsing logic

# Define the SAME placeholder used in pruning.py
NULL_REPR_PLACEHOLDER = "__NULL__"

class FeatureCompiler:
    """Generates features for the HoloClean probabilistic model."""

    def __init__(self, db_conn, relax_constraints=True, constraints_filepath="hospital_constraints.txt"):
        self.db_conn = db_conn
        self.relax_constraints = relax_constraints
        self.constraints_filepath = constraints_filepath
        # Load and parse constraints immediately if needed for relaxed mode
        if self.relax_constraints:
             parser = ConstraintViolationDetector(db_conn, constraints_filepath)
             self.constraints = parser.constraints
        else:
             self.constraints = []


    def _clear_features(self):
        """Clears existing features from the table."""
        # (Keep this method as is)
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
        # (Keep this method as is)
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
                 psycopg2.extras.execute_batch(cur, sql_insert, features_list, page_size=1000)
                 inserted_count = len(features_list)
            self.db_conn.commit()
            insert_time = time.time() - insert_start_time
            print(f"[Compiler] Feature insertion complete in {insert_time:.2f} seconds (estimated {inserted_count} insertions).")
            return inserted_count
        except Exception as e:
            self.db_conn.rollback()
            print(f"[Compiler] Error inserting features: {e}")
            raise


    def generate_cooccurrence_features(self):
        """Generates features based on co-occurrence of candidate values with other cell values in the tuple."""
        # This function should be mostly okay, as it joins domains (which now contain the placeholder)
        # with cells (where val IS NOT NULL). It implicitly won't generate co-occurrence features
        # involving a NULL value in the *other* cell (c.val), which is reasonable.
        # We assume candidate_val in domains is already the correct string/placeholder.
        print("[Compiler] Generating co-occurrence features...")
        start_time = time.time()
        sql = """
            SELECT
                d.tid,
                d.attr,
                d.candidate_val, -- Assumes this correctly contains placeholder string if needed
                'cooc_' || c.attr || '=' || c.val AS feature_name -- c.val is guaranteed not null here
            FROM domains d
            JOIN cells c ON d.tid = c.tid AND d.attr <> c.attr
            WHERE c.val IS NOT NULL; -- Only co-occur with non-null values
        """
        try:
            features_df = pd.read_sql(sql, self.db_conn)
            # Ensure candidate_val is treated as string if it came from DB
            features_list = [(row[0], row[1], str(row[2]), row[3]) for row in features_df.to_numpy()]
            duration = time.time() - start_time
            print(f"[Compiler-Debug] Generated {len(features_list)} co-occurrence feature candidates in {duration:.2f}s.")
            inserted = self._insert_features(features_list)
            print(f"[Compiler-Debug] Inserted approx {inserted} co-occurrence features.")
            return inserted
        except Exception as e:
            print(f"Error generating co-occurrence features: {e}")
            import traceback
            traceback.print_exc()
            return 0


    def generate_minimality_features(self):
        """Generates features favoring the original value (minimality prior)."""
        print("[Compiler] Generating minimality prior features...")
        start_time = time.time()
        # --- MODIFIED SQL to handle NULL original values ---
        # Check case where original cell value was NULL and domain candidate is the placeholder
        # OR check case where original cell value was not NULL and domain candidate matches it
        sql = f"""
            SELECT
                d.tid,
                d.attr,
                d.candidate_val,
                'prior_minimality' AS feature_name
            FROM domains d
            JOIN cells c ON d.tid = c.tid AND d.attr = c.attr
            WHERE
                (c.val IS NULL AND d.candidate_val = '{NULL_REPR_PLACEHOLDER}')
                OR
                (c.val IS NOT NULL AND d.candidate_val = c.val);
        """
        # --- END MODIFIED SQL ---
        try:
            features_df = pd.read_sql(sql, self.db_conn)
            # Ensure candidate_val is treated as string
            features_list = [(row[0], row[1], str(row[2]), row[3]) for row in features_df.to_numpy()]
            duration = time.time() - start_time
            print(f"[Compiler-Debug] Generated {len(features_list)} minimality feature candidates.")
            inserted = self._insert_features(features_list)
            print(f"[Compiler-Debug] Inserted approx {inserted} minimality features.")
            return inserted
        except Exception as e:
            print(f"Error generating minimality features: {e}")
            import traceback
            traceback.print_exc()
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

        # Fetch all cell data once, including NULLs, for checking original values
        try:
            # Fetch including nulls this time
            all_cells_df = pd.read_sql("SELECT tid, attr, val FROM cells", self.db_conn)
            all_cells_df.set_index('tid', inplace=True)
            tuple_data_cache = {}
            for tid, group in all_cells_df.groupby(level=0):
                 # Store None directly in cache
                 tuple_data_cache[tid] = group.set_index('attr')['val'].to_dict()
            print("Created tuple data cache for DC checks (includes None).")
        except Exception as e:
            print(f"Error fetching data for DC checks: {e}")
            return 0

        dc_features_list = []
        total_violating_pairs = 0
        for constraint in self.constraints:
             constraint_violations = 0
             # print(f"Processing relaxed features for constraint {constraint['id']}...") # Verbose
             eq_preds = [p for p in constraint['predicates'] if p['type'] == 'EQ']
             if not eq_preds: continue

             key_attr1 = eq_preds[0]['a1']
             # Fetch only non-null keys for grouping potential pairs efficiently
             sql_fetch_keys = f"SELECT tid, val FROM cells WHERE attr = %s AND val IS NOT NULL;"
             try:
                 key_df = pd.read_sql(sql_fetch_keys, self.db_conn, params=(key_attr1,))
             except Exception as e: continue

             grouped_tids = key_df.groupby('val')['tid'].apply(list)
             violating_pairs = set()

             for key_val, tids in grouped_tids.items():
                 if len(tids) > 1:
                     for i in range(len(tids)):
                         for j in range(i + 1, len(tids)):
                             tid1, tid2 = tids[i], tids[j]
                             pair_key = tuple(sorted((tid1, tid2)))
                             if pair_key in violating_pairs: continue

                             # Use the full tuple cache (which includes None) for violation check
                             tuple1_data = tuple_data_cache.get(tid1, {})
                             tuple2_data = tuple_data_cache.get(tid2, {})
                             if not tuple1_data or not tuple2_data: continue

                             temp_parser = ConstraintViolationDetector(self.db_conn, self.constraints_filepath)
                             if temp_parser._check_violation(tuple1_data, tuple2_data, constraint):
                                violating_pairs.add(pair_key)
                                constraint_violations += 1
                                feature_name = f"dc_relax_{constraint['id']}"

                                for pred in constraint['predicates']:
                                     # --- MODIFIED: Use consistent representation for None ---
                                     if pred['a1'] in tuple1_data:
                                         original_val1 = tuple1_data[pred['a1']]
                                         # Use placeholder if original value was None
                                         val1_repr = NULL_REPR_PLACEHOLDER if original_val1 is None else str(original_val1)
                                         dc_features_list.append((tid1, pred['a1'], val1_repr, feature_name))
                                     if pred['a2'] in tuple2_data:
                                         original_val2 = tuple2_data[pred['a2']]
                                         # Use placeholder if original value was None
                                         val2_repr = NULL_REPR_PLACEHOLDER if original_val2 is None else str(original_val2)
                                         dc_features_list.append((tid2, pred['a2'], val2_repr, feature_name))
                                     # --- END MODIFIED ---

             # print(f"[Compiler-Debug] Found {constraint_violations} violating pairs for constraint {constraint['id']}.") # Verbose
             total_violating_pairs += constraint_violations

        print(f"[Compiler-Debug] Total violating pairs found across all DCs: {total_violating_pairs}")
        unique_dc_features = list(set(dc_features_list))
        duration = time.time() - start_time
        print(f"Generated {len(unique_dc_features)} unique relaxed DC feature candidates in {duration:.2f}s.")
        inserted = self._insert_features(unique_dc_features)
        print(f"[Compiler-Debug] Inserted approx {inserted} relaxed DC features.")
        return inserted


    def compile_all(self):
        """Runs all feature generation steps."""
        # (Keep this method as is)
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
        print(f"\n[Compiler] Total features inserted (estimated): {total_features}")