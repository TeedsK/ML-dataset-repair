# File: compiler/compile.py
# VERSION 6: Refines DC features (includes attribute) and adds frequency features.

import numpy as np
import pandas as pd
import time
import psycopg2
import psycopg2.extras
from collections import defaultdict, Counter
import itertools
import logging
import gc # Garbage collector
import math

from tqdm import tqdm # For log frequency

from detectors.constraints import ConstraintViolationDetector

NULL_REPR_PLACEHOLDER = "__NULL__"

# Define bins for log frequency features (adjust as needed)
# Bins represent ranges of log10(frequency+1)
FREQ_BINS = [0, 1, 2, 3, 4, 5] # Corresponds to freq ranges ~0, 1-9, 10-99, 100-999, etc.

# --- Add Cells to Log Detailed DC Info ---
# Keep the set for debugging specific examples if needed
DETAILED_LOG_CELLS = {
    (75, 'PhoneNumber'), (9, 'ZipCode'), (5, 'MeasureCode'), # False Positives
    (72, 'ZipCode'), (16, 'City'), (99, 'MeasureName'), (61, 'MeasureName') # False Negatives
}
# --- End Cells to Log ---


class FeatureCompiler:
    """Generates features for the HoloClean probabilistic model using unique names."""

    def __init__(self, db_conn, relax_constraints=True, constraints_filepath="hospital_constraints.txt"):
        self.db_conn = db_conn
        self.constraints_filepath = constraints_filepath
        self.constraints = []
        self.detector_helper = None
        self.tuple_data_cache = None
        self.cell_domains = None
        self.dc_indexes = {}
        # --- Added cache for global frequencies ---
        self.global_value_counts = defaultdict(Counter)
        # --- End Added cache ---

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

    def _load_data_and_stats(self):
        """Loads tuple data, domains, and calculates global value counts."""
        # Use existing cache if available
        if self.tuple_data_cache is not None and self.cell_domains is not None and self.global_value_counts:
             logging.info("[Compiler-Data] Using cached tuple data, domains, and stats.")
             return True

        logging.info("[Compiler-Data] Fetching data and calculating stats...")
        load_start = time.time()
        self.tuple_data_cache = defaultdict(dict)
        self.cell_domains = defaultdict(set)
        self.global_value_counts = defaultdict(Counter) # Reset counts

        try:
            # Fetch tuple data and calculate global counts simultaneously
            with self.db_conn.cursor(name='cell_data_cursor_v5') as cur:
                cur.itersize = 50000
                cur.execute("SELECT tid, attr, val FROM cells")
                for tid, attr, val in cur:
                     val_str = NULL_REPR_PLACEHOLDER if pd.isna(val) else str(val)
                     self.tuple_data_cache[int(tid)][attr] = val_str
                     # Count non-null values for frequency features
                     if val_str != NULL_REPR_PLACEHOLDER:
                          self.global_value_counts[attr][val_str] += 1
            logging.info(f"[Compiler-Data] Built initial value cache for {len(self.tuple_data_cache)} tuples.")
            logging.info(f"[Compiler-Data] Calculated global value counts for {len(self.global_value_counts)} attributes.")

            # Fetch domains
            with self.db_conn.cursor(name='domain_data_cursor_v5') as cur:
                cur.itersize = 50000
                cur.execute("SELECT tid, attr, candidate_val FROM domains")
                for tid, attr, candidate_val in cur:
                     cand_str = str(candidate_val)
                     if cand_str != NULL_REPR_PLACEHOLDER:
                         self.cell_domains[(int(tid), attr)].add(cand_str)
            logging.info(f"[Compiler-Data] Fetched domains for {len(self.cell_domains)} cells.")
            load_end = time.time()
            logging.info(f"[Compiler-Data] Data loading and stats calculation took {load_end - load_start:.2f}s.")
            return True

        except Exception as e:
            logging.error(f"Error fetching data/calculating stats: {e}", exc_info=True)
            self.tuple_data_cache = None
            self.cell_domains = None
            self.global_value_counts = defaultdict(Counter)
            return False

    def _build_dc_indexes(self):
        # (Implementation remains the same as V4)
        if not self.tuple_data_cache: return
        if self.dc_indexes: return
        logging.info("[Compiler-DC] Building helper indexes from cached data...")
        build_start = time.time()
        self.dc_indexes['by_zip'] = defaultdict(list)
        self.dc_indexes['by_phone'] = defaultdict(list)
        self.dc_indexes['by_measurecode'] = defaultdict(list)
        self.dc_indexes['by_provider_measure'] = defaultdict(list)
        self.dc_indexes['by_state_measure'] = defaultdict(list)
        for tid, data in self.tuple_data_cache.items():
            zip_val = data.get('ZipCode')
            if zip_val and zip_val != NULL_REPR_PLACEHOLDER: self.dc_indexes['by_zip'][zip_val].append(tid)
            phone_val = data.get('PhoneNumber')
            if phone_val and phone_val != NULL_REPR_PLACEHOLDER: self.dc_indexes['by_phone'][phone_val].append(tid)
            measure_val = data.get('MeasureCode')
            if measure_val and measure_val != NULL_REPR_PLACEHOLDER: self.dc_indexes['by_measurecode'][measure_val].append(tid)
            prov_val = data.get('ProviderNumber')
            state_val = data.get('State')
            if prov_val and prov_val != NULL_REPR_PLACEHOLDER and measure_val and measure_val != NULL_REPR_PLACEHOLDER:
                 self.dc_indexes['by_provider_measure'][(prov_val, measure_val)].append(tid)
            if state_val and state_val != NULL_REPR_PLACEHOLDER and measure_val and measure_val != NULL_REPR_PLACEHOLDER:
                 self.dc_indexes['by_state_measure'][(state_val, measure_val)].append(tid)
        build_end = time.time()
        logging.info(f"[Compiler-DC] Finished building indexes in {build_end - build_start:.2f}s.")
        for name, index in self.dc_indexes.items(): logging.debug(f"  Index '{name}' size: {len(index)} keys")

    def _insert_features(self, features_list):
        # (Implementation remains the same)
        if not features_list:
            logging.warning("[Compiler] No features generated to insert.")
            return 0
        unique_features_set = set(features_list)
        unique_features_list = list(unique_features_set)
        inserted_count = 0
        sql_insert = """INSERT INTO features (tid, attr, candidate_val, feature) VALUES (%s, %s, %s, %s) ON CONFLICT (tid, attr, candidate_val, feature) DO NOTHING;"""
        logging.info(f"[Compiler] Attempting to insert {len(unique_features_list)} unique named features...")
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
            logging.error(f"[Compiler] Error inserting named features: {e}", exc_info=True)
            return 0

    def generate_cooccurrence_features(self):
        # (Implementation remains the same)
        logging.info("[Compiler] Generating NAMED co-occurrence features...")
        start_time = time.time()
        sql = f"""
            SELECT d.tid, d.attr, d.candidate_val, c.attr AS other_attr, c.val AS other_val
            FROM domains d JOIN cells c ON d.tid = c.tid AND d.attr <> c.attr
            WHERE c.val IS NOT NULL AND d.candidate_val <> '{NULL_REPR_PLACEHOLDER}';
        """
        features_list = []
        try:
            with self.db_conn.cursor(name='named_cooc_feature_cursor_v5') as cur:
                 cur.itersize = 50000
                 cur.execute(sql)
                 processed_rows = 0
                 for row in cur:
                     tid, attr, candidate_val_str, other_attr, other_val_str = int(row[0]), row[1], str(row[2]), row[3], str(row[4])
                     feature_name = f'cooc_{attr}={candidate_val_str}_{other_attr}={other_val_str}'
                     features_list.append((tid, attr, candidate_val_str, feature_name))
                     processed_rows += 1
            inserted = self._insert_features(features_list)
            duration = time.time() - start_time
            logging.info(f"[Compiler] Generated {len(features_list)} named co-occurrence feature instances from {processed_rows} DB rows in {duration:.2f}s.")
            logging.info(f"[Compiler] Inserted approx {inserted} unique named co-occurrence features.")
            return inserted
        except Exception as e:
            logging.error(f"Error generating named co-occurrence features: {e}", exc_info=True)
            return 0

    def generate_minimality_features(self):
        # (Implementation remains the same)
        logging.info("[Compiler] Generating minimality prior features...")
        start_time = time.time()
        prior_feature_name = 'prior_minimality'
        sql = f"""
            SELECT d.tid, d.attr, d.candidate_val
            FROM domains d JOIN cells c ON d.tid = c.tid AND d.attr = c.attr
            WHERE (c.val IS NULL AND d.candidate_val = '{NULL_REPR_PLACEHOLDER}')
               OR (c.val IS NOT NULL AND d.candidate_val = CAST(c.val AS TEXT));
        """
        features_list = []
        try:
            with self.db_conn.cursor(name='minimality_feature_cursor_v5') as cur:
                 cur.execute(sql)
                 processed_rows = 0
                 for row in cur:
                      features_list.append((int(row[0]), row[1], str(row[2]), prior_feature_name))
                      processed_rows += 1
            inserted = self._insert_features(features_list)
            duration = time.time() - start_time
            logging.info(f"[Compiler] Generated {len(features_list)} minimality feature instances from {processed_rows} DB rows in {duration:.2f}s.")
            logging.info(f"[Compiler] Inserted approx {inserted} minimality features.")
            return inserted
        except Exception as e:
            logging.error(f"Error generating minimality features: {e}", exc_info=True)
            return 0

    # --- ADDED: generate_frequency_features ---
    def generate_frequency_features(self):
        """
        Generates features based on the global frequency of candidate values.
        Uses logarithmic bins for stability.
        Feature Name: LOGFREQ_{attr}_BIN={bin_index}
        """
        if not self.cell_domains or not self.global_value_counts:
             logging.error("Cannot generate frequency features: domains or global counts not available.")
             return 0

        logging.info("[Compiler] Generating Log-Frequency features...")
        start_time = time.time()
        freq_features_list = []

        for (tid, attr), candidates in tqdm(self.cell_domains.items(), desc="Generating Freq Features"):
             if not candidates: continue
             attr_counts = self.global_value_counts.get(attr)
             if not attr_counts: continue # Skip if no counts for this attribute

             for cand_val_str in candidates:
                 count = attr_counts.get(cand_val_str, 0)
                 # Calculate log10(count + 1) to handle count=0
                 log_freq = math.log10(count + 1)
                 # Assign to a bin
                 bin_index = np.digitize(log_freq, FREQ_BINS, right=False) - 1 # 0-based index
                 bin_index = max(0, bin_index) # Ensure bin index is not negative

                 feature_name = f"LOGFREQ_{attr}_BIN={bin_index}"
                 freq_features_list.append((tid, attr, cand_val_str, feature_name))

        duration = time.time() - start_time
        logging.info(f"Generated {len(freq_features_list)} raw log-frequency feature instances in {duration:.2f}s.")
        inserted = self._insert_features(freq_features_list)
        logging.info(f"[Compiler] Inserted approx {inserted} unique log-frequency features.")
        return inserted
    # --- END ADDED ---


    # --- MODIFIED: generate_dc_violation_features ---
    def generate_dc_violation_features(self):
        """
        Generates NAMED features indicating if assigning a candidate value
        violates a specific denial constraint with ANY other tuple.
        Feature Name: DC_VIOLATES_{constraint_id}_ON_{attr}
        """
        if not self.constraints or self.detector_helper is None: return 0
        if not self._load_data_and_stats(): return 0 # Load data/stats needed
        self._build_dc_indexes()

        logging.info("[Compiler] Generating NAMED DC violation features (ATTR specific)...")
        start_time = time.time()
        dc_features_list = []
        total_checks = 0
        total_violations_found = 0
        skipped_constraints = 0
        logged_details_count = 0

        for (tid, attr), candidates in tqdm(self.cell_domains.items(), desc="Checking DC Violations"):
            if not candidates: continue
            original_tuple_data = self.tuple_data_cache.get(tid)
            if not original_tuple_data: continue

            log_this_cell_detail = (tid, attr) in DETAILED_LOG_CELLS
            if log_this_cell_detail: logging.info(f"[DC Detail Log] Cell ({tid}, {attr}): Candidates = {candidates}")

            for cand_val_str in candidates:
                 hypothetical_tuple = original_tuple_data.copy()
                 hypothetical_tuple[attr] = cand_val_str

                 if log_this_cell_detail: logging.info(f"  [DC Detail Log] Checking Candidate: '{cand_val_str}'")

                 for constraint in self.constraints:
                     constraint_id_str = str(constraint['id'])
                     attrs_in_constraint = set(p['a1'] for p in constraint['predicates']).union(set(p['a2'] for p in constraint['predicates']))
                     if attr not in attrs_in_constraint: continue

                     potential_partners = []
                     partner_found_method = False

                     # Find potential partners using indexes (logic from V4)
                     # (Keep the same partner finding logic)
                     if constraint_id_str in ['1', '2'] and 'ZipCode' in attrs_in_constraint:
                          key_val = hypothetical_tuple.get('ZipCode')
                          if key_val and key_val != NULL_REPR_PLACEHOLDER:
                              potential_partners = [t2_id for t2_id in self.dc_indexes['by_zip'].get(key_val, []) if t2_id != tid]
                              partner_found_method = True
                     elif constraint_id_str in ['3', '4', '5'] and 'PhoneNumber' in attrs_in_constraint:
                          key_val = hypothetical_tuple.get('PhoneNumber')
                          if key_val and key_val != NULL_REPR_PLACEHOLDER:
                              potential_partners = [t2_id for t2_id in self.dc_indexes['by_phone'].get(key_val, []) if t2_id != tid]
                              partner_found_method = True
                     elif constraint_id_str == '6' and {'ProviderNumber', 'MeasureCode'}.issubset(attrs_in_constraint):
                          prov = hypothetical_tuple.get('ProviderNumber')
                          meas = hypothetical_tuple.get('MeasureCode')
                          if prov and prov != NULL_REPR_PLACEHOLDER and meas and meas != NULL_REPR_PLACEHOLDER:
                               potential_partners = [t2_id for t2_id in self.dc_indexes['by_provider_measure'].get((prov, meas), []) if t2_id != tid]
                               partner_found_method = True
                     elif constraint_id_str in ['7', '8'] and 'MeasureCode' in attrs_in_constraint:
                          key_val = hypothetical_tuple.get('MeasureCode')
                          if key_val and key_val != NULL_REPR_PLACEHOLDER:
                               potential_partners = [t2_id for t2_id in self.dc_indexes['by_measurecode'].get(key_val, []) if t2_id != tid]
                               partner_found_method = True
                     elif constraint_id_str == '9' and {'State', 'MeasureCode'}.issubset(attrs_in_constraint):
                          state = hypothetical_tuple.get('State')
                          meas = hypothetical_tuple.get('MeasureCode')
                          if state and state != NULL_REPR_PLACEHOLDER and meas and meas != NULL_REPR_PLACEHOLDER:
                               potential_partners = [t2_id for t2_id in self.dc_indexes['by_state_measure'].get((state, meas), []) if t2_id != tid]
                               partner_found_method = True

                     if not partner_found_method:
                          if constraint_id_str not in getattr(self, '_logged_missing_dc_logic', set()):
                              logging.warning(f"[Compiler-DC] No specific partner finding logic for constraint {constraint_id_str}.")
                              if not hasattr(self, '_logged_missing_dc_logic'): self._logged_missing_dc_logic = set()
                              self._logged_missing_dc_logic.add(constraint_id_str)
                          skipped_constraints += 1
                          continue

                     if not potential_partners: continue

                     violation_found = False
                     violating_partner = None
                     for tid2 in potential_partners:
                         tuple2_data = self.tuple_data_cache.get(tid2)
                         if not tuple2_data: continue
                         total_checks += 1
                         try:
                             if self.detector_helper._check_violation(hypothetical_tuple, tuple2_data, constraint):
                                 violation_found = True
                                 violating_partner = tid2
                                 break
                         except Exception as check_err:
                              logging.error(f"Error checking violation between tid={tid} (cand='{cand_val_str}') and tid={tid2} for constraint {constraint_id_str}: {check_err}", exc_info=True)

                     if violation_found:
                          # --- Refined Feature Name ---
                          feature_name = f"DC_VIOLATES_{constraint_id_str}_ON_{attr}"
                          # --- End Refined Feature Name ---
                          dc_features_list.append((tid, attr, cand_val_str, feature_name))
                          total_violations_found += 1
                          if log_this_cell_detail:
                              logging.info(f"    [DC Detail Log] Candidate '{cand_val_str}' VIOLATES Constraint {constraint_id_str} (Partner TID: {violating_partner}). Added feature: {feature_name}")
                              logged_details_count += 1
                          # break # Optional optimization

                 if log_this_cell_detail and not any(f[0]==tid and f[1]==attr and f[2]==cand_val_str and f[3].startswith("DC_VIOLATES_") for f in dc_features_list[-total_violations_found:]):
                     logging.info(f"    [DC Detail Log] Candidate '{cand_val_str}' did NOT violate any checked constraints.")

        duration = time.time() - start_time
        logging.info(f"Checked {total_checks} potential constraint violations.")
        if skipped_constraints > 0: logging.warning(f"Skipped DC checks for {skipped_constraints} constraint-candidate pairs...")
        logging.info(f"Generated {len(dc_features_list)} raw DC violation feature instances in {duration:.2f}s.")
        if logged_details_count > 0: logging.info(f"Logged details for {logged_details_count} DC violation checks.")
        inserted = self._insert_features(dc_features_list)
        logging.info(f"[Compiler] Inserted approx {inserted} unique DC violation features.")
        return inserted
    # --- END MODIFIED ---


    def compile_all(self):
        """Runs all feature generation steps."""
        if not self._clear_features(): return
        # --- Load data needed for multiple steps ---
        if not self._load_data_and_stats():
             logging.error("Aborting compilation due to error loading data/stats.")
             return
        # --- End Load data ---

        total_features_est = 0
        compile_start_time = time.time()
        try:
            cooc_feat_count = self.generate_cooccurrence_features()
            min_feat_count = self.generate_minimality_features()
            # --- Add Frequency Features ---
            freq_feat_count = self.generate_frequency_features()
            # --- End Add ---
            dc_feat_count = self.generate_dc_violation_features() # Uses cached data

            logging.info("Clearing compiler caches...")
            self.tuple_data_cache = None
            self.cell_domains = None
            self.dc_indexes = {}
            self.global_value_counts = defaultdict(Counter) # Clear counts too
            gc.collect()

            final_feature_count = 0
            try:
                 with self.db_conn.cursor() as cur:
                     cur.execute("SELECT COUNT(DISTINCT feature) FROM features;")
                     final_feature_count = cur.fetchone()[0]
                 logging.info(f"[Compiler] Final total unique features in table: {final_feature_count}")
            except Exception as e:
                 logging.error(f"Could not query final feature count: {e}")
                 total_features_est = cooc_feat_count + min_feat_count + freq_feat_count + dc_feat_count
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

    log_level = logging.INFO # Change to logging.DEBUG for DC details
    logging.basicConfig(level=log_level, format='[%(asctime)s] {%(levelname)s} %(name)s: %(message)s')

    constraints_file = getattr(config, 'CONSTRAINTS_FILE', 'hospital_constraints.txt')
    logging.info(f"Running Feature Compiler (compile.py - Refined DC + Freq Features)")

    try:
        db_conn = psycopg2.connect(**config.DB_SETTINGS)
        logging.info("Database connection successful.")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        compiler = FeatureCompiler(db_conn, constraints_filepath=constraints_file)
        compiler.compile_all()
    except Exception as e:
        logging.error(f"An unexpected error occurred during compilation: {e}", exc_info=True)
    finally:
        if db_conn:
            db_conn.close()
            logging.info("Database connection closed.")
