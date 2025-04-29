# File: detectors/constraints.py
# Detects errors based on denial constraints (DCs), including FDs.
# VERSION 7: Formats tids correctly for bigint[] array literal ('{tid1,tid2}').

import pandas as pd
import itertools
import logging
import time
from collections import Counter, defaultdict

import psycopg2
import psycopg2.extras

# --- Add Support Threshold ---
MIN_VIOLATION_SUPPORT = 2 # Tune this value (e.g., 2, 3)
# --- End Support Threshold ---


class ConstraintViolationDetector:
    """Detects cells involved in denial constraint violations."""

    def __init__(self, db_conn, constraints_file):
        self.db_conn = db_conn
        self.constraints_file = constraints_file
        self.constraints = self._parse_constraints()
        self._tuple_data_cache = None
        self._fd_support_cache = {} # {(attr_key, target_attr): {key_value: Counter(target_val: count)}}
        self._potential_fds = set() # Store identified (key_attr, target_attr) pairs


    def _parse_constraints(self):
        # (Keep parsing logic as is)
        constraints = []
        try:
            with open(self.constraints_file, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split('&')
                    constraint = {'id': i + 1, 'predicates': []}
                    for part in parts:
                        part = part.strip()
                        if part.startswith('EQ('):
                            attrs = part[3:-1].split(',')
                            constraint['predicates'].append({'type': 'EQ', 't1': attrs[0].split('.')[0], 'a1': attrs[0].split('.')[1], 't2': attrs[1].split('.')[0], 'a2': attrs[1].split('.')[1]})
                        elif part.startswith('IQ('):
                             attrs = part[3:-1].split(',')
                             constraint['predicates'].append({'type': 'IQ', 't1': attrs[0].split('.')[0], 'a1': attrs[0].split('.')[1], 't2': attrs[1].split('.')[0], 'a2': attrs[1].split('.')[1]})
                    constraints.append(constraint)
            logging.info(f"Parsed {len(constraints)} constraints from {self.constraints_file}")
        except FileNotFoundError:
            logging.error(f"Constraints file not found: {self.constraints_file}")
        except Exception as e:
            logging.error(f"Error parsing constraints file: {e}", exc_info=True)
        return constraints

    def _fetch_tuple_data(self):
        # (Implementation remains the same)
        if self._tuple_data_cache is not None: return
        logging.info("[ConstraintDetector] Fetching all tuple data into memory...")
        self._tuple_data_cache = defaultdict(dict)
        try:
            # Use a unique cursor name
            with self.db_conn.cursor(name='constraint_data_cursor_v7') as cur: # Use new cursor name
                cur.itersize = 50000
                cur.execute("SELECT tid, attr, val FROM cells ORDER BY tid")
                for tid, attr, val in cur:
                    self._tuple_data_cache[int(tid)][attr] = str(val) if not pd.isna(val) else None
            logging.info(f"Cached data for {len(self._tuple_data_cache)} tuples.")
        except Exception as e:
            logging.error(f"Error fetching tuple data: {e}", exc_info=True)
            self._tuple_data_cache = None

    def _precompute_fd_support(self):
        # (Implementation remains the same)
        if not self._tuple_data_cache: return
        if self._fd_support_cache: return
        logging.info("[ConstraintDetector] Precomputing FD support counts...")
        self._fd_support_cache = {}
        self._potential_fds = set()
        for const in self.constraints:
            eq_preds = [p for p in const['predicates'] if p['type'] == 'EQ']
            iq_preds = [p for p in const['predicates'] if p['type'] == 'IQ']
            if len(eq_preds) == 1 and len(iq_preds) == 1 and \
               eq_preds[0]['a1'] == eq_preds[0]['a2'] and iq_preds[0]['a1'] == iq_preds[0]['a2']:
                  key_attr = eq_preds[0]['a1']; target_attr = iq_preds[0]['a1']
                  self._potential_fds.add((key_attr, target_attr))
        logging.info(f"Identified potential FD keys/targets for support calc: {self._potential_fds}")
        for key_attr, target_attr in self._potential_fds:
            support_key = (key_attr, target_attr)
            self._fd_support_cache[support_key] = defaultdict(Counter)
            for tid, data in self._tuple_data_cache.items():
                key_val = data.get(key_attr); target_val = data.get(target_attr)
                if key_val is not None and target_val is not None:
                    self._fd_support_cache[support_key][key_val][target_val] += 1
            logging.debug(f"  Computed support for {key_attr} -> {target_attr}. Index size: {len(self._fd_support_cache[support_key])} keys.")
        logging.info("[ConstraintDetector] FD support precomputation finished.")


    def _check_violation(self, tuple1_data, tuple2_data, constraint):
        # (Implementation remains the same - no support check here)
        violated = True
        for pred in constraint['predicates']:
            val1 = tuple1_data.get(pred['a1']); val2 = tuple2_data.get(pred['a2'])
            if val1 is None or val2 is None: violated = False; break
            pred_result = False
            try:
                if pred['type'] == 'EQ': pred_result = (str(val1) == str(val2))
                elif pred['type'] == 'IQ': pred_result = (str(val1) != str(val2))
                else: logging.warning(f"Unsupported predicate type {pred['type']}..."); violated = False; break
                if not pred_result: violated = False; break
            except Exception as e: logging.error(f"Error evaluating predicate {pred}..."); violated = False; break
        return violated

    def _has_sufficient_support(self, key_attr, target_attr, key_val, conflicting_target_val):
        # (Implementation remains the same)
        support_key = (key_attr, target_attr)
        if key_val is None or conflicting_target_val is None: return False
        if support_key not in self._fd_support_cache: return False
        support_counts = self._fd_support_cache[support_key].get(key_val, Counter())
        conflicting_support = support_counts.get(conflicting_target_val, 0)
        return conflicting_support >= MIN_VIOLATION_SUPPORT

    def find_violations(self):
        # (Implementation remains the same as V3)
        if not self.constraints: return [], set()
        self._fetch_tuple_data()
        if self._tuple_data_cache is None: return [], set()
        self._precompute_fd_support()
        violations_db = []
        noisy_cells = set()
        start_time = time.time(); processed_constraints = 0
        total_potential_violations = 0; total_ignored_low_support = 0
        logging.info(f"Checking {len(self.constraints)} constraints with MIN_VIOLATION_SUPPORT={MIN_VIOLATION_SUPPORT}...")
        for constraint in self.constraints:
            constraint_id = constraint['id']
            logging.info(f"[ConstraintDetector] Checking constraint {constraint_id}: {constraint['predicates']}")
            const_start_time = time.time(); violation_count_for_constraint = 0; ignored_count_for_constraint = 0
            is_potential_fd = False; fd_key_attr = None; fd_target_attr = None
            eq_preds = [p for p in constraint['predicates'] if p['type'] == 'EQ']
            iq_preds = [p for p in constraint['predicates'] if p['type'] == 'IQ']
            if len(eq_preds) == 1 and len(iq_preds) == 1 and eq_preds[0]['a1'] == eq_preds[0]['a2'] and iq_preds[0]['a1'] == iq_preds[0]['a2']:
                 fd_key_attr = eq_preds[0]['a1']; fd_target_attr = iq_preds[0]['a1']
                 if (fd_key_attr, fd_target_attr) in self._potential_fds: is_potential_fd = True
            key_attrs = [eq_preds[0]['a1']] if len(eq_preds) > 0 else []
            if not key_attrs: logging.warning(f"Cannot determine key attributes for constraint {constraint_id}. Skipping."); continue
            key_attr = key_attrs[0]; logging.debug(f"Using key attribute '{key_attr}' for constraint {constraint_id}")
            grouped_by_key = defaultdict(list)
            for tid, data in self._tuple_data_cache.items():
                key_val = data.get(key_attr)
                if key_val is not None: grouped_by_key[key_val].append(tid)
            checked_pairs = set()
            for key_val, tids in grouped_by_key.items():
                if len(tids) > 1:
                    for tid1, tid2 in itertools.combinations(tids, 2):
                        pair_key = tuple(sorted((tid1, tid2)));
                        if pair_key in checked_pairs: continue
                        checked_pairs.add(pair_key)
                        t1_data = self._tuple_data_cache[tid1]; t2_data = self._tuple_data_cache[tid2]
                        if self._check_violation(t1_data, t2_data, constraint):
                            total_potential_violations += 1
                            apply_support_check = is_potential_fd; passes_support_check = True
                            t1_noisy_target = False; t2_noisy_target = False
                            if apply_support_check:
                                t1_target = t1_data.get(fd_target_attr); t2_target = t2_data.get(fd_target_attr)
                                t1_maybe_noisy = self._has_sufficient_support(fd_key_attr, fd_target_attr, key_val, t2_target)
                                t2_maybe_noisy = self._has_sufficient_support(fd_key_attr, fd_target_attr, key_val, t1_target)
                                if not t1_maybe_noisy and not t2_maybe_noisy:
                                    passes_support_check = False; total_ignored_low_support += 1; ignored_count_for_constraint += 1
                                    logging.debug(f"FD Violation {constraint_id} between {tid1} & {tid2} ignored by support.")
                                else:
                                    if t1_maybe_noisy: t1_noisy_target = True
                                    if t2_maybe_noisy: t2_noisy_target = True
                            if passes_support_check:
                                violations_db.append((constraint_id, tid1, tid2)) # Store original violation info
                                violation_count_for_constraint += 1
                                if apply_support_check:
                                    if t1_noisy_target: noisy_cells.add((tid1, fd_target_attr))
                                    if t2_noisy_target: noisy_cells.add((tid2, fd_target_attr))
                                else:
                                    for pred in constraint['predicates']:
                                        if pred['a1'] in t1_data: noisy_cells.add((tid1, pred['a1']))
                                        if pred['a2'] in t2_data: noisy_cells.add((tid2, pred['a2']))
            const_end_time = time.time()
            logging.info(f"Constraint {constraint_id} finished in {const_end_time - const_start_time:.2f}s, found {violation_count_for_constraint} violations passing support (ignored {ignored_count_for_constraint} due to low support).")
            processed_constraints += 1
        end_time = time.time()
        logging.info(f"Finished checking all constraints in {end_time - start_time:.2f} seconds.")
        logging.info(f"Total potential violations found (before support): {total_potential_violations}")
        logging.info(f"Total violations ignored due to low support: {total_ignored_low_support}")
        logging.info(f"Total violations stored in DB: {len(violations_db)}")
        logging.info(f"Total unique cells marked as noisy (after support): {len(noisy_cells)}")
        return violations_db, noisy_cells

    def insert_violations(self, violations, noisy_cells):
        """Inserts violations and updates noisy cells in the database."""
        if not violations and not noisy_cells:
            logging.info("[ConstraintDetector] No violations or noisy cells to insert.")
            return

        try:
            with self.db_conn.cursor() as cur:
                logging.info("[ConstraintDetector] Clearing previous violations...")
                cur.execute("DELETE FROM violations;")
                cur.execute("DELETE FROM noisy_cells;")
                cur.execute("UPDATE cells SET is_noisy = FALSE;")

                if violations:
                    logging.info(f"Inserting {len(violations)} violation instances...")
                    # Use 'tids' column which is bigint[]
                    insert_sql_violations = "INSERT INTO violations (constraint_id, tids) VALUES (%s, %s);"
                    # --- Corrected Data Formatting for Array Literal ---
                    # Format as '{tid1,tid2}' which PostgreSQL understands for integer/bigint arrays
                    violations_to_insert = [(v[0], f"{{{v[1]},{v[2]}}}") for v in violations]
                    # --- End Correction ---
                    psycopg2.extras.execute_batch(cur, insert_sql_violations, violations_to_insert, page_size=5000)
                    logging.info(f"Inserted {len(violations_to_insert)} violation instances.")

                if noisy_cells:
                    noisy_list = list(noisy_cells)
                    logging.info(f"Inserting {len(noisy_list)} unique noisy cell entries...")
                    insert_sql_noisy = "INSERT INTO noisy_cells (tid, attr) VALUES (%s, %s) ON CONFLICT DO NOTHING;"
                    psycopg2.extras.execute_batch(cur, insert_sql_noisy, noisy_list, page_size=5000)
                    logging.info("Updating is_noisy flag in cells table...")
                    cur.execute("CREATE TEMP TABLE temp_noisy (tid INT, attr TEXT);")
                    psycopg2.extras.execute_batch(cur, "INSERT INTO temp_noisy (tid, attr) VALUES (%s, %s);", noisy_list, page_size=5000)
                    cur.execute("CREATE INDEX ON temp_noisy (tid, attr);")
                    cur.execute("UPDATE cells c SET is_noisy = TRUE FROM temp_noisy t WHERE c.tid = t.tid AND c.attr = t.attr;")
                    cur.execute("DROP TABLE temp_noisy;")
                    logging.info(f"Updated is_noisy flag for {len(noisy_list)} cells.")

            self.db_conn.commit()
            logging.info("[ConstraintDetector] Violations and noisy cells committed to database.")
        except psycopg2.errors.InvalidTextRepresentation as e:
             # Specific handling for the array literal error
             logging.error(f"Database schema error during insertion: {e}")
             logging.error("Please ensure the 'violations' table has column 'tids' of type bigint[] or int[] and the format '{tid1,tid2}' is correct.")
             self.db_conn.rollback()
        except psycopg2.errors.UndefinedColumn as e:
             logging.error(f"Database schema error during insertion: {e}")
             logging.error("Please ensure the 'violations' table has columns 'constraint_id', 'tids' and 'noisy_cells' has 'tid', 'attr'. Check schema.sql.")
             self.db_conn.rollback()
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"Error inserting violations/noisy cells: {e}", exc_info=True)


    def run(self):
        """Runs the full detection and insertion process."""
        violations, noisy_cells = self.find_violations()
        self.insert_violations(violations, noisy_cells)
        logging.info("[ConstraintDetector] Finished.")

# Example usage remains the same
if __name__ == '__main__':
    import config
    import sys
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

    constraints_file = getattr(config, 'CONSTRAINTS_FILE', 'hospital_constraints.txt')
    logging.info(f"Running Constraint Detector with Support Threshold = {MIN_VIOLATION_SUPPORT}")

    try:
        db_conn = psycopg2.connect(**config.DB_SETTINGS)
        logging.info("Database connection successful.")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        detector = ConstraintViolationDetector(db_conn, constraints_file)
        detector.run()
    except Exception as e:
        logging.error(f"An unexpected error occurred during detection: {e}", exc_info=True)
    finally:
        if db_conn:
            db_conn.close()
            logging.info("Database connection closed.")
