# File: detectors/constraints.py
# Detects violations of denial constraints.

import re
import pandas as pd
import time
from .base_detector import BaseDetector

class ConstraintViolationDetector(BaseDetector):
    """Detects errors based on denial constraints."""

    def __init__(self, db_conn, constraints_filepath="hospital_constraints.txt"):
        super().__init__(db_conn)
        self.constraints_filepath = constraints_filepath
        self.constraints = self._parse_constraints()

    def _parse_constraints(self):
        """Parses constraints from the file."""
        constraints = []
        try:
            with open(self.constraints_filepath, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parsed = self._parse_single_constraint(line, constraint_id=i + 1)
                    if parsed:
                        constraints.append(parsed)
            print(f"Parsed {len(constraints)} constraints from {self.constraints_filepath}")
            return constraints
        except FileNotFoundError:
            print(f"Error: Constraint file not found at {self.constraints_filepath}")
            return []
        except Exception as e:
            print(f"Error parsing constraints file: {e}")
            return []

    def _parse_single_constraint(self, line, constraint_id):
        """Parses a single constraint line into a structured dictionary."""
        # Example parsing logic - needs to be robust for different predicate types
        # Format: t1&t2&PRED(t1.Attr1, t2.Attr2)&PRED(...)
        predicates = []
        # Simple regex to find EQ(tX.Attr,tY.Attr) or IQ(tX.Attr,tY.Attr)
        pattern = re.compile(r"(EQ|IQ)\(t(\d+)\.(\w+),t(\d+)\.(\w+)\)")
        matches = pattern.findall(line)

        if not matches:
            print(f"Warning: Could not parse predicates in constraint line {constraint_id}: {line}")
            return None

        involved_attrs = set()
        for pred_type, t1_idx, a1, t2_idx, a2 in matches:
            if t1_idx != '1' or t2_idx != '2':
                 print(f"Warning: Constraint {constraint_id} currently only supports t1 and t2. Found t{t1_idx}, t{t2_idx}.")
                 # For simplicity, we only handle t1, t2 constraints here. Extend if needed.
                 # return None # Or adapt logic
            predicates.append({
                'type': pred_type, # 'EQ' or 'IQ'
                'a1': a1,
                'a2': a2
            })
            involved_attrs.add(a1)
            involved_attrs.add(a2)

        return {
            'id': constraint_id,
            'text': line,
            'predicates': predicates,
            'involved_attrs': list(involved_attrs)
        }

    def _check_violation(self, tuple1_data, tuple2_data, constraint):
        """Checks if two tuples (as dicts attr->val) violate a constraint."""
        for pred in constraint['predicates']:
            val1 = tuple1_data.get(pred['a1'])
            val2 = tuple2_data.get(pred['a2'])

            # Handle comparison with potential None values carefully
            is_eq = (val1 is not None) and (val2 is not None) and (str(val1) == str(val2))
            # IQ often implies case-insensitivity in data cleaning contexts
            is_iq = (val1 is not None) and (val2 is not None) and (str(val1).lower() != str(val2).lower())

            if pred['type'] == 'EQ':
                if not is_eq: return False # If EQ is required but not met, constraint is satisfied
            elif pred['type'] == 'IQ':
                if not is_iq: return False # If IQ is required but not met, constraint is satisfied
            else:
                 # Handle other predicate types if needed (>, <, etc.)
                 print(f"Warning: Unsupported predicate type {pred['type']}")
                 return False # Assume satisfied if predicate unknown

        # If we went through all predicates and none failed, the conjunction is TRUE, so the DC is violated.
        return True

    def detect_errors(self):
        """Detects constraint violations and populates tables."""
        total_violations_found = 0
        total_noisy_cells_added = 0
        start_time = time.time()

        # Clear previous violations for idempotency
        try:
            with self.db_conn.cursor() as cur:
                 print(f"[{self.detector_name}] Clearing previous violations...")
                 cur.execute("DELETE FROM violations WHERE constraint_id IN %s;", (tuple(c['id'] for c in self.constraints),))
                 # Optionally clear noisy_cells added by this detector
                 # cur.execute("DELETE FROM noisy_cells WHERE detection_method = %s;", (self.detector_name,))
                 self.db_conn.commit()
        except Exception as e:
             print(f"[{self.detector_name}] Error clearing previous violations: {e}")
             self.db_conn.rollback()
             # Decide whether to proceed or stop

        for constraint in self.constraints:
            print(f"\n[{self.detector_name}] Checking constraint {constraint['id']}: {constraint['text']}")
            violation_count = 0
            noisy_cells_for_constraint = set() # Use a set to avoid duplicates

            # --- Strategy: Fetch relevant data, then check in Python ---
            # 1. Identify potential violating keys based on EQ predicates
            #    E.g., if EQ(t1.ZipCode, t2.ZipCode), fetch groups of tids sharing the same ZipCode.
            #    This requires dynamic SQL based on constraint structure.
            #    Simplification: Iterate through all pairs (can be slow).
            #    Better Simplification: For each constraint, identify a 'key' attribute (e.g., the first EQ attribute).
            #                           Fetch all cells for that attribute, group by value, then compare pairs within groups.

            eq_preds = [p for p in constraint['predicates'] if p['type'] == 'EQ']
            if not eq_preds:
                 print(f"Warning: Constraint {constraint['id']} has no EQ predicates. Skipping pairwise check for now.")
                 continue # Naive pairwise check is too slow without filtering

            # Use the first EQ predicate's attributes as a potential key
            key_attr1 = eq_preds[0]['a1']
            key_attr2 = eq_preds[0]['a2'] # Assume keys match for simplicity for now

            # Fetch all tuples (tid and value) for the key attribute
            sql_fetch_keys = f"SELECT tid, val FROM cells WHERE attr = %s AND val IS NOT NULL;"
            try:
                key_df = pd.read_sql(sql_fetch_keys, self.db_conn, params=(key_attr1,))
                print(f"Fetched {len(key_df)} potential key values for attr '{key_attr1}'.")
            except Exception as e:
                print(f"Error fetching key values for constraint {constraint['id']}: {e}")
                continue

            # Group TIDs by the key value
            grouped_tids = key_df.groupby('val')['tid'].apply(list)

            # Fetch all involved attributes for faster lookup
            involved_attrs_str = ', '.join(f"'{a}'" for a in constraint['involved_attrs'])
            sql_fetch_data = f"SELECT tid, attr, val FROM cells WHERE attr IN ({involved_attrs_str});"
            try:
                all_data_df = pd.read_sql(sql_fetch_data, self.db_conn)
                all_data_df.set_index('tid', inplace=True) # Index by tid for faster lookup
                print(f"Fetched data for {len(constraint['involved_attrs'])} involved attributes.")
                 # Convert to a dictionary for faster lookups: {tid: {attr: val}}
                tuple_data_cache = {}
                for tid, group in all_data_df.groupby(level=0): # Group by index (tid)
                     tuple_data_cache[tid] = group.set_index('attr')['val'].to_dict()
                print("Created tuple data cache.")
            except Exception as e:
                 print(f"Error fetching tuple data for constraint {constraint['id']}: {e}")
                 continue


            # Iterate through groups where more than one TID shares the key value
            violation_pairs_tids = set() # Store (t1, t2) where t1 < t2
            for key_val, tids in grouped_tids.items():
                if len(tids) > 1:
                    # Check all pairs within this group
                    for i in range(len(tids)):
                        for j in range(i + 1, len(tids)):
                            tid1 = tids[i]
                            tid2 = tids[j]

                            # Avoid re-checking pairs already found
                            pair_key = tuple(sorted((tid1, tid2)))
                            if pair_key in violation_pairs_tids:
                                continue

                            # Get data for t1 and t2 from cache
                            tuple1_data = tuple_data_cache.get(tid1, {})
                            tuple2_data = tuple_data_cache.get(tid2, {})

                            if not tuple1_data or not tuple2_data:
                                # Should not happen if data was fetched correctly
                                continue

                            # Check if this pair violates the constraint
                            if self._check_violation(tuple1_data, tuple2_data, constraint):
                                violation_count += 1
                                violation_pairs_tids.add(pair_key)

                                # Identify specific noisy cells based on predicates
                                for pred in constraint['predicates']:
                                    # Flag both cells involved in the specific predicate causing violation
                                    # (Could refine this based on EQ/IQ logic)
                                    if pred['a1'] in tuple1_data: noisy_cells_for_constraint.add((tid1, pred['a1']))
                                    if pred['a2'] in tuple2_data: noisy_cells_for_constraint.add((tid2, pred['a2']))

            # Insert violations into the database
            if violation_pairs_tids:
                sql_insert_violation = """
                    INSERT INTO violations (constraint_id, tids) VALUES (%s, %s);
                """
                violations_to_insert = [(constraint['id'], list(pair)) for pair in violation_pairs_tids]
                try:
                    with self.db_conn.cursor() as cur:
                        cur.executemany(sql_insert_violation, violations_to_insert)
                    self.db_conn.commit()
                    print(f"Inserted {len(violations_to_insert)} violation instances for constraint {constraint['id']}.")
                    total_violations_found += len(violations_to_insert)
                except Exception as e:
                    self.db_conn.rollback()
                    print(f"Error inserting violations for constraint {constraint['id']}: {e}")

            # Add identified noisy cells
            added_count = self._add_noisy_cells(list(noisy_cells_for_constraint))
            total_noisy_cells_added += added_count

        end_time = time.time()
        print(f"\n[{self.detector_name}] Finished in {end_time - start_time:.2f} seconds.")
        print(f"[{self.detector_name}] Total violations found: {total_violations_found}")
        print(f"[{self.detector_name}] Total unique noisy cells added: {total_noisy_cells_added}") # May differ from sum if cells violate multiple constraints
        return total_violations_found