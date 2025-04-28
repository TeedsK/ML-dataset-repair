# File: model/factor_graph.py (Modified with Debugging)
# Defines the HoloClean factor graph, handling weight learning and inference logic.

import sys
import pandas as pd
import time
import psycopg2
import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor, State
from collections import defaultdict
import random
import itertools
import math

log_count = 0

# --- DEBUGGING SETUP ---
# Choose a few representative features to track during gradient calculation/fitting
# Ensure these feature names actually exist in your 'features' table after compilation
# You might need to adjust these based on your compiler output/data
FEATURES_TO_DEBUG = {
    'prior_minimality',
    'cooc_State=al',        # Example co-occurrence, adjust if needed
    'dc_relax_1'            # Example relaxed DC, adjust if needed
}
MAX_EVIDENCE_VARS_TO_LOG = 5 # Limit per-variable gradient logging to avoid flooding output
# --- END DEBUGGING SETUP ---


class HoloCleanFactorGraph:
    """
    Represents the HoloClean Factor Graph, handling weight learning and inference.
    Uses Relaxed DC approach (features) by default.
    """
    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.graph: FactorGraph = FactorGraph()

        # maps & caches
        self.cell_to_variable_map: dict[tuple, str] = {}
        self.variable_to_cell_map: dict[str, tuple] = {}
        self.variable_states: dict[str, dict[int, str]] = {}
        self.candidate_map: dict[tuple, dict[str, int]] = {}
        self.features_data: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.state_to_features_map: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.weights: dict[str, float] = {}
        self.evidence_dict: dict[str, int] = {}
        self.query_vars: list[str] = []

        # Adam optimizer state
        self.adam_m = None
        self.adam_v = None
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8

        # cfg / constants
        self.initial_weight_value = 0.1
        self.l2_penalty = 1e-8
        self.log_potential_clip = 15.0
        self.epsilon = 1e-10

        # cached DB tables
        self.domains_df = None
        self.features_df = None
        self.cells_df = None

        self.variable_factors: dict[str, DiscreteFactor] = {}

        # --- DEBUGGING STATE ---
        self.evidence_vars_logged_count = 0 # Counter for limiting per-var logs
        # --- END DEBUGGING STATE ---

    # ──────────────────────────────────────────────────────────────────────
    # DATA LOADING (Keep as is)
    # ──────────────────────────────────────────────────────────────────────
    def _load_data(self) -> bool:
        """Fetch domains / features / cells from Postgres."""
        print("[Model] Loading data for graph construction …")
        t0 = time.time()
        try:
            # ... (loading logic remains the same) ...
            self.domains_df = pd.read_sql(
                "SELECT tid, attr, candidate_val FROM domains "
                "ORDER BY tid, attr, candidate_val",
                self.db_conn,
            )
            print(f"Loaded {len(self.domains_df):,} domain rows.")

            self.features_df = pd.read_sql(
                "SELECT tid, attr, candidate_val, feature FROM features", self.db_conn
            )
            print(f"Loaded {len(self.features_df):,} feature rows.")

            self.cells_df = pd.read_sql(
                "SELECT tid, attr, val, is_noisy FROM cells", self.db_conn
            )
            print(f"Loaded {len(self.cells_df):,} cell rows.")
            if self.cells_df.empty: print("ERROR: cells table empty."); return False

            print(f"[Model] Data loaded in {time.time() - t0:.2f}s.")
            return True
        except Exception as e:
            print(f"ERROR loading data: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL STRUCTURE BUILDING (Keep as is)
    # ──────────────────────────────────────────────────────────────────────
    def _build_internal_structures(self):
        """Populates internal mappings based on loaded DataFrames."""
        # ... (structure building logic remains the same, including state_to_features_map) ...
        if self.cells_df is None or self.domains_df is None or self.features_df is None: print("ERROR: Data not loaded."); return False
        print("[Model] Building internal structures (variables, states, features)...")
        start_time = time.time()
        self.cell_to_variable_map.clear(); self.variable_to_cell_map.clear(); self.variable_states.clear()
        self.candidate_map.clear(); self.features_data.clear(); self.weights.clear()
        self.evidence_dict.clear(); self.query_vars.clear(); self.state_to_features_map.clear()
        noisy_cell_candidates = defaultdict(list)
        for _, row in self.domains_df.iterrows(): noisy_cell_candidates[(row['tid'], row['attr'])].append(row['candidate_val'])
        cell_info_lookup = self.cells_df.set_index(['tid', 'attr']).to_dict('index')
        unique_cells = self.cells_df[['tid', 'attr']].drop_duplicates().to_records(index=False)
        total_vars_created = 0
        for tid, attr in unique_cells:
            tid_attr = (tid, attr);
            safe_attr = str(attr).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_').replace('.', '_')
            var_name = f"v_{tid}_{safe_attr}"
            if var_name in self.variable_to_cell_map: print(f"WARNING: Duplicate var name {var_name}"); continue
            self.cell_to_variable_map[tid_attr] = var_name; self.variable_to_cell_map[var_name] = tid_attr; total_vars_created += 1
            cell_info = cell_info_lookup.get(tid_attr);
            if not cell_info: continue
            is_noisy = cell_info['is_noisy']; original_val = cell_info['val']
            candidates = []
            if is_noisy:
                candidates = sorted(list(set(noisy_cell_candidates.get(tid_attr, []))))
                if not candidates:
                    if original_val is not None: candidates = [original_val]; print(f"WARNING: Noisy {tid_attr} no candidates. Using original.")
                    else: candidates = ["__EMPTY_DOMAIN__"]; print(f"WARNING: Noisy NULL {tid_attr} no candidates. Using dummy.")
                self.query_vars.append(var_name)
            else:
                if original_val is not None:
                    candidates = [original_val]
                    self.evidence_dict[var_name] = 0
                else:
                    candidates = ["__NULL__"]
                    self.evidence_dict[var_name] = 0
            if candidates:
                num_states = len(candidates); self.variable_states[var_name] = {i: str(val) for i, val in enumerate(candidates)}
                self.candidate_map[tid_attr] = {str(val): i for i, val in enumerate(candidates)}
            else: print(f"CRITICAL WARNING: Var {var_name} has NO states.")
        print(f"Defined {total_vars_created} variables.")
        print(f"Identified {len(self.evidence_dict)} evidence variables and {len(self.query_vars)} query variables.")
        if len(self.evidence_dict) + len(self.query_vars) != total_vars_created: print(f"WARNING: Var count mismatch!")
        if not self.evidence_dict and total_vars_created > 0: print("WARNING: No evidence variables identified.")
        all_feature_names = set(self.features_df['feature'].unique()) if self.features_df is not None else set()
        # --- NEW: Fixed Prior + Random Others ---
        MINIMALITY_WEIGHT = 1.5 # Define the fixed weight (tune this value if needed)
        INIT_STD_DEV_OTHERS = 0.05 # Std dev for other random weights (tune if needed)
        print(f"Initializing weights: 'prior_minimality' fixed to {MINIMALITY_WEIGHT}, others random (mean=0, std={INIT_STD_DEV_OTHERS}).")
        np.random.seed(42)
        self.weights.clear() # Clear previous weights
        learnable_feature_names = [] # Keep track of features Adam should update

        for feature_name in all_feature_names:
            if feature_name == 'prior_minimality':
                self.weights[feature_name] = MINIMALITY_WEIGHT
            else:
                # Initialize other features randomly
                self.weights[feature_name] = np.random.normal(loc=0.0, scale=INIT_STD_DEV_OTHERS)
                learnable_feature_names.append(feature_name) # Mark as learnable

        # Store the list of learnable features for the optimizer
        self._learnable_features = learnable_feature_names
        print(f"Set {len(self._learnable_features)} features as learnable.")
        # --- END NEW ---
        feature_lookup = defaultdict(list)
        for _, row in self.features_df.iterrows(): feature_lookup[row['feature']].append(((row['tid'], row['attr']), row['candidate_val']))
        print(f"Found {len(all_feature_names)} unique feature types.")
        self.features_data.clear(); self.state_to_features_map.clear()
        processed_feature_links = 0; missing_feature_links = 0
        for feature_name in all_feature_names:
            self.features_data[feature_name] = []
            affected_cells_candidates = feature_lookup.get(feature_name, [])
            for tid_attr, candidate_val in affected_cells_candidates:
                 var_name = self.cell_to_variable_map.get(tid_attr)
                 if var_name and var_name in self.variable_states:
                     state_map_for_var = self.candidate_map.get(tid_attr, {})
                     candidate_val_str = str(candidate_val)
                     if candidate_val_str in state_map_for_var:
                         state_int = state_map_for_var[candidate_val_str]
                         if state_int in self.variable_states[var_name]:
                              self.features_data[feature_name].append((var_name, state_int));
                              self.state_to_features_map[(var_name, state_int)].append(feature_name)
                              processed_feature_links += 1
                         else: missing_feature_links += 1
                     else: missing_feature_links += 1
                 else: missing_feature_links += 1
        map_build_time = time.time() - start_time
        print(f"State-to-features map built in {map_build_time:.2f}s. Contains {len(self.state_to_features_map)} state entries and {processed_feature_links} feature links.")
        build_time = time.time() - start_time
        print(f"Processed {processed_feature_links} feature links. Could not map {missing_feature_links} links.")
        print(f"[Model] Internal structures built in {build_time:.2f}s.")
        return True

    # ──────────────────────────────────────────────────────────────────────
    # POTENTIAL CALCULATION (Keep as is)
    # ──────────────────────────────────────────────────────────────────────
    def _get_potential_sum(self, var_name, state_int, current_weights):
        """Calculate sum of (clipped) log potentials for a specific variable state."""
        # ... (potential calculation logic remains the same) ...
        log_potential_sum = 0.0
        var_state_key = (var_name, state_int)
        relevant_feature_names = self.state_to_features_map.get(var_state_key, [])
        for feature_name in relevant_feature_names:
            weight = current_weights.get(feature_name, 0.0)
            clipped_weight = np.clip(weight, -self.log_potential_clip, self.log_potential_clip)
            log_potential_sum += clipped_weight
        return log_potential_sum

    # ──────────────────────────────────────────────────────────────────────
    # LIKELIHOOD CALCULATION (Keep as is)
    # ──────────────────────────────────────────────────────────────────────
    def _calculate_objective(self, current_weights):
        """Calculates NEGATIVE Pseudo-Log-Likelihood + L2 penalty."""
        # ... (objective calculation logic remains the same) ...
        pseudo_log_likelihood = 0.0
        num_evidence = len(self.evidence_dict)
        if num_evidence == 0:
             l2_norm_sq = sum(w**2 for w in current_weights.values())
             penalty = 0.5 * self.l2_penalty * l2_norm_sq
             return penalty
        for var_name, evidence_state_idx in self.evidence_dict.items():
            if var_name not in self.variable_states: continue
            log_potential_evidence_state = self._get_potential_sum(var_name, evidence_state_idx, current_weights)
            num_states = len(self.variable_states[var_name])
            if num_states <= 0: continue
            log_potential_all_states = np.zeros(num_states)
            for state_int in range(num_states): log_potential_all_states[state_int] = self._get_potential_sum(var_name, state_int, current_weights)
            max_log_potential = np.max(log_potential_all_states)
            if np.isneginf(max_log_potential): max_log_potential = -700
            shifted_potentials = log_potential_all_states - max_log_potential
            sum_exp_terms = np.sum(np.exp(shifted_potentials))
            log_Z = max_log_potential + np.log(sum_exp_terms + self.epsilon)
            log_prob_evidence = log_potential_evidence_state - log_Z
            if not np.isfinite(log_prob_evidence):
                 # print(f"WARN: Non-finite PLL for {var_name}. Setting to -20.") # Too verbose
                 log_prob_evidence = -20.0
            pseudo_log_likelihood += log_prob_evidence
        l2_norm_sq = sum(w**2 for w in current_weights.values())
        penalty = 0.5 * self.l2_penalty * l2_norm_sq
        objective_value = -pseudo_log_likelihood + penalty
        if not np.isfinite(objective_value):
             print(f"WARNING: Objective function returned non-finite value. PLL={pseudo_log_likelihood}, Penalty={penalty}")
             return 1e10 * (1 + l2_norm_sq)
        return objective_value


    # ──────────────────────────────────────────────────────────────────────
    # GRADIENT CALCULATION (Modified with Debugging)
    # ──────────────────────────────────────────────────────────────────────
    def _calculate_gradient(self, current_weights):
        """ Calculates the gradient of the objective function w.r.t each weight."""
        gradients = {fname: 0.0 for fname in self.weights.keys()}
        num_evidence = len(self.evidence_dict)

        # --- DEBUG: Reset per-variable log counter for this gradient calculation ---
        self.evidence_vars_logged_count = 0
        # --- END DEBUG ---

        if num_evidence == 0:
            for feature_name, weight in current_weights.items():
                 gradients[feature_name] = self.l2_penalty * weight
            return gradients

        # Calculate gradient from PLL term
        for var_name, evidence_state_idx in self.evidence_dict.items():
            if var_name not in self.variable_states: continue
            num_states = len(self.variable_states[var_name])
            if num_states <= 0: continue

            # --- DEBUG: Check if we should log details for this variable ---
            log_this_var = self.evidence_vars_logged_count < MAX_EVIDENCE_VARS_TO_LOG
            if log_this_var:
                print(f"  [Grad Debug] Var: {var_name} (Evid Idx: {evidence_state_idx})")
            # --- END DEBUG ---

            # 1. Potentials
            log_potentials = np.zeros(num_states)
            for state_int in range(num_states):
                log_potentials[state_int] = self._get_potential_sum(var_name, state_int, current_weights)

            # 2. Probabilities (Softmax)
            max_log_potential = np.max(log_potentials)
            if np.isneginf(max_log_potential): max_log_potential = -700
            shifted_potentials = log_potentials - max_log_potential
            exp_potentials = np.exp(shifted_potentials)
            sum_exp = np.sum(exp_potentials)
            probabilities = (exp_potentials / sum_exp) if sum_exp > self.epsilon else (np.ones(num_states) / num_states)

            # --- DEBUG: Log probabilities if logging this var ---
            if log_this_var:
                prob_str = np.array2string(probabilities, precision=4, suppress_small=True, max_line_width=120)
                print(f"    Probs: {prob_str}")
            # --- END DEBUG ---

            # 3. Expected vs Observed feature counts
            involved_features = set()
            for state_int in range(num_states):
                involved_features.update(self.state_to_features_map.get((var_name, state_int), []))

            # --- DEBUG: Prepare dict for per-var gradient contributions ---
            var_grad_contributions = {fname: 0.0 for fname in FEATURES_TO_DEBUG if fname in involved_features} if log_this_var else None
            # --- END DEBUG ---

            for feature_name in involved_features:
                # Expected count
                expected_count = 0.0
                for state_int in range(num_states):
                    if feature_name in self.state_to_features_map.get((var_name, state_int), []):
                        expected_count += probabilities[state_int]

                # Observed count
                observed_count = 1.0 if feature_name in self.state_to_features_map.get((var_name, evidence_state_idx), []) else 0.0

                # Gradient contribution from Neg LL for this variable
                grad_contrib = expected_count - observed_count
                gradients[feature_name] += grad_contrib

                # --- DEBUG: Log contribution if tracked feature and logging this var ---
                if log_this_var and feature_name in FEATURES_TO_DEBUG:
                     var_grad_contributions[feature_name] = grad_contrib # Store contribution
                     print(f"      Feature '{feature_name}': Exp={expected_count:.4f}, Obs={observed_count:.1f}, Grad Contrib={grad_contrib:.4f}")
                # --- END DEBUG ---

            # --- DEBUG: Increment log counter ---
            if log_this_var:
                self.evidence_vars_logged_count += 1
            # --- END DEBUG ---

        # Add L2 regularization gradient
        for feature_name, weight in current_weights.items():
            gradients[feature_name] += self.l2_penalty * weight

        # --- DEBUG: Log final gradient for tracked features ---
        print("  [Grad Debug] Final Gradients (Summed + L2) for tracked features:")
        for fname in FEATURES_TO_DEBUG:
             if fname in gradients:
                  print(f"    '{fname}': {gradients[fname]:.6f}")
             else:
                  print(f"    '{fname}': (Not present)")
        # --- END DEBUG ---

        # Check for non-finite gradients
        for feature_name, grad_val in gradients.items():
             if not np.isfinite(grad_val):
                  print(f"WARNING: Non-finite gradient for '{feature_name}'. Setting to 0.")
                  gradients[feature_name] = 0.0

        return gradients


    # ──────────────────────────────────────────────────────────────────────
    # WEIGHT LEARNING (Modified with Debugging)
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, max_iter=100, learning_rate=0.01, log_interval=10):
        """Learns the feature weights using evidence via Adam optimizer."""
        if not self.evidence_dict:
             print("WARNING: No evidence variables found. Skipping weight learning...")
             # Initialize weights if empty
             if not self.weights:
                 all_feature_names = list(set(itertools.chain.from_iterable(self.state_to_features_map.values())))
                 for name in all_feature_names: self.weights[name] = self.initial_weight_value
             return

        print(f"[Model] Starting weight learning (Adam, Max Iter: {max_iter}, LR: {learning_rate})...")
        start_time = time.time()

        # --- MODIFIED: Use _learnable_features ---
        if not hasattr(self, '_learnable_features') or not self._learnable_features:
             print("WARNING: No learnable features identified. Skipping weight learning.")
             return

        learnable_features = self._learnable_features
        print(f"Optimizing {len(learnable_features)} weights (excluding fixed priors).")

        # Initialize Adam state only for learnable features
        if self.adam_m is None or self.adam_v is None:
             self.adam_m = {fname: 0.0 for fname in learnable_features}
             self.adam_v = {fname: 0.0 for fname in learnable_features}
        # --- END MODIFIED ---

        feature_names = list(self.weights.keys())
        if not feature_names:
             print("WARNING: No features found. Skipping weight learning.")
             return

        for i in range(1, max_iter + 1):
            iter_start_time = time.time()
            print(f"--- Iteration {i}/{max_iter} ---") # Log start of iteration

            # 1. Calculate Gradients
            print("  Calculating gradients...") # Log gradient step start
            gradients = self._calculate_gradient(self.weights)

            # 2. Update ONLY Learnable Weights using Adam
            # --- MODIFIED: Update only learnable features ---
            self.adam_m = {
                fname: self.adam_beta1 * self.adam_m.get(fname, 0.0) + (1 - self.adam_beta1) * gradients.get(fname, 0.0)
                for fname in learnable_features # Only iterate over learnable
            }
            self.adam_v = {
                fname: self.adam_beta2 * self.adam_v.get(fname, 0.0) + (1 - self.adam_beta2) * (gradients.get(fname, 0.0) ** 2)
                for fname in learnable_features # Only iterate over learnable
            }
            m_hat = {fname: m / (1 - self.adam_beta1 ** i) for fname, m in self.adam_m.items()}
            v_hat = {fname: v / (1 - self.adam_beta2 ** i) for fname, v in self.adam_v.items()}
    
            weight_updates = {}
            for fname in learnable_features: # Only iterate over learnable
                 update_val = learning_rate * m_hat.get(fname,0.0) / (math.sqrt(v_hat.get(fname,0.0)) + self.adam_epsilon)
                 if not np.isfinite(update_val):
                      # print(f"WARN: Skipping weight update for {fname} (non-finite value)")
                      weight_updates[fname] = 0.0
                      continue
                 weight_updates[fname] = update_val
                 # Apply update ONLY to learnable weights
                 self.weights[fname] -= update_val
            # --- END MODIFIED ---


            iter_time = time.time() - iter_start_time

            # Log objective and tracked weights
            if i % log_interval == 0 or i == max_iter or i == 1:
                current_objective = self._calculate_objective(self.weights)
                print(f"  Iter: {i:>4}/{max_iter} | Objective: {current_objective:.6f} | Iter Time: {iter_time:.3f}s")
                print("    Tracked Weights:")
                for fname in FEATURES_TO_DEBUG:
                     current_weight = self.weights.get(fname, "N/A")
                     is_learnable = hasattr(self, '_learnable_features') and fname in self._learnable_features
                     update = weight_updates.get(fname, "--") if is_learnable else "(fixed)"
                     grad = gradients.get(fname, "N/A")
                     if isinstance(current_weight, float): current_weight = f"{current_weight:.6f}"
                     #if isinstance(update, float): update = f"{update:.6f}" # update is already string or float
                     if isinstance(grad, float): grad = f"{grad:.6f}"
                     learn_status = "" if is_learnable else "(Fixed)"
                     print(f"      '{fname}': Weight={current_weight} {learn_status}, Update={update}, Grad={grad}")

                # --- END DEBUG ---

            if any(not np.isfinite(w) for w in self.weights.values()):
                print(f"ERROR: Non-finite weights encountered at iteration {i}. Stopping.")
                break

        learn_time = time.time() - start_time
        print(f"[Model] Weight learning finished after {i} iterations.") # Use actual iteration count

        if self.weights:
            try:
                weights_values = list(self.weights.values())
                print(f"Learned {len(weights_values)} weights.")
                print(f"Weight stats: Min={np.min(weights_values):.4f}, Max={np.max(weights_values):.4f}, Mean={np.mean(weights_values):.4f}, Std={np.std(weights_values):.4f}")
            except Exception as e_print: print(f"Could not print weight stats: {e_print}")
        else: print("No weights were learned.")
        print(f"[Model] Weight learning phase finished in {learn_time:.2f}s.")


    # ──────────────────────────────────────────────────────────────────────
    # PGMpy GRAPH BUILDING (Keep as is)
    # ──────────────────────────────────────────────────────────────────────
    def _build_pgmpy_graph(self):
        """ Calculates unary factors for each variable and stores them. """
        # ... (factor calculation logic remains the same) ...
        print("[Model] Calculating factors for manual sampling...")
        start_time = time.time()
        self.variable_factors.clear()
        self.graph = FactorGraph()
        variable_node_names = list(self.variable_to_cell_map.keys())
        if not variable_node_names: print("ERROR: No variables defined."); return False
        self.graph.add_nodes_from(variable_node_names)
        # print(f"Added {len(variable_node_names)} variable nodes to graph (for potential check).") # Less verbose
        factors_list = []; problematic_factors = 0; skipped_vars = 0
        # Debug selection
        vars_to_debug = []; evidence_vars_debugged = 0; query_vars_debugged = 0
        available_query_vars = [v for v in variable_node_names if v not in self.evidence_dict]
        for v_name in variable_node_names:
            is_evidence = v_name in self.evidence_dict
            if is_evidence and evidence_vars_debugged < 2: vars_to_debug.append(v_name); evidence_vars_debugged += 1
            elif not is_evidence and query_vars_debugged < 3 and v_name in available_query_vars: vars_to_debug.append(v_name); query_vars_debugged += 1
            if evidence_vars_debugged >= 2 and query_vars_debugged >= 3: break
        if query_vars_debugged < 3 and len(available_query_vars) > query_vars_debugged:
             needed = 3 - query_vars_debugged; vars_to_debug.extend(available_query_vars[query_vars_debugged : query_vars_debugged+needed])
        if vars_to_debug: print(f"Debugging potentials for variables: {vars_to_debug}")

        for var_name in variable_node_names:
            if var_name not in self.variable_states or not self.variable_states[var_name]: skipped_vars += 1; continue
            state_map = self.variable_states[var_name]; num_states = len(state_map)
            if num_states == 0: continue
            is_problematic = False; factor_values = np.zeros(num_states)
            try:
                log_potentials = np.zeros(num_states)
                for state_int in range(num_states): log_potentials[state_int] = self._get_potential_sum(var_name, state_int, self.weights)
                max_log_potential = np.max(log_potentials);
                if np.isneginf(max_log_potential): max_log_potential = -700
                stable_log_potentials = log_potentials - max_log_potential
                factor_values = np.exp(stable_log_potentials)
                if var_name in vars_to_debug:
                     print(f"\n--- DEBUG Potentials for {var_name} (States: {num_states}) ---")
                     tid_attr=self.variable_to_cell_map.get(var_name, "N/A"); print(f"  Cell: {tid_attr}")
                     # print(f"  Log Pots: {log_potentials}") # Can be long
                     print(f"  Max Log: {max_log_potential:.4f}")
                     stable_str = np.array2string(stable_log_potentials, precision=4, max_line_width=120, threshold=10)
                     # print(f"  Stable Logs: {stable_str}") # Can be long
                     vals_str = np.array2string(factor_values, precision=4, max_line_width=120, threshold=10, suppress_small=True)
                     print(f"  Factor Vals (Raw): {vals_str}"); print(f"  Sum: {np.sum(factor_values):.4f}")
                if not np.all(np.isfinite(factor_values)): is_problematic = True # print(f"WARN: {var_name} non-finite.");
                elif np.all(factor_values < self.epsilon): is_problematic = True # print(f"WARN: {var_name} all near-zero.");
                elif np.sum(factor_values) < self.epsilon: is_problematic = True # print(f"WARN: {var_name} near-zero sum.");
                if is_problematic:
                     problematic_factors += 1; factor_values = np.ones(num_states) / num_states
                     if var_name in vars_to_debug: print(f"  --> Assigned Uniform Fallback.")
            except Exception as calc_e:
                print(f"ERROR calculating potentials for {var_name}: {calc_e}. Using uniform."); problematic_factors += 1
                factor_values = np.ones(num_states) / num_states
            ordered_state_names = [state_map[i] for i in range(num_states)]; state_names_dict = {var_name: ordered_state_names}
            try:
                factor = DiscreteFactor(variables=[var_name], cardinality=[num_states], values=factor_values, state_names=state_names_dict)
                self.variable_factors[var_name] = factor
                factors_list.append(factor)
            except Exception as factor_e: print(f"ERROR creating factor {var_name}: {factor_e}"); problematic_factors += 1; import traceback; traceback.print_exc()
        if skipped_vars > 0: print(f"Skipped factor creation for {skipped_vars} vars.")
        if problematic_factors > 0: print(f"WARNING: Assigned uniform fallback to {problematic_factors} factors.")
        # Optional graph check part remains the same...
        # if factors_list:
        #     self.graph.add_nodes_from(factors_list) # Add factors as nodes for check_model if needed
        #     try: self.graph.add_factors(*factors_list)
        #     except Exception as e: print(f"ERROR during add_factors: {e}");
        build_time = time.time() - start_time; print(f"[Model] Factor calculation finished in {build_time:.2f}s.")
        if not self.variable_factors: print("ERROR: No factors were created."); return False
        return True


    # ──────────────────────────────────────────────────────────────────────
    # INFERENCE (Keep Manual Sampling as is)
    # ──────────────────────────────────────────────────────────────────────
    def infer(self, n_samples=1000, n_burn_in=200):
        """ Performs inference using MANUAL independent sampling. """
        # ... (inference logic remains the same) ...
        if not self.variable_factors: print("ERROR: Factors not calculated."); return {}
        print(f"[Model] Starting MANUAL sampling ({n_samples} samples, {n_burn_in} burn-in)...")
        start_time = time.time()
        variables_in_graph_order = sorted(list(self.variable_factors.keys()))
        if not variables_in_graph_order: print("ERROR: No variables with factors."); return {}
        num_vars = len(variables_in_graph_order)
        print(f"Will sample {num_vars} variables independently.")
        samples_list = []
        total_samples_to_gen = n_samples + n_burn_in
        print(f"Generating {total_samples_to_gen} samples manually...")
        progress_interval = max(1, total_samples_to_gen // 10)
        for i in range(total_samples_to_gen):
            current_sample = np.zeros(num_vars, dtype=int)
            for idx, var_name in enumerate(variables_in_graph_order):
                if var_name in self.evidence_dict: current_sample[idx] = self.evidence_dict[var_name]; continue
                factor = self.variable_factors.get(var_name)
                if factor is None: current_sample[idx] = -1; continue
                potentials = factor.values; num_states = len(potentials)
                if num_states == 0: current_sample[idx] = -1; continue
                sum_potentials = np.sum(potentials)
                probs = (potentials / sum_potentials) if (sum_potentials > self.epsilon and np.isfinite(sum_potentials)) else (np.ones(num_states) / num_states)
                probs /= np.sum(probs) # Re-normalize
                try: current_sample[idx] = np.random.choice(a=num_states, p=probs)
                except ValueError as ve: print(f"ERR choice {var_name}: {ve} p={probs}"); current_sample[idx] = -1
            samples_list.append(current_sample)
            if (i + 1) % progress_interval == 0: sys.stdout.write(f"\r  Collected {i + 1}/{total_samples_to_gen} samples..."); sys.stdout.flush()
        print("\nFinished collecting samples.")
        print(f"Total samples collected: {len(samples_list)}")
        if not samples_list: print("ERROR: samples_list empty."); return {}
        print(f"Attempting DataFrame creation ({len(samples_list)}x{num_vars})...")
        try: samples_df = pd.DataFrame(samples_list, columns=variables_in_graph_order); print(f"OK. DataFrame shape: {samples_df.shape}")
        except Exception as e: print(f"ERROR creating DataFrame: {e}"); import traceback; traceback.print_exc(); return {}
        if samples_df is None or samples_df.empty: print("ERROR: Sample DataFrame empty."); return {}
        if n_burn_in >= len(samples_df): print(f"ERROR: Burn-in >= samples."); return {}
        samples_df = samples_df.iloc[n_burn_in:]
        if samples_df.empty: print("ERROR: No samples post-burn-in."); return {}
        print(f"Using {len(samples_df)} samples post-burn-in.")
        marginals = {}
        print(f"Calculating marginals for {len(self.query_vars)} query vars...")
        processed_marginals = 0; skipped_marginals = 0
        query_vars_in_samples = [qv for qv in self.query_vars if qv in samples_df.columns]
        skipped_marginals = len(self.query_vars) - len(query_vars_in_samples)
        if skipped_marginals > 0: print(f"Note: {skipped_marginals} query vars not in sample columns.")
        for var_name in query_vars_in_samples:
             tid_attr = self.variable_to_cell_map.get(var_name);
             if not tid_attr: continue
             state_idx_to_cand_map = self.variable_states.get(var_name, {}); num_states = len(state_idx_to_cand_map)
             if num_states == 0: continue
             valid_samples = samples_df[var_name][samples_df[var_name] != -1]
             if valid_samples.empty:
                  # print(f"WARN: No valid samples for {var_name}. Uniform marginals.") # Verbose
                  marginals[tid_attr] = {c: 1.0/num_states for c in state_idx_to_cand_map.values()}
                  processed_marginals += 1; continue
             state_counts = valid_samples.value_counts(normalize=True); marginals[tid_attr] = {}
             total_prob = 0.0
             for state_int, candidate_val in state_idx_to_cand_map.items():
                 state_prob = state_counts.get(state_int, 0.0); marginals[tid_attr][candidate_val] = state_prob
                 total_prob += state_prob
             processed_marginals += 1
             if total_prob > self.epsilon and abs(total_prob - 1.0) > self.epsilon:
                 norm_factor = 1.0 / total_prob
                 for cand in marginals[tid_attr]: marginals[tid_attr][cand] *= norm_factor
        infer_time = time.time() - start_time
        print(f"Calculated marginals for {processed_marginals} query variables.")
        print(f"[Model] Inference complete in {infer_time:.2f}s.")
        return marginals

    # ──────────────────────────────────────────────────────────────────────
    # PIPELINE RUNNER (Keep as is)
    # ──────────────────────────────────────────────────────────────────────
    def run_pipeline(self, n_samples=1000, n_burn_in=200, learn_iter=100, learning_rate=0.01):
        """Runs the full load, build structures, fit, build graph, infer pipeline."""
        start_pipeline = time.time()
        marginals = None
        if not self._load_data(): print("PIPELINE FAILED: Data loading error."); return None
        if not self._build_internal_structures(): print("PIPELINE FAILED: Error building internal structures."); return None
        self.fit(max_iter=learn_iter, learning_rate=learning_rate) # Pass LR here
        if not self._build_pgmpy_graph(): print("PIPELINE FAILED: Error building pgmpy graph."); return None
        marginals = self.infer(n_samples=n_samples, n_burn_in=n_burn_in)
        pipeline_time = time.time() - start_pipeline
        print(f"\n--- Pipeline execution finished in {pipeline_time:.2f} seconds ---")
        return marginals