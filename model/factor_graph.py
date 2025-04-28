# File: model/factor_graph.py (Refactored)
# Defines the HoloClean factor graph, learning, and inference logic using pgmpy.

import sys
import pandas as pd
import time
import psycopg2
import numpy as np
from pgmpy.models import FactorGraph, MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor, State
from pgmpy.sampling import GibbsSampling
from scipy.optimize import minimize
from collections import defaultdict
import random
import itertools # Added for combinations if needed later, not currently used directly here

log_count = 0

class HoloCleanFactorGraph:
    """
    Represents the HoloClean Factor Graph, handling weight learning and inference.
    Uses Relaxed DC approach (features) by default.
    """
    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.graph: FactorGraph = FactorGraph()  # built in _build_pgmpy_graph

        # maps & caches
        self.cell_to_variable_map: dict[tuple, str] = {}
        self.variable_to_cell_map: dict[str, tuple] = {}
        self.variable_states: dict[str, dict[int, str]] = {}
        self.candidate_map: dict[tuple, dict[str, int]] = {}
        self.features_data: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.weights: dict[str, float] = {}
        self.evidence_dict: dict[str, int] = {}
        self.query_vars: list[str] = []

        # cfg / constants
        self.initial_weight_value = 0.1
        self.l2_penalty = 0.0001
        self.log_potential_clip = 15.0
        self.epsilon = 1e-10

        # cached DB tables
        self.domains_df = None
        self.features_df = None
        self.cells_df = None
        
        self.variable_factors: dict[str, DiscreteFactor] = {}

    # ──────────────────────────────────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────────────────────────────────
    def _load_data(self) -> bool:
        """Fetch domains / features / cells from Postgres."""
        print("[Model] Loading data for graph construction …")
        t0 = time.time()
        try:
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
            if self.cells_df.empty:
                print("ERROR: cells table empty.")
                return False

            print(f"[Model] Data loaded in {time.time() - t0:.2f}s.")
            return True
        except Exception as e:
            print(f"ERROR loading data: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL STRUCTURE BUILDING
    # ──────────────────────────────────────────────────────────────────────
    def _build_internal_structures(self):
        """
        Populates internal mappings (variables, states, features, evidence)
        based on loaded DataFrames. Treats clean NULLs as evidence.
        """
        # ... (Keep initialization and data loading checks as before) ...
        if self.cells_df is None or self.domains_df is None or self.features_df is None: print("ERROR: Data not loaded."); return False
        print("[Model] Building internal structures (variables, states, features)...")
        start_time = time.time()
        # (Clear structures as before)
        self.cell_to_variable_map.clear(); self.variable_to_cell_map.clear(); self.variable_states.clear()
        self.candidate_map.clear(); self.features_data.clear(); self.weights.clear()
        self.evidence_dict.clear(); self.query_vars.clear()

        # (Group domains as before)
        noisy_cell_candidates = defaultdict(list)
        for _, row in self.domains_df.iterrows(): noisy_cell_candidates[(row['tid'], row['attr'])].append(row['candidate_val'])

        # (Define Variables loop as before)
        cell_info_lookup = self.cells_df.set_index(['tid', 'attr']).to_dict('index')
        unique_cells = self.cells_df[['tid', 'attr']].drop_duplicates().to_records(index=False)
        total_vars_created = 0
        for tid, attr in unique_cells:
            tid_attr = (tid, attr); # (Create var_name as before)
            safe_attr = str(attr).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_').replace('.', '_')
            var_name = f"v_{tid}_{safe_attr}"
            if var_name in self.variable_to_cell_map: print(f"WARNING: Duplicate var name {var_name}"); continue
            self.cell_to_variable_map[tid_attr] = var_name; self.variable_to_cell_map[var_name] = tid_attr; total_vars_created += 1
            cell_info = cell_info_lookup.get(tid_attr);
            if not cell_info: continue
            is_noisy = cell_info['is_noisy']; original_val = cell_info['val']

            # --- Define States and Evidence/Query Status ---
            candidates = []
            if is_noisy:
                # (Keep logic for noisy cells as before)
                candidates = sorted(list(set(noisy_cell_candidates.get(tid_attr, []))))
                if not candidates:
                    if original_val is not None: candidates = [original_val]; print(f"WARNING: Noisy {tid_attr} no candidates. Using original.")
                    else: candidates = ["__EMPTY_DOMAIN__"]; print(f"WARNING: Noisy NULL {tid_attr} no candidates. Using dummy.")
                self.query_vars.append(var_name)
            else: # Clean cell
                if original_val is not None:
                    # Clean cell with a non-NULL value: evidence
                    candidates = [original_val]
                    self.evidence_dict[var_name] = 0 # State index 0 is evidence
                else:
                    # === FIX FOR CLEAN NULL ===
                    # Clean cell with NULL value: Treat as EVIDENCE for __NULL__ state
                    candidates = ["__NULL__"] # Define the single state
                    print(f"INFO: Clean cell {tid_attr} is NULL. Treating as EVIDENCE for state '{candidates[0]}'.")
                    self.evidence_dict[var_name] = 0 # State index 0 is evidence for __NULL__
                    # self.query_vars.append(var_name) # <<< REMOVED from query vars

            # (Keep state mapping population as before)
            if candidates:
                num_states = len(candidates); self.variable_states[var_name] = {i: str(val) for i, val in enumerate(candidates)}
                self.candidate_map[tid_attr] = {str(val): i for i, val in enumerate(candidates)}
            else: print(f"CRITICAL WARNING: Var {var_name} has NO states.")
        # (End variable loop)

        print(f"Defined {total_vars_created} variables.")
        print(f"Identified {len(self.evidence_dict)} evidence variables and {len(self.query_vars)} query variables.") # Counts should reflect change
        # (Checks and Feature Prep remain the same)
        if len(self.evidence_dict) + len(self.query_vars) != total_vars_created: print(f"WARNING: Var count mismatch!")
        if not self.evidence_dict and total_vars_created > 0: print("WARNING: No evidence variables identified.")

        # --- Initialize weights if not done by fit() later ---
        if not self.weights:
            all_feature_names = set(self.features_df['feature'].unique()) if self.features_df is not None else set()
            print(f"Initializing {len(all_feature_names)} weights to {self.initial_weight_value} (as fit might be skipped).")
            for feature_name in all_feature_names: self.weights[feature_name] = self.initial_weight_value

        # --- Feature Prep (remains same) ---
        feature_lookup = defaultdict(list); # ... (rest of feature prep loop) ...
        for _, row in self.features_df.iterrows(): feature_lookup[row['feature']].append(((row['tid'], row['attr']), row['candidate_val']))
        all_feature_names = set(self.features_df['feature'].unique()) # Re-get keys based on features_df
        print(f"Found {len(all_feature_names)} unique feature types.")
        self.features_data.clear() # Clear before populating
        processed_feature_links = 0; missing_feature_links = 0
        for feature_name in all_feature_names:
            # self.weights[feature_name] = self.weights.get(feature_name, self.initial_weight_value) # Ensure weight exists
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
                              self.features_data[feature_name].append((var_name, state_int)); processed_feature_links += 1
                         else: missing_feature_links += 1
                     else: missing_feature_links += 1
                 else: missing_feature_links += 1

        # --- AFTER the loop populating self.features_data ---

        print("Building state-to-features reverse map...")
        self.state_to_features_map = defaultdict(list)
        feature_count_in_map = 0
        for feature_name, affected_states_list in self.features_data.items():
            if not isinstance(affected_states_list, list): # Basic check
                 print(f"Warning: Expected list for feature {feature_name}, got {type(affected_states_list)}. Skipping.")
                 continue
            for var_state_tuple in affected_states_list:
                # Ensure var_state_tuple is indeed a tuple of (var_name, state_int)
                if isinstance(var_state_tuple, tuple) and len(var_state_tuple) == 2:
                    self.state_to_features_map[var_state_tuple].append(feature_name)
                    feature_count_in_map += 1
                else:
                    # This case indicates a potential issue upstream in feature generation/loading
                     print(f"Warning: Invalid item '{var_state_tuple}' in affected states list for feature '{feature_name}'. Expected (var_name, state_int). Skipping.")
        

        map_build_time = time.time() - start_time # Reset start_time if needed or use a new one
        print(f"State-to-features map built in {map_build_time:.2f}s. Contains {len(self.state_to_features_map)} state entries and {feature_count_in_map} feature links.")
        # --- End of new section ---

        # (End feature prep)
        build_time = time.time() - start_time
        print(f"Processed {processed_feature_links} feature links. Could not map {missing_feature_links} links.")
        print(f"[Model] Internal structures built in {build_time:.2f}s.")
        return True


    def _get_potential_sum(self, var_name, state_int, current_weights):
        """Calculate sum of (clipped) log potentials for a specific variable state using the precomputed map."""
        log_potential_sum = 0.0
        var_state_key = (var_name, state_int)
    
        # Directly look up the features affecting this state
        relevant_feature_names = self.state_to_features_map.get(var_state_key, []) # Use .get for safety
    
        for feature_name in relevant_feature_names:
            weight = current_weights.get(feature_name, 0.0) # Get current weight from optimizer
            # Clip weight to prevent large values before exp() later
            clipped_weight = np.clip(weight, -self.log_potential_clip, self.log_potential_clip)
            log_potential_sum += clipped_weight
    
        return log_potential_sum

    # ──────────────────────────────────────────────────────────────────────
    # WEIGHT LEARNING
    # ──────────────────────────────────────────────────────────────────────
    def _calculate_log_likelihood(self, weights_array, feature_names_list):
        """
        Calculates Pseudo-Log-Likelihood based on evidence variables.
        Uses log-sum-exp for stability. Includes L2 regularization.
        Returns *negative* PLL for minimization.
        """
        current_weights = dict(zip(feature_names_list, weights_array))
        pseudo_log_likelihood = 0.0
        global log_count # Add 'global log_count' at top of file or handle state differently
        first_var_logged = False # Flag to control detailed logging for the first variable

        if not self.evidence_dict: # If no evidence, PLL is 0 (or could be based on priors)
             # Regularization still applies
             l2_norm_sq = np.sum(weights_array**2)
             penalty = 0.5 * self.l2_penalty * l2_norm_sq
             return penalty

        for var_name, evidence_state_idx in self.evidence_dict.items():
            if var_name not in self.variable_states: continue # Skip if var somehow missing states

            # Calculate potential sum for the actual evidence state
            log_potential_evidence_state = self._get_potential_sum(var_name, evidence_state_idx, current_weights)

            # Calculate potential sums for all possible states of this variable
            log_potential_all_states = []
            num_states = len(self.variable_states[var_name])
            state_map = self.variable_states[var_name] # Get the state map {idx: value}

            if num_states <= 0: continue # Should not happen

            for state_int in range(num_states):
                log_potential_all_states.append(self._get_potential_sum(var_name, state_int, current_weights))

            log_potential_all_states = np.array(log_potential_all_states)
            max_log_potential = np.max(log_potential_all_states)
            if np.isneginf(max_log_potential): max_log_potential = -700 # Avoid issues with all -inf

            #  Calculate log_Z carefully
            shifted_potentials = log_potential_all_states - max_log_potential
            sum_exp_terms = np.sum(np.exp(shifted_potentials))
            log_Z = max_log_potential + np.log(sum_exp_terms + self.epsilon) # Epsilon avoids log(0)

            # Log probability of evidence state: log(exp(pot_evidence) / Z) = pot_evidence - log_Z
            log_prob_evidence = log_potential_evidence_state - log_Z
            pseudo_log_likelihood += log_prob_evidence


            # --- ADDED DETAILED DEBUG FOR FIRST VAR ON FIRST CALL ---
            if log_count == 0 and not first_var_logged:
                 print(f"\n--- Detailed Likelihood Debug (Var: {var_name}, Evidence State Idx: {evidence_state_idx}, Evidence Val: '{state_map.get(evidence_state_idx, 'N/A')}') ---")
                 print(f"  Number of States: {num_states}")
                 # Limit printing potentials if too many states
                 max_states_to_print = 20
                 print(f"  Log Potentials (All States): {np.array2string(log_potential_all_states[:max_states_to_print], precision=4)} {'...' if num_states > max_states_to_print else ''}")
                 print(f"  Log Potential (Evidence State): {log_potential_evidence_state:.6f}")
                 print(f"  Max Log Potential: {max_log_potential:.6f}")
                 print(f"  Log Sum Exp Term (sum(exp(shifted))): {sum_exp_terms:.6f}")
                 print(f"  Log Partition Function (log_Z): {log_Z:.6f}")
                 print(f"  Log Prob Evidence (Pot_ev - log_Z): {log_prob_evidence:.6f}")
                 print(f"--- End Detailed Debug ---")
                 first_var_logged = True # Ensure we on

            if not np.isfinite(pseudo_log_likelihood):
                 print(f"WARNING: Non-finite PLL contribution from {var_name}. Potentials: {log_potential_all_states}, log_Z: {log_Z}")
                 # Return a large penalty to move away from this weight region?
                 return 1e10 * (1 + np.sum(weights_array**2)) # Large value

        # Regularization Penalty (L2)
        l2_norm_sq = np.sum(weights_array**2)
        penalty = 0.5 * self.l2_penalty * l2_norm_sq
        pll_term = -pseudo_log_likelihood
        objective_value = pll_term + penalty

        # We want to MAXIMIZE PLL, so MINIMIZE negative PLL + penalty
        # objective_value = -pseudo_log_likelihood + penalty

        
        if 'log_count' not in globals(): log_count = 0
        if log_count < 2:
             print(f"\n[Likelihood-Debug] Iter {log_count}:")
             print(f"  Neg PLL Term = {pll_term:.6f}")
             print(f"  L2 Penalty   = {penalty:.6f}")
             print(f"  Objective    = {objective_value:.6f}")
             log_count += 1

        if not np.isfinite(objective_value):
             print(f"WARNING: Objective function returned non-finite value. PLL={pseudo_log_likelihood}, Penalty={penalty}")
             return 1e10 * (1 + np.sum(weights_array**2))
        
        return objective_value

    def fit(self, max_iter=100):
        """Learns the feature weights using evidence via L-BFGS-B."""
        if not self.evidence_dict:
             print("WARNING: No evidence variables found. Skipping weight learning. Using initial weights.")
             if not self.weights:
                 all_feature_names = list(self.features_data.keys())
                 for name in all_feature_names:
                     self.weights[name] = self.initial_weight_value
             return

        print(f"[Model] Starting weight learning (Max Iter: {max_iter})...")
        start_time = time.time()

        feature_names_list = list(self.features_data.keys())
        if not feature_names_list:
             print("WARNING: No features found connecting to variable states. Skipping weight learning.")
             if not self.weights:
                  all_feature_names_from_df = set(self.features_df['feature'].unique()) if self.features_df is not None else set()
                  for name in all_feature_names_from_df:
                      self.weights[name] = self.initial_weight_value
             return

        initial_weights_array = np.array([self.weights.get(name, self.initial_weight_value) for name in feature_names_list])

        objective_func = lambda w: self._calculate_log_likelihood(w, feature_names_list)

        # --- Callback for logging progress ---
        iteration_count = [0] # Use a list to allow modification within callback
        callback_start_time = time.time()

        def optimization_callback(xk):
            iteration_count[0] += 1
            current_time = time.time()
            elapsed = current_time - callback_start_time
            # Optionally, calculate objective at xk to show progress (adds overhead)
            # current_objective = objective_func(xk)
            # print(f"\r  Iter: {iteration_count[0]:>4}/{max_iter} | Objective: {current_objective:.4f} | Time: {elapsed:.2f}s", end="")
            print(f"\r  Iter: {iteration_count[0]:>4}/{max_iter} | Time elapsed: {elapsed:.2f}s", end="")
            sys.stdout.flush() # Ensure it prints immediately
        # --- End callback definition ---

        try:
            opt_result = minimize(
                objective_func,
                initial_weights_array,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-7, 'gtol': 1e-5},
                callback=optimization_callback # Pass the callback function
            )
            print() # Newline after optimization finishes

            learned_weights_array = opt_result.x
            if not np.all(np.isfinite(learned_weights_array)):
                print("WARNING: Learned weights contain non-finite values (NaN/inf). Optimization likely failed. Reverting to initial weights.")
                for name in feature_names_list:
                    self.weights[name] = self.initial_weight_value
            else:
                self.weights = dict(zip(feature_names_list, learned_weights_array))

            if opt_result.success:
                print(f"[Model] Weight learning successful after {opt_result.nit} iterations.")
            else:
                print(f"[Model] Weight learning finished ({opt_result.status} - {opt_result.message}). May not have fully converged.")

        except Exception as e:
            print(f"\nERROR during weight optimization: {e}") # Add newline in case callback didn't finish line
            import traceback
            traceback.print_exc()
            print("Reverting to initial weights due to optimization error.")
            for name in feature_names_list:
                self.weights[name] = self.initial_weight_value

        # ... (Keep the rest of the method: weight stats printing, timing) ...
        if self.weights:
            try:
                weights_values = list(self.weights.values())
                print(f"Learned {len(weights_values)} weights.")
                print(f"Weight stats: Min={np.min(weights_values):.4f}, Max={np.max(weights_values):.4f}, Mean={np.mean(weights_values):.4f}, Std={np.std(weights_values):.4f}")
            except Exception as e_print:
                print(f"Could not print weight stats: {e_print}")
        else:
             print("No weights were learned.")

        learn_time = time.time() - start_time
        print(f"[Model] Weight learning phase finished in {learn_time:.2f}s.")




    def _build_pgmpy_graph(self):
        """
        Calculates unary factors for each variable and stores them.
        Optionally builds pgmpy graph structure for validation check.
        """
        print("[Model] Calculating factors for manual sampling...")
        start_time = time.time()
        self.variable_factors.clear() # Clear previous factors
        self.graph = FactorGraph() # Re-initialize graph for optional check

        variable_node_names = list(self.variable_to_cell_map.keys())
        if not variable_node_names: print("ERROR: No variables defined."); return False

        # --- Build graph nodes (optional, for check_model) ---
        self.graph.add_nodes_from(variable_node_names)
        print(f"Added {len(variable_node_names)} variable nodes to graph (for potential check).")

        factors_list = [] # Keep temporary list for check_model
        problematic_factors = 0
        skipped_vars = 0

        # --- Debug selection (keep as before) ---
        vars_to_debug = []; evidence_vars_debugged = 0; query_vars_debugged = 0
        available_query_vars = [v for v in variable_node_names if v not in self.evidence_dict]
        for v_name in variable_node_names:
            is_evidence = v_name in self.evidence_dict
            if is_evidence and evidence_vars_debugged < 2: vars_to_debug.append(v_name); evidence_vars_debugged += 1
            elif not is_evidence and query_vars_debugged < 3 and v_name in available_query_vars: vars_to_debug.append(v_name); query_vars_debugged += 1
            if evidence_vars_debugged >= 2 and query_vars_debugged >= 3: break
        if query_vars_debugged < 3 and len(available_query_vars) > query_vars_debugged:
             needed = 3 - query_vars_debugged; vars_to_debug.extend(available_query_vars[query_vars_debugged : query_vars_debugged+needed])
        print(f"Debugging potentials for variables: {vars_to_debug}")
        # --- End debug selection ---

        for var_name in variable_node_names:
            if var_name not in self.variable_states or not self.variable_states[var_name]:
                skipped_vars += 1; continue
            state_map = self.variable_states[var_name]; num_states = len(state_map)
            if num_states == 0: continue

            is_problematic = False; factor_values = np.zeros(num_states)
            try:
                # (Keep potential calculation and validation/fallback logic as before)
                log_potentials = np.zeros(num_states)
                for state_int in range(num_states): log_potentials[state_int] = self._get_potential_sum(var_name, state_int, self.weights)
                max_log_potential = np.max(log_potentials);
                if np.isneginf(max_log_potential): max_log_potential = -700
                stable_log_potentials = log_potentials - max_log_potential
                factor_values = np.exp(stable_log_potentials)
                # Debug Print
                if var_name in vars_to_debug:
                     print(f"\n--- DEBUG Potentials for {var_name} (States: {num_states}) ---")
                     tid_attr=self.variable_to_cell_map.get(var_name, "N/A"); print(f"  Cell: {tid_attr}")
                     print(f"  Log Pots: {log_potentials}"); print(f"  Max Log: {max_log_potential:.4f}")
                     stable_str = np.array2string(stable_log_potentials, precision=4, max_line_width=120, threshold=10)
                     print(f"  Stable Logs: {stable_str}")
                     vals_str = np.array2string(factor_values, precision=4, max_line_width=120, threshold=10, suppress_small=True)
                     print(f"  Factor Vals (Raw): {vals_str}"); print(f"  Sum: {np.sum(factor_values):.4f}")
                # Validation
                if not np.all(np.isfinite(factor_values)): print(f"WARNING: {var_name} non-finite. Using uniform."); is_problematic = True
                elif np.all(factor_values < self.epsilon): print(f"WARNING: {var_name} all near-zero. Using uniform."); is_problematic = True
                elif np.sum(factor_values) < self.epsilon: print(f"WARNING: {var_name} near-zero sum. Using uniform."); is_problematic = True
                if is_problematic:
                     problematic_factors += 1; factor_values = np.ones(num_states) / num_states
                     if var_name in vars_to_debug: print(f"  --> Assigned Uniform Fallback.")

            except Exception as calc_e:
                print(f"ERROR calculating potentials for {var_name}: {calc_e}. Using uniform."); problematic_factors += 1
                factor_values = np.ones(num_states) / num_states

            # Create the DiscreteFactor
            ordered_state_names = [state_map[i] for i in range(num_states)]; state_names_dict = {var_name: ordered_state_names}
            try:
                factor = DiscreteFactor(variables=[var_name], cardinality=[num_states], values=factor_values, state_names=state_names_dict)
                # --- STORE FACTOR --- <<< MODIFIED
                self.variable_factors[var_name] = factor
                factors_list.append(factor) # Keep for optional check_model
            except Exception as factor_e: print(f"ERROR creating factor {var_name}: {factor_e}"); problematic_factors += 1; import traceback; traceback.print_exc()
        # (End factor creation loop)

        if skipped_vars > 0: print(f"Skipped factor creation for {skipped_vars} vars.")
        if problematic_factors > 0: print(f"WARNING: Assigned uniform fallback to {problematic_factors} factors.")

        # --- Optional: Build rest of graph for check_model ---
        # This part is now less critical as we aren't using pgmpy's sampler
        # but we can keep it for validation if desired.
        if factors_list:
            print(f"Adding {len(factors_list)} factor objects as nodes (for check)...")
            self.graph.add_nodes_from(factors_list)
            print(f"Registering {len(factors_list)} factors internally (for check)...")
            try: self.graph.add_factors(*factors_list); print("OK.")
            except Exception as e: print(f"ERROR during add_factors: {e}"); # Don't fail pipeline here
        # --- End Optional Part ---

        build_time = time.time() - start_time; print(f"[Model] Factor calculation finished in {build_time:.2f}s.")

        # --- Optional: Check Model Validity ---
        # print("Checking graph model validity (optional)...")
        # try:
        #     if self.graph.check_model(): print("Graph model check passed.")
        #     else: print("Graph model check failed.")
        # except Exception as e: print(f"Graph check raised exception: {e}")
        # --- End Optional Check ---

        # Ensure factors were actually created
        if not self.variable_factors:
            print("ERROR: No factors were created/stored. Cannot proceed.")
            return False
        return True # Success if factors were created

    # ──────────────────────────────────────────────────────────────────────
    # INFERENCE  (⇐ **fully rewritten for “Option A”**)
    # ──────────────────────────────────────────────────────────────────────
    def infer(self, n_samples=1000, n_burn_in=200):
        """
        Performs inference using MANUAL independent sampling for each variable
        based on its calculated unary factor.
        """
        # Check if factors were created
        if not self.variable_factors:
             print("ERROR: Factors not calculated before inference. Run _build_pgmpy_graph() first.")
             return {}

        print(f"[Model] Starting MANUAL sampling ({n_samples} samples, {n_burn_in} burn-in)...")
        start_time = time.time()

        # Get the ordered list of variables we need to sample
        # Use keys from variable_factors which should match graph nodes if check was done
        variables_in_graph_order = sorted(list(self.variable_factors.keys()))
        if not variables_in_graph_order:
             print("ERROR: No variables found with factors to sample.")
             return {}
        num_vars = len(variables_in_graph_order)
        print(f"Will sample {num_vars} variables independently.")

        # --- Manual Sampling Loop ---
        samples_list = []
        total_samples_to_gen = n_samples + n_burn_in
        print(f"Generating {total_samples_to_gen} samples manually...")
        progress_interval = max(1, total_samples_to_gen // 10)

        for i in range(total_samples_to_gen):
            current_sample = np.zeros(num_vars, dtype=int) # Store state indices

            for idx, var_name in enumerate(variables_in_graph_order):
                # Evidence variables: Keep their fixed state
                if var_name in self.evidence_dict:
                    current_sample[idx] = self.evidence_dict[var_name]
                    continue

                # Query variables: Sample from their factor distribution
                factor = self.variable_factors[var_name]
                potentials = factor.values # Get potential values
                num_states = len(potentials)

                if num_states == 0: # Should not happen if factor exists
                     print(f"WARNING: Variable {var_name} factor has no states? Skipping sample.")
                     # What state to assign? Maybe random? Or mark as error? Put -1?
                     current_sample[idx] = -1 # Indicate error/missing
                     continue

                # Normalize potentials to get probabilities
                sum_potentials = np.sum(potentials)
                if sum_potentials < self.epsilon or not np.isfinite(sum_potentials):
                     # If sum is zero or non-finite, use uniform probability
                     probs = np.ones(num_states) / num_states
                     if i == 0 and idx < 5: # Print warning only once per var
                          print(f"WARNING: Zero/invalid potential sum for {var_name}. Sampling uniformly.")
                else:
                     probs = potentials / sum_potentials

                # Sample state index using calculated probabilities
                try:
                     sampled_state_idx = np.random.choice(a=num_states, p=probs)
                     current_sample[idx] = sampled_state_idx
                except ValueError as ve:
                     print(f"ERROR during np.random.choice for {var_name}: {ve}")
                     print(f"  Num states: {num_states}, Probs: {probs}, Sum(Probs): {np.sum(probs)}")
                     current_sample[idx] = -1 # Indicate error

            # Add the completed sample (list/array of state indices) to our list
            samples_list.append(current_sample)

            # Progress Update
            if (i + 1) % progress_interval == 0:
                import sys
                sys.stdout.write(f"\r  Collected {i + 1}/{total_samples_to_gen} samples...")
                sys.stdout.flush()

        print("\nFinished collecting samples.")

        # --- Create DataFrame (should work now) ---
        print(f"Total samples collected: {len(samples_list)}")
        if not samples_list: print("ERROR: samples_list empty."); return {}
        print(f"Attempting DataFrame creation ({len(samples_list)}x{num_vars})...")
        try:
             samples_df = pd.DataFrame(samples_list, columns=variables_in_graph_order)
             print(f"OK. DataFrame shape: {samples_df.shape}")
        except Exception as e:
             print(f"ERROR creating DataFrame: {e}"); import traceback; traceback.print_exc(); return {}

        # --- Process Samples (Burn-in) ---
        if samples_df is None or samples_df.empty: print("ERROR: Sample DataFrame empty."); return {}
        if n_burn_in >= len(samples_df): print(f"ERROR: Burn-in >= samples."); return {}
        samples_df = samples_df.iloc[n_burn_in:]
        if samples_df.empty: print("ERROR: No samples post-burn-in."); return {}
        print(f"Using {len(samples_df)} samples post-burn-in.")

        # --- Calculate Marginals ---
        marginals = {}
        print(f"Calculating marginals for {len(self.query_vars)} query vars...")
        processed_marginals = 0; skipped_marginals = 0
        query_vars_in_samples = [qv for qv in self.query_vars if qv in samples_df.columns]
        skipped_marginals = len(self.query_vars) - len(query_vars_in_samples)
        if skipped_marginals > 0: print(f"Note: {skipped_marginals} query vars not in sample columns.")
        # (Marginal calculation loop - check for state index -1)
        for var_name in query_vars_in_samples:
             tid_attr = self.variable_to_cell_map.get(var_name); # ... (rest of loop)
             if not tid_attr: continue
             state_idx_to_cand_map = self.variable_states.get(var_name, {}); num_states = len(state_idx_to_cand_map)
             if num_states == 0: continue
             # Filter out error state -1 before calculating value_counts
             valid_samples = samples_df[var_name][samples_df[var_name] != -1]
             if valid_samples.empty:
                  print(f"WARNING: No valid samples found for {var_name}. Assigning uniform marginals.")
                  marginals[tid_attr] = {c: 1.0/num_states for c in state_idx_to_cand_map.values()}
                  processed_marginals += 1
                  continue
             # Calculate marginals from valid samples
             state_counts = valid_samples.value_counts(normalize=True); marginals[tid_attr] = {}
             total_prob = 0.0
             for state_int, candidate_val in state_idx_to_cand_map.items():
                 state_prob = state_counts.get(state_int, 0.0); marginals[tid_attr][candidate_val] = state_prob
                 total_prob += state_prob
             processed_marginals += 1
             if total_prob > self.epsilon and abs(total_prob - 1.0) > self.epsilon:
                 norm_factor = 1.0 / total_prob
                 for cand in marginals[tid_attr]: marginals[tid_attr][cand] *= norm_factor
        # (End marginal calculation loop)
        infer_time = time.time() - start_time
        print(f"Calculated marginals for {processed_marginals} query variables.")
        print(f"[Model] Inference complete in {infer_time:.2f}s.")
        return marginals



    def run_pipeline(self, n_samples=1000, n_burn_in=200, learn_iter=100):
        """Runs the full load, build structures, fit, build graph, infer pipeline."""
        start_pipeline = time.time()
        marginals = None

        # 1. Load Data
        if not self._load_data():
            print("PIPELINE FAILED: Data loading error.")
            return None

        # 2. Build Internal Structures
        if not self._build_internal_structures():
            print("PIPELINE FAILED: Error building internal structures.")
            return None

        # 3. Learn Weights
        self.fit(max_iter=learn_iter)
        # To skip learning and use initial weights:
        # if not self.weights: # Initialize if fit was skipped and they are empty
        #     all_feature_names = list(self.features_data.keys())
        #     for name in all_feature_names: self.weights[name] = self.initial_weight_value
        # print("Skipped weight learning, using initial values.")

        # 4. Build pgmpy Graph Object
        if not self._build_pgmpy_graph():
            print("PIPELINE FAILED: Error building pgmpy graph or graph check failed.")
            return None

        # 5. Run Inference
        marginals = self.infer(n_samples=n_samples, n_burn_in=n_burn_in)

        pipeline_time = time.time() - start_pipeline
        print(f"\n--- Pipeline execution finished in {pipeline_time:.2f} seconds ---")
        return marginals