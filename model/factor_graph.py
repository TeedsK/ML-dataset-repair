# File: model/factor_graph.py (Corrected)
# Defines the HoloClean factor graph, learning, and inference logic using pgmpy.

import pandas as pd
import time
import psycopg2
import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.sampling import GibbsSampling
from pgmpy.utils import get_example_model # For state names if needed
from scipy.optimize import minimize
from collections import defaultdict
import itertools
import random

class HoloCleanFactorGraph:
    """
    Represents the HoloClean Factor Graph, handling weight learning and inference.
    Uses Relaxed DC approach (features) by default.
    """
    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.graph = FactorGraph()
        self.variables = {} # Maps (tid, attr) -> variable_name string
        self.variable_states = {} # Maps variable_name -> {state_int: candidate_val_str}
        self.candidate_map = {} # Maps (tid, attr) -> {candidate_val_str: state_int}
        self.features_data = {} # Maps feature_name -> list of (var_name, state_int) affected
        self.weights = {} # Maps feature_name -> learned_weight
        self.evidence_dict = {} # Maps var_name -> state_int for clean cells (evidence)
        self.query_vars = [] # List of variable_names for noisy cells (query)

        # Tunable parameters
        self.initial_weight_value = 0.01
        self.l2_penalty = 0.01
        self.epsilon = 1e-10

        # Store dataframes
        self.domains_df = None
        self.features_df = None
        self.cells_df = None

    def _load_data(self):
        """Loads domains, features, and cell info from the database."""
        print("[Model] Loading data for graph construction...")
        start_time = time.time()
        try:
            # Domains define potential states for *noisy* variables
            self.domains_df = pd.read_sql("SELECT tid, attr, candidate_val FROM domains ORDER BY tid, attr, candidate_val", self.db_conn)
            print(f"Loaded {len(self.domains_df)} domain entries (for noisy cells).")
            # No need to error check if empty, as clean cells might exist

            # Features define unary factors - they might relate to clean or noisy cells implicitly
            self.features_df = pd.read_sql("SELECT tid, attr, candidate_val, feature FROM features", self.db_conn)
            print(f"Loaded {len(self.features_df)} feature entries.")

            # Cell info needed for ALL cells to build full graph
            self.cells_df = pd.read_sql("SELECT tid, attr, val, is_noisy FROM cells", self.db_conn)
            print(f"Loaded {len(self.cells_df)} total cell entries.")
            if self.cells_df.empty:
                 print("Error: Cells table is empty. Did ingestion run correctly?")
                 return False

            load_time = time.time() - start_time
            print(f"[Model] Data loading complete in {load_time:.2f}s.")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _build_graph_structure(self):
        """Constructs the pgmpy FactorGraph structure for ALL relevant cells."""
        if self.cells_df is None:
             print("Error: Cells data not loaded before building graph structure.")
             return

        print("[Model] Building factor graph structure (representing all cells)...")
        start_time = time.time()
        self.variables.clear()
        self.variable_states.clear()
        self.candidate_map.clear()
        self.evidence_dict.clear()
        self.query_vars.clear()
        self.features_data.clear()
        self.weights.clear()
        self.graph = FactorGraph()

        # Create lookup for domains of noisy cells
        # Group domains by (tid, attr) to define states for noisy variables
        noisy_cell_candidates = defaultdict(list)
        if self.domains_df is not None and not self.domains_df.empty:
             for index, row in self.domains_df.iterrows():
                 noisy_cell_candidates[(row['tid'], row['attr'])].append(row['candidate_val'])

        # Create lookup for feature data
        # Map features to the (tid, attr, candidate_val) they affect
        feature_lookup = defaultdict(list)
        if self.features_df is not None and not self.features_df.empty:
             for index, row in self.features_df.iterrows():
                 feature_lookup[row['feature']].append(((row['tid'], row['attr']), row['candidate_val']))


        # --- Define Variables (Nodes) for ALL cells ---
        # Iterate through all unique cells found in the cells table
        unique_cells = self.cells_df[['tid', 'attr']].drop_duplicates().to_records(index=False)
        cell_info_lookup = self.cells_df.set_index(['tid', 'attr']).to_dict('index')

        total_vars = 0
        for tid, attr in unique_cells:
            tid_attr = (tid, attr)
            var_name = f"v_{tid}_{attr}"
            self.variables[tid_attr] = var_name
            self.graph.add_node(var_name)
            total_vars += 1

            cell_info = cell_info_lookup.get(tid_attr)
            if not cell_info: continue # Should not happen

            is_noisy = cell_info['is_noisy']
            original_val = cell_info['val'] # Can be None

            if is_noisy:
                # *** Noisy Cell -> Query Variable ***
                candidates = sorted(list(set(noisy_cell_candidates.get(tid_attr, []))))
                # Ensure original value is a candidate if domain pruning missed it but cell is noisy?
                # Or rely entirely on domain pruning? Let's rely on pruning for now.
                # if original_val is not None and original_val not in candidates:
                #      candidates.append(original_val) # Optionally force include original value
                #      candidates.sort()

                if not candidates:
                     # If a noisy cell has no candidates (e.g., due to aggressive pruning)
                     # We must give it at least one state, perhaps its original value?
                     if original_val is not None:
                         candidates = [original_val]
                         # print(f"Warning: Noisy cell {tid_attr} had no candidates from pruning. Using original value '{original_val}' as only state.")
                     else:
                         # print(f"Warning: Noisy cell {tid_attr} with NULL original value has no candidates. Skipping variable state definition.")
                         # This variable might cause issues later if it has no states.
                         # Let's assign a placeholder state? Or handle downstream?
                         # For now, skip adding states, but it might need a default factor later.
                         continue # Skip state definition for now

                num_states = len(candidates)
                self.variable_states[var_name] = {i: val for i, val in enumerate(candidates)}
                self.candidate_map[tid_attr] = {val: i for i, val in enumerate(candidates)}
                self.query_vars.append(var_name)

            else:
                # *** Clean Cell -> Evidence Variable ***
                if original_val is None:
                     # How to represent evidence for NULL? Maybe skip creating evidence?
                     # Or create a special state? For now, treat as query if original is NULL.
                     # print(f"Warning: Clean cell {tid_attr} has NULL value. Treating as query variable with NULL state?")
                     # Let's define it as a query variable with one state 'None' for now.
                     # This might need refinement based on how NULLs should be handled.
                     candidates = ['__NULL__'] # Special placeholder state
                     num_states = 1
                     self.variable_states[var_name] = {0: candidates[0]}
                     self.candidate_map[tid_attr] = {candidates[0]: 0}
                     self.query_vars.append(var_name) # Treat as query for now
                else:
                    # Define variable with 1 state: the original value
                    candidates = [original_val]
                    num_states = 1
                    state_int = 0
                    self.variable_states[var_name] = {state_int: original_val}
                    self.candidate_map[tid_attr] = {original_val: state_int}
                    # Set evidence
                    self.evidence_dict[var_name] = state_int


        print(f"Defined {total_vars} variables (representing all cells).")
        print(f"Identified {len(self.evidence_dict)} evidence variables and {len(self.query_vars)} query variables.")
        if not self.evidence_dict:
             print("CRITICAL WARNING: No evidence variables identified. Weight learning will be skipped.")


        # --- Prepare Feature Data for Factor Creation ---
        # (This part needs adjustment to map features to the potentially new variable states)
        all_feature_names = set(self.features_df['feature'].unique()) if self.features_df is not None else set()
        print(f"Found {len(all_feature_names)} unique feature types.")
        self.features_data.clear()
        self.weights.clear()

        for feature_name in all_feature_names:
            self.weights[feature_name] = self.initial_weight_value
            self.features_data[feature_name] = [] # Reset list for this feature

            # Find all (tid_attr, candidate_val) affected by this feature
            affected_cells_candidates = feature_lookup.get(feature_name, [])

            for tid_attr, candidate_val in affected_cells_candidates:
                 var_name = self.variables.get(tid_attr)
                 if var_name:
                     state_map = self.candidate_map.get(tid_attr, {})
                     if candidate_val in state_map:
                         state_int = state_map[candidate_val]
                         # Ensure this var/state actually exists before adding
                         if var_name in self.variable_states and state_int in self.variable_states[var_name]:
                             self.features_data[feature_name].append((var_name, state_int))


        build_time = time.time() - start_time
        print(f"[Model] Graph structure built in {build_time:.2f}s.")


    def _get_potential_sum(self, var_name, state_int, current_weights):
        """Calculate sum of weighted potentials for a specific variable state."""
        potential_sum = 0.0
        # Find all features affecting this specific state
        for feature_name, affected_states in self.features_data.items():
             # Check if this feature affects the var_name and state_int
             if (var_name, state_int) in affected_states:
                  weight = current_weights.get(feature_name, 0.0)
                  potential_sum += weight # In log space, potentials sum
        return potential_sum


    def _calculate_log_likelihood(self, weights_array, feature_names_list):
        """
        Calculates Pseudo-Log-Likelihood based on evidence variables.
        PLL(theta) = sum_{evidence_var} log P(evidence_var | MB(evidence_var), theta)
        Approximation: Use sum of weights affecting the evidence state.
        Return *negative* PLL for minimization.
        """
        current_weights = dict(zip(feature_names_list, weights_array))
        pseudo_log_likelihood = 0.0

        # *** FIX: Improved PLL Approximation ***
        for var_name, evidence_state in self.evidence_dict.items():
            # Calculate potential for the actual evidence state
            log_potential_evidence_state = self._get_potential_sum(var_name, evidence_state, current_weights)

            # Calculate potentials for all possible states of this variable
            log_potential_all_states = []
            num_states = len(self.variable_states[var_name])
            for state_int in range(num_states):
                log_potential_all_states.append(self._get_potential_sum(var_name, state_int, current_weights))

            # Calculate log partition function (log-sum-exp for numerical stability)
            max_log_potential = np.max(log_potential_all_states)
            log_Z = max_log_potential + np.log(np.sum(np.exp(np.array(log_potential_all_states) - max_log_potential)) + self.epsilon)

            # Log probability of evidence state P(var=evidence | MB(var)) approx log(exp(pot_ev) / Z)
            log_prob_evidence = log_potential_evidence_state - log_Z
            pseudo_log_likelihood += log_prob_evidence


        # Regularization Penalty (L2)
        l2_norm_sq = np.sum(weights_array**2)
        penalty = 0.5 * self.l2_penalty * l2_norm_sq

        # We want to MAXIMIZE PLL, so MINIMIZE negative PLL + penalty
        objective_value = -pseudo_log_likelihood + penalty

        # Debug print (optional, can be verbose)
        # print(f"NegPLL: {objective_value:.4f} (PLL part: {-pseudo_log_likelihood:.4f}, Penalty: {penalty:.4f})")
        return objective_value


    def fit(self, max_iter=100):
        """Learns the feature weights using evidence."""
        if not self.evidence_dict:
             print("Warning: No evidence variables found. Skipping weight learning. Using initial weights.")
             # Ensure weights are initialized if skipping
             if not self.weights:
                 all_feature_names = set(self.features_df['feature'].unique())
                 for feature_name in all_feature_names:
                     self.weights[feature_name] = self.initial_weight_value
             return

        print("[Model] Starting weight learning...")
        start_time = time.time()

        feature_names_list = list(self.weights.keys())
        initial_weights_array = np.array([self.weights.get(name, self.initial_weight_value) for name in feature_names_list])

        objective_func = lambda w: self._calculate_log_likelihood(w, feature_names_list)

        opt_result = minimize(
            objective_func,
            initial_weights_array,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-7, 'gtol': 1e-5} # Adjust tolerances
        )

        learned_weights_array = opt_result.x
        # *** FIX: Check for overflow/NaN in weights after optimization ***
        if not np.all(np.isfinite(learned_weights_array)):
             print("Warning: Learned weights contain non-finite values (NaN/inf). Learning may have failed. Reverting to initial weights.")
             # Revert to initial weights to avoid exp() overflow later
             for name in feature_names_list:
                 self.weights[name] = self.initial_weight_value
        else:
             self.weights = dict(zip(feature_names_list, learned_weights_array))


        if opt_result.success:
            print("[Model] Weight learning successful.")
        else:
            print(f"[Model] Weight learning may not have fully converged ({opt_result.status}). Message: {opt_result.message}")


        # Optional: Print some learned weights
        try:
            sorted_weights = sorted(self.weights.items(), key=lambda item: abs(item[1]), reverse=True)
            print("Top 5 learned weights (abs value):", sorted_weights[:5])
            # print("Bottom 5 learned weights (abs value):", sorted_weights[-5:])
            min_w = min(self.weights.values())
            max_w = max(self.weights.values())
            print(f"Weight range: [{min_w:.4f}, {max_w:.4f}]")
        except Exception as e:
             print(f"Could not print sorted weights: {e}") # Handle cases with few weights etc.


        learn_time = time.time() - start_time
        print(f"[Model] Weight learning finished in {learn_time:.2f}s.")


    def _add_factors_to_graph(self):
        """Adds factors to the pgmpy graph using the learned weights."""
        print("[Model] Adding factors to pgmpy graph...")
        start_time = time.time()
        self.graph = FactorGraph() # Recreate graph to clear old factors/nodes

        # Add nodes first
        for var_name in self.variables.values():
            self.graph.add_node(var_name)

        factors_added_count = 0
        # --- Add Unary Factors from Features ---
        var_potentials = defaultdict(lambda: defaultdict(float)) # {var_name: {state: log_potential_sum}}

        for feature_name, affected_states in self.features_data.items():
            weight = self.weights.get(feature_name, 0.0)
            # Clip weights to prevent overflow in exp()
            clipped_weight = np.clip(weight, -700, 700) # exp(700) is large but avoids immediate overflow

            for var_name, state_int in affected_states:
                 # Sum weights in log space
                 var_potentials[var_name][state_int] += clipped_weight


        # Create one factor per variable from the summed log potentials
        for var_name, state_log_potentials in var_potentials.items():
             if var_name not in self.variable_states: continue # Skip if variable definition missing

             var_cardinality = len(self.variable_states[var_name])
             # Initialize factor log values to 0
             factor_log_values = np.zeros(var_cardinality)

             for state_int, log_potential_sum in state_log_potentials.items():
                 if 0 <= state_int < var_cardinality:
                     factor_log_values[state_int] = log_potential_sum

             # Convert log potentials to potentials: exp(log_values)
             # Handle potential underflow/overflow with stabilization
             max_log_val = np.max(factor_log_values)
             if np.isinf(max_log_val): max_log_val = 700 # Avoid inf
             stable_log_values = factor_log_values - max_log_val
             factor_values = np.exp(stable_log_values) + self.epsilon # Add epsilon for stability

             try:
                factor = DiscreteFactor(
                    variables=[var_name],
                    cardinality=[var_cardinality],
                    values=factor_values
                )
                self.graph.add_factors(factor)
                factors_added_count += 1
             except ValueError as ve:
                 print(f"Error creating factor for {var_name}: {ve}")
                 print(f"  Cardinality: {var_cardinality}")
                 print(f"  Values shape: {factor_values.shape}")
                 print(f"  Values: {factor_values}")


        # --- Add Cardinality Factors (Now potentially necessary) ---
        # Since we combined feature weights into single factors per variable,
        # the sampling might not implicitly enforce the categorical constraint.
        # Add a factor for each variable ensuring only one state is '1'.
        # This is hard to enforce directly in standard Gibbs. Let's rely on the
        # categorical nature of the sampling process itself for now. If results
        # are poor, explicit constraint factors might be needed, making inference harder.

        duration = time.time() - start_time
        print(f"Added {factors_added_count} factors (one per variable with features) in {duration:.2f}s.")


    def infer(self, n_samples=1000, n_burn_in=200):
        """Performs inference using Gibbs sampling."""
        if not self.query_vars:
             print("[Model] No query variables to perform inference on. Skipping.")
             return {}

        print(f"[Model] Starting Gibbs sampling ({n_samples} samples, {n_burn_in} burn-in)...")
        start_time = time.time()

        self._add_factors_to_graph()
        if not self.graph.get_factors():
             print("Warning: No factors added to the graph. Inference might yield uniform probabilities.")
             # If no factors, maybe return uniform marginals? Or empty?
             return {} # Return empty if no factors to sample from

        gibbs = GibbsSampling(self.graph)

        start_state = {}
        # Initialize ALL variables: evidence fixed, query random
        for var_name in self.graph.nodes(): # Iterate through all nodes added
            if var_name in self.evidence_dict:
                start_state[var_name] = self.evidence_dict[var_name]
            else:
                num_states = len(self.variable_states.get(var_name, {}))
                if num_states > 0:
                     start_state[var_name] = random.randint(0, num_states - 1)
                else:
                     # Handle variable with no states - assign a dummy state? Or skip?
                     # Assigning 0, but this variable shouldn't exist ideally.
                     start_state[var_name] = 0
                     print(f"Warning: Variable {var_name} has no defined states. Assigning state 0.")


        start_state_list = [tuple([var, state]) for var, state in start_state.items()]
        # Ensure start_state_list covers all nodes in the graph
        if len(start_state_list) != len(self.graph.nodes()):
             print(f"Error: Start state length ({len(start_state_list)}) doesn't match graph nodes ({len(self.graph.nodes())}).")
             # Find missing nodes maybe?
             missing_nodes = set(self.graph.nodes()) - set(s[0] for s in start_state_list)
             print(f"Missing nodes: {missing_nodes}")
             return {} # Cannot proceed


        print("Generating samples (this may take time)...")
        samples_list = []
        try:
            # Set show_progress=False as manual printing is done
            sample_generator = gibbs.generate_sample(start_state=start_state_list, size=n_samples + n_burn_in, seed=42)

            progress_interval = (n_samples + n_burn_in) // 10
            if progress_interval == 0: progress_interval = 1

            for i, sample in enumerate(sample_generator):
                # Check sample format - should be a list/tuple of states
                if not isinstance(sample, (list, tuple)) or len(sample) != len(start_state_list):
                     print(f"Error: Unexpected sample format at iteration {i}. Got: {sample}")
                     # Should be list of states like [0, 1, 0, ...]
                     # If it's state names, need to map back based on variable_order
                     # Let's assume it returns states corresponding to start_state order
                     return {} # Cannot proceed
                samples_list.append(sample)
                if (i + 1) % progress_interval == 0:
                    print(f"  Generated {i + 1}/{n_samples + n_burn_in} samples...")

        except Exception as e:
             print(f"Error during Gibbs sampling generation: {e}")
             import traceback
             traceback.print_exc()
             return {} # Stop inference on error


        if not samples_list:
             print("Error: No samples generated by Gibbs sampler.")
             return {}

        # Convert list of samples (tuples/lists of state values) to DataFrame
        variable_order = [s[0] for s in start_state_list] # Get variable names in order
        samples_array = np.array(samples_list)

        try:
            # Ensure correct shape before creating DataFrame
            if samples_array.shape != (len(samples_list), len(variable_order)):
                 raise ValueError(f"Shape mismatch: samples_array {samples_array.shape}, expected {(len(samples_list), len(variable_order))}")
            samples_df = pd.DataFrame(samples_array, columns=variable_order)
        except ValueError as e:
             print(f"Error creating DataFrame from samples: {e}")
             print(f"Sample list length: {len(samples_list)}")
             if samples_list: print(f"First sample example: {samples_list[0][:10]}...") # Print only first few states
             print(f"Variable order length: {len(variable_order)}")
             return {} # Cannot proceed


        samples_df = samples_df.iloc[n_burn_in:]
        if samples_df.empty:
             print("Error: No samples remaining after burn-in.")
             return {}
        print(f"Generated {len(samples_df)} samples after burn-in.")

        # --- Calculate Marginals (ONLY for Query Vars)---
        marginals = {}
        for var_name in self.query_vars: # Iterate only over query variables
             if var_name not in samples_df.columns:
                 print(f"Warning: Query variable '{var_name}' not found in sample columns. Skipping.")
                 continue

             tid_attr = next((key for key, val in self.variables.items() if val == var_name), None)
             if not tid_attr: continue

             state_to_cand_map = self.variable_states.get(var_name, {})
             num_states = len(state_to_cand_map)
             if num_states == 0: continue

             state_counts = samples_df[var_name].value_counts(normalize=True)
             state_probs = state_counts.to_dict()

             marginals[tid_attr] = {}
             total_prob = 0.0
             for state_int, candidate_val in state_to_cand_map.items():
                 prob = state_probs.get(state_int, 0.0)
                 marginals[tid_attr][candidate_val] = prob
                 total_prob += prob

             if total_prob > self.epsilon and abs(total_prob - 1.0) > self.epsilon:
                 for cand in marginals[tid_attr]:
                     if total_prob != 0: # Avoid division by zero
                        marginals[tid_attr][cand] /= total_prob
                     else: # If total_prob is zero, keep probs as zero
                        marginals[tid_attr][cand] = 0.0


        infer_time = time.time() - start_time
        print(f"[Model] Inference complete in {infer_time:.2f}s.")
        return marginals

    def run_pipeline(self, n_samples=1000, n_burn_in=200, learn_iter=100):
        """Runs the full load, build, fit, infer pipeline."""
        if self._load_data():
            self._build_graph_structure()

            print("!!! SKIPPING WEIGHT LEARNING FOR DEBUGGING !!!")
            # Ensure weights are initialized if skipping fit
            if not self.weights:
                 all_feature_names = set(self.features_df['feature'].unique()) if self.features_df is not None else set()
                 for feature_name in all_feature_names:
                     self.weights[feature_name] = self.initial_weight_value
                 print(f"Using initial weights: {self.initial_weight_value}")
                 
            # self.fit(max_iter=learn_iter) # Uncomment HERE to enable weight learning


            marginals = self.infer(n_samples=n_samples, n_burn_in=n_burn_in)
            return marginals
        else:
            print("Pipeline aborted due to data loading failure.")
            return None