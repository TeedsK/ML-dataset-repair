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
        print("Sample variable names being generated:") # Add log
        sample_vars_printed = 0

        for tid, attr in unique_cells:
            tid_attr = (tid, attr)
            # var_name = f"v_{tid}_{attr}"
            # self.variables[tid_attr] = var_name
            # self.graph.add_node(var_name)
            
            # *** Check for problematic characters in attr? ***
            # Basic sanitation: replace spaces, maybe other chars if needed
            safe_attr = str(attr).replace(' ', '_').replace('/', '_').replace('\\', '_')
            var_name = f"v_{tid}_{safe_attr}" # Use safe_attr

            # Print first few generated names
            if sample_vars_printed < 5: # Add log
                 print(f"  - ({tid}, {attr}) -> '{var_name}'") # Add log
                 sample_vars_printed += 1 # Add log counter

            self.variables[tid_attr] = var_name
            # self.graph.add_node(var_name) # Node addition moved to _add_factors_to_graph
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

        print(f"Defined {total_vars} variables mapping (tid, attr) -> var_name.")

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
        """Adds factors to the pgmpy graph using the learned weights.
           *** HYBRID APPROACH: Use add_factors first, then ensure structure manually ***
        """
        print("[Model] Adding factors to pgmpy graph (Hybrid add_factors + Manual method)...")
        start_time = time.time()
        self.graph = FactorGraph() # Re-initialize

        # --- Add Variable Nodes ---
        variable_node_names = list(self.variables.values())
        self.graph.add_nodes_from(variable_node_names)
        print(f"Added {len(self.graph.nodes())} variable nodes to graph.")
        variable_nodes_set = set(self.graph.nodes()) # Keep track

        # --- Create Factor Objects (Feature + Default) ---
        # Store all created factor objects in a list
        all_factor_objects = []
        # Map variable name to its corresponding factor object for easy lookup later
        variable_to_factor_map = {}
        factors_created_count = 0
        default_factors_created = 0
        problematic_factors = 0

        # --- Step 1a: Create Feature Factor Objects ---
        var_potentials = defaultdict(lambda: defaultdict(float))
        # ... (calculation of var_potentials - same as before) ...
        for feature_name, affected_states in self.features_data.items():
             weight = self.weights.get(feature_name, 0.0)
             clipped_weight = np.clip(weight, -700, 700)
             for var_name, state_int in affected_states:
                  if var_name in self.variable_states:
                     var_potentials[var_name][state_int] += clipped_weight

        print("Creating factors from features...")
        for var_name, state_log_potentials in var_potentials.items():
             if var_name not in self.variable_states: continue
             if var_name not in variable_nodes_set: continue # Ensure node exists

             var_cardinality = len(self.variable_states[var_name])
             if var_cardinality == 0: continue

             # ... (factor value calculation - same as before) ...
             factor_log_values = np.zeros(var_cardinality)
             for state_int, log_potential_sum in state_log_potentials.items():
                 if 0 <= state_int < var_cardinality:
                     factor_log_values[state_int] = log_potential_sum
             max_log_val = np.max(factor_log_values)
             if np.isinf(max_log_val): max_log_val = 700
             stable_log_values = factor_log_values - max_log_val
             factor_values = np.exp(stable_log_values)

             factor_vars = [var_name]
             is_problematic = False
             # ... (check factor_values for NaN/inf/zero - same as before) ...
             if not np.all(np.isfinite(factor_values)):
                 print(f"DEBUG: Feature factor for {var_name} has non-finite values. Replacing with ones.")
                 factor_values = np.ones(var_cardinality)
                 is_problematic = True
             elif np.sum(factor_values) < self.epsilon:
                 print(f"DEBUG: Feature factor for {var_name} has near-zero sum. Replacing with ones.")
                 factor_values = np.ones(var_cardinality)
                 is_problematic = True

             if is_problematic: problematic_factors += 1

             try:
                 factor = DiscreteFactor(factor_vars, [var_cardinality], factor_values)
                 all_factor_objects.append(factor) # Add to list
                 variable_to_factor_map[var_name] = factor # Add to map
                 factors_created_count += 1
             except Exception as e:
                 print(f"Error creating feature factor object for {var_name}: {e}")
                 problematic_factors += 1

        # --- Step 1b: Create Default Uniform Factor Objects ---
        print(f"Creating default factors for variables without feature factors...")
        variables_needing_default = variable_nodes_set - set(variable_to_factor_map.keys())

        for var_name in variables_needing_default:
            if var_name not in self.variable_states: continue
            if var_name not in variable_nodes_set: continue

            var_cardinality = len(self.variable_states[var_name])
            if var_cardinality == 0: continue

            factor_vars = [var_name]
            factor_values = np.ones(var_cardinality)

            try:
                 factor = DiscreteFactor(factor_vars, [var_cardinality], factor_values)
                 all_factor_objects.append(factor) # Add to list
                 variable_to_factor_map[var_name] = factor # Add to map
                 default_factors_created += 1
            except Exception as e:
                 print(f"Error creating default factor object for {var_name}: {e}")
                 problematic_factors += 1

        duration_creation = time.time() - start_time
        print(f"Created {factors_created_count} factors from features and {default_factors_created} default factors in {duration_creation:.2f}s (Object creation only).")
        print(f"Total factor objects created: {len(all_factor_objects)}")
        if problematic_factors > 0:
            print(f"WARNING: Encountered {problematic_factors} potential problems during factor object creation.")
        if len(all_factor_objects) != len(variable_nodes_set):
             print(f"CRITICAL WARNING: Number of factors created ({len(all_factor_objects)}) does not match number of variables ({len(variable_nodes_set)}).")


        # --- Step 2: Use add_factors to register factors internally ---
        print(f"Registering {len(all_factor_objects)} factors using self.graph.add_factors...")
        start_add_factors_time = time.time()
        if all_factor_objects:
            try:
                 self.graph.add_factors(*all_factor_objects)
                 add_factors_duration = time.time() - start_add_factors_time
                 print(f"Finished add_factors registration in {add_factors_duration:.2f}s.")
                 # Check if get_factors works now
                 print(f"DEBUG: Factors found by get_factors() after add_factors: {len(self.graph.get_factors())}")
            except Exception as e:
                 print(f"ERROR during self.graph.add_factors: {e}")
                 import traceback
                 traceback.print_exc()
                 print("Aborting factor addition.")
                 return # Don't proceed if add_factors failed
        else:
             print("Warning: No factors were created to register.")


        # --- Step 3: Manually add Factor Nodes and Edges (potentially redundant but ensures structure) ---
        print("Ensuring bipartite structure by manually adding factor nodes and edges...")
        start_manual_add_time = time.time()
        edges_added_count = 0
        factor_nodes_added_manually = set() # Track nodes added in this step
        variables_connected_by_edges = set()

        # Iterate through the map we created earlier
        for var_name, factor in variable_to_factor_map.items():
             # Ensure factor is valid
             if not isinstance(factor, DiscreteFactor): continue
             # Ensure variable node exists
             if not self.graph.has_node(var_name):
                  print(f"ERROR: Variable node '{var_name}' missing before manual edge add. Skipping.")
                  continue

             try:
                # 1. Add the factor object AS A NODE if not already present
                #    (add_factors might or might not have added it as a node)
                if not self.graph.has_node(factor):
                    self.graph.add_node(factor)
                factor_nodes_added_manually.add(factor) # Track attempt

                # 2. Add edge between the variable node and the factor node if not already present
                edge = (var_name, factor)
                if not self.graph.has_edge(*edge):
                     # Double check nodes exist before adding edge
                    if self.graph.has_node(var_name) and self.graph.has_node(factor):
                         self.graph.add_edge(*edge)
                         edges_added_count += 1
                    else:
                         print(f"ERROR: Nodes for edge {edge} not found before adding edge.")

                # Always mark the variable as intended to be connected
                variables_connected_by_edges.add(var_name)

             except Exception as e:
                 print(f"Error during manual node/edge addition for var '{var_name}': {e}")

        manual_add_duration = time.time() - start_manual_add_time
        print(f"Finished manual node/edge structure check/addition in {manual_add_duration:.2f}s.")
        print(f"DEBUG: Manually added/ensured {edges_added_count} edges.")


        # --- Step 4: Log final graph state ---
        print("--- Graph State After Hybrid Addition ---")
        current_nodes = self.graph.nodes()
        current_edges = self.graph.edges()
        current_factors = self.graph.get_factors() # Check again
        print(f"Nodes: {len(current_nodes)}")
        print(f"Edges: {len(current_edges)}")
        print(f"Factors (via get_factors()): {len(current_factors)}")

        # Check node types
        final_variable_nodes = set(n for n in current_nodes if isinstance(n, str))
        final_factor_nodes = set(n for n in current_nodes if isinstance(n, DiscreteFactor))
        print(f"Final variable nodes identified: {len(final_variable_nodes)}")
        print(f"Final factor nodes identified: {len(final_factor_nodes)}")

        # Verify counts and connectivity
        expected_node_count = len(variable_nodes_set) + len(final_factor_nodes) # Should include factors as nodes
        if len(current_nodes) != expected_node_count:
             print(f"WARNING: Node count mismatch! Actual: {len(current_nodes)}, Expected (vars + unique factors): {expected_node_count}")
             other_nodes = set(n for n in current_nodes if not isinstance(n, (str, DiscreteFactor)))
             if other_nodes: print(f"  Nodes with unexpected types found: {set(type(n) for n in other_nodes)}")


        if len(current_edges) != len(variable_nodes_set): # Expect one edge per variable ultimately
             print(f"WARNING: Final edge count ({len(current_edges)}) doesn't match variable count ({len(variable_nodes_set)})")
             # Check for unconnected variables again
             vars_in_edges = set(u for u, v in current_edges if isinstance(u, str)) | set(v for u, v in current_edges if isinstance(v, str))
             unconnected_vars = variable_nodes_set - vars_in_edges
             if unconnected_vars:
                  print(f"CRITICAL ERROR: {len(unconnected_vars)} variable nodes still seem unconnected!")
                  print(f"  Unconnected examples: {list(unconnected_vars)[:20]}...")

        print("---------------------------------------")
        # The rest of the infer method (check_model, sampling) will proceed from here...



    def infer(self, n_samples=1000, n_burn_in=200):
        """Performs inference using Gibbs sampling."""
        if not self.query_vars:
             print("[Model] No query variables to perform inference on. Skipping.")
             return {}

        print(f"[Model] Starting Gibbs sampling ({n_samples} samples, {n_burn_in} burn-in)...")
        start_time = time.time()

        self._add_factors_to_graph() # Now adds default factors

        # Check factors again after adding defaults
        # if not self.graph.get_factors():
        #      print("Error: No factors found in the graph even after adding defaults. Aborting.")
        #      return {}
        # if len(self.graph.get_factors()) != len(self.graph.nodes()):
        #     print(f"Error: Factor count ({len(self.graph.get_factors())}) still doesn't match node count ({len(self.graph.nodes())}). Aborting.")
        #     # Find nodes without factors
        #     nodes_with_factors = set()
        #     for factor in self.graph.get_factors():
        #         nodes_with_factors.update(factor.variables)
        #     missing_factor_nodes = set(self.graph.nodes()) - nodes_with_factors
        #     print(f"Nodes missing factors: {list(missing_factor_nodes)[:20]}...") # Print first few
        #     return {}

        # # *** REFINED MANUAL CHECK: Compare complete variable sets ***
        # print("Manually checking factors vs nodes BEFORE check_model()...")
        # manual_check_failed = False
        # current_nodes_set = set(self.graph.nodes())
        # if not current_nodes_set:
        #      print("ERROR: Graph has no nodes!")
        #      manual_check_failed = True

        # actual_factors = self.graph.get_factors()
        # print(f"Found {len(actual_factors)} factors in graph object.")

        # # Collect all unique variables mentioned in all factors
        # all_vars_in_factors = set()
        # factors_with_empty_vars = 0
        # for factor in actual_factors:
        #     if not factor.variables:
        #         factors_with_empty_vars += 1
        #     else:
        #         # Add variables from this factor to the set
        #         all_vars_in_factors.update(factor.variables)

        # if factors_with_empty_vars > 0:
        #     print(f"ERROR: Found {factors_with_empty_vars} factors with no associated variables!")
        #     manual_check_failed = True

        # print(f"Total unique variables mentioned across all factors: {len(all_vars_in_factors)}")
        # print(f"Total nodes in graph: {len(current_nodes_set)}")

        # # Check 1: Are there variables in factors that are not nodes?
        # vars_in_factors_not_nodes = all_vars_in_factors - current_nodes_set
        # if vars_in_factors_not_nodes:
        #     print(f"ERROR: Variables found in factors but not in graph nodes: {list(vars_in_factors_not_nodes)[:20]}...")
        #     manual_check_failed = True

        # # Check 2: Are there nodes that are not mentioned in any factor?
        # # (This check corresponds to the *previous* error, should pass now)
        # nodes_not_in_factors = current_nodes_set - all_vars_in_factors
        # if nodes_not_in_factors:
        #     print(f"ERROR: Nodes found in graph but not in any factor's scope: {list(nodes_not_in_factors)[:20]}...")
        #     # This indicates the default factor addition might have missed some nodes
        #     manual_check_failed = True


        # if manual_check_failed:
        #     print("Manual check failed. Aborting before pgmpy check_model().")
        #     return {}
        # else:
        #     print("Manual check comparing factor variables and graph nodes passed.")

        print("Checking graph model validity...")
        try:
            # This should now pass if the bipartite structure is correct
            print('--------------')
            print("Graph model before check_model():")
            print("Nodes:", len(self.graph.nodes()))
            print("Factors:", len(self.graph.get_factors()))
            print("Edges:", len(self.graph.edges()))
            print("Variable states:", len(self.variable_states))
            print("Candidate map:", len(self.candidate_map))
            print("Weights:", len(self.weights))
            print("Features data:", len(self.features_data))
            print("Evidence dict:", len(self.evidence_dict))
            print("Query vars:", len(self.query_vars))
            print("Graph object:", len(self.graph))
            print('--------------')
            check_result = self.graph.check_model()
            print("Graph model check passed.")
        except Exception as e:
            print(f"Graph model check raised an exception: {e}")
            import traceback
            traceback.print_exc()
            print("Aborting inference due to graph check failure.")
            return {}


        gibbs = GibbsSampling(self.graph)

        # --- Build start_state_list (Corrected Debug Logic) ---
        start_state_dict = {} # First build the dictionary
        # Get variable nodes *from the graph* after check_model passed
        # Or rely on self.variables keys mapping to the names added
        variable_nodes_in_graph = [node for node in self.graph.nodes() if isinstance(node, str)] # Assuming var names are strings
        variable_nodes_in_graph.sort()

        print("Building start state dictionary...")
        for var_name in variable_nodes_in_graph:
            state_assigned = -1
            if var_name in self.evidence_dict:
                state_assigned = self.evidence_dict[var_name]
            else:
                var_states = self.variable_states.get(var_name, {})
                num_states = len(var_states)
                if num_states > 0:
                     state_assigned = random.randint(0, num_states - 1)
                else:
                     state_assigned = 0 # Default for vars with no states

            # Verify assigned state is valid *before* adding to dict
            var_states = self.variable_states.get(var_name, {})
            num_states = len(var_states)
            if not (0 <= state_assigned < num_states):
                 print(f"  - WARNING: Assigned start state {state_assigned} for {var_name} is OUT OF BOUNDS for cardinality {num_states}. Resetting to 0.")
                 state_assigned = 0 if num_states > 0 else 0 # Assign 0 if possible, else keep 0

            start_state_dict[var_name] = state_assigned

        # Now build the list of tuples from the verified dictionary
        start_state_list = []
        print("Verifying start state consistency (first 10 variables)...")
        verified_count = 0
        for var_name in variable_nodes_in_graph: # Iterate in sorted order
             state_assigned = start_state_dict[var_name] # Get from the verified dict
             start_state_list.append(tuple([var_name, state_assigned]))

             # Print details for the first few
             if verified_count < 10:
                  var_states = self.variable_states.get(var_name, {})
                  num_states = len(var_states)
                  state_map_str = str(var_states)
                  if len(state_map_str) > 100: state_map_str = state_map_str[:100] + "..."
                  print(f"  - {var_name}: States={state_map_str} (Card={num_states}), StartState={state_assigned}")
             verified_count += 1


        if len(start_state_list) != len(variable_nodes_in_graph):
             print(f"Error: Start state list length ({len(start_state_list)}) doesn't match graph variable nodes ({len(variable_nodes_in_graph)}).")
             return {}

        variable_order = [s[0] for s in start_state_list]

        print("Generating samples (this may time)...")
        # ... (Sampling loop remains the same as the last working version - expecting list/tuple samples) ...
        samples_list = []
        try:
            sample_generator = gibbs.generate_sample(start_state=start_state_list, size=n_samples + n_burn_in, seed=42)
            # ... (rest of the loop from previous correct version) ...
            progress_interval = (n_samples + n_burn_in) // 10
            if progress_interval == 0: progress_interval = 1

            for i, sample in enumerate(sample_generator):
                if isinstance(sample, (list, tuple)) and len(sample) == len(variable_order):
                     samples_list.append(sample)
                elif sample == []:
                     print(f"Error: Gibbs sampler returned empty list [] at iteration {i}. Aborting.")
                     return {}
                else:
                     print(f"Error: Unexpected sample format at iteration {i}. Expected list/tuple of length {len(variable_order)}, Got type: {type(sample)}, Length: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
                     return {} # Cannot proceed

                if (i + 1) % progress_interval == 0:
                    print(f"  Generated {i + 1}/{n_samples + n_burn_in} samples...")

        except Exception as e:
             print(f"Error during Gibbs sampling generation: {e}")
             import traceback
             traceback.print_exc()
             return {}


        if not samples_list:
             print("Error: No valid samples generated by Gibbs sampler.")
             return {}

        # --- Convert list of samples (lists of state values) to DataFrame ---
        # variable_order is already defined and sorted before the loop
        samples_array = np.array(samples_list)

        try:
            # Ensure correct shape before creating DataFrame
            if len(samples_list) == 0: # Check if list is empty after potential skips
                 print("Error: No samples collected after filtering.")
                 return {}
            if samples_array.shape[1] != len(variable_order):
                 raise ValueError(f"Shape mismatch: samples_array columns ({samples_array.shape[1]}) != variable_order length ({len(variable_order)})")

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