# File: model/factor_graph.py
# Defines the HoloClean factor graph, learning, and inference logic using pgmpy.

import pandas as pd
import time
import psycopg2
import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.sampling import GibbsSampling
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
        self.evidence = {} # Maps variable_name -> state_int for clean cells
        self.query_vars = [] # List of variable_names for noisy cells

        # Tunable parameters (could be arguments)
        self.initial_weight_value = 0.01 # Small initial weight for features
        self.l2_penalty = 0.01 # Regularization strength for weight learning

    def _load_data(self):
        """Loads domains, features, and cell info from the database."""
        print("[Model] Loading data for graph construction...")
        start_time = time.time()
        try:
            # Domains define variables and their states
            self.domains_df = pd.read_sql("SELECT tid, attr, candidate_val FROM domains ORDER BY tid, attr, candidate_val", self.db_conn)
            print(f"Loaded {len(self.domains_df)} domain entries.")

            # Features define unary factors
            self.features_df = pd.read_sql("SELECT tid, attr, candidate_val, feature FROM features", self.db_conn)
            print(f"Loaded {len(self.features_df)} feature entries.")

            # Cell info needed to identify clean/noisy (evidence/query)
            self.cells_df = pd.read_sql("SELECT tid, attr, val, is_noisy FROM cells", self.db_conn)
            print(f"Loaded {len(self.cells_df)} cell entries.")

            load_time = time.time() - start_time
            print(f"[Model] Data loading complete in {load_time:.2f}s.")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _build_graph_structure(self):
        """Constructs the pgmpy FactorGraph structure (nodes and factor scopes)."""
        if self.domains_df is None or self.features_df is None or self.cells_df is None:
             print("Error: Data not loaded before building graph structure.")
             return

        print("[Model] Building factor graph structure...")
        start_time = time.time()

        # --- Define Variables (Nodes) ---
        # Group domains by (tid, attr) to define each variable
        grouped_domains = self.domains_df.groupby(['tid', 'attr'])
        for (tid, attr), group in grouped_domains:
            var_name = f"v_{tid}_{attr}"
            self.variables[(tid, attr)] = var_name
            candidates = group['candidate_val'].tolist()
            num_states = len(candidates)
            self.graph.add_node(var_name) # Add node to graph

            # Map states (0 to k-1) to candidate values
            self.variable_states[var_name] = {i: val for i, val in enumerate(candidates)}
            # Create reverse map for easy lookup
            self.candidate_map[(tid, attr)] = {val: i for i, val in enumerate(candidates)}

            # Add variable with cardinality to graph for pgmpy factors
            # This requires handling during factor creation, pgmpy FactorGraph itself doesn't store cardinality explicitly on nodes like BayesianModel

        print(f"Defined {len(self.variables)} variables.")

        # --- Identify Evidence and Query Variables ---
        cells_map = self.cells_df.set_index(['tid', 'attr']).to_dict('index')
        for tid_attr, var_name in self.variables.items():
             cell_info = cells_map.get(tid_attr)
             if cell_info:
                 if not cell_info['is_noisy'] and cell_info['val'] is not None:
                     # This is an evidence variable
                     original_val = cell_info['val']
                     candidate_to_state = self.candidate_map.get(tid_attr, {})
                     if original_val in candidate_to_state:
                         self.evidence[var_name] = candidate_to_state[original_val]
                     # Else: original value not in pruned domain? Handle this case (e.g., warning)
                 else:
                     # This is a query variable (noisy or null original value)
                     self.query_vars.append(var_name)
             # Else: Domain exists but cell doesn't? Data inconsistency.

        print(f"Identified {len(self.evidence)} evidence variables and {len(self.query_vars)} query variables.")

        # --- Prepare Feature Data for Factor Creation ---
        # Group features by feature name for easier weight mapping later
        feature_groups = self.features_df.groupby('feature')
        all_feature_names = set(self.features_df['feature'].unique())
        print(f"Found {len(all_feature_names)} unique feature types.")

        # Initialize weights for all unique features
        for feature_name in all_feature_names:
            self.weights[feature_name] = self.initial_weight_value
            self.features_data[feature_name] = []

        # Map features to the variable states they affect
        for index, row in self.features_df.iterrows():
            tid_attr = (row['tid'], row['attr'])
            var_name = self.variables.get(tid_attr)
            candidate_val = row['candidate_val']
            feature_name = row['feature']

            if var_name:
                state_map = self.candidate_map.get(tid_attr, {})
                if candidate_val in state_map:
                    state_int = state_map[candidate_val]
                    self.features_data[feature_name].append((var_name, state_int))

        # Note: We don't add factors to the pgmpy graph yet, as their values depend on weights learned later.
        # We will create them dynamically during likelihood calculation or inference.

        build_time = time.time() - start_time
        print(f"[Model] Graph structure built in {build_time:.2f}s.")


    def _calculate_log_likelihood(self, weights_array, feature_names_list):
        """
        Calculates the log-likelihood of the evidence given the current weights.
        This is the objective function for weight learning (needs to be maximized).
        We return the *negative* log-likelihood for minimization.
        """
        # Update self.weights dictionary from the optimizer's array
        current_weights = dict(zip(feature_names_list, weights_array))

        log_likelihood = 0.0

        # --- Sum log potentials of factors involving evidence ---
        # Simplified approach: Assume features are unary factors.
        # Calculate potential for each feature based on current weights.
        # Need to consider how evidence affects this.

        # A more rigorous approach requires calculating the partition function Z,
        # which is intractable. HoloClean uses approximations or Pseudo-Likelihood.
        # Let's approximate using Pseudo-Likelihood: P(evidence | Markov blanket).
        # Even simpler for unary factors: Sum weights of features active under evidence.

        # Simplistic Placeholder: Sum weights of features connected to evidence vars being in their evidence state.
        # This ignores interactions and normalization, but is a starting point.
        for feature_name, affected_states in self.features_data.items():
            weight = current_weights.get(feature_name, 0.0)
            for var_name, state_int in affected_states:
                if var_name in self.evidence and self.evidence[var_name] == state_int:
                    # This feature supports the observed evidence state
                    log_likelihood += weight # Simplified potential: exp(weight) -> log(exp(weight)) = weight

        # --- Add Regularization Penalty ---
        # L2 regularization: Penalize large weights
        l2_norm_sq = np.sum(weights_array**2)
        penalty = 0.5 * self.l2_penalty * l2_norm_sq # Factor 0.5 is common

        negative_log_likelihood = -log_likelihood + penalty
        # print(f"Current NLL: {negative_log_likelihood:.4f}, Likelihood part: {-log_likelihood:.4f}, Penalty: {penalty:.4f}") # Debug print
        return negative_log_likelihood

    def fit(self, max_iter=100):
        """Learns the feature weights using evidence."""
        if not self.evidence:
             print("Warning: No evidence variables found. Skipping weight learning.")
             # Keep initial weights
             return

        print("[Model] Starting weight learning...")
        start_time = time.time()

        # Prepare data for optimizer
        feature_names_list = list(self.weights.keys())
        initial_weights_array = np.array([self.weights[name] for name in feature_names_list])

        # Define the objective function for the optimizer
        objective_func = lambda w: self._calculate_log_likelihood(w, feature_names_list)

        # Use scipy.optimize.minimize with L-BFGS-B
        # We are minimizing the negative log-likelihood
        opt_result = minimize(
            objective_func,
            initial_weights_array,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': True} # Show optimizer progress
        )

        if opt_result.success:
            # Update self.weights with the learned values
            learned_weights_array = opt_result.x
            self.weights = dict(zip(feature_names_list, learned_weights_array))
            print("[Model] Weight learning successful.")
            # Optionally print some learned weights
            # sorted_weights = sorted(self.weights.items(), key=lambda item: abs(item[1]), reverse=True)
            # print("Top 5 learned weights (abs value):", sorted_weights[:5])
            # print("Bottom 5 learned weights (abs value):", sorted_weights[-5:])
        else:
            print("[Model] Weight learning failed or did not converge.")
            print("Optimizer message:", opt_result.message)
            # Keep initial or last weights? For now, keep the result even if not converged.
            learned_weights_array = opt_result.x
            self.weights = dict(zip(feature_names_list, learned_weights_array))


        learn_time = time.time() - start_time
        print(f"[Model] Weight learning finished in {learn_time:.2f}s.")


    def _add_factors_to_graph(self):
        """Adds factors to the pgmpy graph using the learned weights."""
        print("[Model] Adding factors to pgmpy graph...")
        # Clear existing factors if any
        self.graph.remove_factors(*self.graph.get_factors())

        # --- Add Unary Factors from Features ---
        feature_factors = defaultdict(lambda: {}) # {var_name: {state: accumulated_potential}}

        for feature_name, affected_states in self.features_data.items():
            weight = self.weights.get(feature_name, 0.0)
            potential = np.exp(weight) # Convert log-weight back to potential factor

            for var_name, state_int in affected_states:
                 # Accumulate potential for this state from different features
                 # This simple accumulation might not be the standard way factor potentials combine.
                 # Usually, one factor per feature.
                 # Let's create one factor per (var_name, state_int) pair, summing weights? No.
                 # Correct approach: Each feature implies a factor.
                 # For a feature f affecting (var_name, state_int), create a unary factor
                 # whose potential is exp(weight) if var=state_int, and 1 otherwise (or exp(0)).

                 var_cardinality = len(self.variable_states[var_name])
                 # Initialize factor values to 1 (exp(0))
                 factor_values = np.ones(var_cardinality)
                 # Set the potential for the affected state
                 factor_values[state_int] = potential

                 # Create the DiscreteFactor
                 factor = DiscreteFactor(
                     variables=[var_name],
                     cardinality=[var_cardinality],
                     values=factor_values
                 )
                 self.graph.add_factors(factor)

        # --- Add Cardinality Constraint Factor (Implicit in Gibbs for Categorical?) ---
        # Pgmpy's Gibbs sampling on a single variable samples from P(X | MB(X)).
        # For a categorical variable, this implicitly handles the "sum to 1" constraint.
        # We don't need explicit cardinality factors *if* variables are correctly defined
        # and sampling is done variable by variable.

        print(f"Added {len(self.graph.get_factors())} factors to the graph.")


    def infer(self, n_samples=1000, n_burn_in=200):
        """Performs inference using Gibbs sampling."""
        print(f"[Model] Starting Gibbs sampling ({n_samples} samples, {n_burn_in} burn-in)...")
        start_time = time.time()

        # Ensure factors are added with learned weights
        self._add_factors_to_graph()

        # Initialize the Gibbs sampler
        gibbs = GibbsSampling(self.graph)

        # Prepare evidence dictionary in {var_name: state_int} format
        pgmpy_evidence = self.evidence

        # Run sampling
        # seed for reproducibility
        samples_df = gibbs.sample(size=n_samples + n_burn_in, evidence=pgmpy_evidence, seed=42)

        # Discard burn-in samples
        samples_df = samples_df.iloc[n_burn_in:]
        print(f"Generated {len(samples_df)} samples after burn-in.")

        # --- Calculate Marginals ---
        marginals = {} # Output format: {(tid, attr): {candidate_val: prob}}
        for var_name in self.query_vars:
             if var_name not in samples_df.columns:
                 print(f"Warning: Query variable '{var_name}' not found in samples. Skipping.")
                 continue

             # Calculate probability for each state
             state_counts = samples_df[var_name].value_counts(normalize=True)
             state_probs = state_counts.to_dict()

             # Convert back to (tid, attr) and candidate_val format
             tid_attr = next((key for key, val in self.variables.items() if val == var_name), None)
             if tid_attr:
                 marginals[tid_attr] = {}
                 state_to_cand_map = self.variable_states.get(var_name, {})
                 for state_int, prob in state_probs.items():
                     candidate_val = state_to_cand_map.get(state_int)
                     if candidate_val is not None:
                         marginals[tid_attr][candidate_val] = prob
                 # Ensure all possible states have a probability (even if 0)
                 for state_int, candidate_val in state_to_cand_map.items():
                     if candidate_val not in marginals[tid_attr]:
                          marginals[tid_attr][candidate_val] = 0.0

        infer_time = time.time() - start_time
        print(f"[Model] Inference complete in {infer_time:.2f}s.")
        return marginals

    def run_pipeline(self, n_samples=1000, n_burn_in=200, learn_iter=100):
        """Runs the full load, build, fit, infer pipeline."""
        if self._load_data():
            self._build_graph_structure()
            self.fit(max_iter=learn_iter)
            marginals = self.infer(n_samples=n_samples, n_burn_in=n_burn_in)
            return marginals
        else:
            return None