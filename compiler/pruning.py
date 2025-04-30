import pandas as pd
import time
from collections import Counter, defaultdict
import logging
from scipy.stats import entropy
import psycopg2
import psycopg2.extras
from tqdm import tqdm

try:
    import config
    NULL_REPR_PLACEHOLDER = config.NULL_REPR_PLACEHOLDER
    DEFAULT_CORRELATION_THRESHOLD = config.PRUNING_CORRELATION_THRESHOLD
    DEFAULT_DOMAIN_COOCCURRENCE_THRESHOLD = config.PRUNING_DOMAIN_COOCCURRENCE_THRESHOLD
    DEFAULT_MAX_CANDIDATES_PER_CORR_ATTR = config.PRUNING_MAX_CANDIDATES_PER_CORR_ATTR
    DEFAULT_MAX_DOMAIN_SIZE_PER_CELL = config.PRUNING_MAX_DOMAIN_SIZE_PER_CELL
    DEFAULT_MIN_DOMAIN_SIZE = config.PRUNING_MIN_DOMAIN_SIZE

except (ImportError, AttributeError) as e:
    logging.warning(f"Could not import settings from config.py ({e}). Using defaults.")
    NULL_REPR_PLACEHOLDER = "__NULL__"
    DEFAULT_CORRELATION_THRESHOLD = 0.1
    DEFAULT_DOMAIN_COOCCURRENCE_THRESHOLD = 0.01
    DEFAULT_MAX_CANDIDATES_PER_CORR_ATTR = 10
    DEFAULT_MAX_DOMAIN_SIZE_PER_CELL = 50
    DEFAULT_MIN_DOMAIN_SIZE = 2


#calculates candidate domains using correlations and co-occurrence statistics,
#trying to mimic the HoloClean implementation or something similar
class DomainPruner:

    def __init__(self, db_conn,
                 correlation_threshold=DEFAULT_CORRELATION_THRESHOLD,
                 domain_cooccurrence_threshold=DEFAULT_DOMAIN_COOCCURRENCE_THRESHOLD,
                 max_candidates_per_corr_attr=DEFAULT_MAX_CANDIDATES_PER_CORR_ATTR,
                 max_domain_size_per_cell=DEFAULT_MAX_DOMAIN_SIZE_PER_CELL,
                 min_domain_size=DEFAULT_MIN_DOMAIN_SIZE
                 ):
        self.db_conn = db_conn
        self.correlation_threshold = correlation_threshold
        self.domain_cooccurrence_threshold = domain_cooccurrence_threshold
        self.max_candidates_per_corr_attr = max_candidates_per_corr_attr
        self.max_domain_size_per_cell = max_domain_size_per_cell
        self.min_domain_size = min_domain_size

        self.raw_df = None
        self.attributes = []
        self.raw_records_by_tid = {}

        self.single_stats = defaultdict(Counter)
        self.pair_stats_raw = defaultdict(lambda: defaultdict(Counter))
        self.total_tuples = 0
        self.correlations = defaultdict(dict)
        self._corr_attrs_cache = {}
        self.noisy_cells_set = set()


    #fetches raw data and noisy cells
    def _fetch_data(self):
        logging.info("[PrunerV2] Fetching data from database...")
        fetch_start = time.time()
        try:
            self.raw_df = pd.read_sql("SELECT tid, attr, val FROM cells ORDER BY tid, attr", self.db_conn)
            logging.info(f"Fetched {len(self.raw_df)} total cell entries.")
            if self.raw_df.empty:
                 logging.error("Fetched DataFrame is empty. Cannot proceed.")
                 return False

            try:
                 self.raw_df['tid'] = self.raw_df['tid'].astype(int)
            
            except ValueError as e:
                 logging.error(f"Could not convert 'tid' column to integer: {e}. Check data.")
                 return False

            self.attributes = sorted(self.raw_df['attr'].unique())
            logging.info(f"Found {len(self.attributes)} unique attributes.")

            #create tuple {tid: {attr: val_str}} like a cache
            logging.info("Building tuple context cache...")
            self.raw_records_by_tid = {}
            for tid, group in self.raw_df.groupby('tid'):
                 
                 self.raw_records_by_tid[tid] = {
                     row['attr']: NULL_REPR_PLACEHOLDER if pd.isna(row['val']) else str(row['val'])
                     for _, row in group.iterrows()
                 }

            self.total_tuples = len(self.raw_records_by_tid)
            logging.info(f"Built cache for {self.total_tuples} tuples.")

            noisy_df = pd.read_sql("SELECT tid, attr FROM noisy_cells", self.db_conn)

            try:
                 noisy_df['tid'] = noisy_df['tid'].astype(int)
            except ValueError as e:
                 logging.warning(f"Could not convert 'tid' in noisy_cells to integer: {e}. Matching might fail.")

            self.noisy_cells_set = set(tuple(x) for x in noisy_df[['tid', 'attr']].to_numpy())
            logging.info(f"Fetched {len(self.noisy_cells_set)} unique noisy cell identifiers.")

            fetch_end = time.time()
            logging.info(f"[PrunerV2] Data fetching completed in {fetch_end - fetch_start:.2f} seconds.")
            return True

        except Exception as e:
            logging.error(f"Error fetching data: {e}", exc_info=True)
            return False

    #computes single and pairwise statistics from the raw data
    def _compute_statistics(self):

        if self.raw_df is None or self.raw_df.empty:
            logging.error("Raw data not loaded. Cannot compute statistics.")
            return

        logging.info("[PrunerV2] Calculating single and pairwise statistics...")
        stats_start = time.time()

        self.single_stats = defaultdict(Counter)
        for _, row in self.raw_df.iterrows():
            attr = row['attr']
            val_str = NULL_REPR_PLACEHOLDER if pd.isna(row['val']) else str(row['val'])
            self.single_stats[attr][val_str] += 1

        self.pair_stats_raw = defaultdict(lambda: defaultdict(Counter))
        grouped_by_tid = self.raw_df.groupby('tid')

        for tid, group in tqdm(grouped_by_tid, desc="Calculating Pair Stats", unit=" tuple"):

            non_null_cells = group.dropna(subset=['val'])
            attributes_in_tuple = list(non_null_cells['attr'])

            # Create combinations of attributes inside of the tuple
            for i in range(len(attributes_in_tuple)):
                for j in range(i + 1, len(attributes_in_tuple)):
                    
                    attr1 = attributes_in_tuple[i]
                    attr2 = attributes_in_tuple[j]

                    val1_str = str(non_null_cells[non_null_cells['attr'] == attr1]['val'].iloc[0])
                    val2_str = str(non_null_cells[non_null_cells['attr'] == attr2]['val'].iloc[0])

                    self.pair_stats_raw[attr1][attr2][(val1_str, val2_str)] += 1
                    self.pair_stats_raw[attr2][attr1][(val2_str, val1_str)] += 1

        stats_end = time.time()

        single_count_total = sum(sum(c.values()) for c in self.single_stats.values())
        pair_count_total = sum(sum(sum(c.values()) for c in inner.values()) for inner in self.pair_stats_raw.values()) // 2
        
        logging.info(f"[PrunerV2] Statistics calculated in {stats_end - stats_start:.2f} seconds.")
        logging.info(f"Total single counts: {single_count_total}, Total unique pair counts: {pair_count_total}")


    #Normalizes conditional entropy
    def _normalize_entropy(self, h_cond, h_base):
        if h_base == 0:
             return 1.0
        
        if h_cond < 1e-9:
             h_cond = 0.0

        norm_h_cond = h_cond / h_base

        return 1.0 - norm_h_cond

    #computes pairwise attribute correlations
    def _compute_correlations(self):
        if self.raw_df is None or self.raw_df.empty:
             logging.error("Raw data not loaded. Cannot compute correlations.")
             return

        logging.info("[PrunerV2] Calculating attribute correlations (using conditional entropy)...")
        corr_start = time.time()
        self.correlations = defaultdict(dict)

        try:
            pivot_df = self.raw_df.pivot(index='tid', columns='attr', values='val').fillna(NULL_REPR_PLACEHOLDER).astype(str)
        except Exception as e:
            logging.error(f"Failed to pivot DataFrame for correlation calculation: {e}. Check for duplicate (tid, attr) pairs.")
            return

        #pre-calculate
        base_entropies = {}
        for attr in self.attributes:
            if attr not in pivot_df.columns: continue
            
            counts = pivot_df[attr].value_counts(normalize=True)
            base_entropies[attr] = entropy(counts, base=2)

        logging.info(f"Calculated base entropies for {len(base_entropies)} attributes.")

        #compute pairwise conditional thingies
        for attr_x in tqdm(self.attributes, desc="Calculating Correlations", unit=" attr"):
            if attr_x not in pivot_df.columns: continue
            h_x = base_entropies.get(attr_x, 0)
            self.correlations[attr_x][attr_x] = 1.0

            for attr_y in self.attributes:
                if attr_x == attr_y or attr_y not in pivot_df.columns:
                    continue

                try:

                    #calculate H(X, Y)
                    joint_counts = pivot_df.groupby([attr_x, attr_y]).size()
                    h_xy = entropy(joint_counts, base=2)

                    #calculate H(Y)
                    h_y = base_entropies.get(attr_y, 0)

                    #calculate H(X|Y) = H(X, Y) - H(Y)
                    h_x_given_y = h_xy - h_y
                    if h_x_given_y < 0: h_x_given_y = 0

                    #normalize
                    corr_xy = self._normalize_entropy(h_x_given_y, h_x)
                    self.correlations[attr_x][attr_y] = corr_xy

                except Exception as e:
                    logging.warning(f"Could not calculate correlation between '{attr_x}' and '{attr_y}': {e}")
                    self.correlations[attr_x][attr_y] = 0.0

        corr_end = time.time()
        logging.info(f"[PrunerV2] Correlations calculated in {corr_end - corr_start:.2f} seconds.")

    #gets attributes correlated with target_attr above the threshold.
    def get_correlated_attributes(self, target_attr):
        cache_key = (target_attr, self.correlation_threshold)
        
        if cache_key not in self._corr_attrs_cache:
            correlated = []
            
            if target_attr in self.correlations:
                for other_attr, score in self.correlations[target_attr].items():
                    if other_attr != target_attr and score >= self.correlation_threshold:
                        correlated.append(other_attr)
            
            self._corr_attrs_cache[cache_key] = sorted(correlated)
        
        return self._corr_attrs_cache[cache_key]


    #gets candidate values for target_attr that co-occur with
    #cond_val_str in cond_attr, pruned by threshold P(target_val | cond_val).
    def _get_cooccurrence_candidates(self, target_attr, cond_attr, cond_val_str):
        candidates = []
        cond_val_count = self.single_stats[cond_attr].get(cond_val_str, 0)
        if cond_val_count == 0:
            return candidates # Cannot calculate probability

        #iterate through potential target values that cooccurred with cond_val
        if cond_attr in self.pair_stats_raw and target_attr in self.pair_stats_raw[cond_attr]:

             relevant_pairs = {pair: count for pair, count in self.pair_stats_raw[cond_attr][target_attr].items() if pair[0] == cond_val_str}

             cand_probs = []
             for (cv, target_val), pair_count in relevant_pairs.items():
                 prob_target_given_cond = pair_count / cond_val_count
                 if prob_target_given_cond >= self.domain_cooccurrence_threshold:
                      cand_probs.append((target_val, prob_target_given_cond))

             #sort by probability
             cand_probs.sort(key=lambda x: x[1], reverse=True)
             candidates = [cand for cand, prob in cand_probs[:self.max_candidates_per_corr_attr]]

        return candidates


    #generates the candidate domain for a single cell (tid, attr)
    def _generate_domain_for_cell(self, tid, attr):
        if tid not in self.raw_records_by_tid:
            logging.warning(f"TID {tid} not found in tuple cache. Skipping domain for ({tid}, {attr}).")
            return set()

        row_dict = self.raw_records_by_tid[tid]
        init_value_str = row_dict.get(attr, NULL_REPR_PLACEHOLDER)

        #start domain with initial value if not null
        domain_set = set()
        if init_value_str != NULL_REPR_PLACEHOLDER:
            domain_set.add(init_value_str)

        #candidates from correlated attributes cooccurrence
        correlated_attributes = self.get_correlated_attributes(attr)
        logging.debug(f"Cell ({tid}, {attr}): Correlated Attrs (>{self.correlation_threshold}): {correlated_attributes}")

        for cond_attr in correlated_attributes:
            cond_val_str = row_dict.get(cond_attr)
            if cond_val_str is None or cond_val_str == NULL_REPR_PLACEHOLDER:
                continue

            cooc_candidates = self._get_cooccurrence_candidates(attr, cond_attr, cond_val_str)
            if cooc_candidates:
                logging.debug(f"  Cell ({tid}, {attr}): Adding {len(cooc_candidates)} candidates from P(target|{cond_attr}='{cond_val_str}') >= {self.domain_cooccurrence_threshold}: {cooc_candidates[:5]}...")
                domain_set.update(cooc_candidates)

        is_noisy = (tid, attr) in self.noisy_cells_set
        if is_noisy and len(domain_set) < self.min_domain_size:
            
            logging.debug(f"  Cell ({tid}, {attr}): Domain size {len(domain_set)} < min {self.min_domain_size}. Adding frequent global values.")
            num_to_add = self.min_domain_size - len(domain_set)
            added_count = 0

            global_candidates_sorted = sorted(
                [(val, count) for val, count in self.single_stats[attr].items() if val != NULL_REPR_PLACEHOLDER],
                key=lambda x: x[1], reverse=True
            )

            for val, count in global_candidates_sorted:
                if val not in domain_set:
                    domain_set.add(val)
                    added_count += 1
                    logging.debug(f"    Added global candidate '{val}' (Count: {count})")
                    if added_count >= num_to_add:
                        break

        domain_set.discard(NULL_REPR_PLACEHOLDER)

        logging.debug(f"  Cell ({tid}, {attr}): Final domain size = {len(domain_set)}, Domain = {list(domain_set)[:10]}...")
        return domain_set


    #calculates domains for all cells and inserts them into the database.
    def calculate_and_insert_domains(self):

        if not self._fetch_data(): return False
        
        self._compute_statistics()
        self._compute_correlations()

        all_domain_entries = []
        logging.info(f"[PrunerV2] Starting domain calculation for ALL {self.total_tuples * len(self.attributes)} potential cells...")
        domain_start_time = time.time()

        processed_cells = 0
        all_tids = list(self.raw_records_by_tid.keys())

        for tid in tqdm(all_tids, desc="Generating Domains", unit=" tuple"):
            
            for attr in self.attributes:
                domain_set = self._generate_domain_for_cell(tid, attr)
                
                for candidate_val in domain_set:
                    all_domain_entries.append((tid, attr, str(candidate_val)))
                processed_cells += 1

        domain_end_time = time.time()
        logging.info(f"[PrunerV2] Domain calculation finished in {domain_end_time - domain_start_time:.2f} seconds.")
        logging.info(f"Generated {len(all_domain_entries)} domain entries for {processed_cells} cells.")

        return self.insert_domains_to_db(all_domain_entries)

    #inserts calculated domain entries into the database
    def insert_domains_to_db(self, domain_entries):
        if not domain_entries:
            logging.warning("No domain entries generated. Skipping database insertion.")

            return True

        logging.info("[PrunerV2] Inserting candidate domains into database...")
        insert_start = time.time()
        
        try:
            
            with self.db_conn.cursor() as cur:
                logging.info("Clearing previous domains...")
                cur.execute("DELETE FROM domains;")
                logging.info(f"Attempting to insert {len(domain_entries)} domain entries...")
                insert_sql = "INSERT INTO domains (tid, attr, candidate_val) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
                psycopg2.extras.execute_batch(cur, insert_sql, domain_entries, page_size=10000)
            
            self.db_conn.commit()
            
            insert_end = time.time()
            logging.info(f"[PrunerV2] Database commit successful for domains in {insert_end - insert_start:.2f} seconds.")
            
            return True
        
        except Exception as e:
            self.db_conn.rollback()
            logging.error(f"Error inserting domains: {e}", exc_info=True)
            
            return False

    #executes the full pruning and insertion process
    def run(self):
        success = self.calculate_and_insert_domains()
        return success
