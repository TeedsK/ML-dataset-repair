from .base_detector import BaseDetector
import pandas as pd
import numpy as np
from scipy import stats

#detects statistical outliers in number type attributes.
class StatisticalOutlierDetector(BaseDetector):

    def __init__(self, db_conn, attributes=None, method='zscore', threshold=3.0):
        super().__init__(db_conn)

        self.attributes = attributes if attributes else ['Score', 'Sample']
        self.method = method
        self.threshold = threshold

    def detect_errors(self):
        print(f"\n[{self.detector_name}] Running for attributes: {self.attributes} using {self.method}...")
        total_noisy_cells_added = 0

        if self.method == 'zscore':
            for attr in self.attributes:

                print(f"[{self.detector_name}] Processing attribute '{attr}' with Z-score...")
                sql = "SELECT tid, val FROM cells WHERE attr = %s;"

                try:
                    df = pd.read_sql(sql, self.db_conn, params=(attr,))

                    df['val_numeric'] = pd.to_numeric(df['val'], errors='coerce')
                    df.dropna(subset=['val_numeric'], inplace=True)

                    if len(df) < 2:
                         print(f"Not enough numeric data for Z-score on '{attr}'.")
                         continue

                    #Z-scores
                    df['zscore'] = np.abs(stats.zscore(df['val_numeric']))

                    #find outliers
                    outliers = df[df['zscore'] > self.threshold]
                    noisy_cells = list(zip(outliers['tid'], [attr] * len(outliers)))

                    if noisy_cells:
                        print(f"Found {len(noisy_cells)} potential outliers for '{attr}'.")
                        added = self._add_noisy_cells(noisy_cells)
                        total_noisy_cells_added += added
                    else:
                        print(f"No outliers found for '{attr}' with threshold {self.threshold}.")

                except Exception as e:
                    print(f"Error processing outliers for attribute '{attr}': {e}")

        elif self.method == 'isolation_forest':
             #implement in the future
             pass
        else:
            print(f"[{self.detector_name}] Unknown outlier detection method: {self.method}")

        print(f"[{self.detector_name}] Total unique noisy cells added by outliers: {total_noisy_cells_added}")
        return total_noisy_cells_added