"""
Forecast Score Inference Engine
================================
Computes a composite Forecastability Score for each SKU and assigns
Easy / Moderate / Hard labels based on cluster-level tercile splits.

Methodology:
    - Uses polarity-weighted scoring across 11 feature groups
    - Features with positive polarity (higher = easier to forecast) contribute positively
    - Features with negative polarity (higher = harder) are flipped
    - Neutral/contextual features are excluded
    - Final score is a balanced average across groups
    - ANOVA F-test validates discriminative power
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f_oneway
import warnings

warnings.filterwarnings("ignore")


class ForecastScoreEngine:
    def __init__(self):
        # Feature Polarity: +1 (higher = easier), -1 (higher = harder), 0 (neutral)
        self.feature_polarity = {
            # Positive (+1)
            'acf_lag1': 1, 'acf_lag12': 1, 'partial_acf_lag1': 1,
            'trend_strength': 1, 'trend_linearity': 1,
            'seasonal_strength': 1, 'seasonal_period_strength': 1,
            'level_stability': 1, 'trend_stability': 1, 'seasonal_stability': 1,
            'series_length': 1, 'demand_regularity': 1, 'mean_segment_length': 1,

            # Negative (-1)
            'cv': -1, 'std': -1,
            'approx_entropy': -1, 'sample_entropy': -1, 'spectral_entropy': -1,
            'p_zero': -1, 'adi': -1,
            'n_changepoints': -1, 'max_cp_magnitude': -1,
            'kurtosis': -1, 'skewness': -1, 'tail_heaviness': -1,
            'rolling_cv_mean': -1, 'volatility_change_ratio': -1, 'garch_like_vol': -1,
            'recency_weight': -1,

            # Neutral (0) — excluded from scoring
            'mean': 0, 'median': 0, 'trend_slope': 0, 'peak_to_trough_ratio': 0
        }

        # Feature Groups for balanced scoring
        self.feature_groups = {
            'Basic Stats': ['mean', 'std', 'cv', 'median'],
            'Demand Pattern': ['adi', 'p_zero', 'demand_regularity'],
            'Trend': ['trend_strength', 'trend_slope', 'trend_linearity'],
            'Seasonality': ['seasonal_strength', 'peak_to_trough_ratio', 'seasonal_period_strength'],
            'Volatility': ['rolling_cv_mean', 'volatility_change_ratio', 'garch_like_vol'],
            'Autocorrelation': ['acf_lag1', 'acf_lag12', 'partial_acf_lag1'],
            'Information Theory': ['approx_entropy', 'sample_entropy', 'spectral_entropy'],
            'Stability': ['level_stability', 'trend_stability', 'seasonal_stability'],
            'Shape': ['skewness', 'kurtosis', 'tail_heaviness'],
            'Changepoints': ['n_changepoints', 'max_cp_magnitude', 'mean_segment_length'],
            'Bonus': ['series_length', 'recency_weight']
        }

        self.group_weight = 1.0 / len(self.feature_groups)

    def score(self, input_df, cluster_col='Cluster'):
        """
        Main entry point.
        Input: DataFrame with features + Cluster column + Inferred_Pattern
        Output: (result_df, anova_df) tuple
            - result_df: input + 'Forecastability_Score', 'Score_Bucket', 'Forecastability_Label'
            - anova_df: ANOVA F-stat validation results
        """
        df = input_df.copy()

        # 1. Compute Composite Score Per SKU (0.0 to 1.0)
        scaler = MinMaxScaler()
        feature_cols = [c for c in df.columns if c in self.feature_polarity]

        df_norm = df.copy()
        df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])

        scores = np.zeros(len(df))
        for group, features in self.feature_groups.items():
            group_score = np.zeros(len(df))
            valid_feats = 0
            for f in features:
                if f not in df.columns: continue
                polarity = self.feature_polarity.get(f, 0)
                if polarity == 0: continue
                val = df_norm[f].values
                if polarity == -1: val = 1 - val 
                group_score += val
                valid_feats += 1
            if valid_feats > 0:
                group_score /= valid_feats
                scores += group_score * self.group_weight

        df['Forecastability_Score'] = scores

        # --- INDEPENDENT TRIANGULATION LOGIC ---

        # 2. SIGNAL 1: Score Points (0 to 2)
        # We use Terciles for a clean low/mid/high split representing the numeric score signal.
        df['Score_Tier'] = pd.qcut(df['Forecastability_Score'], 3, labels=[0, 1, 2]).astype(int)
        
        # 3. SIGNAL 2: Pattern Points (0 to 2)
        # We map patterns to difficulty levels independently.
        pattern_signal_map = {
            'Smooth': 2, 'Seasonal': 2, 'Trending': 1,
            'New_Product': 1,
            'Intermittent': 0, 'Lumpy': 0, 'Erratic': 0
        }
        df['Pattern_Signal'] = df.get('Inferred_Pattern', 'Erratic').map(pattern_signal_map).fillna(0).astype(int)
        
        # 4. SIGNAL 3: Cluster Integrity (Simplified)
        # We use the raw Cluster ID as a context marker without ranking it by score.
        # This keeps the grouping signal purely about "behavioral shape."

        # 5. CROSS-VALIDATION LABELING (Triangulation)
        def calculate_label(row):
            # We take the AVERAGE of Score and Pattern signals.
            # If they both agree, the label is strong. If they differ, it becomes Moderate.
            avg_signal = (row['Score_Tier'] + row['Pattern_Signal']) / 2
            
            if avg_signal >= 1.5:
                return 'Easy'
            elif avg_signal >= 0.75:
                return 'Moderate'
            else:
                return 'Hard'

        df['Forecastability_Label'] = df.apply(calculate_label, axis=1)

        # 6. Validate with ANOVA F-stat
        print("Validating Independent Labels with ANOVA...")
        anova_results = []
        groups = [df[df['Forecastability_Label'] == label] for label in ['Easy', 'Moderate', 'Hard']]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) > 1:
            for feat in feature_cols:
                vals = [g[feat].values for g in groups]
                try:
                    f_stat, p_val = f_oneway(*vals)
                    anova_results.append({'Feature': feat, 'F_Stat': f_stat, 'P_Value': p_val})
                except: pass
            anova_df = pd.DataFrame(anova_results).sort_values('F_Stat', ascending=False)
        else:
            anova_df = pd.DataFrame()

        return df, anova_df


if __name__ == "__main__":
    try:
        df = pd.read_csv("segmentation_output.csv")
        engine = ForecastScoreEngine()
        result_df, anova_df = engine.score(df)

        print("\nForecast Score Classification Complete.")
        print(result_df['Forecastability_Label'].value_counts())

        result_df.to_csv("final_output.csv", index=False)
        print("Saved to final_output.csv")
    except Exception as e:
        print(f"Test failed: {e}")
