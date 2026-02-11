"""
Pattern Inference Engine
========================
Robust, exhaustive demand pattern classification using ALL 35 extracted features.
Replaces the simplistic ADI/CV² Syntetos-Boylan quadrant with a multi-signal,
confidence-scored approach.

Patterns detected:
    - Smooth: Stable, low volatility, high autocorrelation
    - Seasonal: Strong periodic patterns
    - Intermittent: Sporadic demand with many zeros
    - Lumpy: Intermittent + high volatility spikes
    - Trending: Significant upward/downward trend
    - Erratic: High volatility without intermittency
    - New_Product: Very short history with ramp-up signature

Each SKU receives:
    - Inferred_Pattern: The winning pattern label
    - Pattern_Confidence: Confidence score (0-1) of the winning pattern
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


class PatternInferenceEngine:
    """
    Multi-signal pattern classification engine.
    Uses weighted evidence scoring across all 35 features to assign
    each SKU its most likely demand pattern with a confidence score.
    """

    def __init__(self):
        # Define pattern scoring rules: each pattern has a list of
        # (feature_name, condition_func, weight) tuples.
        # condition_func takes the raw feature value and returns a score 0-1.
        self.pattern_rules = self._build_rules()

    # -----------------------------------------------------------------
    # Rule Definitions
    # -----------------------------------------------------------------
    def _build_rules(self):
        """
        For each pattern, define which features contribute evidence
        and how to score them. Higher score = stronger evidence.
        """
        rules = {}

        # --- INTERMITTENT ---
        rules['Intermittent'] = [
            # Primary signals
            ('p_zero',              lambda v: self._sigmoid(v, center=0.30, steepness=12),  3.0),
            ('adi',                 lambda v: self._sigmoid(v, center=1.5,  steepness=3),   2.5),
            ('demand_regularity',   lambda v: 1 - self._sigmoid(v, center=0.5, steepness=6), 1.5),
            # Secondary confirmations
            ('acf_lag1',            lambda v: 1 - self._sigmoid(v, center=0.3, steepness=6), 1.0),
            ('mean_segment_length', lambda v: 1 - self._sigmoid(v, center=10, steepness=0.5), 1.0),
            ('cv',                  lambda v: 1 - self._sigmoid(v, center=0.8, steepness=5), 0.5),  # NOT high CV (else Lumpy)
            ('spectral_entropy',    lambda v: self._sigmoid(v, center=2.0, steepness=1.5),  0.8),
            ('approx_entropy',      lambda v: self._sigmoid(v, center=1.5, steepness=2),    0.5),
        ]

        # --- LUMPY (Intermittent + High Volatility) ---
        rules['Lumpy'] = [
            # Primary: intermittent AND volatile
            ('p_zero',              lambda v: self._sigmoid(v, center=0.25, steepness=10),  2.5),
            ('adi',                 lambda v: self._sigmoid(v, center=1.3,  steepness=3),   2.0),
            ('cv',                  lambda v: self._sigmoid(v, center=0.8,  steepness=6),   3.0),
            # Spike / tail signals
            ('kurtosis',            lambda v: self._sigmoid(v, center=2.0,  steepness=1.5), 2.0),
            ('tail_heaviness',      lambda v: self._sigmoid(v, center=0.05, steepness=40),  1.5),
            ('garch_like_vol',      lambda v: self._sigmoid(v, center=1.0,  steepness=2),   1.5),
            ('skewness',            lambda v: self._sigmoid(abs(v), center=1.0, steepness=3), 1.0),
            ('peak_to_trough_ratio', lambda v: self._sigmoid(v, center=5.0, steepness=0.5), 1.0),
            ('volatility_change_ratio', lambda v: self._sigmoid(v, center=1.5, steepness=3), 0.8),
        ]

        # --- SEASONAL ---
        rules['Seasonal'] = [
            # Primary
            ('seasonal_strength',       lambda v: self._sigmoid(v, center=0.25, steepness=12), 3.0),
            ('seasonal_period_strength', lambda v: self._sigmoid(v, center=0.15, steepness=15), 2.5),
            ('acf_lag12',               lambda v: self._sigmoid(v, center=0.15, steepness=12), 2.0),
            # Secondary
            ('peak_to_trough_ratio',    lambda v: self._sigmoid(v, center=1.5, steepness=2),  1.5),
            ('seasonal_stability',      lambda v: self._sigmoid(v, center=0.5, steepness=5),  1.0),
            ('spectral_entropy',        lambda v: 1 - self._sigmoid(v, center=3.0, steepness=1), 0.8),
            ('p_zero',                  lambda v: 1 - self._sigmoid(v, center=0.2, steepness=10), 0.5),  # NOT intermittent
        ]

        # --- TRENDING ---
        rules['Trending'] = [
            # Primary
            ('trend_strength',    lambda v: self._sigmoid(v, center=0.35, steepness=10),   3.0),
            ('trend_linearity',   lambda v: self._sigmoid(v, center=0.25, steepness=10),   2.5),
            ('trend_slope',       lambda v: self._sigmoid(abs(v), center=5.0, steepness=0.3), 2.0),
            # Secondary
            ('level_stability',   lambda v: self._sigmoid(v, center=0.3, steepness=5),     1.5),
            ('volatility_change_ratio', lambda v: self._sigmoid(v, center=1.3, steepness=3), 1.0),
            ('acf_lag1',          lambda v: self._sigmoid(v, center=0.4, steepness=6),      1.0),
            ('recency_weight',    lambda v: self._sigmoid(abs(v - 1), center=0.3, steepness=5), 0.8),
            ('n_changepoints',    lambda v: 1 - self._sigmoid(v, center=5, steepness=0.5),  0.5),  # Few CPs = clean trend
            ('p_zero',            lambda v: 1 - self._sigmoid(v, center=0.15, steepness=10), 0.5),
        ]

        # --- ERRATIC (Volatile but not intermittent) ---
        rules['Erratic'] = [
            # Primary
            ('cv',                  lambda v: self._sigmoid(v, center=0.5, steepness=8),    3.0),
            ('approx_entropy',      lambda v: self._sigmoid(v, center=1.8, steepness=2),    2.5),
            ('spectral_entropy',    lambda v: self._sigmoid(v, center=2.5, steepness=1.5),  2.0),
            # Secondary
            ('garch_like_vol',      lambda v: self._sigmoid(v, center=0.5, steepness=3),    1.5),
            ('rolling_cv_mean',     lambda v: self._sigmoid(v, center=0.4, steepness=6),    1.5),
            ('sample_entropy',      lambda v: self._sigmoid(v, center=0.5, steepness=5),    1.0),
            ('trend_stability',     lambda v: self._sigmoid(v, center=1.0, steepness=3),    1.0),
            ('n_changepoints',      lambda v: self._sigmoid(v, center=3, steepness=0.5),    0.8),
            ('p_zero',              lambda v: 1 - self._sigmoid(v, center=0.15, steepness=10), 1.0),  # NOT intermittent
            ('acf_lag1',            lambda v: 1 - self._sigmoid(v, center=0.3, steepness=6),  0.5),  # Low autocorrelation
        ]

        # --- NEW PRODUCT ---
        rules['New_Product'] = [
            # Primary
            ('series_length',     lambda v: 1 - self._sigmoid(v, center=18, steepness=0.4), 4.0),
            ('recency_weight',    lambda v: self._sigmoid(v, center=1.5, steepness=3),      3.0),
            # Secondary: ramp-up signature
            ('trend_strength',    lambda v: self._sigmoid(v, center=0.3, steepness=8),      2.0),
            ('trend_slope',       lambda v: self._sigmoid(v, center=3.0, steepness=0.5),    1.5),
            ('volatility_change_ratio', lambda v: self._sigmoid(v, center=1.5, steepness=3), 1.0),
            ('level_stability',   lambda v: self._sigmoid(v, center=0.4, steepness=5),      1.0),
        ]

        # --- SMOOTH ---
        rules['Smooth'] = [
            # Primary: low volatility, high predictability
            ('cv',                  lambda v: 1 - self._sigmoid(v, center=0.25, steepness=12), 3.0),
            ('acf_lag1',            lambda v: self._sigmoid(v, center=0.4, steepness=8),       2.5),
            ('approx_entropy',      lambda v: 1 - self._sigmoid(v, center=1.5, steepness=2),   2.0),
            # Secondary
            ('spectral_entropy',    lambda v: 1 - self._sigmoid(v, center=2.5, steepness=1.5), 1.5),
            ('garch_like_vol',      lambda v: 1 - self._sigmoid(v, center=0.3, steepness=5),   1.5),
            ('n_changepoints',      lambda v: 1 - self._sigmoid(v, center=3, steepness=0.5),   1.0),
            ('level_stability',     lambda v: 1 - self._sigmoid(v, center=0.3, steepness=5),   1.0),
            ('rolling_cv_mean',     lambda v: 1 - self._sigmoid(v, center=0.3, steepness=6),   1.0),
            ('p_zero',              lambda v: 1 - self._sigmoid(v, center=0.05, steepness=20), 1.0),
            ('trend_stability',     lambda v: 1 - self._sigmoid(v, center=0.8, steepness=3),   0.8),
            ('sample_entropy',      lambda v: 1 - self._sigmoid(v, center=0.6, steepness=5),   0.5),
        ]

        return rules

    # -----------------------------------------------------------------
    # Sigmoid utility for smooth thresholding
    # -----------------------------------------------------------------
    @staticmethod
    def _sigmoid(x, center=0.0, steepness=1.0):
        """
        Smooth sigmoid activation: returns ~0 when x << center, ~1 when x >> center.
        Steepness controls the sharpness of the transition.
        """
        z = steepness * (x - center)
        z = np.clip(z, -20, 20)  # Avoid overflow
        return 1.0 / (1.0 + np.exp(-z))

    # -----------------------------------------------------------------
    # Main Inference
    # -----------------------------------------------------------------
    def infer_patterns(self, features_df, sku_col='SKU'):
        """
        Main entry point.
        Input:  DataFrame from FeatureExtractionEngine (SKU + 35 numeric features)
        Output: Same DataFrame with 'Inferred_Pattern' and 'Pattern_Confidence' added
        """
        df = features_df.copy()
        pattern_names = list(self.pattern_rules.keys())

        # Compute evidence score for each pattern for each SKU
        all_scores = {p: np.zeros(len(df)) for p in pattern_names}

        for pattern, rules in self.pattern_rules.items():
            total_weight = sum(w for _, _, w in rules)

            for feat, score_fn, weight in rules:
                if feat not in df.columns:
                    continue
                # Score each row
                vals = df[feat].values
                scores = np.array([score_fn(v) for v in vals])
                all_scores[pattern] += scores * (weight / total_weight)

        # Build score matrix
        score_matrix = np.column_stack([all_scores[p] for p in pattern_names])

        # Winner-takes-all: highest evidence score
        best_idx = np.argmax(score_matrix, axis=1)
        df['Inferred_Pattern'] = [pattern_names[i] for i in best_idx]
        df['Pattern_Confidence'] = np.max(score_matrix, axis=1)

        # Normalize confidence to 0-1 range across all SKUs
        if df['Pattern_Confidence'].max() > 0:
            max_conf = df['Pattern_Confidence'].max()
            min_conf = df['Pattern_Confidence'].min()
            if max_conf > min_conf:
                df['Pattern_Confidence'] = (df['Pattern_Confidence'] - min_conf) / (max_conf - min_conf)

        # Print summary
        print("\n--- Pattern Inference Summary ---")
        print(df['Inferred_Pattern'].value_counts().to_string())
        avg_conf = df.groupby('Inferred_Pattern')['Pattern_Confidence'].mean()
        print("\nAverage Confidence per Pattern:")
        print(avg_conf.round(3).to_string())

        return df


if __name__ == "__main__":
    try:
        df = pd.read_csv("features_output.csv")
        engine = PatternInferenceEngine()
        result = engine.infer_patterns(df)

        print(f"\nPattern Inference Complete — {len(result)} SKUs classified.")
        print(f"Columns: {list(result.columns)}")

        result.to_csv("pattern_output.csv", index=False)
        print("Saved to pattern_output.csv")
    except Exception as e:
        print(f"Test failed: {e}")
