import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f_oneway
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class ForecastabilityClassifier:
    def __init__(self):
        # Define Polarity: +1 (Higher = Easier), -1 (Higher = Harder)
        # 0 means ambiguous or context dependent (we'll skipping or normalizing them neutrally)
        # Actually plan says: normalize context-dependent to neutral? 
        # Better: if irrelevant, weight=0. 
        # But for 'mean', 'median', 'peak_to_trough' -> let's treat as neutral (0)
        # or maybe slightly positive? Higher mean = usually better signal-to-noise. Let's stick to Plan.
        
        self.feature_polarity = {
            # --- Positive (+1) ---
            'acf_lag1': 1, 'acf_lag12': 1, 'partial_acf_lag1': 1,
            'trend_strength': 1, 'trend_linearity': 1,
            'seasonal_strength': 1, 'seasonal_period_strength': 1,
            'level_stability': 1, 'trend_stability': 1, 'seasonal_stability': 1,
            'series_length': 1, 'demand_regularity': 1, 'mean_segment_length': 1,
            
            # --- Negative (-1) ---
            'cv': -1, 'std': -1, 
            'approx_entropy': -1, 'sample_entropy': -1, 'spectral_entropy': -1,
            'p_zero': -1, 'adi': -1, 
            'n_changepoints': -1, 'max_cp_magnitude': -1,
            'kurtosis': -1, 'skewness': -1, 'tail_heaviness': -1,
            'rolling_cv_mean': -1, 'volatility_change_ratio': -1, 'garch_like_vol': -1,
            'recency_weight': -1,
            
            # --- Neutral/Contextual (0) --- 
            # We won't use these for the score, as they don't universally indicate difficulty
            'mean': 0, 'median': 0, 'trend_slope': 0, 'peak_to_trough_ratio': 0
        }
        
        # Define Groups for Balancing
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
        
        # Group weights (1/10 for main groups, maybe split bonus?)
        # Let's give equal weight to all groups including Bonus for simplicity
        # Total groups = 11. 
        self.group_weight = 1.0 / len(self.feature_groups)

    def classify(self, segmentation_df, cluster_col='Cluster'):
        """
        Main entry point.
        Input: DataFrame from SegmentationEngine (with features + Cluster col)
        Output: DataFrame with 'Forecastability_Score', 'Forecastability_Label' added.
        """
        df = segmentation_df.copy()
        
        # 1. Compute Composite Score Per SKU
        scaler = MinMaxScaler()
        feature_cols = [c for c in df.columns if c in self.feature_polarity]
        
        # Normalize all features 0-1 first
        df_norm = df.copy()
        df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Calculate score from groups
        scores = np.zeros(len(df))
        
        for group, features in self.feature_groups.items():
            group_score = np.zeros(len(df))
            valid_feats = 0
            
            for f in features:
                if f not in df.columns: continue
                polarity = self.feature_polarity.get(f, 0)
                
                if polarity == 0:
                    continue
                
                val = df_norm[f]
                if polarity == -1:
                    val = 1 - val # Flip so higher is better
                
                group_score += val
                valid_feats += 1
            
            if valid_feats > 0:
                # Average within group
                group_score /= valid_feats
                # Add weighted group score to total
                scores += group_score * self.group_weight
                
        df['Forecastability_Score'] = scores
        
        # 2. Assign Labels based on Cluster Means
        cluster_scores = df.groupby(cluster_col)['Forecastability_Score'].mean()
        
        # Rank clusters
        sorted_clusters = cluster_scores.sort_values(ascending=False).index.tolist()
        
        # Tercile split
        n_clusters = len(sorted_clusters)
        chunk_size = np.ceil(n_clusters / 3).astype(int)
        
        # Easy (Top 1/3)
        easy_clusters = sorted_clusters[:chunk_size]
        # Moderate (Middle 1/3)
        mod_clusters = sorted_clusters[chunk_size : 2*chunk_size]
        # Hard (Bottom 1/3)
        hard_clusters = sorted_clusters[2*chunk_size:]
        
        label_map = {}
        for c in easy_clusters: label_map[c] = 'Easy'
        for c in mod_clusters: label_map[c] = 'Moderate'
        for c in hard_clusters: label_map[c] = 'Hard'
        
        df['Forecastability_Label'] = df[cluster_col].map(label_map)
        
        # 3. Validate with ANOVA F-stat
        print("Validating with ANOVA F-stats...")
        anova_results = []
        
        groups = [df[df['Forecastability_Label'] == label] for label in ['Easy', 'Moderate', 'Hard']]
        # Filter groups that might be empty if K<3
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) > 1:
            for feat in feature_cols:
                # Get values for each group
                vals = [g[feat].values for g in groups]
                try:
                    f_stat, p_val = f_oneway(*vals)
                    anova_results.append({'Feature': feat, 'F_Stat': f_stat, 'P_Value': p_val})
                except:
                    pass
            
            anova_df = pd.DataFrame(anova_results).sort_values('F_Stat', ascending=False)
            print("Top 5 Discriminative Features:")
            print(anova_df.head(5))
        else:
            print("Not enough categories for ANOVA (K < 2).")
            anova_df = pd.DataFrame()

        return df, anova_df

if __name__ == "__main__":
    try:
        df = pd.read_csv("segmentation_output.csv")
        clf = ForecastabilityClassifier()
        result_df, anova_df = clf.classify(df)
        
        print("\nClassification Complete.")
        print(result_df['Forecastability_Label'].value_counts())
        
        result_df.to_csv("final_output.csv", index=False)
        print("Saved to final_output.csv")
    except Exception as e:
        print(f"Test failed: {e}")
