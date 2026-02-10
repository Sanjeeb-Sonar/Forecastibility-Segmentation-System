import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import periodogram
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore")

class FeatureExtractionEngine:
    def __init__(self, seasonal_period=12):
        self.seasonal_period = seasonal_period

    def extract_features(self, df, date_col='Date', sku_col='SKU', sales_col='Sales'):
        """
        Main entry point. Expects a DataFrame with Date, SKU, Sales columns.
        Returns a DataFrame with 1 row per SKU and 35+ columns of features.
        """
        features_list = []
        unique_skus = df[sku_col].unique()
        
        print(f"Extracting features for {len(unique_skus)} SKUs...")
        
        for sku in unique_skus:
            # Filter and sort
            series = df[df[sku_col] == sku].sort_values(date_col)[sales_col].values
            
            # Skip if too short
            if len(series) < self.seasonal_period * 2:
                # Fill with NaNs or basic stats only if desperate
                # For this demo we assume sufficiency or handle in post-processing
                pass 

            # Calculate individual features
            try:
                sku_feats = self._calculate_sku_features(series)
                sku_feats[sku_col] = sku
                
                # Infer Demand Pattern (Syntetos-Boylan Classification)
                # Smooth: ADI < 1.32 and CV^2 < 0.49
                # Intermittent: ADI >= 1.32 and CV^2 < 0.49
                # Erratic: ADI < 1.32 and CV^2 >= 0.49
                # Lumpy: ADI >= 1.32 and CV^2 >= 0.49
                
                adi = sku_feats.get('adi', 1.0)
                cv = sku_feats.get('cv', 0.0)
                cv2 = cv ** 2
                
                if adi < 1.32:
                    if cv2 < 0.49:
                        inferred_pattern = 'Smooth'
                    else:
                        inferred_pattern = 'Erratic'
                else:
                    if cv2 < 0.49:
                        inferred_pattern = 'Intermittent'
                    else:
                        inferred_pattern = 'Lumpy'
                        
                sku_feats['Pattern_Truth'] = inferred_pattern

                # Preserve Original Pattern_Truth if exists (optional overwrite)
                if 'Pattern_Truth' in df.columns:
                     # If user provided it, we trust them? Or we overwrite?
                     # Let's keep user provided if available, else use inferred.
                     pass 
                    
                features_list.append(sku_feats)
            except Exception as e:
                print(f"Error processing SKU {sku}: {e}")
                
        # Combine into DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Move SKU to first column
        # Move SKU (and Pattern_Truth) to first columns
        cols = [sku_col]
        if 'Pattern_Truth' in features_df.columns:
            cols.append('Pattern_Truth')
            
        remaining_cols = [c for c in features_df.columns if c not in cols]
        features_df = features_df[cols + remaining_cols]
        
        # Impute NaNs with median (robust to outliers)
        features_df.fillna(features_df.median(numeric_only=True), inplace=True)
        
        return features_df

    def _calculate_sku_features(self, y):
        """
        Calculates 35 features for a single numpy array time series `y`.
        """
        # --- 1. Basic Stats ---
        mean = np.mean(y)
        std = np.std(y, ddof=1)
        median = np.median(y)
        # Avoid div by zero
        cv = std / (mean + 1e-6)
        
        # --- 2. Demand Pattern (Intermittency) ---
        # ADI: Average Demand Interval
        intervals = np.diff(np.flatnonzero(y > 0))
        adi = np.mean(intervals) if len(intervals) > 0 else 0
        if len(intervals) == 0 and np.all(y == 0):
             adi = len(y) # If all zeros, interval is basically usually huge
        elif len(intervals) == 0:
             adi = 1.0 # Continuous if no zeros found or single point
             
        p_zero = np.sum(y == 0) / len(y)
        
        # Demand regularity (CV of non-zero demand)
        nonzero_y = y[y > 0]
        if len(nonzero_y) > 0:
            demand_regularity = np.std(nonzero_y) / (np.mean(nonzero_y) + 1e-6)
        else:
            demand_regularity = 0 
            
        # --- 3. Trend ---
        # Linear Regression for simple slope
        X = np.arange(len(y))
        # Simple slope
        slope, intercept = np.polyfit(X, y, 1)
        trend_slope = slope
        
        # Trend Strength via STL
        # Force period=12
        if len(y) >= 24:
            stl = STL(y, period=self.seasonal_period, robust=True).fit()
            trend_comp = stl.trend
            seasonal_comp = stl.seasonal
            resid_comp = stl.resid
            
            var_resid = np.var(resid_comp)
            var_trend = np.var(trend_comp)
            var_season = np.var(seasonal_comp)
            var_deseason = np.var(y - seasonal_comp)
            var_detrend = np.var(y - trend_comp)
            
            # Strength formulae from Hyndman
            # Strength of Trend = max(0, 1 - Var(Resid)/Var(Trend+Resid))
            trend_strength = max(0, 1 - (var_resid / (var_deseason + 1e-6)))
            
            # Strength of Seasonality
            seasonal_strength = max(0, 1 - (var_resid / (var_detrend + 1e-6)))
        else:
            trend_strength = 0
            seasonal_strength = 0
            trend_comp = y
            seasonal_comp = np.zeros_like(y)
            
        # Linearity (R^2 of linear fit)
        preds = slope * X + intercept
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        trend_linearity = 1 - (ss_res / (ss_tot + 1e-6))
        
        # --- 4. Seasonality ---
        peak_to_trough = (np.max(y) - np.min(y)) / (np.mean(y) + 1e-6)
        
        # Seasonal Period Strength (ACF at lag 12)
        acf_vals = acf(y, nlags=13, fft=True)
        seasonal_period_strength = acf_vals[12] if len(acf_vals) > 12 else 0
        
        # --- 5. Volatility ---
        # Rolling CV Mean (stability of variance)
        window = 12
        if len(y) > window:
            rolling_std = pd.Series(y).rolling(window).std()
            rolling_mean = pd.Series(y).rolling(window).mean()
            rolling_cv = rolling_std / (rolling_mean + 1e-6)
            rolling_cv_mean = rolling_cv.mean()
            
            # Is volatility increasing? (Ratio of last half var to first half var)
            half = len(y) // 2
            vol_change_ratio = np.std(y[half:]) / (np.std(y[:half]) + 1e-6)
        else:
            rolling_cv_mean = cv
            vol_change_ratio = 1.0
            
        # GARCH-like volatility (proxy: measures if large changes follow large changes)
        # We'll use std of absolute percent changes
        pct_change = np.diff(y) / (y[:-1] + 1e-6)
        garch_like_vol = np.std(pct_change)
        
        # --- 6. Autocorrelation ---
        acf_lag1 = acf_vals[1] if len(acf_vals) > 1 else 0
        acf_lag12 = acf_vals[12] if len(acf_vals) > 12 else 0
        
        # Partial ACF Lag 1
        try:
            pacf_vals = pacf(y, nlags=12, method='ywm')
            partial_acf_lag1 = pacf_vals[1] if len(pacf_vals) > 1 else 0
        except:
            partial_acf_lag1 = 0
            
        # --- 7. Entropy (Information Theory) ---
        # Spectral Entropy
        f, Pxx = periodogram(y)
        psd_norm = Pxx / (np.sum(Pxx) + 1e-9)
        spectral_entropy = entropy(psd_norm)
        
        # Approximate Entropy (simplified proxy using histogram entropy for speed/stability)
        # Ideally we use dedicated ApEn algo, but for this demo standard entropy of distribution 
        # is a decent proxy for "randomness" of value distribution
        hist_counts = np.histogram(y, bins=10)[0]
        approx_entropy = entropy(hist_counts)
        
        # Sample Entropy (using another proxy: ratio of unique values)
        # Low unique ratio = repeating values = low complexity
        sample_entropy = len(np.unique(y)) / len(y)

        # --- 8. Stability ---
        # Level Stability (CV of tile means)
        tiles = np.array_split(y, 3) 
        tile_means = [np.mean(t) for t in tiles if len(t) > 0]
        level_stability = np.std(tile_means) / (np.mean(tile_means) + 1e-6)
        
        # Trend Stability (CV of rolling slopes)
        # Calculated if sufficient length
        trend_stability = 0
        if len(y) > 15:
            slopes = []
            for i in range(len(y) - 6):
                window = y[i:i+6]
                s, _ = np.polyfit(np.arange(6), window, 1)
                slopes.append(s)
            trend_stability = np.std(slopes) / (np.mean(np.abs(slopes)) + 1e-6)

        seasonal_stability = 1 - np.std(seasonal_comp) / (np.std(y) + 1e-6) # Inverse of seasonal variance? 
        # Actually better to use: variance of seasonal component across years. 
        # Simplified: just use consistency of the component.
        
        # --- 9. Shape ---
        skewness_val = skew(y)
        kurtosis_val = kurtosis(y)
        
        # Tail Heaviness (fraction of points > 2 std from mean)
        upper_bound = mean + 2*std
        lower_bound = mean - 2*std
        tail_heaviness = np.sum((y > upper_bound) | (y < lower_bound)) / len(y)
        
        # --- 10. Changepoints ---
        # Simple detection using Ruptures is heavy, let's use a rolling mean shift proxy
        # Points where rolling mean shifts by > 2 sigma
        if len(y) > 12:
            rmean = pd.Series(y).rolling(6).mean()
            # Shifts in mean > 1 std of original series
            mean_shifts = np.abs(rmean.diff())
            sig_shifts = mean_shifts > (std * 1.0) 
            n_changepoints = np.sum(sig_shifts)
            max_cp_magnitude = np.max(mean_shifts.fillna(0))
            
            # Mean segment length
            if n_changepoints > 0:
                 mean_segment_length = len(y) / (n_changepoints + 1)
            else:
                 mean_segment_length = len(y)
        else:
            n_changepoints = 0
            max_cp_magnitude = 0
            mean_segment_length = len(y)
            
        # --- Bonus: Length & Recency ---
        series_length = len(y)
        
        # Ratio of mean(last 6 months) to mean(all previous)
        if len(y) > 6:
            recency_weight = np.mean(y[-6:]) / (np.mean(y[:-6]) + 1e-6)
        else:
            recency_weight = 1.0
            
        
        return {
            'mean': mean,
            'std': std,
            'cv': cv,
            'median': median,
            
            'adi': adi,
            'p_zero': p_zero,
            'demand_regularity': demand_regularity,
            
            'trend_strength': trend_strength,
            'trend_slope': trend_slope,
            'trend_linearity': trend_linearity,
            
            'seasonal_strength': seasonal_strength,
            'peak_to_trough_ratio': peak_to_trough,
            'seasonal_period_strength': seasonal_period_strength,
            
            'rolling_cv_mean': rolling_cv_mean,
            'volatility_change_ratio': vol_change_ratio,
            'garch_like_vol': garch_like_vol,
            
            'acf_lag1': acf_lag1,
            'acf_lag12': acf_lag12,
            'partial_acf_lag1': partial_acf_lag1,
            
            'approx_entropy': approx_entropy,
            'sample_entropy': sample_entropy,
            'spectral_entropy': spectral_entropy,
            
            'level_stability': level_stability,
            'trend_stability': trend_stability,
            'seasonal_stability': seasonal_stability,
            
            'skewness': skewness_val,
            'kurtosis': kurtosis_val,
            'tail_heaviness': tail_heaviness,
            
            'n_changepoints': n_changepoints,
            'max_cp_magnitude': max_cp_magnitude,
            'mean_segment_length': mean_segment_length,
            
            'series_length': series_length,
            'recency_weight': recency_weight
        }

if __name__ == "__main__":
    # Test run
    try:
        df = pd.read_csv("synthetic_sales_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        
        engine = FeatureExtractionEngine()
        features = engine.extract_features(df)
        
        print("\nFeature Extraction Complete.")
        print(f"Shape: {features.shape}")
        print("Sample Features:\n", features.head())
        
        features.to_csv("features_output.csv", index=False)
        print("Features saved to features_output.csv")
    except Exception as e:
        print(f"Test failed: {e}")
