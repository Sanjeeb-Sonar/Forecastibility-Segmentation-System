class MethodologyDocumentation:
    def print_report(self):
        report = """
================================================================================
FORECASTABILITY SEGMENTATION SYSTEM — METHODOLOGY REPORT
================================================================================

1. FEATURE EXTRACTION STRATEGY
--------------------------------------------------------------------------------
We extract 35 time series features across 10 dimensions to capture the full 
"DNA" of each SKU's demand pattern. A single metric (like CV) is insufficient.

- Basic Stats: Mean, Std, CV, Median (Distribution shape)
- Demand Pattern: ADI, % Zeros (Intermittency detection for Croston/SBA)
- Trend: Strength (STL), Slope, Linearity (Direction & consistency)
- Seasonality: Strength, Periodicity, Peak-to-Trough (Cyclical patterns)
- Volatility: Rolling CV, GARCH-proxy (Is variance changing?)
- Autocorrelation: ACF Lag 1/12, PACF (Memory & predictability)
- Information Theory: Approx/Sample/Spectral Entropy (Randomness/Complexity)
- Stability: Level/Trend/Seasonal stability over time
- Shape: Skewness, Kurtosis, Tail Heaviness (Extreme value risk)
- Changepoints: Structural breaks in demand history

2. ROBUST SEGMENTATION ENGINE
--------------------------------------------------------------------------------
Standard K-Means assumes spherical clusters and is sensitive to outliers. 
We use a robust multi-algorithm approach:

- Outlier Capping: Features clipped at 3-sigma to prevent distortion
- PCA: Dimensionality reduction (90% variance) to remove noise
- Multi-Algorithm: We test K-Means, Agglomerative (Ward), and GMM
- Multi-Metric Selection: Optimal K selected via majority vote of 
  Silhouette, Calinski-Harabasz, and Davies-Bouldin indices.
- Stability Analysis: Bootstrap resampling (20x) confirms clusters are real,
  not artifacts of sampling.

3. FORECASTABILITY CLASSIFIER (Composite Score)
--------------------------------------------------------------------------------
We determine "Easy", "Moderate", and "Hard" labels using a purely feature-driven
composite score, avoiding black-box dependencies.

- Polarity Alignment: All 35 features are aligned so that Higher = Easier.
  (e.g., Entropy is flipped: 1 - Entropy).
- Group-Balanced Weighting: Each of the 10 feature groups gets equal weight 
  (1/10). This ensures no single dimension (like Volatility) dominates.
- Labelling: Clusters are ranked by mean score and split into terciles.
- Validation: ANOVA F-stats confirm which features truly discriminate.

4. INTERPRETING THE CLASSES
--------------------------------------------------------------------------------
[Easy to Forecast]
- Characteristics: High seasonality, strong linear trend, low entropy, low CV.
- Recommended Models: Holt-Winters, SARIMA, Prophet, or even Naive.
- Strategy: Automate completely.

[Moderate to Forecast]
- Characteristics: Some signal but high noise, or changing regimes.
- Recommended Models: SES, Theta, MAPA, or LightGBM/XGBoost.
- Strategy: Monitor errors, lightweight human review on exceptions.

[Hard to Forecast]
- Characteristics: Intermittent/Lumpy (High % Zero), High Entropy, Structural Breaks.
- Recommended Models: Croston, SBA, ADIDA (if intermittent), or just Moving Average.
- Strategy: Don't trust the point forecast. Use safety stock buffering. 
  "Forecast the risk, not the number."

================================================================================
"""
        print(report)

if __name__ == "__main__":
    docs = MethodologyDocumentation()
    docs.print_report()
