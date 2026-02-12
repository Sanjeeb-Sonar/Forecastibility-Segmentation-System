# Technical Deep-Dive: Forecastability Segmentation System 🔬

## 1. Objective & Strategic Vision
The Forecastability Segmentation System is designed to solve the "modeling exhaustion" problem in supply chain analytics. Instead of applying complex forecasting models to every SKU, this system uses **Feature-Driven Inference** to segment a portfolio into **Easy**, **Moderate**, and **Hard** categories. 

This allows organizations to:
*   **Allocate Resources**: Focus human analysts and expensive compute on "Moderate" SKUs.
*   **Automate "Easy"**: Use simple AI/ML pipelines for predictable items.
*   **Manage Expectations on "Hard"**: Use naive baselines for chaotic items where complex models would likely overfit.

---

## 2. The Case for 35 Features (The "DNA" Rationale)
Traditional demand classification (like the Syntetos-Boylan quadrant) uses only 2 metrics: **ADI** (Average Demand Interval) and **CV²** (Coefficient of Variation). 

Our system extracts **35 features** across **11 dimensions** to capture a "High Definition" view of demand:
1.  **Multi-Signal Capture**: A SKU can be highly volatile (Bad) but also highly seasonal (Good). A 2-feature system misses this interaction.
2.  **Information Theory**: We use **Entropy** (Spectral, Sample, Approximate) to measure the fundamental randomness of the data, which is a stronger indicator of forecastability than simple variance.
3.  **Structural Integrity**: We detect **Changepoints** and **Stability** to determine if historical patterns are still relevant for future projections.

---

## 3. Pipeline Stages & Technical Thresholds

### Stage 1: Data Preprocessing
*   **Minimum History**: SKUs with less than **24 months** of data (2 * seasonal period) are flagged as "Short History" or "New Product."
*   **Scaling**: All features are scaled using `StandardScaler` and **clipped at ±3 standard deviations** to prevent extreme outliers from skewing the clustering engine.

### Stage 2: Dimensionality Reduction (PCA)
*   **Threshold**: The system applies Principal Component Analysis (PCA) to noise-reduce the 35 features, retaining **90% of the total variance**. 
*   **Action**: This typically reduces the 35 raw features down to 8–12 "eigen-features" which represent the core behavior of the SKU.

### Stage 3: Robust Segmentation (Clustering Ensemble)
*   **Multi-Algorithm Sweep**: We run **K-Means**, **Gaussian Mixture Models (GMM)**, and **Agglomerative Clustering** for $K \in [2, 8]$.
*   **Recursive Divisive Splitting (The 10% Rule)**: 
    *   To ensure high-resolution portfolio management, no segment is allowed to contain more than **10% of total SKUs**.
    *   If a cluster exceeds this threshold, the engine recursively triggers a "sub-split" using K-Means (K=2) until all final segments satisfy the size constraint.
    *   This prevents "mega-clusters" from hiding diverse behaviors.
*   **Stability Threshold**: We run **Bootstrap Stability** checks to ensure groups are reliable.

### Stage 4: Pattern Inference Logic (Sigmoid Rules)
We use a **Weighted Multi-Signal Evidence Scoring** system. Each feature's contribution is put through a **Sigmoid Activation Function**:
$$f(x) = \frac{1}{1 + e^{-\text{steepness} \cdot (x - \text{center})}}$$

**Key Thresholds (Centers):**
*   **Intermittent**: `p_zero` (30% zeros), `adi` (1.5 interval).
*   **Lumpy**: `cv` (0.8), `p_zero` (25%).
*   **Seasonal**: `seasonal_strength` (0.25), `acf_lag12` (0.15).
*   **Trending**: `trend_strength` (0.35), `trend_linearity` (0.25).
*   **New Product**: `series_length` (18 months - negative weight).

---

## 4. The Final Labeling Algorithm (Triangulated Independence)

To ensure the highest accuracy, the system treats **Score**, **Pattern**, and **Cluster** as three independent dimensions. They are cross-referenced (Triangulated) to find the final label, rather than being dependent on each other.

### Signal A: Numeric Forecastability Score (Independent)
*   **Calculation**: A balanced average across 11 feature groups (Trend, Seasonality, Entropy, etc.).
*   **Polarity**: Each feature is aligned so that higher = easier to forecast.
*   **Result**: A score from 0.0 to 1.0.

### Signal B: Demand Pattern Diagnostics (Independent)
*   **Calculation**: Logic-driven inference (Sigmoid centers) that detects "Smooth," "Seasonal," "Lumpy," etc.
*   **Result**: A categorical behavior label.

### Signal C: Behavioral Clusters (Independent)
*   **Calculation**: Unsupervised machine learning (K-Means/GMM) that groups SKUs by "mathematical shape."
*   **Constraint**: No cluster contains >10% of the portfolio.
*   **Result**: A cluster ID (Cluster 1, Cluster 2, etc.).

### The "Consensus" Voting System
The final **Forecastability Label** is determined by the convergence of Signal A and Signal B:
1.  **Score Signal (0–2 pts)**: Based on the SKU's numeric score relative to the portfolio (Terciles).
2.  **Pattern Signal (0–2 pts)**: Based on the inherent difficulty of the detected pattern (e.g., Smooth = 2, Lumpy = 0).
3.  **Label Calculation**:
    *   **Easy**: Average Signal $\ge$ 1.5 (Both signals agree it's predictable).
    *   **Moderate**: Average Signal $\ge$ 0.75 (Mixed signals).
    *   **Hard**: Average Signal $<$ 0.75 (Both signals agree it's chaotic).

**Why this is better**:
By keeping these components independent, we avoid "circular logic" where a bad cluster ruins a good score. Instead, if a SKU has a high score and a stable pattern, it is marked **Easy** even if it sits next to "Moderate" items in a cluster. This is the **Triangulation of Truth**.

---

## 5. Maintenance & Validation
*   **ANOVA F-Stats**: The system calculates the **F-Statistic** to validate that the labels (Easy vs. Hard) are statistically distinct across the 35 features.
*   **Calibration**: If the portfolio shifts (e.g., market volatility increases), the quartile splits in Stage 4 will automatically adjust, making the labeling relative to the *current* state of the business.
