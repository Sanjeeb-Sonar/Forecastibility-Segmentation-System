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
*   **Metric Optimization**: We calculate **Silhouette**, **Calinski-Harabasz**, and **Davies-Bouldin** scores.
*   **Stability Threshold**: We run **20 Bootstrap Iterations** (subsampling 80% of data).
*   **Final selection formula**: `0.6 * Silhouette_Score + 0.4 * Bootstrap_Stability_Score`.

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

## 4. The Final Labeling Algorithm (The Logic)

We leverage three independent signals to determine the final **Forecastability Label**:

### Signal A: Composite Score (Feature Polarity)
Each feature is assigned a **Polarity** (+1 = Easier, -1 = Harder).
*   **Positive (+1)**: Trend/Seasonal Strength, Autocorrelation, Stability.
*   **Negative (-1)**: CV, Entropy, Changepoints, Intermittency.
*   **Calculation**: We calculate the balanced average of 11 feature groups to get a score from 0.0 to 1.0.

### Signal B: The Points Equation
We assign "Points" to each SKU based on its performance:
1.  **Bucketing (0–3 pts)**: SKUs are split into Quartiles based on their composite score. `Q4 (Top 25%) = 3 pts`, `Q1 (Bottom 25%) = 0 pts`.
2.  **Pattern Points (+1 or -1)**:
    *   **+1 Bonus**: If Pattern $\in$ {Smooth, Seasonal, Trending}.
    *   **-1 Penalty**: If Pattern $\in$ {Intermittent, Lumpy, Erratic}.
3.  **Cluster Adjustment (+1 or -1)**:
    *   **Tier Ranking**: Clusters are ranked by their mean composite score.
    *   **+1**: SKUs in "High Performance" clusters.
    *   **-1**: SKUs in "Low Performance" clusters.

### Final Classification Thresholds:
$$\text{Total Points} = \text{Bucket Points} + \text{Pattern Adjustment} + \text{Cluster Adjustment}$$

| Total Points | Final Label | Forecast Strategy Recommendation |
| :--- | :--- | :--- |
| **4 or higher** | **Easy** | Full AI Automation (Prophet, NeuralProphet). |
| **2 or 3** | **Moderate** | Standard ML Models with human-in-the-loop review. |
| **Less than 2** | **Hard** | Use Naive/Simple baselines; avoid overfitting mess. |

---

## 5. Maintenance & Validation
*   **ANOVA F-Stats**: The system calculates the **F-Statistic** to validate that the labels (Easy vs. Hard) are statistically distinct across the 35 features.
*   **Calibration**: If the portfolio shifts (e.g., market volatility increases), the quartile splits in Stage 4 will automatically adjust, making the labeling relative to the *current* state of the business.
