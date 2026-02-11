class MethodologyDocumentation:
    def print_report(self):
        report = """
================================================================================
FORECASTABILITY SEGMENTATION SYSTEM — METHODOLOGY REPORT
================================================================================

1. DATA FOUNDATION
--------------------------------------------------------------------------------
The system accepts raw transactional data with exactly 3 columns: Date, SKU,
and Sales. This ensures maximum flexibility for external data uploads.

2. FEATURE EXTRACTION STRATEGY
--------------------------------------------------------------------------------
We extract 35 time series features across 10 dimensions to capture the full 
"DNA" of each SKU. A single metric (like CV) is insufficient for complex demand.

3. MULTI-SIGNAL PATTERN INFERENCE
--------------------------------------------------------------------------------
Instead of relying on a simplistic 2-feature quadrant (ADI/CV²), we use a 
dedicated Pattern Inference Engine. It evaluates all 35 features using 
multi-signal sigmoid scoring to classify SKUs into 7 archetypes: 
(Smooth, Seasonal, Intermittent, Lumpy, Trending, Erratic, New Product).

4. ROBUST SEGMENTATION ENGINE
--------------------------------------------------------------------------------
Standard K-Means is sensitive to outliers. We use a robust approach:
- PCA: Dimensionality reduction (90% variance) to remove noise.
- Multi-Algorithm Sweep: Testing K-Means, Agglomerative, and GMM.
- Stability Analysis: Resampling (20x) to ensure cluster reliability.

5. FORECASTABILITY CLASSIFIER (Composite Score)
--------------------------------------------------------------------------------
We determine "Easy", "Moderate", and "Hard" labels using a feature-driven 
composite score. Each feature is polarity-aligned (Higher = Easier) and 
averaged within 11 groups for balanced representation.

6. INTERPRETING THE CLASSES
--------------------------------------------------------------------------------
... [Same as before] ...

================================================================================
"""
        print(report)

if __name__ == "__main__":
    docs = MethodologyDocumentation()
    docs.print_report()
