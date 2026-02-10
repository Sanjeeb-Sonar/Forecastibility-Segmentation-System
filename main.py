import pandas as pd
import os
import time

# Import modules
from data_generator import generate_synthetic_data
from feature_extraction import FeatureExtractionEngine
from segmentation import SegmentationEngine
from forecastability import ForecastabilityClassifier
from visualization import VisualizationEngine
from methodology import MethodologyDocumentation

def run_pipeline():
    print("================================================================================")
    print("STARTING FORECASTIBILITY SEGMENTATION PIPELINE")
    print("================================================================================")
    start_time = time.time()
    
    # 1. Data Generation
    print("\n[STEP 1] Generating Data...")
    if not os.path.exists("synthetic_sales_data.csv"):
        df = generate_synthetic_data(n_skus=150) # Use slightly more SKUs
        df.to_csv("synthetic_sales_data.csv", index=False)
        print(f"Generated {len(df)} rows for {df['SKU'].nunique()} SKUs.")
    else:
        print("Using existing synthetic_sales_data.csv")
        df = pd.read_csv("synthetic_sales_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])

    # 2. Feature Extraction
    print("\n[STEP 2] Extracting Features...")
    feature_engine = FeatureExtractionEngine()
    features_df = feature_engine.extract_features(df)
    features_df.to_csv("features_output.csv", index=False)
    print(f"Extracted 35 features for {len(features_df)} SKUs.")

    # 3. Segmentation
    print("\n[STEP 3] Running Robust Segmentation...")
    seg_engine = SegmentationEngine(min_k=2, max_k=8, n_bootstrap=20)
    segmented_df = seg_engine.run_segmentation(features_df)
    segmented_df.to_csv("segmentation_output.csv", index=False)
    print(f"Segmentation complete. Optimal K={seg_engine.best_k} using {seg_engine.best_algo_name}.")

    # 4. Forecastability Classification
    print("\n[STEP 4] Classifying Forecastability...")
    classifier = ForecastabilityClassifier()
    final_df, anova_df = classifier.classify(segmented_df)
    final_df.to_csv("final_output.csv", index=False)
    
    print("\nForecastability counts:")
    print(final_df['Forecastability_Label'].value_counts())
    
    if not anova_df.empty:
        print("\nTop 5 Discriminative Features (ANOVA):")
        print(anova_df.head(5)[['Feature', 'F_Stat']])

    # 5. Methodology & Documentation
    print("\n[STEP 5] Generating Methodology Report...")
    docs = MethodologyDocumentation()
    docs.print_report()

    # 6. Visualization
    print("\n[STEP 6] Generating Visualizations...")
    viz_engine = VisualizationEngine()
    # Pass segmentation engine to plot K selection
    viz_engine.run_all_visualizations(final_df, segmentation_engine=seg_engine)
    
    elapsed = time.time() - start_time
    print("\n================================================================================")
    print(f"PIPELINE COMPLETE in {elapsed:.2f} seconds.")
    print("Outputs saved:")
    print("- final_output.csv (Results)")
    print("- viz_*.png (Charts)")
    print("================================================================================")

if __name__ == "__main__":
    run_pipeline()
