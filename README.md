# Forecastability Segmentation System 🎯

A data-driven machine learning system that segments SKU portfolios based on "Forecastability"—determining which products are Easy, Moderate, or Hard to forecast before modeling begins.

## 🚀 Key Features

*   **Multi-Dimensional Feature Extraction**: Extracts 35+ time series features (Seasonality, Trend, Entropy, Intermittency, Volatility) for every SKU.
*   **Robust Segmentation**: Uses an ensemble of K-Means, GMM, and Agglomerative clustering to find natural SKU groupings.
*   **Smart Pattern Inference**: Automatically detects demand patterns (Smooth, Intermittent, Lumpy, Erratic) using ADI and CV² rules—even on raw data.
*   **Cluster Naming**: Assigns descriptive labels to clusters (e.g., "Seasonal", "Volatile", "Lumpy") based on their centroid characteristics.
*   **Interactive Dashboard**: A Streamlit-based UI for exploring the portfolio, visualizing segments, and drilling down into individual SKUs.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Sanjeeb-Sonar/Forecastibility-Segmentation-System.git
    cd Forecastibility-Segmentation-System
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### Option 1: Interactive Dashboard (Recommended)

Double-click `run_dashboard.bat` or run:
```bash
streamlit run dashboard.py
```
This launches the web interface where you can:
*   Upload your own sales data (CSV with `Date`, `SKU`, `Sales`).
*   Visualize portfolio health (Easy vs. Hard split).
*   Filter by specific patterns or clusters.

### Option 2: Full Pipeline (CLI)

To regenerate all data and models from scratch:
```bash
python main.py
```
This script will:
1.  Generate synthetic data (if no input provided).
2.  Run feature extraction and segmentation.
3.  Save results to `final_output.csv`.
4.  Generate static analysis charts in the root folder.

## 📊 Methodology

The system assesses forecastability by analyzing:
1.  **Demand Patterns**: Is the demand continuous or intermittent? (ADI/CV²)
2.  **Signal Strength**: How strong are the Trend and Seasonality components? (STL Decomposition)
3.  **Noise & Entropy**: How random is the series? (Spectral Entropy, Volatility)

Clusters are labeled as **Easy**, **Moderate**, or **Hard** based on a composite score derived from these features.

## 📁 Project Structure

*   `dashboard.py`: Streamlit application code.
*   `feature_extraction.py`: Engine for calculating time series metrics.
*   `segmentation.py`: Clustering logic (K-Means/GMM/Agglomerative).
*   `forecastability.py`: Logic for scoring and labeling SKUs.
*   `data_generator.py`: Creates synthetic retail sales data for testing.
*   `main.py`: Orchestrator script for the batch pipeline.
