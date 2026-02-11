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

The system assesses forecastability in 4 distinct inference stages:
1.  **Pattern Inference**: Detects archetypes (Smooth, Seasonal, Intermittent, etc.) using multi-signal weights from all 35 features.
2.  **Cluster Inference**: Segments SKUs into natural groupings using GMM and KMeans with bootstrap stability validation.
3.  **Score Inference**: Computes a balanced "Forecastability Score" based on feature polarity (Higher = Easier to forecast).
4.  **Label Inference**: Categorizes clusters into **Easy**, **Moderate**, or **Hard** based on score rankings.

## 📁 Project Structure

1.  **`data_generator.py`**: Generates 3-column raw sales data (`Date, SKU, Sales`).
2.  **`feature_extraction.py`**: Pure 35-feature extraction engine.
3.  **`pattern_inference.py`**: Robust multi-signal demand pattern classification.
4.  **`segmentation.py`**: Multi-algorithm cluster segmentation engine.
5.  **`forecast_score.py`**: Composite forecastability scoring and labeling.
6.  **`main.py`**: CLI orchestrator for the full 7-step pipeline.
7.  **`dashboard.py`**: Interactive Streamlit application.

Support modules:
*   `methodology.py`: Report generation logic.
*   `visualization.py`: Static chart generation logic.
*   `requirements.txt`: Dependencies.
*   `run_dashboard.bat`: Windows shortcut.
