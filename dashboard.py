import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

# Import Analysis Pipeline
from feature_extraction import FeatureExtractionEngine
from pattern_inference import PatternInferenceEngine
from segmentation import SegmentationEngine
from forecast_score import ForecastScoreEngine

# Page Config
st.set_page_config(page_title="Demand Forecasting Manager", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .big-font { font-size:24px !important; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Demand Forecasting Segmentation System")

# --- SESSION STATE INITIALIZATION ---
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "Demo"

# --- PIPELINE FUNCTION ---
def run_full_pipeline(df):
    """Run the complete modular pipeline: Features → Patterns → Clusters → Scores"""
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # 1. Feature Extraction
        status_text.text("Step 1/4: Extracting 35 features...")
        feature_engine = FeatureExtractionEngine()
        features_df = feature_engine.extract_features(df)
        features_df.to_csv("features_output.csv", index=False)
        progress_bar.progress(25)
        
        # 2. Pattern Inference
        status_text.text("Step 2/4: Inferring demand patterns (multi-signal)...")
        pattern_engine = PatternInferenceEngine()
        pattern_df = pattern_engine.infer_patterns(features_df)
        pattern_df.to_csv("pattern_output.csv", index=False)
        progress_bar.progress(50)
        
        # 3. Segmentation (Clustering)
        status_text.text("Step 3/4: Running cluster segmentation...")
        seg_engine = SegmentationEngine(min_k=2, max_k=8, n_bootstrap=20)
        segmented_df = seg_engine.run_segmentation(pattern_df)
        segmented_df.to_csv("segmented_output.csv", index=False)
        progress_bar.progress(75)
        
        # 4. Forecast Score
        status_text.text("Step 4/4: Computing forecastability scores...")
        score_engine = ForecastScoreEngine()
        final_df, _ = score_engine.score(segmented_df)
        final_df.to_csv("final_output.csv", index=False)
        progress_bar.progress(100)
        
        status_text.success("Analysis Complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return final_df
        
    except Exception as e:
        status_text.error(f"Pipeline Failed: {str(e)}")
        return None

# --- SIDEBAR: DATA LOADING & FILTERS ---
with st.sidebar:
    st.header("📂 Data Source")
    source_option = st.radio("Select Source:", ["Use Demo Data", "Upload CSV"])
    
    if source_option == "Upload CSV":
        st.caption("Upload a CSV with exactly 3 columns: **Date**, **SKU**, **Sales**")
        uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type=['csv'])
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                required_cols = ['Date', 'SKU', 'Sales']
                if not all(col in raw_df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {required_cols}")
                else:
                    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
                    st.success(f"Loaded {len(raw_df)} rows, {raw_df['SKU'].nunique()} SKUs.")
                    
                    if st.button("🚀 Run Segmentation Pipeline"):
                        with st.spinner("Processing your data..."):
                            results = run_full_pipeline(raw_df)
                            if results is not None:
                                st.session_state.results_df = results
                                st.session_state.sales_df = raw_df
                                st.session_state.data_source = "Upload"
                                st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        # Demo Data Logic
        if st.session_state.data_source != "Demo" or st.session_state.sales_df is None:
            try:
                st.session_state.sales_df = pd.read_csv("synthetic_sales_data.csv")
                st.session_state.sales_df['Date'] = pd.to_datetime(st.session_state.sales_df['Date'])
                st.session_state.results_df = pd.read_csv("final_output.csv")
                st.session_state.data_source = "Demo"
            except:
                st.warning("Demo data not found. Please run main.py or upload data.")

    st.write("---")
    
    # --- FILTERS ---
    st.header("🎯 Filters")
    
    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        sales_df = st.session_state.sales_df
        
        # 1. Forecastability Filter
        cats = ['Easy', 'Moderate', 'Hard']
        selected_cats = st.multiselect("Forecastability", cats, default=cats)
        
        # 2. Cluster Filter
        all_clusters = sorted(results_df['Cluster'].unique())
        selected_clusters = st.multiselect("Cluster", all_clusters, default=all_clusters)
        
        # 3. Pattern Filter (from pattern inference)
        if 'Inferred_Pattern' in results_df.columns:
            all_patterns = sorted(results_df['Inferred_Pattern'].astype(str).unique())
            selected_patterns = st.multiselect("Demand Pattern (Inferred)", all_patterns, default=all_patterns)
        else:
            selected_patterns = []

        # 4. Score Bucket Filter
        if 'Score_Bucket' in results_df.columns:
            all_buckets = sorted(results_df['Score_Bucket'].astype(str).unique())
            selected_buckets = st.multiselect("Score Bucket (Rank Grade)", all_buckets, default=all_buckets)
        else:
            selected_buckets = []
            
        # Apply Filters
        mask = (results_df['Forecastability_Label'].isin(selected_cats)) & \
               (results_df['Cluster'].isin(selected_clusters))
        
        if selected_patterns:
            mask = mask & (results_df['Inferred_Pattern'].isin(selected_patterns))
        if selected_buckets:
            mask = mask & (results_df['Score_Bucket'].isin(selected_buckets))
            
        filtered_results = results_df[mask]
        filtered_skus = filtered_results['SKU'].unique()
        filtered_sales = sales_df[sales_df['SKU'].isin(filtered_skus)]
        
        st.metric("SKUs Selected", len(filtered_skus))
        
    else:
        st.info("Load data to see filters.")
        filtered_results = None
        filtered_sales = None

# --- MAIN DASHBOARD CONTENT ---
if filtered_results is not None:
    
    has_pattern = 'Inferred_Pattern' in filtered_results.columns

    # TABS
    tab_overview, tab_drivers, tab_deepdive = st.tabs([
        "1️⃣ Portfolio Diagnosis", 
        "2️⃣ Forecastability Drivers", 
        "3️⃣ SKU Deep Dive"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab_overview:
        st.markdown("### 🏥 Portfolio Health")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total SKUs", len(filtered_results))
        counts = filtered_results['Forecastability_Label'].value_counts()
        c2.metric("Easy", counts.get('Easy', 0))
        c3.metric("Moderate", counts.get('Moderate', 0))
        c4.metric("Hard", counts.get('Hard', 0))
        
        st.divider()
        
        col_pca, col_stats = st.columns([2, 1])
        with col_pca:
            hover_cols = ['SKU']
            if has_pattern:
                hover_cols.append('Inferred_Pattern')
            
            # Determine symbol col
            sym_col = 'Inferred_Pattern' if has_pattern else 'Cluster'
            
            fig_pca = px.scatter(
                filtered_results, x='PCA1', y='PCA2', 
                color='Forecastability_Label', symbol=sym_col,
                hover_data=hover_cols,
                color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'},
                title="Segmentation Map (PCA)"
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
        with col_stats:
            st.subheader("Cluster Profile")
            st.caption("Average features per cluster")
            cols_to_show = ['p_zero', 'cv', 'seasonal_strength', 'approx_entropy', 'trend_strength']
            cols_to_show = [c for c in cols_to_show if c in filtered_results.columns]
            
            means = filtered_results.groupby('Cluster')[cols_to_show].mean()
            st.dataframe(means.style.format("{:.2f}").background_gradient(cmap='Blues'), height=400)

        # Pattern Distribution
        if has_pattern:
            st.divider()
            st.markdown("### 📊 Pattern Distribution")
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                pattern_counts = filtered_results['Inferred_Pattern'].value_counts().reset_index()
                pattern_counts.columns = ['Pattern', 'Count']
                fig_pat = px.bar(pattern_counts, x='Pattern', y='Count', color='Pattern',
                                title="Inferred Pattern Distribution")
                st.plotly_chart(fig_pat, use_container_width=True)
            
            with col_p2:
                # Pattern vs Forecastability cross-tab
                cross = pd.crosstab(filtered_results['Inferred_Pattern'], 
                                    filtered_results['Forecastability_Label'])
                fig_cross = px.imshow(cross, text_auto=True, 
                                     color_continuous_scale='Blues',
                                     title="Pattern × Forecastability Heatmap")
                st.plotly_chart(fig_cross, use_container_width=True)

    # --- TAB 2: DRIVERS ---
    with tab_drivers:
        st.markdown("### 🔍 What Drives Forecastability?")
        col_rad, col_box = st.columns(2)
        
        with col_rad:
            feat_candidates = ['cv', 'p_zero', 'trend_strength', 'seasonal_strength', 
                             'approx_entropy', 'skewness', 'acf_lag1', 'garch_like_vol']
            present_feats = [f for f in feat_candidates if f in filtered_results.columns]
            
            if present_feats:
                # Group by label and get means
                df_rad = filtered_results.groupby('Forecastability_Label')[present_feats].mean()
                
                # Apply scaling so all axes are 0-1 (Radar Chart Fix)
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                if not df_rad.empty:
                    df_rad_scaled = pd.DataFrame(
                        scaler.fit_transform(df_rad),
                        columns=df_rad.columns,
                        index=df_rad.index
                    ).reset_index()
                    
                    df_melt = df_rad_scaled.melt(id_vars='Forecastability_Label', var_name='Feature', value_name='Value')
                    
                    fig_rad = px.line_polar(
                        df_melt, r='Value', theta='Feature', color='Forecastability_Label', line_close=True,
                        color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'},
                        title="Normalized Feature DNA (Radar)"
                    )
                    st.plotly_chart(fig_rad, use_container_width=True)
                else:
                    st.info("No data available for radar chart.")
        
        with col_box:
            st.caption("Distribution of features by forecastability")
            exclude_cols = ['Date', 'SKU', 'Cluster', 'Forecastability_Label', 'Forecastability_Score',
                          'PCA1', 'PCA2', 'Inferred_Pattern', 'Pattern_Confidence', 
                          'Model_Used', 'Algorithm_Stability', 'Optimal_K']
            all_feats = [c for c in filtered_results.columns if c not in exclude_cols 
                        and filtered_results[c].dtype in ['float64', 'int64', 'float32', 'int32']]
            sel_feat = st.selectbox("Compare Feature:", all_feats, index=0 if len(all_feats) > 0 else None)
            
            if sel_feat:
                fig_box = px.box(
                    filtered_results, x='Forecastability_Label', y=sel_feat, color='Forecastability_Label',
                    color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'},
                    title=f"Distribution: {sel_feat}"
                )
                st.plotly_chart(fig_box, use_container_width=True)

    # --- TAB 3: DEEP DIVE ---
    with tab_deepdive:
        st.markdown("### 📉 Historical Analysis")
        st.caption("Filtered View of Historical Sales")
        
        st.info(f"Viewing {len(filtered_skus)} SKUs | Categories: {', '.join(selected_cats)}")
        
        MAX_LINES = 100
        plot_data = filtered_sales
        if len(filtered_skus) > MAX_LINES:
            st.warning(f"⚠️ High volume! Showing random sample of {MAX_LINES} SKUs.")
            sample_skus = np.random.choice(filtered_skus, MAX_LINES, replace=False)
            plot_data = filtered_sales[filtered_sales['SKU'].isin(sample_skus)]
        
        plot_data = plot_data.merge(filtered_results[['SKU', 'Forecastability_Label']], on='SKU', how='left')
        
        fig_hist = px.line(
            plot_data, x='Date', y='Sales', 
            color='Forecastability_Label', line_group='SKU',
            color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'},
            title="Sales History (Filtered)"
        )
        fig_hist.update_traces(opacity=0.5, line=dict(width=1))
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.divider()
        st.markdown("### 🧊 3D Feature Explorer")
        st.caption("Interact with the raw feature space for selected SKUs. Choose any 3 metrics as axes.")
        
        # Identify numeric feature columns
        exclude = ['Date', 'SKU', 'Cluster', 'Model_Used', 'Algorithm_Stability', 'PCA1', 'PCA2', 'Forecastability_Score', 'Forecastability_Label', 'Inferred_Pattern', 'Pattern_Confidence', 'Score_Bucket', 'Optimal_K']
        feat_cols = [c for c in filtered_results.columns if c not in exclude and filtered_results[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        if len(feat_cols) >= 3:
            f1, f2, f3 = st.columns(3)
            with f1: x_axis = st.selectbox("X Axis", feat_cols, index=feat_cols.index('cv') if 'cv' in feat_cols else 0, key='x3d')
            with f2: y_axis = st.selectbox("Y Axis", feat_cols, index=feat_cols.index('adi') if 'adi' in feat_cols else 1, key='y3d')
            with f3: z_axis = st.selectbox("Z Axis", feat_cols, index=feat_cols.index('seasonal_strength') if 'seasonal_strength' in feat_cols else 2, key='z3d')
            
            fig_3d = px.scatter_3d(
                filtered_results, 
                x=x_axis, y=y_axis, z=z_axis,
                color='Forecastability_Label',
                symbol='Cluster',
                hover_data=['SKU', 'Inferred_Pattern', 'Score_Bucket'],
                color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'},
                opacity=0.8,
                title=f"3D Analysis: {x_axis} vs {y_axis} vs {z_axis}"
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=50), scene=dict(aspectmode='cube'))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Not enough numeric features found for 3D analysis.")

        st.divider()
        st.subheader("Individual Drill-down")
        sel_sku = st.selectbox("Select SKU:", filtered_skus)
        
        if sel_sku:
            c1, c2 = st.columns([3, 1])
            with c1:
                sku_data = sales_df[sales_df['SKU'] == sel_sku]
                fig_s = px.line(sku_data, x='Date', y='Sales', markers=True, title=f"SKU: {sel_sku}")
                st.plotly_chart(fig_s, use_container_width=True)
            with c2:
                row = filtered_results[filtered_results['SKU'] == sel_sku].iloc[0]
                st.write(f"**Segment:** {row['Forecastability_Label']}")
                st.write(f"**Cluster:** {row['Cluster']}")
                if has_pattern:
                    st.write(f"**Pattern:** {row['Inferred_Pattern']}")
                    st.write(f"**Confidence:** {row.get('Pattern_Confidence', 0):.2f}")
                st.metric("Intermittency", f"{row.get('p_zero', 0):.2f}")
                st.metric("Volatility (CV)", f"{row.get('cv', 0):.2f}")
                st.metric("Trend Strength", f"{row.get('trend_strength', 0):.2f}")
                st.metric("Seasonality", f"{row.get('seasonal_strength', 0):.2f}")
                st.metric("Forecast Score", f"{row.get('Forecastability_Score', 0):.3f}")
                st.write(f"**Score Bucket:** {row.get('Score_Bucket', 'N/A')}")
                
                st.divider()
                st.caption("🔍 Label Evidence")
                st.write("- **Pattern:** " + ("✅ Stable" if row.get('Inferred_Pattern') in ['Smooth', 'Seasonal', 'Trending'] else "⚠️ Erratic"))
                st.write("- **Cluster:** " + ("✅ Strong" if row.get('Cluster') in [0, 1] else "⚖️ Neutral")) # Simplified placeholder for cluster power
                st.write(f"- **Bucket:** {row.get('Score_Bucket')}")

else:
    st.info("👈 Please load data using the sidebar.")
