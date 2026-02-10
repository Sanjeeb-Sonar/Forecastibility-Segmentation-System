import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

# Import Analysis Pipeline
from feature_extraction import FeatureExtractionEngine
from segmentation import SegmentationEngine
from forecastability import ForecastabilityClassifier

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
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # 1. Feature Extraction
        status_text.text("Extracting Features (this may take a moment)...")
        feature_engine = FeatureExtractionEngine()
        features_df = feature_engine.extract_features(df)
        progress_bar.progress(40)
        
        # 2. Segmentation
        status_text.text("Running Segmentation Engine...")
        seg_engine = SegmentationEngine(min_k=2, max_k=8, n_bootstrap=20)
        segmented_df = seg_engine.run_segmentation(features_df)
        progress_bar.progress(70)
        
        # 3. Classification
        status_text.text("Classifying Forecastability...")
        classifier = ForecastabilityClassifier()
        final_df, _ = classifier.classify(segmented_df)
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
        uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type=['csv'])
        if uploaded_file is not None:
            # Load Raw Data
            try:
                raw_df = pd.read_csv(uploaded_file)
                # Validation
                required_cols = ['Date', 'SKU', 'Sales']
                if not all(col in raw_df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {required_cols}")
                else:
                    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
                    st.success(f"Loaded {len(raw_df)} rows.")
                    
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
        selected_cats = st.multiselect("Category", cats, default=cats)
        
        # 2. Cluster Filter (Use Nature if available)
        if 'Cluster_Nature' in results_df.columns:
            cluster_col = 'Cluster_Nature'
        else:
            cluster_col = 'Cluster'
            
        all_clusters = sorted(results_df[cluster_col].unique())
        selected_clusters = st.multiselect("Cluster (Nature)", all_clusters, default=all_clusters)
        
        # 3. Pattern Filter (Always available now via inference)
        # User said "Demand Pattern should be calculated based on all 35 features.. that is what we are saying the clusters"
        # So we might want to hide the ADI/CV inferred pattern to avoid confusion? 
        # Or keep it as "Rules-based Pattern" vs "Cluster Pattern"
        # Let's keep it but rename it "Rule-based Pattern" for clarity
        if 'Pattern_Truth' in results_df.columns:
            all_patterns = sorted(results_df['Pattern_Truth'].astype(str).unique())
            selected_patterns = st.multiselect("Demand Pattern (Inferred)", all_patterns, default=all_patterns)
        else:
            selected_patterns = []
            
        # Apply Filters
        mask = (results_df['Forecastability_Label'].isin(selected_cats)) & \
               (results_df[cluster_col].isin(selected_clusters))
        
        if selected_patterns and 'Pattern_Truth' in results_df.columns:
            mask = mask & (results_df['Pattern_Truth'].isin(selected_patterns))
            
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
    
    # Check if we have Pattern info for display
    has_pattern = 'Pattern_Truth' in filtered_results.columns

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
            # PCA Scatter
            hover_cols = ['SKU']
            if has_pattern: hover_cols.append('Pattern_Truth')
            
            # Determine symbol col
            sym_col = 'Cluster_Nature' if 'Cluster_Nature' in filtered_results.columns else 'Cluster'
            
            fig_pca = px.scatter(
                filtered_results, x='PCA1', y='PCA2', 
                color='Forecastability_Label', symbol=sym_col,
                hover_data=hover_cols,
                color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'},
                title="Consumer Segmentation Map"
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
        with col_stats:
            st.subheader("Cluster Profile")
            st.caption("Average features per cluster")
            cols_to_show = ['p_zero', 'cv', 'seasonal_strength', 'approx_entropy']
            # Filter cols that exist
            cols_to_show = [c for c in cols_to_show if c in filtered_results.columns]
            
            group_col = 'Cluster_Nature' if 'Cluster_Nature' in filtered_results.columns else 'Cluster'
            means = filtered_results.groupby(group_col)[cols_to_show].mean()
            st.dataframe(means.style.format("{:.2f}").background_gradient(cmap='Blues'), height=400)

    # --- TAB 2: DRIVERS ---
    with tab_drivers:
        st.markdown("### 🔍 Why describes the segments?")
        col_rad, col_box = st.columns(2)
        
        with col_rad:
            feat_candidates = ['cv', 'p_zero', 'trend_strength', 'seasonal_strength', 'approx_entropy', 'skewness']
            present_feats = [f for f in feat_candidates if f in filtered_results.columns]
            
            if present_feats:
                df_rad = filtered_results.groupby('Forecastability_Label')[present_feats].mean().reset_index()
                df_melt = df_rad.melt(id_vars='Forecastability_Label', var_name='Feature', value_name='Value')
                
                fig_rad = px.line_polar(
                    df_melt, r='Value', theta='Feature', color='Forecastability_Label', line_close=True,
                    color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'}
                )
                st.plotly_chart(fig_rad, use_container_width=True)
        
        with col_box:
            # Feature Boxplots
            st.caption("Distribution of features")
            all_feats = [c for c in filtered_results.columns if c not in ['Date','SKU','Cluster','Forecastability_Label','PCA1','PCA2','Pattern_Truth']]
            sel_feat = st.selectbox("Compare Feature:", all_feats, index=0 if len(all_feats)>0 else None)
            
            if sel_feat:
                fig_box = px.box(
                    filtered_results, x='Forecastability_Label', y=sel_feat, color='Forecastability_Label',
                    color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'}
                )
                st.plotly_chart(fig_box, use_container_width=True)

    # --- TAB 3: DEEP DIVE ---
    with tab_deepdive:
        st.markdown("### 📉 Historical Analysis")
        st.caption("Filtered View of Historical Sales")
        
        # Active Filters Display
        st.info(f"Viewing {len(filtered_skus)} SKUs. Filters: {', '.join(selected_cats)} | Clusters: {selected_clusters}")
        
        # Max Lines Check
        MAX_LINES = 100
        plot_data = filtered_sales
        if len(filtered_skus) > MAX_LINES:
            st.warning(f"⚠️ High volume! showing random sample of {MAX_LINES} SKUs.")
            sample_skus = np.random.choice(filtered_skus, MAX_LINES, replace=False)
            plot_data = filtered_sales[filtered_sales['SKU'].isin(sample_skus)]
        
        # Merge Segments onto Sales for Coloring
        plot_data = plot_data.merge(filtered_results[['SKU', 'Forecastability_Label']], on='SKU', how='left')
        
        # Multi-line Chart
        fig_hist = px.line(
            plot_data, x='Date', y='Sales', 
            color='Forecastability_Label', line_group='SKU', # One line per SKU
            color_discrete_map={'Easy': '#00CC96', 'Moderate': '#FFA15A', 'Hard': '#EF553B'},
            title="Sales History (Filtered)"
        )
        fig_hist.update_traces(opacity=0.5, line=dict(width=1))
        st.plotly_chart(fig_hist, use_container_width=True)
        
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
                if has_pattern: st.write(f"**Pattern:** {row['Pattern_Truth']}")
                st.metric("Intermittency", f"{row.get('p_zero', 0):.2f}")
                st.metric("Volatility (CV)", f"{row.get('cv', 0):.2f}")

else:
    st.info("👈 Please load data using the sidebar.")
