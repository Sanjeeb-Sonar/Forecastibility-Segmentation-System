import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class VisualizationEngine:
    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams.update({'figure.max_open_warning': 0})
        
        self.segment_colors = {
            'Easy': '#2ecc71',      # Green
            'Moderate': '#f1c40f',  # Yellow/Orange
            'Hard': '#e74c3c'       # Red
        }
        
        self.cluster_cmap = 'viridis'

    def run_all_visualizations(self, full_df, segmentation_engine=None):
        """
        Runs all 8 visualizations utilizing the full dataframe (with features + labels)
        and optionally the segmentation engine object for internal metrics.
        """
        print("Generating visualizations...")
        
        # 1. Multi-Metric K Selection (if engine provided)
        if segmentation_engine and hasattr(segmentation_engine, 'results_summary'):
            self.plot_k_selection(segmentation_engine.results_summary)
            
        # 2. PCA Scatter
        self.plot_pca_scatter(full_df)
        
        # 3. Radar Chart (Cluster Profiles)
        self.plot_radar_profiles(full_df)
        
        # 4. Feature Heatmap
        self.plot_feature_heatmap(full_df)
        
        # 5. Box Plots (Key Features)
        self.plot_boxplots(full_df)
        
        # 7. Feature Importance (ANOVA) - computed in forecastability.py but let's visualize if we can
        # For now, we'll skip the bar chart unless passed, or compute simple F-stats here?
        # Let's just do a correlation/separation plot instead or rely on the console output from forecastability.py
        
        # 8. Summary Table (saved as CSV, maybe plot as image?)
        self.save_summary_table(full_df)
        
        print("Visualizations saved.")

    def plot_k_selection(self, results_summary):
        # We need to reshape the results summary
        # It's a dict: {'KMeans': [{'k':2, 'sil':...}, ...], ...}
        
        plt.figure(figsize=(15, 5))
        
        metrics = ['sil', 'ch', 'db']
        titles = ['Silhouette (Higher is better)', 'Calinski-Harabasz (Higher is better)', 'Davies-Bouldin (Lower is better)']
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            for algo_name, res_list in results_summary.items():
                ks = [r['k'] for r in res_list]
                vals = [r[metric] for r in res_list]
                plt.plot(ks, vals, marker='o', label=algo_name)
            
            plt.title(titles[i])
            plt.xlabel('K')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/viz_01_k_selection.png", dpi=300)
        plt.close()

    def plot_pca_scatter(self, df):
        plt.figure(figsize=(10, 8))
        
        # Color by Cluster, Shape by Forecastability Label
        # Map labels to markers
        markers = {'Easy': '^', 'Moderate': 'o', 'Hard': 's'} # Triangle, Circle, Square
        
        # Create a combined hue/style plot
        # But seaborn scatterplot is easier
        
        # Sort so Hard is on top?
        df_sorted = df.sort_values('Forecastability_Label', key=lambda x: x.map({'Easy':0, 'Moderate':1, 'Hard':2}))
        
        sns.scatterplot(
            data=df_sorted, 
            x='PCA1', y='PCA2', 
            hue='Forecastability_Label', 
            style='Forecastability_Label',
            markers=markers,
            palette=self.segment_colors,
            s=100, alpha=0.8, edgecolor='k'
        )
        
        plt.title('PCA Projection of SKUs\nColored by Forecastability', fontsize=14)
        plt.savefig(f"{self.output_dir}/viz_02_pca_scatter.png", dpi=300)
        plt.close()

    def plot_radar_profiles(self, df):
        # Aggregate by cluster
        # Normalize features to 0-1 for radar
        feature_cols = [c for c in df.columns if c not in ['Date', 'SKU', 'Cluster', 'Model_Used', 'Algorithm_Stability', 'Forecastability_Score', 'Forecastability_Label', 'PCA1', 'PCA2', 'Pattern_Truth']]
        # Select typical representative features to avoid clutter
        selected_feats = ['cv', 'p_zero', 'trend_strength', 'seasonal_strength', 'acf_lag1', 'approx_entropy', 'level_stability', 'skewness']
        
        # Filter only existing columns
        selected_feats = [f for f in selected_feats if f in df.columns]
        
        if not selected_feats: return
        
        scaler = MinMaxScaler()
        df_norm = df.copy()
        df_norm[selected_feats] = scaler.fit_transform(df[selected_feats])
        
        # Group by label? Or Cluster? Let's do Label for clearer story
        means = df_norm.groupby('Forecastability_Label')[selected_feats].mean()
        
        # Reorder index to Easy -> Moderate -> Hard
        order = [l for l in ['Easy', 'Moderate', 'Hard'] if l in means.index]
        means = means.loc[order]
        
        # Plot
        # Number of variables
        N = len(selected_feats)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # Draw one axe per variable + labels
        plt.xticks(angles[:-1], selected_feats, color='grey', size=10)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)
        
        for idx, row in means.iterrows():
            values = row.values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=idx, color=self.segment_colors.get(idx, 'blue'))
            ax.fill(angles, values, alpha=0.1, color=self.segment_colors.get(idx, 'blue'))
            
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Forecastability Profiles (Normalized Means)', size=15, y=1.1)
        plt.savefig(f"{self.output_dir}/viz_03_radar_profiles.png", dpi=300)
        plt.close()
        
    def plot_feature_heatmap(self, df):
        # Heatmap of Cluster Centers (Z-scores) (Features x Clusters)
        feature_cols = [c for c in df.columns if c not in ['Date', 'SKU', 'Cluster', 'Model_Used', 'Algorithm_Stability', 'Forecastability_Score', 'Forecastability_Label', 'PCA1', 'PCA2', 'Pattern_Truth']]
        
        # Select top 20 features by variance or ANOVA?
        # Let's just take the main representatives from each group to keep it readable
        repr_feats = [
            'mean', 'cv', 'p_zero', 'adi', 
            'trend_strength', 'trend_linearity', 
            'seasonal_strength', 'peak_to_trough_ratio',
            'rolling_cv_mean', 'volatility_change_ratio',
            'acf_lag1', 'acf_lag12',
            'approx_entropy', 'spectral_entropy',
            'level_stability', 'kurtosis'
        ]
        repr_feats = [f for f in repr_feats if f in df.columns]
        
        # Calculate Z-scores of means per cluster relative to global mean/std
        means = df.groupby('Cluster')[repr_feats].mean()
        # Scale for visualization (StandardScaler on the means? No, on original data)
        # Better: (ClusterMean - GlobalMean) / GlobalStd
        global_mean = df[repr_feats].mean()
        global_std = df[repr_feats].std()
        
        z_scores = (means - global_mean) / (global_std + 1e-6)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(z_scores.T, cmap='RdBu_r', center=0, annot=True, fmt='.1f')
        plt.title('Cluster Feature Profiles (Z-Scores)\nBlue = Below Avg, Red = Above Avg')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/viz_04_feature_heatmap.png", dpi=300)
        plt.close()
        
    def plot_boxplots(self, df):
        # Key features distribution across Easy/Moderate/Hard
        key_feats = ['cv', 'approx_entropy', 'trend_strength', 'seasonal_strength', 'p_zero', 'acf_lag1']
        key_feats = [f for f in key_feats if f in df.columns]
        
        plt.figure(figsize=(15, 10))
        count = 1
        for f in key_feats:
            plt.subplot(2, 3, count)
            # Sort order
            order = ['Easy', 'Moderate', 'Hard']
            sns.boxplot(x='Forecastability_Label', y=f, data=df, order=order, palette=self.segment_colors)
            plt.title(f)
            count += 1
            
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/viz_05_boxplots.png", dpi=300)
        plt.close()

    def plot_representative_timeseries(self, df, sales_df_path="synthetic_sales_data.csv"):
        # We need the original time series data. 
        # Assuming it's in the same directory or passed. 
        # For this script we will try to load it.
        try:
            ts_df = pd.read_csv(sales_df_path)
            ts_df['Date'] = pd.to_datetime(ts_df['Date'])
        except:
            print("Could not load sales data for time series plot.")
            return

        # Select 1 representative SKU from each Category (closest to centroid?)
        # Or just random
        plt.figure(figsize=(15, 6))
        
        categories = ['Easy', 'Moderate', 'Hard']
        
        for i, cat in enumerate(categories):
            skus_in_cat = df[df['Forecastability_Label'] == cat]['SKU'].values
            if len(skus_in_cat) == 0: continue
            
            # Pick one with high series length
            sample_sku = skus_in_cat[0]
            # Better: Pick one that is most typical? 
            # Random is fine for now
            
            ts_data = ts_df[ts_df['SKU'] == sample_sku].sort_values('Date')
            
            plt.subplot(1, 3, i+1)
            plt.plot(ts_data['Date'], ts_data['Sales'], color=self.segment_colors.get(cat, 'blue'))
            plt.title(f"{cat} to Forecast\n({sample_sku})")
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/viz_06_representative_series.png", dpi=300)
        plt.close()

    def save_summary_table(self, full_df):
        # Cluster ID | Forecastability Label | Size | Top Defining Features | Recommended Models
        
        summary = []
        feature_cols = [c for c in full_df.columns if c not in ['Date', 'SKU', 'Cluster', 'Model_Used', 'Algorithm_Stability', 'Forecastability_Score', 'Forecastability_Label', 'PCA1', 'PCA2',  'Pattern_Truth']]
        
        global_mean = full_df[feature_cols].mean()
        global_std = full_df[feature_cols].std()
        
        for cluster_id in sorted(full_df['Cluster'].unique()):
            cluster_data = full_df[full_df['Cluster'] == cluster_id]
            size = len(cluster_data)
            label = cluster_data['Forecastability_Label'].mode()[0]
            
            # Top defining features (highest absolute Z-scores)
            means = cluster_data[feature_cols].mean()
            z_scores = (means - global_mean) / (global_std + 1e-6)
            top_feats = z_scores.abs().sort_values(ascending=False).head(3)
            
            feat_desc = []
            for f, z in top_feats.items():
                direction = "High" if z_scores[f] > 0 else "Low"
                feat_desc.append(f"{direction} {f}")
            
            top_features_str = ", ".join(feat_desc)
            
            # Rec model logic (simple rules)
            # If High p_zero -> Croston/SBA
            # If High Seasonality -> SARIMA / HW / Prophet
            # If High Entropy -> ML (Random Forest/XGB) or Naive
            
            means_dict = means.to_dict()
            rec_model = "Unknown"
            
            if means_dict.get('p_zero', 0) > 0.3:
                rec_model = "Croston / SBA (Intermittent)"
            elif means_dict.get('seasonal_strength', 0) > 0.6:
                rec_model = "SARIMA / Prophet (Seasonal)"
            elif means_dict.get('trend_strength', 0) > 0.6:
                rec_model = "Holt / ARIMA (Trend)"
            elif means_dict.get('approx_entropy', 0) > 0.8:
                rec_model = "Naive / Simple Mean (High Entropy)"
            else:
                rec_model = "SES / ARIMA (General)"
            
            summary.append({
                'Cluster': cluster_id,
                'Forecastability': label,
                'Size': size,
                'Defining Characteristics': top_features_str,
                'Recommended Models': rec_model
            })
            
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{self.output_dir}/viz_08_summary_table.csv", index=False)
        print("Summary table saved.")

if __name__ == "__main__":
    try:
        df = pd.read_csv("final_output.csv")
        viz = VisualizationEngine()
        viz.run_all_visualizations(df) # Note: can't plot K selection without segmentation object, but others will work
    except Exception as e:
        print(f"Test failed: {e}")
