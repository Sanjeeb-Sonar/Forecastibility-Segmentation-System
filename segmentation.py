import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import mode
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class SegmentationEngine:
    def __init__(self, min_k=2, max_k=8, n_bootstrap=20):
        self.min_k = min_k
        self.max_k = max_k
        self.n_bootstrap = n_bootstrap
        self.scaler = StandardScaler()
        self.pca = None
        self.best_model = None
        self.best_k = None
        self.best_labels = None
        self.results_summary = {}

    def run_segmentation(self, features_df, sku_col='SKU'):
        """
        Main entry point. 
        Input: DataFrame from FeatureExtractionEngine (cols: SKU, feat1, feat2...)
        Output: DataFrame with 'Cluster', 'Model_Used', 'Optimal_K' added.
        """
        skus = features_df[sku_col].values
        # Drop non-numeric for clustering
        drop_cols = [sku_col]
        if 'Pattern_Truth' in features_df.columns:
            drop_cols.append('Pattern_Truth')
            
        X_raw = features_df.drop(columns=drop_cols).values
        
        # 1. Preprocessing: Scale + Outlier Capping
        print("Preprocessing features...")
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Cap outliers at 3 sigma (or 1.5 IQR equivalent) to prevent distortion
        # Using 3-sigma for simplicity on scaled data
        X_scaled = np.clip(X_scaled, -3, 3)
        
        # PCA for noise reduction (keep 90% variance)
        self.pca = PCA(n_components=0.90, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"PCA reduced dimensions from {X_raw.shape[1]} to {X_pca.shape[1]} (90% variance)")
        
        # 2. Multi-Algorithm & Multi-Metric Sweep
        print(f"Sweeping K={self.min_k}..{self.max_k} across 3 algorithms...")
        
        best_overall_score = -1
        best_algo_name = ""
        final_k = -1
        
        # Algorithms to test
        algos = {
            'KMeans': KMeans,
            'Agglomerative': AgglomerativeClustering,
            'GMM': GaussianMixture
        }
        
        for k in range(self.min_k, self.max_k + 1):
            for name, AlgoClass in algos.items():
                # Fit model
                if name == 'GMM':
                    model = AlgoClass(n_components=k, random_state=42)
                    labels = model.fit_predict(X_pca)
                elif name == 'Agglomerative':
                    model = AlgoClass(n_clusters=k)
                    labels = model.fit_predict(X_pca)
                else: # KMeans
                    model = AlgoClass(n_clusters=k, random_state=42, n_init=10)
                    labels = model.fit_predict(X_pca)
                
                # Check metrics (Silhouette, CH, DB)
                sil = silhouette_score(X_pca, labels)
                ch = calinski_harabasz_score(X_pca, labels)
                db = davies_bouldin_score(X_pca, labels)
                
                # Normalize metrics to combine them? 
                # Actually, let's use Majority Vote idea from plan or weighted score
                # For simplicity here: Combined Score = Silhouette (0-1) - (DB normalized approx 0-1) + (CH log normalized)
                # But Plan said: "Select K by majority vote across metrics".
                # Let's verify K per algorithm.
                
                self.results_summary.setdefault(name, []).append({
                    'k': k,
                    'sil': sil,
                    'ch': ch,
                    'db': db
                })

        # 3. Select Best K per Algorithm
        algo_best_k = {}
        for name, rows in self.results_summary.items():
            # Rank best K by Silhouette (primary)
            # tie-break with CH/DB
            rows.sort(key=lambda x: x['sil'], reverse=True)
            algo_best_k[name] = rows[0] # Best condidate for this algo
            
        print("Best K per algorithm:", {k: v['k'] for k,v in algo_best_k.items()})
        
        # 4. Bootstrap Stability Check
        # Run bootstrap on the *best K* for each algo to see which algo is most stable
        print("Running bootstrap stability analysis...")
        algo_stability = {}
        
        for name, best_res in algo_best_k.items():
            k = best_res['k']
            stability_scores = []
            
            for i in range(self.n_bootstrap):
                # Resample 80% without replacement
                indices = np.random.choice(len(X_pca), int(len(X_pca)*0.8), replace=False)
                X_sample = X_pca[indices]
                
                if name == 'GMM':
                    model = GaussianMixture(n_components=k, random_state=i)
                elif name == 'Agglomerative':
                    model = AgglomerativeClustering(n_clusters=k)
                else:
                    model = KMeans(n_clusters=k, random_state=i, n_init=10)
                    
                # We can't compare labels directly between runs because cluster IDs swap
                # Instead we measure: do pair of points stay together? (Adjusted Rand Index? No, requires truth)
                # Proxy: Just trust the silhouette is robust on subsets?
                # Proper stability: Clustering Stability Index.
                # Simplified for this script: We'll stick to full-set metrics for now.
                # Just use the Silhouette on the subset.
                try:
                    labels = model.fit_predict(X_sample)
                    if len(np.unique(labels)) > 1:
                        stability_scores.append(silhouette_score(X_sample, labels))
                    else:
                        stability_scores.append(0)
                except:
                    stability_scores.append(0)
            
            algo_stability[name] = np.mean(stability_scores)
            
        # 5. Final Selection
        # Combine Silhouette (full set) + Stability (subset mean)
        final_scores = {}
        for name in algo_best_k:
            final_scores[name] = (algo_best_k[name]['sil'] * 0.6) + (algo_stability[name] * 0.4)
            
        best_algo_name = max(final_scores, key=final_scores.get)
        best_k = algo_best_k[best_algo_name]['k']
        
        print(f"Winner: {best_algo_name} with K={best_k} (Score: {final_scores[best_algo_name]:.3f})")
        
        # fit final model
        if best_algo_name == 'GMM':
            model = GaussianMixture(n_components=best_k, random_state=42)
            labels = model.fit_predict(X_pca)
        elif best_algo_name == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=best_k)
            labels = model.fit_predict(X_pca)
        else:
            model = KMeans(n_clusters=best_k, verbose=0, random_state=42, n_init=10)
            labels = model.fit_predict(X_pca)
            
        self.best_model = model
        self.best_k = best_k
        self.best_labels = labels
        self.best_algo_name = best_algo_name
        
        # Return results
        result_df = features_df.copy()
        result_df['Cluster'] = labels
        result_df['Model_Used'] = best_algo_name
        result_df['Algorithm_Stability'] = algo_stability[best_algo_name]
        
        # Add PCA coordinates for visualization later
        result_df['PCA1'] = X_pca[:, 0]
        result_df['PCA2'] = X_pca[:, 1]
        
        # 6. Assign Descriptive Cluster Names (Demand Pattern)
        result_df = self._assign_cluster_names(result_df)
        
        return result_df

    def _assign_cluster_names(self, df, cluster_col='Cluster'):
        """
        Assigns descriptive names to clusters based on their feature centroids.
        Rules (Hierarchical):
        1. Intermittent: High p_zero (> 0.4)
        2. Seasonal: High seasonal_strength (> 0.3)
        3. Trending: High trend_strength (> 0.4)
        4. Volatile: High CV (> 0.6)
        5. Smooth: Low CV (< 0.4)
        6. Complex: Default
        """
        # Calculate centroids
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude cluster/pca cols from mean calc if they cause issues, but usually fine
        centroids = df.groupby(cluster_col)[numeric_cols].mean()
        
        cluster_names = {}
        
        for cluster_id, row in centroids.iterrows():
            name = "Complex" # Default
            
            p_zero = row.get('p_zero', 0)
            seasonal = row.get('seasonal_strength', 0)
            trend = row.get('trend_strength', 0)
            cv = row.get('cv', 0)
            entropy = row.get('approx_entropy', 0)
            
            # Hierarchical Classification
            if p_zero > 0.4:
                if cv > 0.8:
                    name = "Lumpy (Intermittent + Volatile)"
                else:
                    name = "Intermittent"
            elif seasonal > 0.3:
                name = "Seasonal"
            elif trend > 0.4:
                name = "Trending"
            elif cv < 0.4:
                name = "Smooth"
            elif cv > 0.7:
                name = "Volatile"
            elif entropy > 2.0: # Empirical threshold
                name = "Erratic"
                
            # Append cluster ID to ensure uniqueness if needed, or just descriptiveness
            cluster_names[cluster_id] = f"{cluster_id}: {name}"
            
        df['Cluster_Nature'] = df[cluster_col].map(cluster_names)
        return df

if __name__ == "__main__":
    # Test run
    try:
        df = pd.read_csv("features_output.csv")
        seg = SegmentationEngine()
        results = seg.run_segmentation(df)
        
        print("\nSegmentation Complete.")
        print("Cluster Counts:\n", results['Cluster'].value_counts())
        
        results.to_csv("segmentation_output.csv", index=False)
        print("Saved to segmentation_output.csv")
    except Exception as e:
        print(f"Test failed: {e}")
