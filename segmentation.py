import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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
        self.best_algo_name = ""
        self.results_summary = {}

    def run_segmentation(self, features_df, sku_col='SKU'):
        """
        Pure clustering engine with size constraints (<10% per cluster).
        Input: DataFrame with SKU + numeric features
        Output: DataFrame with 'Cluster', 'PCA1', 'PCA2', etc.
        """
        skus = features_df[sku_col].values
        
        # Drop non-numeric columns for clustering
        non_feature_cols = [sku_col]
        for col in ['Inferred_Pattern', 'Pattern_Confidence']:
            if col in features_df.columns:
                non_feature_cols.append(col)
            
        X_raw = features_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=[np.number]).values
        
        # 1. Preprocessing
        print("Preprocessing features...")
        X_scaled = self.scaler.fit_transform(X_raw)
        X_scaled = np.clip(X_scaled, -3, 3)
        
        # PCA for noise reduction (keep 90% variance)
        self.pca = PCA(n_components=0.90, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"PCA reduced dimensions from {X_raw.shape[1]} to {X_pca.shape[1]} (90% variance)")
        
        # 2. Recursive Divisive Clustering
        max_allowed_size = int(np.ceil(len(features_df) * 0.10))
        print(f"Targeting max cluster size: {max_allowed_size} SKUs (10% of {len(features_df)})")
        
        # Initial Clustering
        labels = self._find_best_clusters(X_pca)
        
        # Iteratively split large clusters
        final_labels = labels.copy()
        current_max_id = np.max(final_labels)
        
        for iteration in range(5): # Safety limit for recursions
            unique_clusters = np.unique(final_labels)
            clusters_to_split = []
            
            for c in unique_clusters:
                size = np.sum(final_labels == c)
                if size > max_allowed_size:
                    clusters_to_split.append(c)
            
            if not clusters_to_split:
                break
                
            print(f"Iteration {iteration+1}: Splitting {len(clusters_to_split)} large clusters...")
            
            for c in clusters_to_split:
                indices = np.where(final_labels == c)[0]
                X_sub = X_pca[indices]
                
                # Split the large cluster into 2 smaller ones
                sub_model = KMeans(n_clusters=2, random_state=42, n_init=10)
                sub_labels = sub_model.fit_predict(X_sub)
                
                # Assign new IDs
                final_labels[indices[sub_labels == 1]] = current_max_id + 1
                current_max_id += 1
                
        # 3. Build result
        result_df = features_df.copy()
        result_df['Cluster'] = final_labels
        result_df['Model_Used'] = self.best_algo_name
        result_df['PCA1'] = X_pca[:, 0]
        result_df['PCA2'] = X_pca[:, 1]
        
        print(f"Final Clustering: {len(np.unique(final_labels))} clusters created.")
        return result_df

    def _find_best_clusters(self, X_pca):
        """Helper to run the multi-algorithm sweep and pick best initial K."""
        self.results_summary = {}
        algos = {'KMeans': KMeans, 'Agglomerative': AgglomerativeClustering, 'GMM': GaussianMixture}
        
        for k in range(self.min_k, self.max_k + 1):
            for name, AlgoClass in algos.items():
                if name == 'GMM': model = AlgoClass(n_components=k, random_state=42)
                elif name == 'Agglomerative': model = AlgoClass(n_clusters=k)
                else: model = AlgoClass(n_clusters=k, random_state=42, n_init=10)
                
                labels = model.fit_predict(X_pca)
                sil = silhouette_score(X_pca, labels)
                ch = calinski_harabasz_score(X_pca, labels)
                db = davies_bouldin_score(X_pca, labels)
                self.results_summary.setdefault(name, []).append({
                    'k': k, 'sil': sil, 'ch': ch, 'db': db
                })

        algo_best_k = {}
        for name, rows in self.results_summary.items():
            rows.sort(key=lambda x: x['sil'], reverse=True)
            algo_best_k[name] = rows[0]
            
        # Select winner based on simple silhouette for initial phase
        best_algo_name = max(algo_best_k, key=lambda x: algo_best_k[x]['sil'])
        self.best_algo_name = best_algo_name
        k = algo_best_k[best_algo_name]['k']
        
        if best_algo_name == 'GMM': mod = GaussianMixture(n_components=k, random_state=42)
        elif best_algo_name == 'Agglomerative': mod = AgglomerativeClustering(n_clusters=k)
        else: mod = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        return mod.fit_predict(X_pca)

if __name__ == "__main__":
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
