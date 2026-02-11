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
        Pure clustering engine.
        Input: DataFrame with SKU + numeric features (may include Inferred_Pattern, Pattern_Confidence)
        Output: DataFrame with 'Cluster', 'PCA1', 'PCA2', 'Model_Used', 'Algorithm_Stability' added.
        """
        skus = features_df[sku_col].values
        
        # Drop non-numeric columns for clustering
        non_feature_cols = [sku_col]
        for col in ['Inferred_Pattern', 'Pattern_Confidence']:
            if col in features_df.columns:
                non_feature_cols.append(col)
            
        X_raw = features_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=[np.number]).values
        
        # 1. Preprocessing: Scale + Outlier Capping
        print("Preprocessing features...")
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Cap outliers at 3 sigma
        X_scaled = np.clip(X_scaled, -3, 3)
        
        # PCA for noise reduction (keep 90% variance)
        self.pca = PCA(n_components=0.90, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"PCA reduced dimensions from {X_raw.shape[1]} to {X_pca.shape[1]} (90% variance)")
        
        # 2. Multi-Algorithm & Multi-Metric Sweep
        print(f"Sweeping K={self.min_k}..{self.max_k} across 3 algorithms...")
        
        best_overall_score = -1
        final_k = -1
        
        algos = {
            'KMeans': KMeans,
            'Agglomerative': AgglomerativeClustering,
            'GMM': GaussianMixture
        }
        
        for k in range(self.min_k, self.max_k + 1):
            for name, AlgoClass in algos.items():
                if name == 'GMM':
                    model = AlgoClass(n_components=k, random_state=42)
                    labels = model.fit_predict(X_pca)
                elif name == 'Agglomerative':
                    model = AlgoClass(n_clusters=k)
                    labels = model.fit_predict(X_pca)
                else:
                    model = AlgoClass(n_clusters=k, random_state=42, n_init=10)
                    labels = model.fit_predict(X_pca)
                
                sil = silhouette_score(X_pca, labels)
                ch = calinski_harabasz_score(X_pca, labels)
                db = davies_bouldin_score(X_pca, labels)
                
                self.results_summary.setdefault(name, []).append({
                    'k': k, 'sil': sil, 'ch': ch, 'db': db
                })

        # 3. Select Best K per Algorithm
        algo_best_k = {}
        for name, rows in self.results_summary.items():
            rows.sort(key=lambda x: x['sil'], reverse=True)
            algo_best_k[name] = rows[0]
            
        print("Best K per algorithm:", {k: v['k'] for k, v in algo_best_k.items()})
        
        # 4. Bootstrap Stability Check
        print("Running bootstrap stability analysis...")
        algo_stability = {}
        
        for name, best_res in algo_best_k.items():
            k = best_res['k']
            stability_scores = []
            
            for i in range(self.n_bootstrap):
                indices = np.random.choice(len(X_pca), int(len(X_pca)*0.8), replace=False)
                X_sample = X_pca[indices]
                
                if name == 'GMM':
                    model = GaussianMixture(n_components=k, random_state=i)
                elif name == 'Agglomerative':
                    model = AgglomerativeClustering(n_clusters=k)
                else:
                    model = KMeans(n_clusters=k, random_state=i, n_init=10)
                    
                try:
                    labels = model.fit_predict(X_sample)
                    if len(np.unique(labels)) > 1:
                        stability_scores.append(silhouette_score(X_sample, labels))
                    else:
                        stability_scores.append(0)
                except:
                    stability_scores.append(0)
            
            algo_stability[name] = np.mean(stability_scores)
            
        # 5. Final Selection: Silhouette (60%) + Stability (40%)
        final_scores = {}
        for name in algo_best_k:
            final_scores[name] = (algo_best_k[name]['sil'] * 0.6) + (algo_stability[name] * 0.4)
            
        best_algo_name = max(final_scores, key=final_scores.get)
        best_k = algo_best_k[best_algo_name]['k']
        
        print(f"Winner: {best_algo_name} with K={best_k} (Score: {final_scores[best_algo_name]:.3f})")
        
        # Fit final model
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
        
        # Build result
        result_df = features_df.copy()
        result_df['Cluster'] = labels
        result_df['Model_Used'] = best_algo_name
        result_df['Algorithm_Stability'] = algo_stability[best_algo_name]
        
        # Add PCA coordinates for visualization
        result_df['PCA1'] = X_pca[:, 0]
        result_df['PCA2'] = X_pca[:, 1]
        
        return result_df

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

