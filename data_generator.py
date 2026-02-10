import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_synthetic_data(n_skus=100, min_months=36, max_months=60):
    """
    Generates synthetic monthly sales data for a diverse portfolio of SKUs.
    
    Patterns included:
    1. Smooth/Stable (Low Volatility)
    2. Seasonal (Annual cycle)
    3. Intermittent (Sporadic zeros)
    4. Lumpy (High volatility + zeros)
    5. Trending (Up/Down)
    6. New Product (Short history, ramp-up)
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    # Define SKUs per pattern
    patterns = {
        'Smooth': int(n_skus * 0.20),
        'Seasonal': int(n_skus * 0.20),
        'Intermittent': int(n_skus * 0.25),
        'Lumpy': int(n_skus * 0.15),
        'Trending': int(n_skus * 0.10),
        'New_Product': int(n_skus * 0.10)
    }
    
    # Adjust to match exact n_skus if rounding errors
    current_total = sum(patterns.values())
    if current_total < n_skus:
        patterns['Smooth'] += (n_skus - current_total)
        
    sku_id_counter = 1
    start_date_base = datetime(2020, 1, 1)
    
    for pattern_name, count in patterns.items():
        for _ in range(count):
            sku_id = f"SKU_{sku_id_counter:03d}_{pattern_name}"
            sku_id_counter += 1
            
            # Determine series length
            if pattern_name == 'New_Product':
                n_months = np.random.randint(6, 18)  # Short history
                # Start date is recent
                start_date = start_date_base + timedelta(days=30 * (60 - n_months))
            else:
                n_months = np.random.randint(min_months, max_months)
                start_date = start_date_base
            
            dates = [start_date + timedelta(days=30*i) for i in range(n_months)]
            # Normalize to first of month
            dates = [d.replace(day=1) for d in dates]
            
            # Base level
            level = np.random.uniform(100, 1000)
            
            # Generate values based on pattern
            if pattern_name == 'Smooth':
                # Base + small noise
                noise = np.random.normal(0, level * 0.1, n_months)
                sales = level + noise
                
            elif pattern_name == 'Seasonal':
                # Base + Sine wave + noise
                seasonality = level * 0.5 * np.sin(np.linspace(0, 2*np.pi * (n_months/12), n_months))
                noise = np.random.normal(0, level * 0.1, n_months)
                sales = level + seasonality + noise
                
            elif pattern_name == 'Intermittent':
                # Poisson-like but with structural zeros
                sales = np.random.poisson(level * 0.2, n_months).astype(float)
                # Force 40-70% zeros
                mask = np.random.rand(n_months) < np.random.uniform(0.4, 0.7)
                sales[mask] = 0
                
            elif pattern_name == 'Lumpy':
                # Similar to intermittent but with high variance spikes
                sales = np.random.poisson(level * 0.2, n_months).astype(float)
                mask = np.random.rand(n_months) < 0.5
                sales[mask] = 0
                # Add spikes
                spike_mask = np.random.rand(n_months) < 0.1
                sales[spike_mask] += np.random.uniform(level, level*3, np.sum(spike_mask))
                
            elif pattern_name == 'Trending':
                # Linear trend + noise
                trend_dir = np.random.choice([1, -1])
                trend = np.linspace(0, level * 0.8 * trend_dir, n_months)
                noise = np.random.normal(0, level * 0.1, n_months)
                sales = level + trend + noise
                
            elif pattern_name == 'New_Product':
                # Ramp up curve
                ramp = np.logspace(0, 1, n_months) / 10
                noise = np.random.normal(0, level * 0.1, n_months)
                sales = (level * ramp) + noise
            
            # Ensure non-negative and round
            sales = np.maximum(sales, 0)
            sales = np.round(sales)
            
            for d, s in zip(dates, sales):
                data.append({'Date': d, 'SKU': sku_id, 'Sales': s, 'Pattern_Truth': pattern_name})
                
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    print(f"Generated {len(df)} rows for {df['SKU'].nunique()} SKUs.")
    print("Sample:\n", df.head())
    
    # Sanity check distribution
    print("\nPattern Distribution:")
    print(df.groupby('SKU')['Pattern_Truth'].first().value_counts())
    
    df.to_csv("synthetic_sales_data.csv", index=False)
    print("\nSaved to synthetic_sales_data.csv")
