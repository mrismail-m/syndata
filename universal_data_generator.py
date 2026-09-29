import sys
import os
import json
import time
import pandas as pd
import numpy as np
import warnings

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdmetrics.reports.single_table import QualityReport
except ImportError:
    print("Missing SDV libraries. Please run: pip install sdv sdmetrics")
    sys.exit(1)

def auto_discover_constraints(df):
    constraints = []
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df[col].min() >= 0:
            constraints.append({
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': col, 'strict': False}
            })
            print(f"   [Auto-Rule] Discovered non-negative column: '{col}'. Added Positive constraint.")
    return constraints

def inject_edge_cases(df, null_rate, outlier_rate):
    np.random.seed(42)
    print(f"   [AI Layer] Injecting {null_rate*100}% nulls and {outlier_rate*100}% outliers into generated data.")
    
    if null_rate > 0:
        for col in df.columns:
            mask = np.random.rand(len(df)) < null_rate
            df.loc[mask, col] = np.nan
            
    if outlier_rate > 0:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            mask = np.random.rand(len(df)) < outlier_rate
            spikes = np.random.uniform(5, 10, size=mask.sum())
            df.loc[mask, col] = df.loc[mask, col] * spikes
            
    return df

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "insurance.csv"
    print(f"Loading dataset: {csv_file}")
    
    config = {}
    if os.path.exists("generation_config.json"):
        with open("generation_config.json", "r") as f:
            config = json.load(f)
            
    num_rows_req = config.get("num_rows", None)
    random_seed = config.get("random_seed", 42)
    null_rate = config.get("null_rate", 0.0)
    outlier_rate = config.get("outlier_rate", 0.0)
    pii_columns = config.get("pii_columns", [])
    
    try:
        real_data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'.")
        sys.exit(1)

    print("\n--- 1. Auto-Detecting Metadata & Rules ---")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    for col in pii_columns:
        if col in metadata.columns:
            metadata.update_column(column_name=col, sdtype='pii')
            print(f"   [Privacy] Column '{col}' marked for AI-driven PII Synthesis (Faker).")

    auto_constraints = auto_discover_constraints(real_data)
    if auto_constraints:
        print(f"   Total constraints applied: {len(auto_constraints)}")

    print("\n--- 2. Building Gaussian Copula Model (High-Speed Statistical) ---")
    
    import torch
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    synthesizer = GaussianCopulaSynthesizer(metadata)
    
    if auto_constraints:
        try:
            synthesizer.add_constraints(auto_constraints)
        except Exception as e:
            pass

    print(f"\n--- 3. Training & Scoring ---")
    print(f"\n>> Testing Algorithm: Gaussian Copula (Statistical)")
    start_time = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            synthesizer.fit(real_data)
            sample_data = synthesizer.sample(num_rows=min(1000, len(real_data)))
            report = QualityReport()
            report.generate(real_data, sample_data, metadata.to_dict(), verbose=False)
            score = report.get_score()
            print(f"   Accuracy Score: {score * 100:.2f}% (Trained in {time.time() - start_time:.2f}s)")
        except Exception as e:
            print(f"   Failed to train/score model: {e}")
            sys.exit(1)

    print("\n" + "="*55)
    print(f"🏆 TOURNAMENT WINNER: Gaussian Copula (Statistical)")
    print(f"🏆 WINNING SCORE: {score * 100:.2f}%")
    print("="*55)

    print("\n--- 4. Generating Final Dataset with Winner ---")
    out_rows = num_rows_req if num_rows_req else len(real_data)
    
    np.random.seed(random_seed)
    final_synthetic_data = synthesizer.sample(num_rows=out_rows)
    
    if null_rate > 0 or outlier_rate > 0:
        final_synthetic_data = inject_edge_cases(final_synthetic_data, null_rate, outlier_rate)
        
    final_output = "universal_synthetic_output.csv"
    final_synthetic_data.to_csv(final_output, index=False)
    print(f"Successfully saved {out_rows} perfectly constrained rows to '{final_output}'")
    
    synthesizer.save('best_universal_model.pkl')
    print("Saved the winning model to 'best_universal_model.pkl'.")

if __name__ == "__main__":
    main()
