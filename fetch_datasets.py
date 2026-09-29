import sys
import subprocess
import pandas as pd
from sklearn.datasets import fetch_kddcup99, fetch_covtype

def main():
    print("Fetching KDD Cup 99...")
    # KDD Cup 99 is a network intrusion detection dataset (highly imbalanced attacks)
    kdd = fetch_kddcup99(percent10=True, as_frame=True)
    df_kdd = kdd.frame
    # Capping at 5,000 rows so the 500-epoch neural nets don't take 5+ hours on CPU
    df_kdd = df_kdd.sample(n=5000, random_state=42).reset_index(drop=True)
    
    # KDD has byte strings that SDV struggles with, decode them
    for col in df_kdd.select_dtypes([object]).columns:
        df_kdd[col] = df_kdd[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
    df_kdd.to_csv("kddcup99.csv", index=False)
    print(f"✅ Saved kddcup99.csv ({len(df_kdd)} rows)")

    print("Fetching Covertype...")
    # Covertype is a forest cover prediction dataset (54 features, mixed types)
    cov = fetch_covtype(as_frame=True)
    df_cov = cov.frame
    # Capping at 5,000 rows
    df_cov = df_cov.sample(n=5000, random_state=42).reset_index(drop=True)
    df_cov.to_csv("covertype.csv", index=False)
    print(f"✅ Saved covertype.csv ({len(df_cov)} rows)")

    print("\n" + "="*50)
    print("🚀 PIPELINE 1: KDD CUP 99 (Target: 'labels')")
    print("="*50)
    subprocess.run([sys.executable, "tabular_data_analyzer.py", "kddcup99.csv", "labels"])

    print("\n" + "="*50)
    print("🚀 PIPELINE 2: COVERTYPE (Target: 'Cover_Type')")
    print("="*50)
    subprocess.run([sys.executable, "tabular_data_analyzer.py", "covertype.csv", "Cover_Type"])

if __name__ == "__main__":
    main()
