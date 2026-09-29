import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances

def calculate_fidelity(real_df, synth_df):
    """Computes univariate KS statistic and multivariate correlation error."""
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    
    # Univariate: KS Test
    ks_stats = []
    for col in num_cols:
        stat, _ = ks_2samp(real_df[col].dropna(), synth_df[col].dropna())
        ks_stats.append(stat)
        
    avg_ks = np.mean(ks_stats) if ks_stats else 0.0
    
    # Multivariate: Correlation Matrix Absolute Error
    real_corr = real_df[num_cols].corr().fillna(0)
    synth_corr = synth_df[num_cols].corr().fillna(0)
    corr_error = np.mean(np.abs(real_corr.values - synth_corr.values))
    
    return {
        "Average_KS_Statistic": float(avg_ks),
        "Correlation_Matrix_Error": float(corr_error)
    }

def calculate_utility(real_df, synth_df, target_col):
    """Computes TSTR and TRTR baseline using an optimized Random Forest pipeline."""
    if target_col not in real_df.columns:
        return {"Error": f"Target column '{target_col}' not found."}
        
    # Split features and target
    X_real = real_df.drop(columns=[target_col])
    y_real = real_df[target_col]
    X_synth = synth_df.drop(columns=[target_col])
    y_synth = synth_df[target_col]
    
    # Identify column types for optimal preprocessing
    num_cols = X_real.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_real.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Build a robust scikit-learn preprocessing pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    # Extract a 20% hold-out test set from the real data
    test_size = max(1, int(len(real_df) * 0.2))
    X_real_train = X_real.iloc[:-test_size]
    y_real_train = y_real.iloc[:-test_size]
    X_test = X_real.iloc[-test_size:]
    y_test = y_real.iloc[-test_size:]
    
    # Determine if classification or regression is needed
    is_classification = y_real.nunique() < 20 or y_real.dtype == 'object'
    
    if is_classification:
        y_real_train = y_real_train.astype(str)
        y_synth = y_synth.astype(str)
        y_test = y_test.astype(str)
        
        # 100 estimators for higher stability and accuracy
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        metric = "Accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        metric = "RMSE"
        
    # Combine preprocessor and model into final pipelines
    pipeline_real = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline_synth = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # Train pipelines
    pipeline_real.fit(X_real_train, y_real_train)
    pipeline_synth.fit(X_synth, y_synth)
    
    # Evaluate
    if is_classification:
        trtr_score = accuracy_score(y_test, pipeline_real.predict(X_test))
        tstr_score = accuracy_score(y_test, pipeline_synth.predict(X_test))
    else:
        trtr_score = np.sqrt(mean_squared_error(y_test, pipeline_real.predict(X_test)))
        tstr_score = np.sqrt(mean_squared_error(y_test, pipeline_synth.predict(X_test)))
        
    return {
        "Metric": metric,
        "TRTR_Score": float(trtr_score),
        "TSTR_Score": float(tstr_score),
        "Utility_Delta": float(abs(trtr_score - tstr_score))
    }

def calculate_privacy(real_df, synth_df):
    """Computes Distance to Closest Record (DCR) to detect memorization."""
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    
    if len(num_cols) == 0:
        return {"Error": "No numerical columns available for DCR calculation."}
        
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_df[num_cols].fillna(0))
    synth_scaled = scaler.transform(synth_df[num_cols].fillna(0))
    
    # Subsample matrices to prevent OOM errors on massive datasets
    max_rows = 2000
    if len(synth_scaled) > max_rows:
        synth_scaled = synth_scaled[:max_rows]
    if len(real_scaled) > max_rows:
        real_scaled = real_scaled[:max_rows]
        
    dists = euclidean_distances(synth_scaled, real_scaled)
    min_dists = np.min(dists, axis=1)
    
    exact_matches = np.sum(min_dists < 1e-6)
    fifth_percentile = np.percentile(min_dists, 5)
    
    return {
        "Exact_Matches": int(exact_matches),
        "5th_Percentile_DCR": float(fifth_percentile)
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <real_data.csv> <synthetic_data.csv> [target_column]")
        sys.exit(1)
        
    real_path = sys.argv[1]
    synth_path = sys.argv[2]
    target_col = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Loading Real Data: {real_path}")
    print(f"Loading Synthetic Data: {synth_path}")
    
    try:
        real_df = pd.read_csv(real_path)
        synth_df = pd.read_csv(synth_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    
    report = {}
    
    print("Evaluating Fidelity...")
    report["Fidelity"] = calculate_fidelity(real_df, synth_df)
    
    if target_col:
        print(f"Evaluating Utility on target '{target_col}'...")
        report["Utility"] = calculate_utility(real_df, synth_df, target_col)
    else:
        report["Utility"] = "Skipped. Provide a target column name as the 3rd argument to run TSTR."
        
    print("Evaluating Privacy (DCR)...")
    report["Privacy"] = calculate_privacy(real_df, synth_df)
    
    print("\n" + "="*65)
    print(" 📊 SYNTHETIC DATA EVALUATION REPORT")
    print("="*65)
    print(f"{'CATEGORY':<12} | {'METRIC':<32} | {'SCORE'}")
    print("-" * 65)
    print(f"{'Fidelity':<12} | {'Average KS Statistic':<32} | {report['Fidelity']['Average_KS_Statistic']:.4f}")
    print(f"{'Fidelity':<12} | {'Correlation Matrix Error':<32} | {report['Fidelity']['Correlation_Matrix_Error']:.4f}")
    print("-" * 65)
    if target_col:
        print(f"{'Utility':<12} | {f'TRTR Baseline ({report.get('Utility',{}).get('Metric','')})':<32} | {report['Utility']['TRTR_Score']:.4f}")
        print(f"{'Utility':<12} | {f'TSTR Score ({report.get('Utility',{}).get('Metric','')})':<32} | {report['Utility']['TSTR_Score']:.4f}")
        print(f"{'Utility':<12} | {'Utility Delta':<32} | {report['Utility']['Utility_Delta']:.4f}")
    else:
        print(f"{'Utility':<12} | {'Status':<32} | Skipped (No target)")
    print("-" * 65)
    print(f"{'Privacy':<12} | {'Exact Matches (Data Leaks)':<32} | {report['Privacy']['Exact_Matches']}")
    print(f"{'Privacy':<12} | {'5th Percentile DCR':<32} | {report['Privacy']['5th_Percentile_DCR']:.4f}")
    print("="*65 + "\n")
    
if __name__ == "__main__":
    main()
