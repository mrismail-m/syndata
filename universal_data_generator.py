import sys
import os
import json
import time
import pandas as pd
import warnings

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer, CTGANSynthesizer
    from sdmetrics.reports.single_table import QualityReport
except ImportError:
    print("Missing SDV libraries. Please run: pip install sdv sdmetrics")
    sys.exit(1)

def auto_discover_constraints(df):
    """Mathematically scan the dataframe to infer SDV constraints."""
    constraints = []
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # 1. Non-Negative Auto-Rule
    # If a column strictly contains values >= 0, we force the AI to never generate negative numbers.
    for col in num_cols:
        if df[col].min() >= 0:
            constraints.append({
                'constraint_class': 'Positive',
                'constraint_parameters': {
                    'column_name': col,
                    'strict': False
                }
            })
            print(f"   [Auto-Rule] Discovered non-negative column: '{col}'. Added Positive constraint.")
            
    return constraints

def load_manual_rules(json_path):
    """Load manual business rules from a JSON file if it exists."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                rules = json.load(f)
            print(f"   [Manual-Rule] Successfully loaded {len(rules)} custom rules from {json_path}")
            return rules
        except Exception as e:
            print(f"   [Manual-Rule] Error reading {json_path}: {e}")
    return []

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "insurance.csv"
    print(f"Loading dataset: {csv_file}")
    
    try:
        real_data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'.")
        sys.exit(1)

    print("\n--- 1. Auto-Detecting Metadata & Rules ---")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    # Generate hybrid rules (Auto Math + Manual Business Logic)
    auto_constraints = auto_discover_constraints(real_data)
    manual_constraints = load_manual_rules('manual_rules.json')
    all_constraints = auto_constraints + manual_constraints
    
    if all_constraints:
        print(f"   Total constraints applied: {len(all_constraints)}")

    # 2. Define the Tournament Lineup with FAST MODE (30 Epochs)
    print("\n--- 2. Building Models (Fast Mode 30 Epochs) ---")
    models = {
        "Gaussian Copula (Statistical)": GaussianCopulaSynthesizer(metadata),
        "CTGAN (Generative Adversarial Network)": CTGANSynthesizer(metadata, epochs=30),
        "TVAE (Variational Autoencoder)": TVAESynthesizer(metadata, epochs=30)
    }
    
    # Inject our auto-generated rules into the AI models
    for name, synth in models.items():
        if all_constraints:
            try:
                synth.add_constraints(all_constraints)
            except Exception as e:
                print(f"   [Warning] Could not apply constraints to {name}: {e}")

    best_score = 0
    best_model_name = None
    best_model_instance = None

    print(f"\n--- 3. Starting Auto-ML Tournament ---")
    
    for name, synthesizer in models.items():
        print(f"\n>> Testing Algorithm: {name}")
        start_time = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Train the model on the full data
                synthesizer.fit(real_data)
                
                # Generate a sample strictly to grade its performance
                sample_data = synthesizer.sample(num_rows=min(1000, len(real_data)))
                
                # Run the SDMetrics grader
                report = QualityReport()
                report.generate(real_data, sample_data, metadata.to_dict(), verbose=False)
                score = report.get_score()
                print(f"   Accuracy Score: {score * 100:.2f}% (Trained in {time.time() - start_time:.2f}s)")
            except Exception as e:
                print(f"   Failed to train/score {name}: {e}")
                score = 0
                
        if score > best_score:
            best_score = score
            best_model_name = name
            best_model_instance = synthesizer

    print("\n" + "="*55)
    print(f"🏆 TOURNAMENT WINNER: {best_model_name}")
    print(f"🏆 WINNING SCORE: {best_score * 100:.2f}%")
    print("="*55)

    if best_model_instance:
        print("\n--- 4. Generating Final Dataset with Winner ---")
        # Generate the final output exactly matching the size of the original data
        final_synthetic_data = best_model_instance.sample(num_rows=len(real_data))
        final_output = "universal_synthetic_output.csv"
        final_synthetic_data.to_csv(final_output, index=False)
        print(f"Successfully saved {len(real_data)} perfectly constrained rows to '{final_output}'")
        
        best_model_instance.save('best_universal_model.pkl')
        print("Saved the winning model to 'best_universal_model.pkl'.")

if __name__ == "__main__":
    main()
