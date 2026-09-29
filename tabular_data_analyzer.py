import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python tabular_data_analyzer.py <dataset.csv> [target_column]")
        print("Example: python tabular_data_analyzer.py insurance.csv charges")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    target_col = sys.argv[2] if len(sys.argv) > 2 else ""
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}'. Please ensure the file exists.")
        sys.exit(1)
        
    print(f"🚀 STARTING TABULAR DATA ANALYZER FOR: {csv_file}")
    
    # ---------------------------------------------------------
    # PHASE 1: GENERATION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 1: SYNTHETIC DATA GENERATION (Auto-ML)")
    print("="*50)
    
    gen_cmd = [sys.executable, "universal_data_generator.py", csv_file]
    try:
        # We use subprocess to keep memory perfectly clean between generation and evaluation
        subprocess.run(gen_cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Error: Generation failed. Aborting analyzer.")
        sys.exit(1)
        
    synthetic_file = "universal_synthetic_output.csv"
    if not os.path.exists(synthetic_file):
        print("\n❌ Error: Synthetic data was not found after generation phase.")
        sys.exit(1)
        
    # ---------------------------------------------------------
    # PHASE 2: EVALUATION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 2: INDUSTRY STANDARD EVALUATION")
    print("="*50)
    
    eval_cmd = [sys.executable, "evaluator.py", csv_file, synthetic_file]
    if target_col:
        eval_cmd.append(target_col)
        
    try:
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Error: Evaluation failed.")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("✅ ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
