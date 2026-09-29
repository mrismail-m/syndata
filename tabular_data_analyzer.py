import sys
import os
import subprocess
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python tabular_data_analyzer.py <dataset.csv>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}'. Please ensure the file exists.")
        sys.exit(1)
        
    print(f"🚀 STARTING TABULAR DATA ANALYZER FOR: {csv_file}")
    
    # Read config
    target_col = ""
    if os.path.exists("generation_config.json"):
        with open("generation_config.json", "r") as f:
            config = json.load(f)
            target_col = config.get("target_column", "")
            
    # PHASE 1: GENERATION
    print("\n" + "="*50)
    print("PHASE 1: SYNTHETIC DATA GENERATION (Auto-ML)")
    print("="*50)
    
    gen_cmd = [sys.executable, "universal_data_generator.py", csv_file]
    try:
        subprocess.run(gen_cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Error: Generation failed. Aborting analyzer.")
        sys.exit(1)
        
    synthetic_file = "universal_synthetic_output.csv"
    if not os.path.exists(synthetic_file):
        print("\n❌ Error: Synthetic data was not found after generation phase.")
        sys.exit(1)
        
    # PHASE 2: EVALUATION
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
        
    # PHASE 3: VISUALIZATIONS
    print("\n" + "="*50)
    print("PHASE 3: GENERATING VISUALIZATIONS")
    print("="*50)
    
    vis_cmd = [sys.executable, "visualizer.py", csv_file, synthetic_file]
    try:
        subprocess.run(vis_cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Error: Visualizations failed.")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("✅ ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
