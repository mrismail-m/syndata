import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from rdt.transformers.numerical import ClusterBasedNormalizer

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer, GaussianCopulaSynthesizer
    from sdv.evaluation.single_table import evaluate_quality
    from sklearn.datasets import fetch_california_housing
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages using:")
    print("pip install pandas matplotlib seaborn sdv scikit-learn")
    sys.exit(1)

def load_data(csv_path=None):
    """
    Loads tabular data from a CSV file. If none is provided, loads a sample
    dataset (California Housing) to demonstrate the functionality.
    """
    if csv_path:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        if len(df) > 5000:
            print(f"Subsampling data to 5000 rows (from {len(df)}) for faster training...")
            df = df.sample(n=5000, random_state=42)
    else:
        print("No CSV provided. Loading California Housing dataset as a fallback...")
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        # Subsampling for speed during demonstration
        df = df.sample(n=2000, random_state=42) 
    return df

def generate_and_evaluate(csv_path=None, num_rows=1000):
    real_data = load_data(csv_path)
    
    print("\n--- 1. Metadata Detection ---")
    start_time = time.time()
    metadata = SingleTableMetadata()
    # SDV automatically detects column data types and structural constraints
    metadata.detect_from_dataframe(real_data)
    
    # Override automatic PII detection for coordinates if they exist
    if 'longitude' in real_data.columns:
        metadata.update_column(column_name='longitude', sdtype='numerical')
    if 'latitude' in real_data.columns:
        metadata.update_column(column_name='latitude', sdtype='numerical')
        
    print(f"Metadata detected in {time.time() - start_time:.2f} seconds.")

    print("\\n--- 2. Training Synthesizer ---")
    
    # Dynamic Model Selection for maximum quality
    if len(real_data) < 2000:
        print("Small dataset detected (< 2000 rows). Using GaussianCopulaSynthesizer for optimal statistical fidelity.")
        synthesizer = GaussianCopulaSynthesizer(metadata)
    else:
        print(f"Dataset has {len(real_data)} rows. Using TVAESynthesizer with optimized deep-learning hyperparameters.")
        # Lower batch size increases weight update frequency per epoch, great for smaller datasets
        batch_size = 50 if len(real_data) < 5000 else 500
        synthesizer = TVAESynthesizer(metadata, epochs=500, batch_size=batch_size)
    
    print("\\n--- Updating Transformers ---")
    synthesizer.auto_assign_transformers(real_data)
    
    # If dealing with geospatial data, enforce cluster models
    transformers_to_update = {}
    if 'longitude' in real_data.columns:
        transformers_to_update['longitude'] = ClusterBasedNormalizer(enforce_min_max_values=True, max_clusters=10)
    if 'latitude' in real_data.columns:
        transformers_to_update['latitude'] = ClusterBasedNormalizer(enforce_min_max_values=True, max_clusters=10)
        
    if transformers_to_update:
        synthesizer.update_transformers(column_name_to_transformer=transformers_to_update)
    
    start_time = time.time()
    synthesizer.fit(real_data)
    print(f"Model trained in {time.time() - start_time:.2f} seconds.")

    print("\n--- 3. Generating Synthetic Data ---")
    start_time = time.time()
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    print(f"{num_rows} rows generated in {time.time() - start_time:.2f} seconds.")

    print("\\n--- 4. Evaluating Statistical Quality ---")
    try:
        from sdmetrics.reports.single_table import QualityReport
        report = QualityReport()
        report.generate(real_data, synthetic_data, metadata.to_dict())
        
        print(f"\\nOverall Quality Score: {report.get_score():.2%}")
        print("\\nDetailed Properties (Column Shapes - Statistical Similarity):")
        print(report.get_details(property_name='Column Shapes'))
    except Exception as e:
        print(f"Evaluation failed (likely due to SDV/SDMetrics version changes): {e}")

    print("\n--- 5. Plotting Distributions ---")
    # Select the first 4 numerical columns for visual comparison
    numerical_cols = real_data.select_dtypes(include=['number']).columns[:4]
    
    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(nrows=len(numerical_cols), ncols=1, figsize=(10, 3 * len(numerical_cols)))
        if len(numerical_cols) == 1:
            axes = [axes]
            
        for i, col in enumerate(numerical_cols):
            sns.kdeplot(real_data[col], label='Real Data', fill=True, ax=axes[i], color='#1f77b4', alpha=0.6)
            sns.kdeplot(synthetic_data[col], label='Synthetic Data', fill=True, ax=axes[i], color='#ff7f0e', alpha=0.6)
            axes[i].set_title(f'Distribution Comparison: {col}')
            axes[i].legend()
            
        plt.tight_layout()
        plot_path = 'synthetic_vs_real_plot.png'
        plt.savefig(plot_path)
        print(f"\nPlot successfully saved as '{plot_path}'")
        plt.show()

if __name__ == "__main__":
    import sys
    # If a path is provided as a command-line argument, use it.
    csv_input = sys.argv[1] if len(sys.argv) > 1 else "/home/groot/Documents/syndata/insurance.csv"
    generate_and_evaluate(csv_path=csv_input, num_rows=1000)
