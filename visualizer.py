import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# Output directory for artifacts
OUT_DIR = "./"

def plot_pca(real_df, synth_df):
    print("Generating PCA Projection...")
    n_samples = min(2000, len(real_df), len(synth_df))
    r_sample = real_df.sample(n_samples, random_state=42)
    s_sample = synth_df.sample(n_samples, random_state=42)
    
    num_cols = r_sample.select_dtypes(include=[np.number]).columns
    r_num = r_sample[num_cols].fillna(0)
    s_num = s_sample[num_cols].fillna(0)
    
    scaler = StandardScaler()
    r_scaled = scaler.fit_transform(r_num)
    s_scaled = scaler.transform(s_num)
    
    pca = PCA(n_components=2)
    r_pca = pca.fit_transform(r_scaled)
    s_pca = pca.transform(s_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(r_pca[:, 0], r_pca[:, 1], alpha=0.5, label='Real Data', color='blue', s=15)
    plt.scatter(s_pca[:, 0], s_pca[:, 1], alpha=0.5, label='Synthetic Data', color='red', s=15)
    plt.title("PCA Geometric Projection (Blue=Real, Red=Synthetic)")
    plt.legend()
    plt.savefig(OUT_DIR + "pca_projection.png", bbox_inches='tight', dpi=150)
    plt.close()

def plot_correlation_diff(real_df, synth_df):
    print("Generating Correlation Heatmap...")
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    # Keep it readable for the image (max 15 columns)
    cols_to_plot = num_cols[:15]
    
    r_corr = real_df[cols_to_plot].corr()
    s_corr = synth_df[cols_to_plot].corr()
    diff_corr = r_corr - s_corr
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_corr, annot=False, cmap='coolwarm', center=0, vmin=-0.5, vmax=0.5)
    plt.title("Correlation Error Heatmap (0 = Perfect Match)")
    plt.savefig(OUT_DIR + "correlation_heatmap.png", bbox_inches='tight', dpi=150)
    plt.close()

def plot_dcr(real_df, synth_df):
    print("Generating Privacy DCR Histogram...")
    n_samples = min(1000, len(real_df), len(synth_df))
    r_sample = real_df.sample(n_samples, random_state=42)
    s_sample = synth_df.sample(n_samples, random_state=42)
    
    num_cols = r_sample.select_dtypes(include=[np.number]).columns
    
    scaler = StandardScaler()
    r_scaled = scaler.fit_transform(r_sample[num_cols].fillna(0))
    s_scaled = scaler.transform(s_sample[num_cols].fillna(0))
    
    distances = cdist(s_scaled, r_scaled, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(min_distances, bins=40, kde=True, color='purple')
    plt.title("Privacy Proof: Distance to Closest Real Record")
    plt.xlabel("Euclidean Distance (Higher is safer)")
    plt.ylabel("Number of Synthetic Rows")
    plt.savefig(OUT_DIR + "dcr_histogram.png", bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualizer.py <real.csv> <synthetic.csv>")
        sys.exit(1)
        
    real = pd.read_csv(sys.argv[1])
    synth = pd.read_csv(sys.argv[2])
    
    plot_pca(real, synth)
    plot_correlation_diff(real, synth)
    plot_dcr(real, synth)
    print("Done!")
