import streamlit as st
import subprocess
import sys
import os
import json
import pandas as pd

st.set_page_config(page_title="HackDataV2 - Tabular", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        width: 100%;
        border-radius: 8px;
    }
    .log-box {
        background-color: #0E1117;
        color: #00FF00;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border-radius: 8px;
        height: 250px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 14px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR (CONFIGURATION) -----------------
with st.sidebar:
    st.title("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Schema/Sample (CSV)", type=["csv"])
    
    st.markdown("### 📊 Generation Settings")
    num_rows = st.number_input("Row Count", min_value=10, max_value=1000000, value=1000, step=100)
    random_seed = st.number_input("Random Seed (Determinism)", value=42)
    target_col = st.text_input("Target Column (Utility Test)", placeholder="e.g., 'charges'")
    
    st.markdown("### 🔒 Privacy Controls")
    pii_input = st.text_input("Columns to Anonymize (PII)", placeholder="e.g. name, email, address")
    st.caption("AI will synthesize realistic mock data for these columns.")
    
    st.markdown("### 🧪 AI Edge-Case Injection")
    null_rate = st.slider("Null Rate %", 0.0, 50.0, 0.0, help="Randomly inject missing values to simulate messy data.")
    outlier_rate = st.slider("Outlier Rate %", 0.0, 25.0, 0.0, help="Mathematically spike numeric values to test extreme cases.")
    
    process_btn = st.button("🚀 Process & Generate Data")

# ----------------- MAIN WORKSPACE (LIVE PREVIEW) -----------------
st.title("🧬 HackDataV2 Workspace: Tabular Engine")

if not uploaded_file:
    st.info("👈 Upload a sample dataset or schema in the sidebar to begin.")
else:
    # We display the preview immediately
    st.subheader("Live Preview Canvas: Real Data Sample")
    preview_df = pd.read_csv(uploaded_file)
    st.dataframe(preview_df.head(5), use_container_width=True)

    if process_btn:
        st.markdown("---")
        st.subheader("Live Processing Logs")
        log_placeholder = st.empty()
        
        file_path = "uploaded_data.csv"
        uploaded_file.seek(0)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
            
        pii_columns = [c.strip() for c in pii_input.split(",") if c.strip()] if pii_input else []
        
        config = {
            "target_column": target_col,
            "num_rows": int(num_rows),
            "random_seed": int(random_seed),
            "null_rate": null_rate / 100.0,
            "outlier_rate": outlier_rate / 100.0,
            "pii_columns": pii_columns
        }
        with open("generation_config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        cmd = [sys.executable, "tabular_data_analyzer.py", file_path]
        
        log_content = "Initializing Universal AI Pipeline...\n"
        log_placeholder.markdown(f'<div class="log-box">{log_content}</div>', unsafe_allow_html=True)
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        def translate_log(raw_line):
            raw_line = raw_line.strip()
            if not raw_line: return None
            if "Loading dataset:" in raw_line or "Loading Real Data" in raw_line:
                return "📂 Reading schema and learning distributions..."
            if "[Privacy] Column" in raw_line:
                return "🔒 " + raw_line.strip()
            if "Auto-Detecting Metadata" in raw_line:
                return "🔍 Scanning your data to understand its hidden structure..."
            if "Total constraints applied" in raw_line:
                return "🛡️ Adding mathematical rules to prevent impossible data (e.g. no negative ages)."
            if "Building Gaussian Copula Model" in raw_line:
                return "🤖 Preparing the High-Speed Statistical AI Model..."
            if "Testing Algorithm: Gaussian Copula" in raw_line:
                return "⚙️ Training Gaussian Copula Algorithm..."
            if "Accuracy Score:" in raw_line:
                return f"   ✅ Finished testing model! ({raw_line.strip()})"
            if "TOURNAMENT WINNER:" in raw_line:
                winner = raw_line.split(":")[-1].strip()
                return f"🏆 AI Tournament Complete! The winner is: {winner}"
            if "Injecting" in raw_line and "nulls" in raw_line:
                return "🧪 " + raw_line.strip()
            if "Generating Final Dataset" in raw_line:
                return "🏭 The winning AI is now generating your brand new synthetic rows..."
            if "Successfully saved" in raw_line:
                return "💾 Generation complete!"
            if "INDUSTRY STANDARD EVALUATION" in raw_line:
                return "🔬 Sending the generated data to the Quality Lab..."
            if "Evaluating Fidelity" in raw_line:
                return "📐 Checking Fidelity (Does it mathematically match the real data?)..."
            if "Evaluating Utility" in raw_line:
                return "📈 Checking Utility (Can we train Machine Learning on this fake data?)..."
            if "Evaluating Privacy" in raw_line:
                return "🔒 Checking Privacy (Ensuring no real user data was copy-pasted)..."
            if "GENERATING VISUALIZATIONS" in raw_line:
                return "🎨 Generating visual diagnostic charts (PCA, Heatmaps, DCR)..."
            if "SYNTHETIC DATA EVALUATION REPORT" in raw_line:
                return "✅ All tests passed! Data is ready for download."
            if "Error" in raw_line or "Exception" in raw_line:
                return f"⚠️ Warning: {raw_line}"
            return None

        log_lines = []
        for line in process.stdout:
            friendly_msg = translate_log(line)
            if friendly_msg:
                log_lines.append(friendly_msg + "\n")
                if len(log_lines) > 12:
                    log_lines.pop(0)
                display_text = "".join(log_lines)
                log_placeholder.markdown(f'<div class="log-box">{display_text}</div>', unsafe_allow_html=True)
            
        process.wait()
        
        if process.returncode == 0:
            st.success("✅ Analysis Complete! The new synthetic data has been successfully generated.")
            
            st.markdown("---")
            st.subheader("Live Preview Canvas: Generated Synthetic Data")
            try:
                synth_df = pd.read_csv("universal_synthetic_output.csv")
                st.dataframe(synth_df.head(10), use_container_width=True)
                
                csv = synth_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Export Synthetic Data (CSV)",
                    data=csv,
                    file_name='synthetic_data.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Could not load output file: {e}")
                
            st.markdown("---")
            st.header("📊 Evaluation Report")
            try:
                with open("evaluation_report.json", "r") as f:
                    report = json.load(f)
                
                col_f, col_u, col_p = st.columns(3)
                with col_f:
                    st.subheader("📐 Fidelity")
                    st.metric("KS Statistic", f"{report['Fidelity']['Average_KS_Statistic']:.4f}")
                    st.metric("Correlation Error", f"{report['Fidelity']['Correlation_Matrix_Error']:.4f}")
                    
                with col_u:
                    st.subheader("📈 Utility")
                    if 'Utility_Delta' in report.get('Utility', {}):
                        st.metric("TRTR Baseline", f"{report['Utility']['TRTR_Score']:.4f}")
                        st.metric("Utility Delta", f"{report['Utility']['Utility_Delta']:.4f}")
                    else:
                        st.write("Skipped (No Target)")
                        
                with col_p:
                    st.subheader("🔒 Privacy")
                    st.metric("Exact Matches", report['Privacy']['Exact_Matches'])
                    st.metric("5th Percentile DCR", f"{report['Privacy']['5th_Percentile_DCR']:.4f}")
            except Exception as e:
                st.error("Evaluation report not found.")

            st.markdown("---")
            st.header("📈 Visual Diagnostics")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                if os.path.exists("pca_projection.png"):
                    st.image("pca_projection.png", caption="PCA Geometric Overlap (Red=Synthetic, Blue=Real)", use_container_width=True)
            with col_img2:
                if os.path.exists("correlation_heatmap.png"):
                    st.image("correlation_heatmap.png", caption="Correlation Error Heatmap", use_container_width=True)
            if os.path.exists("dcr_histogram.png"):
                st.image("dcr_histogram.png", caption="Privacy: Distance to Closest Real Record", use_container_width=True)
        else:
            st.error("❌ Process failed. Check the logs above for details.")
