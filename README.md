# SynthAI Hub 🧠

A full-stack, multi-modal synthetic data generation engine designed to solve the privacy-utility bottleneck in Machine Learning. 

SynthAI Hub ingests sensitive production data (Tabular, Relational, or Unstructured Documents) and mathematically generates high-fidelity "synthetic twins" that preserve complex statistical correlations while mathematically guaranteeing zero privacy leaks.

![Architecture Presentation](presentation.html)

## 🚀 Features
- **Tabular & Relational Modeling**: Leverages the Synthetic Data Vault (SDV) and Gaussian Copulas to learn exact marginal distributions and cross-column correlations.
- **Document OCR Parsing**: Ingests raw OCR JSON (e.g., invoices), extracts relational components (parents vs. line items), and generates mathematically sound synthetic records (e.g., Quantity × Price = Subtotal).
- **Strict Privacy Guarantees**: Evaluates generated datasets using high-dimensional Euclidean Distance to Closest Record (DCR) to prove zero data memorization.
- **Absolute Sandbox Isolation**: The FastAPI orchestration layer dynamically provisions isolated directories for every project and injects asynchronous Python subprocesses into them, preventing cross-project data bleed.
- **Real-Time Streaming**: Streams live terminal execution logs directly to the React SPA using Server-Sent Events (SSE).

## 🛠️ Tech Stack
- **Frontend**: React.js, Vite
- **Backend Orchestrator**: Python, FastAPI, Uvicorn
- **ML Engines**: PyTorch, SDV, Copulas, scikit-learn, sdmetrics
- **Visualizations**: Matplotlib, Seaborn
- **Infrastructure Ready**: Designed for isolated VPS daemonization via `systemd`.

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mrismail-m/syndata.git
   cd syndata
   ```

2. **Set up the Python Backend:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up the React Frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Start the FastAPI Server:**
   ```bash
   # The FastAPI server automatically mounts and serves the compiled React frontend!
   python server.py
   ```
   Navigate to `http://localhost:8000` in your browser.

## 📊 Evaluation & Metrics
The pipeline doesn't just generate data; it validates it.
- **Fidelity**: Kolmogorov-Smirnov (KS) statistic and Correlation Matrix Error.
- **Privacy**: 5th Percentile DCR and exact-match detection.

## 📝 License
MIT
