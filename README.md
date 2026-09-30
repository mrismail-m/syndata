# SynthAI Hub

SynthAI Hub is a multi-modal synthetic data generation platform designed to solve the privacy bottleneck in software testing and machine learning. 

When developers need realistic data but cannot use actual production data due to privacy laws, SynthAI Hub steps in. It ingests sensitive tabular, relational, or document datasets and mathematically generates "synthetic twins" that maintain the exact statistical properties and cross-column correlations of the original data, without leaking any private information.

## Core Features
- **Tabular & Relational Modeling**: Uses the Synthetic Data Vault (SDV) and Gaussian Copulas to learn the exact distributions and relationships across complex data tables.
- **Document Processing**: Parses nested OCR JSON data (such as invoices), structures it into relational tables, and synthesizes variations while strictly enforcing mathematical logic (e.g., ensuring quantity multiplied by price equals the subtotal).
- **Strict Privacy Validation**: Evaluates generated datasets in high-dimensional space using Distance to Closest Record (DCR). The system mathematically proves zero rows were memorized or leaked from the source data.
- **Absolute Sandbox Isolation**: The FastAPI backend dynamically provisions isolated sandbox directories for every project. It injects asynchronous Python subprocesses directly into these contexts, ensuring data from different projects never mixes.
- **Real-Time Streaming**: Streams live terminal execution logs directly to the React frontend using Server-Sent Events (SSE).

## Tech Stack
- **Frontend**: React.js, Vite
- **Backend Orchestrator**: Python, FastAPI, Uvicorn
- **Machine Learning**: PyTorch, SDV, Copulas, scikit-learn, sdmetrics
- **Visualizations**: Matplotlib, Seaborn

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mrismail-m/syndata.git
   cd syndata
   ```

2. **Set up the backend environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Build the frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Start the server:**
   ```bash
   # The FastAPI backend automatically serves the compiled React app
   python server.py
   ```
   Navigate to `http://localhost:8000` in your browser.

## Evaluation Metrics
The pipeline inherently evaluates the data it generates against industry-standard metrics:
- **Fidelity**: Kolmogorov-Smirnov (KS) statistic and Correlation Matrix Error.
- **Privacy**: Distance to Closest Record (5th Percentile) and exact-match leak detection.

## License
MIT License
