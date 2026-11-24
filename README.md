# 🛡️ DataGuard-Lite: AI Data Quality & Anomaly Detector

**Live App:** [Click here to try DataGuard-Lite live](YOUR_LIVE_STREAMLIT_URL)

## 💡 What is this project?
Big companies collect thousands of sales records every day, and messy mistakes—like missing prices, duplicate orders,
negative numbers, or a ₹45,000 bag of chips—can break their business reports.

**DataGuard-Lite** is a web app that automatically scans a messy spreadsheet, finds those hidden errors using 
Machine Learning, explains what went wrong in simple English using AI, and lets you download a clean version of the data in one click.

---

## ✨ Key Features

1. **Instant Data Health Check**
   - Upload your own CSV file or test the built-in messy retail dataset with one click.
   - Automatically finds **duplicate rows**, **missing values**, and **negative numbers** (like `-5` items sold).
   - Gives your dataset an overall **Data Health Score (0% to 100%)**.

2. **Smart Anomaly Detection (Unsupervised ML)**
   - Uses **Isolation Forest** (`scikit-learn`) to spot weird numbers that stand out from normal sales—without
   - needing pre-labeled examples of bad data.
   - Includes a **Sensitivity Slider** so you can control how strict the detector should be, plus an interactive
   -  chart showing normal rows (blue) vs. weird rows (red).

3. **AI Root-Cause & Auto-Fix Copilot (Google Gemini)**
   - Reads the errors found in your file and explains in plain words how they affect the business.
   - Automatically writes clean Python (`pandas`) code to fix the exact issues found in your dataset.
   - Includes automatic model fallback so the AI tab stays reliable even during heavy traffic.

4. **One-Click Clean Download**
   - Removes duplicates and bad outliers, fills missing numbers safely, and lets you download the ready-to-use `cleaned_sales_data.csv`.

---

## 🛠️ Tools & Technologies Used
- **Python** — Core programming language
- **Streamlit** — Interactive web dashboard
- **Scikit-Learn (Isolation Forest)** — Unsupervised Machine Learning for anomaly detection
- **Pandas & NumPy** — Data cleaning and processing
- **Plotly** — Interactive scatter charts
- **Google Gemini API** — Generative AI explanations and code generation

---

## 📁 Project Files
- `app.py` — Main dashboard interface, charts, and Gemini AI integration
- `ml_engine.py` — Data quality rules, Isolation Forest model, and auto-cleaning logic
- `sample_data.py` — Generates realistic retail sales data with injected errors for testing
- `requirements.txt` — List of Python libraries needed to run the app

---
