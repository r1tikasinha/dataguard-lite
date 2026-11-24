import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from sample_data import generate_messy_sales_data
from ml_engine import run_quality_audit

st.set_page_config(page_title="DataGuard-Lite | AI Data Quality Engine", layout="wide")

st.title("🛡️ DataGuard-Lite: AI Data Quality & Anomaly Engine")
st.caption("Automated Data Health Auditing using Unsupervised ML (Isolation Forest) + GenAI Root-Cause Copilot")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("1. Data Source")
    data_mode = st.radio("Choose input:", ["Use Built-in Messy Retail Dataset", "Upload Your Own CSV"])

    uploaded_file = None
    if data_mode == "Upload Your Own CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.header("2. ML Sensitivity")
    contamination = st.slider(
        "Anomaly Sensitivity (Contamination %)",
        min_value=0.01, max_value=0.15, value=0.03, step=0.01,
        help="Higher values flag more borderline rows as anomalies.",
    )

    st.header("3. GenAI Copilot Key")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get a free key at aistudio.google.com")

# --- LOAD DATA ---
if data_mode == "Upload Your Own CSV" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_messy_sales_data(500)

# --- RUN ML & QUALITY AUDIT ---
audit = run_quality_audit(df, contamination=contamination)

# --- TOP KPI METRICS ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Data Health Score", f"{audit['health_score']}%")
col2.metric("Total Rows Scanned", audit["total_rows"])
col3.metric("ML Anomalies Flagged", audit["anomaly_count"])
col4.metric("Duplicate Rows", audit["duplicate_rows"])
col5.metric("Missing Cells", audit["missing_cells"])

if audit["negative_by_col"]:
    st.warning(f"Negative values detected (rule-based check): {audit['negative_by_col']}")

st.divider()

# --- INTERACTIVE TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Anomaly Visualization", "🚨 Flagged Bad Rows", "🤖 AI Root-Cause & Auto-Fix Code"])

with tab1:
    st.subheader("Unsupervised ML Anomaly Detection (Isolation Forest)")
    num_cols = audit["numeric_cols"]
    if len(num_cols) >= 2:
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X-Axis Feature", num_cols, index=0)
        y_col = c2.selectbox("Y-Axis Feature", num_cols, index=min(1, len(num_cols) - 1))

        fig = px.scatter(
            audit["scored_df"],
            x=x_col,
            y=y_col,
            color="Status",
            color_discrete_map={"Normal": "#2E86C1", "Anomaly": "#E74C3C"},
            hover_data=df.columns.tolist(),
            title=f"Anomaly Clusters: {x_col} vs {y_col}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a dataset with at least 2 numeric columns to view the 2D anomaly scatter plot.")

with tab2:
    st.subheader("Rows Flagged by Isolation Forest")
    if len(audit["anomalies_df"]) > 0:
        st.dataframe(audit["anomalies_df"], use_container_width=True)
    else:
        st.success("No statistical anomalies detected at current sensitivity!")

    st.subheader("Download Auto-Cleaned Dataset")
    csv_clean = audit["clean_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv_clean,
        file_name="cleaned_sales_data.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("AI Data Engineer Copilot")
    if st.button("✨ Explain Errors & Generate Cleaning Code"):
        if not api_key:
            st.warning("Please paste your free Gemini API key in the left sidebar first.")
        else:
            with st.spinner("Analyzing anomalies and writing remediation pipeline..."):
                sample_anomalies = audit["anomalies_df"].head(5).to_string()
                prompt = f"""
                You are a Senior Data Engineer at Sigmoid Analytics. Analyze this data quality audit summary:
                - Total Rows: {audit['total_rows']}
                - Duplicate Rows: {audit['duplicate_rows']}
                - Missing Values by Column: {audit['missing_by_col']}
                - Negative Values by Column: {audit['negative_by_col']}
                - ML-Detected Anomalies ({audit['anomaly_count']} rows). Top examples:
                {sample_anomalies}

                Provide two clear sections:
                1. **Plain-English Business Impact**: Explain what went wrong in simple terms for a retail manager.
                2. **Production Python Cleaning Script**: Write clean pandas code to fix these exact issues.
                """
                # try:
                #     client = genai.Client(api_key=api_key)
                #     response = client.models.generate_content(
                #         # model="gemini-2.5-flash",
                #         model="gemini-3.8-flash",
                #         contents=prompt,
                #     )
                #     st.markdown(response.text)
                # except Exception as e:
                #     st.error(f"API Error: {e}")

                try:
                    client = genai.Client(api_key=api_key)
                    models_to_try = [
                        "gemini-3.8-flash",
                        "gemini-3.5-flash",
                        "gemini-3.5-flash-lite",
                    ]
                    response = None
                    last_error = None

                    for model_name in models_to_try:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt
                            )
                            st.caption(f"Generated using `{model_name}`")
                            break
                        except Exception as model_err:
                            last_error = model_err
                            continue

                    if response:
                        st.markdown(response.text)
                    else:
                        st.error(f"All fallback models are temporarily busy. Please wait 10 seconds and retry. Details: {last_error}")
                except Exception as e:
                    st.error(f"API Error: {e}")

