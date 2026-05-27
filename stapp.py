import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

# Page config
st.set_page_config(
    page_title="AI Cybersecurity Threat Detection",
    layout="wide"
)

# Sidebar
st.sidebar.title("Cybersecurity Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Upload CSV",
        "Live Prediction",
        "Threat Analytics"
    ]
)

# Home page
if menu == "Home":

    st.title("AI-Powered Cybersecurity Threat Detection")

    st.markdown("""
    ### Features
    - Upload network traffic CSV
    - Detect cyber threats
    - Analyze attacks
    - Visualize traffic
    - Real-time predictions
    """)

# Upload CSV
elif menu == "Upload CSV":

    st.title("Cybersecurity CSV Threat Analysis")

    uploaded_files = st.file_uploader(
        "Upload CSV Files",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files is not None and len(uploaded_files) > 0:

        try:

            dataframes = []

            progress = st.progress(0)

            total_files = len(uploaded_files)

            # Read each file
            for index, file in enumerate(uploaded_files):

                df = pd.read_csv(
                    file,
                    low_memory=False
                )

                # Clean column names
                df.columns = df.columns.str.strip()

                dataframes.append(df)

                progress.progress(
                    (index + 1) / total_files
                )

            # Merge all files
            combined_df = pd.concat(
                dataframes,
                ignore_index=True
            )

            st.success(
                f"{total_files} files uploaded successfully"
            )

            # Dataset info
            st.subheader("Dataset Information")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Rows",
                combined_df.shape[0]
            )

            col2.metric(
                "Columns",
                combined_df.shape[1]
            )

            col3.metric(
                "Files",
                total_files
            )

            # Preview
            st.subheader("Dataset Preview")

            st.dataframe(
                combined_df.head()
            )

            # Threat distribution
            if "Label" in combined_df.columns:

                st.subheader(
                    "Threat Distribution"
                )

                attack_counts = (
                    combined_df["Label"]
                    .value_counts()
                    .reset_index()
                )

                attack_counts.columns = [
                    "Attack Type",
                    "Count"
                ]

                # Bar chart
                fig = px.bar(
                    attack_counts,
                    x="Attack Type",
                    y="Count",
                    title="Cyber Threat Counts"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Pie chart
                fig2 = px.pie(
                    attack_counts,
                    names="Attack Type",
                    values="Count",
                    title="Threat Distribution"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

                # Threat table
                st.subheader(
                    "Attack Summary"
                )

                st.dataframe(
                    attack_counts
                )

            # Download merged CSV
            csv = combined_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Merged CSV",
                data=csv,
                file_name="merged_dataset.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(str(e))

# Live prediction
elif menu == "Live Prediction":

    st.title("Real-Time Threat Prediction")

    st.write("Generate random network traffic sample")

    if st.button("Predict Threat"):

        try:

            # Generate 78 random features
            features = np.random.rand(78).tolist()

            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={
                    "features": features
                }
            )

            result = response.json()

            prediction = result.get(
                "prediction",
                "Unknown"
            )

            # Threat result
            if prediction == "BENIGN":

                st.success(
                    f"Prediction: {prediction}"
                )

            else:

                st.error(
                    f"Threat Detected: {prediction}"
                )

            # Show sample values
            st.subheader("Generated Features")

            st.write(features)

        except Exception as e:

            st.error(str(e))
# Threat analytics
elif menu == "Threat Analytics":

    st.title("Threat Analytics Dashboard")

    # Simulated attack data
    attack_types = [
        "BENIGN",
        "DDoS",
        "PortScan",
        "Bot",
        "Brute Force"
    ]

    counts = np.random.randint(
        10,
        500,
        size=5
    )

    analytics_df = pd.DataFrame({
        "Attack": attack_types,
        "Count": counts
    })

    # Bar chart
    st.subheader("Attack Counts")

    fig = px.bar(
        analytics_df,
        x="Attack",
        y="Count"
    )

    st.plotly_chart(fig)

    # Pie chart
    st.subheader("Attack Distribution")

    fig2 = px.pie(
        analytics_df,
        names="Attack",
        values="Count"
    )

    st.plotly_chart(fig2)

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Threats",
        int(np.sum(counts))
    )

    col2.metric(
        "Highest Attack",
        attack_types[np.argmax(counts)]
    )

    col3.metric(
        "Attack Types",
        len(attack_types)
    )