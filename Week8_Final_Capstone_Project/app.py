
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Zynxis Internship Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("zynxis_performance_model.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎓 Zynxis AI / ML Internship Performance Predictor")

st.markdown(
    """
    ### AI-Based Internship Performance Analysis

    Enter your performance information for all **8 internship tasks**.
    The trained machine learning model will predict your overall score
    and provide personalized performance suggestions.
    """
)

st.divider()

# --------------------------------------------------
# TASK SCORES
# --------------------------------------------------

st.subheader("📊 Internship Task Scores")

col1, col2 = st.columns(2)

with col1:
    week1 = st.slider("Week 1 — Python & EDA", 0, 100, 75)
    week2 = st.slider("Week 2 — Data Preprocessing", 0, 100, 75)
    week3 = st.slider("Week 3 — Classification Model", 0, 100, 75)
    week4 = st.slider("Week 4 — Regression Model", 0, 100, 75)

with col2:
    week5 = st.slider("Week 5 — Clustering", 0, 100, 75)
    week6 = st.slider("Week 6 — NLP", 0, 100, 75)
    week7 = st.slider("Week 7 — Model Deployment", 0, 100, 75)
    week8 = st.slider("Week 8 — Final Capstone Project", 0, 100, 75)

st.divider()

# --------------------------------------------------
# SUBMISSION INFORMATION
# --------------------------------------------------

st.subheader("📋 Submission Information")

col1, col2, col3 = st.columns(3)

with col1:
    tasks_delivered = st.number_input(
        "Tasks Delivered",
        min_value=0,
        max_value=8,
        value=8
    )

with col2:
    on_time = st.number_input(
        "On-Time Submissions",
        min_value=0,
        max_value=8,
        value=8
    )

with col3:
    late = st.number_input(
        "Late Submissions",
        min_value=0,
        max_value=8,
        value=0
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🚀 Predict My Performance", use_container_width=True):

    # Feature data
    input_data = pd.DataFrame([{
        "Week1_Python_EDA_Score": week1,
        "Week2_Data_Preprocessing_Score": week2,
        "Week3_Classification_Model_Score": week3,
        "Week4_Regression_Model_Score": week4,
        "Week5_Clustering_Score": week5,
        "Week6_NLP_Score": week6,
        "Week7_Model_Deployment_Score": week7,
        "Week8_Final_Capstone_Project_Score": week8,
        "Tasks_Delivered": tasks_delivered,
        "OnTime_Submissions": on_time,
        "Late_Submissions": late
    }])

    # Prediction
    predicted_score = model.predict(input_data)[0]

    predicted_score = max(0, min(100, predicted_score))

    # Average score
    task_scores = [
        week1, week2, week3, week4,
        week5, week6, week7, week8
    ]

    average_score = np.mean(task_scores)

    # On-time rate
    on_time_rate = (on_time / 8) * 100

    # Performance level
    if predicted_score >= 85:
        performance = "Excellent 🟢"
    elif predicted_score >= 75:
        performance = "Good 🔵"
    elif predicted_score >= 65:
        performance = "Average 🟡"
    else:
        performance = "Needs Improvement 🔴"

    # --------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------

    st.subheader("🎯 Performance Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average Task Score",
            f"{average_score:.1f}"
        )

    with col2:
        st.metric(
            "Tasks Delivered",
            f"{tasks_delivered}/8"
        )

    with col3:
        st.metric(
            "On-Time Rate",
            f"{on_time_rate:.1f}%"
        )

    with col4:
        st.metric(
            "Predicted Score",
            f"{predicted_score:.2f}"
        )

    st.success(
        f"### Performance Level: {performance}"
    )

    # --------------------------------------------------
    # 8-TASK CHART
    # --------------------------------------------------

    st.subheader("📈 8-Task Performance")

    chart_data = pd.DataFrame({
        "Task": [
            "Python & EDA",
            "Preprocessing",
            "Classification",
            "Regression",
            "Clustering",
            "NLP",
            "Deployment",
            "Capstone"
        ],
        "Score": task_scores
    })

    st.bar_chart(
        chart_data.set_index("Task")
    )

    # --------------------------------------------------
    # DELIVERABLE STATUS
    # --------------------------------------------------

    st.subheader("📦 Deliverable Completion")

    if tasks_delivered == 8:
        st.success("✅ All 8 internship tasks have been delivered!")
    else:
        st.warning(
            f"⚠️ {tasks_delivered}/8 tasks delivered. "
            "Complete the remaining tasks."
        )

    # --------------------------------------------------
    # PERSONALIZED SUGGESTIONS
    # --------------------------------------------------

    st.subheader("💡 Personalized Suggestions")

    suggestions = []

    task_dictionary = {
        "Python & EDA": week1,
        "Data Preprocessing": week2,
        "Classification": week3,
        "Regression": week4,
        "Clustering": week5,
        "NLP": week6,
        "Model Deployment": week7,
        "Final Capstone": week8
    }

    weak_tasks = [
        task for task, score in task_dictionary.items()
        if score < 70
    ]

    if weak_tasks:
        suggestions.append(
            "Focus on improving: " +
            ", ".join(weak_tasks)
        )

    if on_time < 6:
        suggestions.append(
            "Try to improve your on-time submission rate."
        )

    if tasks_delivered < 8:
        suggestions.append(
            "Complete all 8 internship tasks."
        )

    if not suggestions:
        suggestions.append(
            "Excellent work! Keep maintaining your performance."
        )

    for suggestion in suggestions:
        st.write("•", suggestion)

    # --------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------

    st.subheader("🤖 ML Model Information")

    st.info(
        """
        **Model:** Linear Regression

        **MAE:** 1.6014

        **MSE:** 4.0229

        **R² Score:** 0.8400

        The model was trained using internship task scores,
        task completion information, and submission behavior.
        """
    )

    # --------------------------------------------------
    # DOWNLOADABLE REPORT
    # --------------------------------------------------

    report = f"""
ZYNXIS INTERNSHIP PERFORMANCE REPORT
====================================

Average Task Score: {average_score:.2f}
Tasks Delivered: {tasks_delivered}/8
On-Time Submission Rate: {on_time_rate:.2f}%
Predicted Overall Score: {predicted_score:.2f}
Performance Level: {performance}

TASK SCORES
-----------
Python & EDA: {week1}
Data Preprocessing: {week2}
Classification: {week3}
Regression: {week4}
Clustering: {week5}
NLP: {week6}
Model Deployment: {week7}
Final Capstone: {week8}

MODEL
-----
Linear Regression
MAE: 1.6014
MSE: 4.0229
R2 Score: 0.8400
"""

    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="zynxis_performance_report.txt",
        mime="text/plain"
    )
