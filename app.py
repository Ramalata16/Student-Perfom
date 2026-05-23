import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')

# -----------------------------
# Mappings (UI → numeric)
# -----------------------------
gender_map = {"Male": 0, "Female": 1}

education_map = {
    "Did not finish school": 0,
    "Finished high school": 1,
    "College / TVET qualification": 2,
    "University degree": 3,
    "Postgraduate degree": 4
}

binary_map = {"No": 0, "Yes": 1}

attendance_map = {
    "Rarely attends": 0,
    "Sometimes attends": 1,
    "Mostly attends": 2,
    "Always attends": 3
}

# -----------------------------
# UI Layout
# -----------------------------
st.set_page_config(page_title="Student Predictor", layout="centered")

st.markdown(
    "<h1 style='text-align:center;'>🎓 Student Performance Predictor</h1>",
    unsafe_allow_html=True
)

st.write("Predict whether a student will PASS or FAIL based on daily habits and background factors.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", list(gender_map.keys()))
    age = st.slider("Age", 15, 30, 18)
    study_hours_per_day = st.slider("Study Hours per Day", 0, 12, 2)
    attendance = st.selectbox("Attendance", list(attendance_map.keys()))

with col2:
    parent_education = st.selectbox("Parent Education", list(education_map.keys()))
    internet = st.selectbox("Internet Access", list(binary_map.keys()))
    extracurricular = st.selectbox("Extracurricular Activities", list(binary_map.keys()))
    previous_score = st.slider("Previous Score", 0, 100, 50)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Result 🚀"):

    # Convert daily → weekly
    study_hours_per_week = study_hours_per_day * 7

    # Feature array
    features = np.array([[
        gender_map[gender],
        age,
        study_hours_per_week,
        attendance_map[attendance],
        education_map[parent_education],
        binary_map[internet],
        binary_map[extracurricular],
        previous_score
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    # -----------------------------
    # Output logic
    # -----------------------------
    st.subheader("Prediction Result")

    if probability >= 0.75:
        st.success(f"🟢 High chance of PASS ({probability:.2f})")
    elif probability >= 0.5:
        st.warning(f"🟡 Moderate risk ({probability:.2f})")
    else:
        st.error(f"🔴 High chance of FAIL ({1 - probability:.2f})")

    st.info(f"Probability of Passing: {probability:.2f}")
    st.info(f"Probability of Failing: {1 - probability:.2f}")

    # -----------------------------
    # Simple insight
    # -----------------------------
    st.subheader("Key Insight")
    st.write("Study habits (study hours, attendance, and previous scores) are the strongest indicators of performance.")