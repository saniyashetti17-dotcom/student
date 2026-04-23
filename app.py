import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import urllib.parse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Student App", layout="wide")

st.title("🎓 Student Performance Prediction System")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("🔐 Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful")
        else:
            st.error("Invalid Credentials")

    st.stop()

st.sidebar.title("📊 Project Info")
st.sidebar.write("Student Prediction System using ML")

@st.cache_resource
def get_collection():
    username = "saniya6user"
    password = urllib.parse.quote_plus("saniya703033")
    uri = f"mongodb+srv://{username}:{password}@cluster0.ym77vo2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    db = client["student_db"]
    collection = db["students"]
    return collection

@st.cache_data
def load_data():
    collection = get_collection()
    data = list(collection.find())
    df = pd.DataFrame(data)

    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    df_model = df[['lectures_attended','midterm_marks','previous_gpa','final_marks']].dropna()
    return df_model

@st.cache_resource
def train_model():
    df_model = load_data()

    X = df_model[['lectures_attended','midterm_marks','previous_gpa']]
    y = df_model['final_marks']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    accuracy = model.score(X_test, y_test)

    return model, r2, rmse, accuracy, df_model

model, r2, rmse, accuracy, df_model = train_model()

st.write("### 📈 Model Performance")
col1, col2, col3 = st.columns(3)

with col1:
    st.success(f"R² Score: {r2:.2f}")
with col2:
    st.info(f"RMSE: {rmse:.2f}")
with col3:
    st.warning(f"Accuracy: {accuracy:.2f}")

st.write("### ✏ Enter Student Details")
c1, c2, c3 = st.columns(3)

with c1:
    attendance = st.number_input("Lectures", 0, 12, 8)
with c2:
    midterm = st.number_input("Midterm", 0.0, 30.0, 20.0)
with c3:
    gpa = st.number_input("GPA", 0.0, 4.0, 2.5)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔍 Predict"):
    sample = [[attendance, midterm, gpa]]
    prediction = model.predict(sample)[0]

    st.success(f"Predicted Marks: {prediction:.2f}")

    if prediction >= 35:
        st.success("PASS")
    else:
        st.error("FAIL")

    if prediction >= 50:
        grade = "A"
    elif prediction >= 40:
        grade = "B"
    elif prediction >= 35:
        grade = "C"
    else:
        grade = "Fail"

    st.info(f"Grade: {grade}")

    if prediction < 25:
        st.error("🔴 Risk Level: High")
    elif prediction < 35:
        st.warning("🟠 Risk Level: Medium")
    else:
        st.success("🟢 Risk Level: Low")

    st.write("### 📌 Suggestions")

    if attendance < 6:
        st.write("- Increase lecture attendance")
    if midterm < 15:
        st.write("- Improve midterm preparation")
    if gpa < 2.0:
        st.write("- Focus on overall academics")
    if attendance >= 6 and midterm >= 15 and gpa >= 2.0:
        st.write("- Good performance, keep it up")

    st.session_state.history.append({
        "Attendance": attendance,
        "Midterm": midterm,
        "GPA": gpa,
        "Prediction": round(prediction, 2)
    })

if st.session_state.history:
    st.write("### 🕘 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

st.write("### 📊 Graph")
fig, ax = plt.subplots()
ax.scatter(df_model["midterm_marks"], df_model["final_marks"])
ax.set_xlabel("Midterm")
ax.set_ylabel("Final")
st.pyplot(fig)