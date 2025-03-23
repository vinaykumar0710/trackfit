import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
warnings.filterwarnings('ignore')

# Apply Custom CSS for Styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Define Inline CSS for Background and User Input Section Colors
def set_page_style():
    st.markdown(
        """
        <style>
        /* Set main background color */
        .stApp {
            background-color: #f08080;
        }

        /* Style for error messages */
        .stError {
            color: yellow !important;
        }

        /* Style buttons */
        .stButton>button {
            background-color: pink !important;
            color: black !important;
        }

        /* Style for progress bar */
        .stProgressBar div[role="progressbar"] {
            background-color: pink !important;
        }

        /* Style for Sidebar (User Input Section) */
        .css-1d391kg {
            background-color: indianred !important; /* Set to #CD5C5C */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_page_style()

# Title and Description
st.write("## TrackFit")
st.write("In this WebApp, you can observe your predicted calories burned based on parameters such as `Age`, `Gender`, `BMI`, etc.")

# Sidebar for User Input
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 20)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 60, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

# Display User Inputs
st.write("---")
st.header("Your Parameters")
def update_progress():
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
update_progress()
st.write(df)

# Load and Preprocess Data
try:
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
except FileNotFoundError:
    st.error("Error: Required CSV files not found!")
    st.stop()

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the Model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

# Display Prediction
st.write("---")
st.header("Prediction")
update_progress()
st.write(f"{round(prediction[0], 2)} **kilocalories**")

# Save Prediction Option
if st.button("Save Prediction"):
    prediction_data = pd.DataFrame({"Predicted Calories": [round(prediction[0], 2)]})
    prediction_data.to_csv("prediction.csv", index=False)
    st.success("Prediction saved as prediction.csv!")

# Display Similar Results
st.write("---")
st.header("Similar Results")
update_progress()
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

# General Insights
st.write("---")
st.header("General Information")
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write(f"You are older than {round(sum(boolean_age) / len(boolean_age), 2) * 100}% of other people.")
st.write(f"Your exercise duration is longer than {round(sum(boolean_duration) / len(boolean_duration), 2) * 100}% of other people.")
st.write(f"Your heart rate is higher than {round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}% of other people during exercise.")
st.write(f"Your body temperature is higher than {round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}% of other people during exercise.")