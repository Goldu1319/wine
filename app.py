import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

# App title
st.title("🍷 Wine Quality Prediction App")
st.markdown("This app predicts the **quality of red wine** based on its chemical attributes using the **KNN algorithm**.")

# Load the dataset with correct separator
df = pd.read_csv("winequality-red.csv", sep=';')

# Strip any extra spaces from the column names
df.columns = df.columns.str.strip()

# Dataset info
with st.expander("📂 Preview Dataset"):
    st.write("Dataset Columns:", list(df.columns))
    st.dataframe(df.head())

# Check for 'quality' column
if 'quality' not in df.columns:
    st.error("❌ Error: 'quality' column not found in the dataset. Please check the column name.")
else:
    # Split features and target
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained with **{accuracy * 100:.2f}%** accuracy.")

    # Sidebar for user input
    st.sidebar.header("Enter Wine Attributes:")
    fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, step=0.1)
    volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, step=0.01)
    citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 1.5, step=0.01)
    residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 15.0, step=0.1)
    chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, step=0.001)
    free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, step=1.0)
    total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, step=1.0)
    density = st.sidebar.number_input("Density", 0.9900, 1.0050, step=0.0001, format="%.4f")
    pH = st.sidebar.number_input("pH", 2.5, 4.5, step=0.01)
    sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, step=0.01)
    alcohol = st.sidebar.number_input("Alcohol", 8.0, 15.0, step=0.1)

    # Prepare input data
    input_data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Predict quality
    prediction = model.predict(input_scaled)[0]

    st.subheader("🔍 Prediction Result")
    if prediction <= 4:
        st.error(f"Predicted Wine Quality: {prediction} (Poor Quality)")
    elif 5 <= prediction <= 6:
        st.warning(f"Predicted Wine Quality: {prediction} (Average Quality)")
    else:
        st.success(f"Predicted Wine Quality: {prediction} (Good Quality)")

    # Show the entered inputs
    with st.expander("📊 Your Input Summary"):
        st.dataframe(input_df)
