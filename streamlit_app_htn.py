import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ========== Load Models and Scaler ==========
with open("input_vars_htn7param.json","r") as f:
    selected_vars = json.load(f)
continuous_vars = ['age','dbp','sbp','bmi','energy','protein']
cat_vars = ['educationyears']


edu_levels = [1,2,3,4,5] 

# Load scaler
scaler = joblib.load("scaler_htn7param.pkl")

# List of all models to display
model_names = [
    'LogisticRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting', 
    'SVM', 'NeuralNet'
]


if os.path.exists("model_XGBoost_htn7param.pkl"):
    model_names.append("XGBoost")
if os.path.exists("model_LightGBM_htn7param.pkl"):
    model_names.append("LightGBM")

ensemble_names = []
if os.path.exists("model_Stacking_htn7param.pkl"):
    ensemble_names.append("Stacking")
if os.path.exists("model_VotingSoft_htn7param.pkl"):
    ensemble_names.append("VotingSoft")

# Load all models in dictionary
models = {}
for name in model_names + ensemble_names:
    models[name] = joblib.load(f"model_{name}_htn7param.pkl")

# ========== Helper - Preprocessing/New Sample ==========
def preprocess_sample(form_dict):
    # Create DataFrame from user input
    df = pd.DataFrame([form_dict])

    # Standardize and clip continuous variables
    df[continuous_vars] = scaler.transform(df[continuous_vars])
    df[continuous_vars] = df[continuous_vars].clip(-3, 3)

    # One-hot educationyears
   
    for lev in edu_levels[1:]:
        df[f"edu_{lev}"] = (df["educationyears"] == lev).astype(int)
    # Remove original educationyears column
    df = df.drop(columns=["educationyears"])
    # Add missing columns as zeros for robustness
    expected_cols = [v for v in continuous_vars] + [f"edu_{lev}" for lev in edu_levels[1:]]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]
    return df

def categorize(prob):
    if prob < 0.10: return 'Low'
    elif prob < 0.20: return 'Medium'
    else: return 'High'

# ========== Streamlit UI ==========

st.set_page_config(page_title="Hypertension Risk Prediction", layout="centered")
st.title("Hypertension Risk Calculator")
st.caption("Research model for incident hypertension risk prediction using machine learning. \n\n **Developed by Parsa Amirian M.D.**")

st.write("## Input Your Parameters")
with st.form(key="htnml_form"):
    age = st.number_input("Age (years)", min_value=16, max_value=99)
    dbp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=180)
    sbp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=260)
    bmi = st.number_input("BMI (kg/m²)", min_value=13.0, max_value=70.0)
    energy = st.number_input("Daily Energy Intake (kcal)", min_value=400, max_value=6000)
    protein = st.number_input("Daily Protein Intake (grams)", min_value=0, max_value=400)
    education_label, education_value = st.selectbox(
        "Education Level Attained (years)", 
        options=[
            ("<9th grade", 1),
            ("9-11 years",2),
            ("High school graduate/GED",3),
            ("Some college/Associate's",4),
            ("Bachelor's or beyond",5)
        ], format_func=lambda x: x[0]
    )
    submit = st.form_submit_button("Predict Risk")
if submit:
    user_input = {
        "age": age, "dbp": dbp, "sbp": sbp, "bmi": bmi, "energy": energy, "protein": protein, "educationyears": education_value
    }
    X_input = preprocess_sample(user_input)

    st.write("## Model Predictions")
    st.caption("All models shown were externally validated (NHANES). Probability = risk of incident hypertension.")
    results = []
    for name in model_names + ensemble_names:
        model = models[name]
       
        try:
            prob = float(model.predict_proba(X_input)[0,1])
        except AttributeError:
            # For SVM/others that lack predict_proba
            df = model.decision_function(X_input)
            df = (df - df.min())/(df.max()-df.min()) if df.max()!=df.min() else np.zeros_like(df)
            prob = float(df[0])
        results.append({
            "Model": name,
            "Risk Probability (%)": f"{prob*100:.1f}",
            "Category": categorize(prob)
        })

    st.dataframe(pd.DataFrame(results).set_index("Model"))

    st.info("**Risk categories are illustrative, not clinical thresholds. Discuss concerns with your healthcare provider.**")
    st.caption("Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, Neural Net, XGBoost, LightGBM, Stacking, VotingSoft Ensemble. Developed by Parsa Amirian M.D.")

with st.expander("Methodology/Model Details"):
    st.markdown("""
    - **Features:** Age, diastolic/systolic BP, BMI, energy/protein intake, education level.
    - **All models** were externally validated with US NHANES data.
    - **Ensemble** models (Stacking and Voting) aggregate the predictions of all base models.
    - Each model is trained using cross-validated and externally validated pipeline.
    - **Education** mapped to: 1=<9th, 2=9-11, 3=HS/GED, 4=Some college/AA, 5=BA/BS+.
    """)

st.write("—")

st.write("Contact: [Parsa Amirian](mailto:parsapj@gmail.com)")
