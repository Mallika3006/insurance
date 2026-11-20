"""Streamlit UI for Indian Health Insurance Premium Prediction."""
import json
from pathlib import Path
import streamlit as st
from src import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, EDA_OUTPUT_DIR, EVAL_OUTPUT_DIR, OUTPUTS_DIR
from src.model_training import train_and_save_pipeline
from src.prediction import load_trained_model, predict_insurance_charge

st.set_page_config(page_title="Indian Health Insurance Premium Prediction",page_icon="🇮🇳",layout="wide")
st.title("Indian Health Insurance Premium Prediction")
st.caption("Educational ML estimate based on an Indian-oriented public dataset. Amounts are shown in INR (₹).")

@st.cache_resource
def get_model():
    # A fresh clone includes the Indian CSV but not generated model binaries.
    # Train once automatically on first prediction, then cache the loaded model.
    if not DEFAULT_MODEL_PATH.exists():
        if not DEFAULT_DATA_PATH.exists():
            raise FileNotFoundError(f"Bundled Indian dataset is missing: {DEFAULT_DATA_PATH}")
        with st.spinner("Training the Indian health-insurance model for the first time. This may take a few minutes..."):
            train_and_save_pipeline()
    return load_trained_model()

page=st.sidebar.radio("Navigation",["Home","Predict Premium","Model Insights"])
if page=="Home":
    st.subheader("Estimate an annual health insurance premium")
    st.write("This model uses age, medical history, height, weight and surgery history to estimate a premium in Indian rupees.")
    st.info("This is a learning/portfolio project, not an insurer quote or financial advice. Actual premiums depend on insurer, plan, city, sum insured, exclusions and underwriting.")
elif page=="Predict Premium":
    st.subheader("Enter applicant details")
    with st.form("premium_form"):
        c1,c2,c3=st.columns(3)
        age=c1.number_input("Age (years)",18,100,30)
        height=c2.number_input("Height (cm)",120.0,220.0,165.0)
        weight=c3.number_input("Weight (kg)",25.0,250.0,65.0)
        labels={"diabetes":"Diabetes","blood_pressure_problems":"Blood pressure problems","any_transplants":"Any organ transplant","any_chronic_diseases":"Any chronic disease","known_allergies":"Known allergies","family_cancer_history":"Family history of cancer"}
        values={}
        cols=st.columns(3)
        for i,(key,label) in enumerate(labels.items()):
            values[key]=cols[i%3].selectbox(label,["No","Yes"],key=key)
        surgeries=st.selectbox("Number of major surgeries",[0,1,2,3,4,5])
        submitted=st.form_submit_button("Predict annual premium",type="primary")
    if submitted:
        try:
            result=predict_insurance_charge(age,values["diabetes"],values["blood_pressure_problems"],
                values["any_transplants"],values["any_chronic_diseases"],height,weight,
                values["known_allergies"],values["family_cancer_history"],surgeries,model=get_model())
            st.success(f"Estimated annual premium: ₹{result['estimated_annual_premium_inr']:,.0f}")
            st.caption("Model estimate only; not an official insurance quotation.")
        except Exception as exc:
            st.error(str(exc))
else:
    st.subheader("Model insights")
    metrics=OUTPUTS_DIR/"tuning_comparison.json"
    if metrics.exists():
        data=json.loads(metrics.read_text(encoding="utf-8"))
        st.write("Selected model:",data.get("best_model_name","Not available"))
        st.json(data.get("final_selected_metrics",data.get("after_tuning",{})))
    else:
        st.info("Train the model first using: python -m src.model_training")
    for folder in [EDA_OUTPUT_DIR,EVAL_OUTPUT_DIR]:
        if folder.exists():
            for image in sorted(folder.glob("*.png")):
                st.image(str(image),caption=image.stem.replace("_"," ").title(),use_container_width=True)
