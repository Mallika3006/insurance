"""Streamlit UI for Indian Health Insurance Premium Prediction."""
import json

import streamlit as st

from src import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, EDA_OUTPUT_DIR, EVAL_OUTPUT_DIR, OUTPUTS_DIR
from src.model_training import train_and_save_pipeline
from src.prediction import load_trained_model, predict_insurance_charge

st.set_page_config(
    page_title="Indian Health Insurance Premium Prediction",
    page_icon="🇮🇳",
    layout="wide",
)
st.title("Indian Health Insurance Premium Prediction")
st.caption(
    "Educational ML estimate based on an Indian-oriented public dataset. "
    "Amounts are shown in INR (₹)."
)


@st.cache_resource
def get_model():
    # A fresh clone includes the Indian CSV but not generated model binaries.
    # Train once automatically on first prediction, then cache the loaded model.
    if not DEFAULT_MODEL_PATH.exists():
        if not DEFAULT_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Bundled Indian dataset is missing: {DEFAULT_DATA_PATH}"
            )
        with st.spinner(
            "Training the Indian health-insurance model for the first time. "
            "This may take a few minutes..."
        ):
            train_and_save_pipeline(run_eda_plots=False, quick_mode=True)
    return load_trained_model()


def calculate_bmi(height_cm, weight_kg):
    """Return BMI and a general adult BMI category."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)

    if bmi < 18.5:
        category = "Below the usual adult BMI range"
    elif bmi < 25:
        category = "Within the usual adult BMI range"
    elif bmi < 30:
        category = "Above the usual adult BMI range"
    else:
        category = "In the higher adult BMI range"

    return bmi, category


page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict Premium", "Model Insights"],
)

if page == "Home":
    st.subheader("Estimate an annual health insurance premium")
    st.write(
        "Enter all applicant details together. The application will calculate "
        "BMI, show a summary of the entered health details, and estimate an "
        "annual premium in one submission."
    )
    st.info(
        "This is a learning/portfolio project, not an insurer quote or financial "
        "advice. Actual premiums depend on insurer, plan, city, sum insured, "
        "exclusions and underwriting."
    )

elif page == "Predict Premium":
    st.subheader("Applicant details")
    st.write(
        "Complete the form once. BMI, your health details summary and the "
        "premium estimate will appear together after submission."
    )

    labels = {
        "diabetes": "Diabetes",
        "blood_pressure_problems": "Blood pressure problems",
        "any_transplants": "Any organ transplant",
        "any_chronic_diseases": "Any chronic disease",
        "known_allergies": "Known allergies",
        "family_cancer_history": "Family history of cancer",
    }

    with st.form("applicant_details_form"):
        st.markdown("#### Personal and body measurements")
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age (years)", 18, 100, 30)
        height = col2.number_input(
            "Height (cm)", min_value=120.0, max_value=220.0,
            value=165.0, step=1.0
        )
        weight = col3.number_input(
            "Weight (kg)", min_value=25.0, max_value=250.0,
            value=65.0, step=1.0
        )

        st.markdown("#### Health history")
        values = {}
        cols = st.columns(3)
        for i, (key, label) in enumerate(labels.items()):
            values[key] = cols[i % 3].selectbox(
                label, ["No", "Yes"], key=key
            )

        surgeries = st.selectbox(
            "Number of major surgeries", [0, 1, 2, 3, 4, 5]
        )

        submitted = st.form_submit_button(
            "Calculate BMI and predict premium", type="primary",
            use_container_width=True
        )

    if submitted:
        # Calculate BMI and present all results together after one submission.
        bmi, bmi_category = calculate_bmi(height, weight)

        st.divider()
        st.subheader("Your results")

        bmi_col, category_col, age_col = st.columns(3)
        bmi_col.metric("Calculated BMI", f"{bmi:.1f}")
        category_col.metric("General adult BMI range", bmi_category)
        age_col.metric("Age", f"{age} years")

        st.caption(
            "BMI is a general height-to-weight screening measure, not a "
            "medical diagnosis. Adult BMI ranges are general guidance."
        )

        st.markdown("#### Health details summary")
        summary_col1, summary_col2 = st.columns(2)
        summary_col1.write(f"**Height:** {height:.0f} cm")
        summary_col1.write(f"**Weight:** {weight:.1f} kg")
        summary_col2.write(f"**Major surgeries reported:** {surgeries}")

        reported_details = [
            label for key, label in labels.items() if values[key] == "Yes"
        ]
        if reported_details:
            st.write("**Health-history items selected:**")
            for detail in reported_details:
                st.write(f"- {detail}")
        else:
            st.write("No listed health-history items were selected.")

        st.caption(
            "This summary only repeats the information entered. It is not a "
            "medical diagnosis or medical risk assessment."
        )

        st.markdown("#### Estimated annual premium")
        try:
            result = predict_insurance_charge(
                age,
                values["diabetes"],
                values["blood_pressure_problems"],
                values["any_transplants"],
                values["any_chronic_diseases"],
                height,
                weight,
                values["known_allergies"],
                values["family_cancer_history"],
                surgeries,
                model=get_model(),
            )
            st.success(
                f"Estimated annual premium: "
                f"₹{result['estimated_annual_premium_inr']:,.0f}"
            )
            st.caption(
                "Model estimate only; not an official insurance quotation."
            )
        except Exception as exc:
            st.error(str(exc))

else:
    st.subheader("Model insights")
    metrics = OUTPUTS_DIR / "tuning_comparison.json"
    if metrics.exists():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        st.write("Selected model:", data.get("best_model_name", "Not available"))
        st.json(data.get("final_selected_metrics", data.get("after_tuning", {})))
    else:
        st.info("Train the model first using: python -m src.model_training")

    for folder in [EDA_OUTPUT_DIR, EVAL_OUTPUT_DIR]:
        if folder.exists():
            for image in sorted(folder.glob("*.png")):
                st.image(
                    str(image),
                    caption=image.stem.replace("_", " ").title(),
                    use_container_width=True,
                )
