
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

DATA_PATH = Path("data/Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")
COMPARISON_PATH = Path("reports/model_comparison.csv")
EVALUATION_PATH = Path("reports/evaluation.json")

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Intelligence")
st.caption(
    "Machine learning dashboard for telecom customer retention analysis"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()
comparison = pd.read_csv(COMPARISON_PATH)

with open(EVALUATION_PATH, encoding="utf-8") as file:
    evaluation = json.load(file)

# Prepare the dataset for exploration.
df["ChurnLabel"] = df["Churn"].map({"No": 0, "Yes": 1})

total_customers = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = int((df["Churn"] == "No").sum())
churn_rate = churned / total_customers * 100

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page",
        [
            "Overview",
            "Customer Explorer",
            "Predict Churn",
            "Model Performance",
        ],
    )

    st.divider()
    st.caption("Dataset: IBM Telco Customer Churn")
    st.caption(f"Records: {total_customers:,}")
    st.caption(f"Selected model: {evaluation['best_model']}")


if page == "Overview":
    st.subheader("Business Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Customers Churned", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Distribution")
        counts = df["Churn"].value_counts().reindex(
            ["No", "Yes"], fill_value=0
        )
        fig, ax = plt.subplots()
        ax.bar(["Retained", "Churned"], counts.values)
        ax.set_ylabel("Number of customers")
        ax.set_title("Retained vs Churned")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Churn by Contract Type")
        contract_churn = (
            df.groupby("Contract", observed=False)["ChurnLabel"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots()
        ax.bar(contract_churn.index.astype(str), contract_churn.values)
        ax.set_ylabel("Churn rate (%)")
        ax.set_title("Churn Rate by Contract")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Customer Explorer":
    st.subheader("Customer Explorer")
    st.write("Filter the customer dataset to explore customer segments.")

    contract_options = sorted(df["Contract"].dropna().unique())
    selected_contracts = st.multiselect(
        "Contract type",
        contract_options,
        default=contract_options,
    )

    churn_filter = st.selectbox(
        "Churn status",
        ["All", "Churned", "Retained"],
    )

    tenure_range = st.slider(
        "Customer tenure (months)",
        min_value=0,
        max_value=int(df["tenure"].max()),
        value=(0, int(df["tenure"].max())),
    )

    filtered = df[
        df["Contract"].isin(selected_contracts)
        & df["tenure"].between(*tenure_range)
    ]

    if churn_filter == "Churned":
        filtered = filtered[filtered["Churn"] == "Yes"]
    elif churn_filter == "Retained":
        filtered = filtered[filtered["Churn"] == "No"]

    st.metric("Matching customers", f"{len(filtered):,}")
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "Download filtered customers",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
    )

elif page == "Predict Churn":
    st.subheader("Predict Churn for a Customer")
    st.write(
        "Enter a customer's details to estimate their churn probability. "
        "This is a model estimate, not a guarantee."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])

        with col2:
            multiple = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online security", ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online backup", ["Yes", "No", "No internet service"]
            )
            protection = st.selectbox(
                "Device protection", ["Yes", "No", "No internet service"]
            )
            tech = st.selectbox(
                "Tech support", ["Yes", "No", "No internet service"]
            )

        with col3:
            tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input(
                "Monthly charges ($)", 0.0, 200.0, 70.0
            )
            total = st.number_input(
                "Total charges ($)", 0.0, 10000.0, 840.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        # The model was trained with this feature order and names.
        probability = float(model.predict_proba(customer)[0, 1])
        prediction = int(model.predict(customer)[0])

        st.divider()
        a, b = st.columns(2)

        with a:
            st.metric("Estimated churn probability", f"{probability:.1%}")
            st.progress(probability)

        with b:
            if prediction == 1:
                st.error("Model prediction: Likely to churn")
            else:
                st.success("Model prediction: Likely to stay")

        if probability >= 0.70:
            st.warning(
                "High estimated churn probability. Consider reviewing "
                "customer feedback, service issues, and contract options."
            )
        elif probability >= 0.40:
            st.info(
                "Moderate estimated churn probability. Further review "
                "may be useful."
            )
        else:
            st.info(
                "Lower estimated churn probability. Continue monitoring "
                "customer satisfaction."
            )

        st.caption(
            "This model is trained on historical data and may be inaccurate "
            "for individual customers. Do not use this prediction as the "
            "sole basis for customer decisions."
        )

elif page == "Model Performance":
    st.subheader("Model Evaluation")

    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{evaluation['accuracy']:.3f}")
    b.metric("Precision", f"{evaluation['precision']:.3f}")
    c.metric("Recall", f"{evaluation['recall']:.3f}")
    d.metric("F1 Score", f"{evaluation['f1_score']:.3f}")
    e.metric("ROC-AUC", f"{evaluation['roc_auc']:.3f}")

    st.caption(
        f"Selected model: {evaluation['best_model']} | "
        f"Test records: {evaluation['test_size']:,}"
    )

    st.subheader("Compare All Models")
    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Confusion Matrix")
    cm = evaluation["confusion_matrix"]
    fig, ax = plt.subplots()
    image = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], labels=["Actual stay", "Actual churn"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(cm[row][col]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    report = pd.DataFrame(evaluation["classification_report"]).T
    st.dataframe(report, use_container_width=True)
