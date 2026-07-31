import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

from main import main

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Runs the ML pipeline only once and caches the results.
    """
    return main()

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Airport Luggage Jam Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

.main{
    background-color:#0b1220;
    color:white;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

.metric-card{
    background:#1b263b;
    padding:18px;
    border-radius:12px;
    border:1px solid #2dd4bf;
    box-shadow:0 0 10px rgba(45,212,191,.3);
}

.title{
    text-align:center;
    color:#2dd4bf;
    font-size:42px;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:#94a3b8;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================
defaults = {
    "dataset": None,
    "model": None,
    "trained": False,
    "prediction": None,
    "accuracy": None,
    "results": None,
    "metrics": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("✈️ Airport Luggage Jam")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Analysis",
        "🤖 Model Training",
        "📈 Evaluation",
        "🔮 Prediction",
        "ℹ️ About"
    ]
)

# ==========================================================
# IMAGE DIRECTORY
# ==========================================================
IMAGE_DIR = Path("images")

# ==========================================================
# HOME PAGE
# ==========================================================
def home():

    st.markdown(
        "<h1 class='title'>Airport Luggage Jam Prediction</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>Industry 4.0 Intelligent Decision Support Dashboard</p>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset", "Loaded" if st.session_state.dataset is not None else "Not Loaded")
    c2.metric("Model", "Ready" if st.session_state.trained else "Not Trained")
    c3.metric("Prediction", "-" if st.session_state.prediction is None else st.session_state.prediction)
    c4.metric("Accuracy", "-" if st.session_state.accuracy is None else f"{st.session_state.accuracy:.2%}")

    st.divider()

st.subheader("Project Overview")

st.markdown("""
### ✈️ Airport Luggage Jam Prediction System

This dashboard predicts potential luggage jams at airport baggage transfer
points using a Random Forest Machine Learning model.

### Objectives

- Detect possible luggage jams before they occur
- Analyse the luggage dataset
- Train and evaluate the prediction model
- Visualize model performance
- Predict jam events on unseen test data

### Workflow

1. Upload the luggage dataset
2. Train the Random Forest model
3. Evaluate model performance
4. View predictions on the test dataset

### Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

### Industry 4.0 Relevance

The system demonstrates predictive maintenance and intelligent decision
support for automated airport baggage handling systems.
""")
st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("Dataset Information")

    if st.session_state.results:

        st.write(
            f"**Samples:** {st.session_state.results['dataset_shape'][0]}"
        )

        st.write(
            f"**Features:** {st.session_state.results['dataset_shape'][1]}"
        )

with col2:

    st.subheader("Model")

    if st.session_state.trained:

        st.success("Random Forest Trained")

        st.write(
            f"Accuracy: {st.session_state.results['metrics']['accuracy']:.2%}"
        )

        st.write(
            f"ROC AUC: {st.session_state.results['metrics']['roc_auc']:.3f}"
        )
    # ==========================================================
# DATASET ANALYSIS PAGE
# ==========================================================
def dataset_analysis():

    st.header("📊 Dataset Analysis")

    uploaded_file = st.file_uploader(
        "Upload Airport Luggage Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.session_state.dataset = df

    if st.session_state.dataset is None:

        st.info("Please upload a CSV dataset to continue.")

        return

    df = st.session_state.dataset

    st.success(f"Dataset Loaded Successfully ({df.shape[0]} rows × {df.shape[1]} columns)")

    # ------------------------------------------------------
    # DATA PREVIEW
    # ------------------------------------------------------
    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # ------------------------------------------------------
    # DATA SUMMARY
    # ------------------------------------------------------
    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True
    )

    # ------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------
    st.subheader("Missing Values")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        st.success("No Missing Values Found ✅")

    else:

        st.dataframe(
            missing.to_frame("Missing Count"),
            use_container_width=True
        )

    # ------------------------------------------------------
    # FEATURE DISTRIBUTION
    # ------------------------------------------------------
    st.subheader("Feature Distribution")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns):

        feature = st.selectbox(
            "Select Numerical Feature",
            numeric_columns
        )

        st.bar_chart(
            df[feature].value_counts().sort_index()
        )

    # ------------------------------------------------------
    # CORRELATION MATRIX
    # ------------------------------------------------------
    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    st.dataframe(
        corr.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

    # ------------------------------------------------------
    # DISPLAY SAVED IMAGES
    # ------------------------------------------------------
    st.subheader("Generated Analysis Plots")

    if IMAGE_DIR.exists():

        images = sorted(
            IMAGE_DIR.glob("*")
        )

        if len(images):

            cols = st.columns(2)

            for i, image in enumerate(images):

                with cols[i % 2]:

                    st.image(
                        str(image),
                        caption=image.stem.replace("_", " "),
                        use_container_width=True
                    )

        else:

            st.warning("No images found inside images/ folder.")

    else:

        st.warning("images/ directory does not exist.")

# ==========================================================
# MODEL TRAINING PAGE
# ==========================================================
def model_training():

    st.header("🤖 Model Training")

    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
        return

    st.write("Train the Random Forest model using the uploaded dataset.")

    if st.button("🚀 Train Model", use_container_width=True):

        with st.spinner("Training model..."):

            results = load_pipeline()

            st.session_state.results = results
            st.session_state.model = results["model"]
            st.session_state.metrics = results["metrics"]
            st.session_state.trained = True
            st.session_state.accuracy = results["metrics"]["accuracy"]

        st.success("✅ Model trained successfully!")

        col1, col2 = st.columns(2)

        col1.metric(
            "Accuracy",
            f"{results['metrics']['accuracy']:.2%}"
        )

        col2.metric(
            "ROC AUC",
            f"{results['metrics']['roc_auc']:.3f}"
        )
        # ==========================================================
# EVALUATION PAGE
# ==========================================================
def evaluation():

    st.header("📈 Model Evaluation")

    if not st.session_state.trained:

        st.warning("Please train the model first.")
        return

    metrics = st.session_state.results["metrics"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "F1 Score",
        f"{metrics['f1_score']:.2%}"
    )

    col5.metric(
        "ROC AUC",
        f"{metrics['roc_auc']:.3f}"
    )

    col6.metric(
        "PR AUC",
        f"{metrics['average_precision']:.3f}"
    )

    st.divider()

    st.subheader("Model Evaluation Plots")

    evaluation_images = [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "feature_importance.png",
        "class_distribution.png",
        "correlation_heatmap.png",
        "feature_distributions.png"
    ]

    cols = st.columns(2)

    for i, image in enumerate(evaluation_images):

        path = IMAGE_DIR / image

        if path.exists():

            with cols[i % 2]:

                st.image(
                    str(path),
                    caption=image.replace("_", " ").replace(".png", "").title(),
                    use_container_width=True
                )

                # ==========================================================
# PREDICTION PAGE
# ==========================================================
def prediction():

    st.header("🔮 Prediction")

    if not st.session_state.trained:
        st.warning("Please train the model first.")
        return

    results = st.session_state.results

    model = results["model"]
    X_test = results["X_test"]
    y_test = results["y_test"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    prediction_df = X_test.copy()
    prediction_df["Actual"] = y_test.values
    prediction_df["Predicted"] = y_pred
    prediction_df["Confidence"] = y_prob

    col1, col2, col3 = st.columns(3)

    col1.metric("Test Samples", len(prediction_df))
    col2.metric("Predicted Jams", int((y_pred == 1).sum()))
    col3.metric("Predicted Normal", int((y_pred == 0).sum()))

    st.divider()

    st.dataframe(
        prediction_df,
        use_container_width=True
    )

    csv = prediction_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )

    # ==========================================================
# ABOUT PAGE
# ==========================================================
def about():

    st.header("ℹ️ About")

    st.write("""
    ## Airport Luggage Jam Prediction

    Design for Industry 4.0

    TU Clausthal

    Deliverable 3b

    Machine Learning:
    - Random Forest

    Libraries:
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    """)

    # ==========================================================
# PAGE ROUTER
# ==========================================================
if page == "🏠 Home":
    home()

elif page == "📊 Dataset Analysis":
    dataset_analysis()

elif page == "🤖 Model Training":
    model_training()

elif page == "📈 Evaluation":
    evaluation()

elif page == "🔮 Prediction":
    prediction()

elif page == "ℹ️ About":
    about()