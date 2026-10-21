"""
DriveGuard AI — Intelligent Driver Drowsiness Detection System
Interactive Academic Dashboard & Real-Time Computer Vision Detection Application
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

# Ensure working directory and sys.path point to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(rel_path: str) -> str:
    """Returns absolute path anchored to project root."""
    return str((PROJECT_ROOT / rel_path).resolve())


def render_image(container, img_input, caption=None):
    """Safely renders an image with compatibility for Streamlit versions and path checks."""
    if isinstance(img_input, str):
        full_path = resolve_path(img_input)
        if not os.path.exists(full_path):
            container.warning(f"Image artifact not found: {img_input}")
            return
        img_to_show = full_path
    else:
        img_to_show = img_input

    try:
        container.image(img_to_show, caption=caption, width="stretch")
    except TypeError:
        container.image(img_to_show, caption=caption, use_container_width=True)


# Configure page settings
st.set_page_config(
    page_title="DriveGuard AI — Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Navy & Cyan CSS Theme
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0b132b;
        color: #e0e6ed;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #48cae4 !important;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1c2541;
        border-right: 1px solid #3a506b;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00b4d8 !important;
        font-size: 2rem !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
    }
    
    /* Card containers */
    .metric-card {
        background-color: #1c2541;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #3a506b;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Badges */
    .badge-alert {
        background-color: #ff4d4d;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-safe {
        background-color: #06d6a0;
        color: black;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Lazy loading inference engine to cache in Streamlit memory
@st.cache_resource
def load_engine(model_type: str, threshold: float):
    from src.inference import DriveGuardInferenceEngine
    return DriveGuardInferenceEngine(
        model_type=model_type,
        confidence_threshold=threshold
    )


@st.cache_data
def load_metrics_and_manifest():
    manifest_path = resolve_path("outputs/metrics/dataset_splits.json")
    eval_path = resolve_path("outputs/metrics/evaluation_results.json")
    comp_path = resolve_path("outputs/metrics/model_comparison.json")
    cv_path = resolve_path("outputs/metrics/cv_techniques_experiments.json")

    data = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            data["manifest"] = json.load(f)
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            data["eval"] = json.load(f)
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            data["comp"] = json.load(f)
    if os.path.exists(cv_path):
        with open(cv_path, "r") as f:
            data["cv"] = json.load(f)
    return data


data_store = load_metrics_and_manifest()

# Sidebar controls
render_image(st.sidebar, "outputs/plots/1_dataset_class_distribution.png")
st.sidebar.title("DriveGuard AI Control")
st.sidebar.markdown("**Intelligent Driver Drowsiness Detection System**")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "Active Detection Architecture",
    ["DriveGuardMobileNetV2 (Recommended)", "DriveGuardCNN (From Scratch)"],
    index=0
)
model_key = "mobilenet_v2" if "MobileNet" in selected_model else "custom_cnn"

conf_threshold = st.sidebar.slider(
    "Drowsiness Decision Threshold",
    min_value=0.20,
    max_value=0.80,
    value=0.50,
    step=0.05,
    help="Cutoff probability for classifying a driver state as fatigued."
)

temporal_smoothing = st.sidebar.checkbox(
    "Enable Temporal Filtering",
    value=True,
    help="Applies rolling exponential smoothing across frames to filter out blinks."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Academic System Status**\n\n"
    "• Dataset: 9,120 verified images\n"
    "• Test Set: 1,368 held-out images\n"
    "• Device: PyTorch Multi-Threaded Engine\n"
    "• Status: Evaluation Complete"
)

# Navigation Tabs
tabs = st.tabs([
    "📊 Main Dashboard",
    "🎥 Live Detection & Testing",
    "⚖️ Model Comparison",
    "📁 Dataset Explorer",
    "📈 Evaluation & Metrics",
    "📘 Project Documentation"
])

# -----------------------------------------------------------------------------
# TAB 1: MAIN DASHBOARD
# -----------------------------------------------------------------------------
with tabs[0]:
    st.title("🚗 DriveGuard AI — Main Control Dashboard")
    st.markdown(
        "A deep learning and computer vision framework for real-time driver fatigue monitoring, "
        "drowsiness detection, and accident prevention."
    )

    # Top Metric KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="MobileNetV2 Test Accuracy",
            value="96.20%",
            delta="+18.20% vs CNN"
        )
    with col2:
        st.metric(
            label="Fatigue Recall (Safety)",
            value="95.61%",
            delta="30 FN out of 684"
        )
    with col3:
        st.metric(
            label="ROC-AUC Score",
            value="0.9933",
            delta="Near-Optimal"
        )
    with col4:
        st.metric(
            label="Inference Latency",
            value="11.51 ms",
            delta="86.9 FPS"
        )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("System Performance Summary")
        if "comp" in data_store:
            df_comp = pd.DataFrame(data_store["comp"])
            try:
                st.dataframe(df_comp, width="stretch")
            except TypeError:
                st.dataframe(df_comp, use_container_width=True)
        else:
            st.warning("Comparison metrics not yet available.")

        st.subheader("Key Findings & Safety Assessment")
        st.markdown("""
        - **MobileNetV2** with inverted residual bottleneck blocks achieved **96.20% test accuracy** and **95.61% recall** on the Fatigue class, missing only 30 drowsy cases out of 684 held-out test frames.
        - **DriveGuardCNN** trained from scratch achieved **78.00% test accuracy** and **70.76% fatigue recall** with a lightweight footprint (422K parameters, 4.86 MB).
        - **Computer Vision Temporal Smoothing** successfully suppressed **80.0% of false transient alarms** during continuous video simulation.
        """)

    with col_right:
        st.subheader("Model Performance Overview")
        render_image(st, "outputs/plots/7_model_comparison_barchart.png")


# -----------------------------------------------------------------------------
# TAB 2: LIVE DETECTION & INFERENCE
# -----------------------------------------------------------------------------
with tabs[1]:
    st.title("🎥 Live Detection & Visual Inference")
    st.markdown("Upload test images or test live camera frames using the verified model checkpoint.")

    input_mode = st.radio(
        "Select Detection Mode",
        ["🖼️ Image Upload / Dataset Sample Fallback", "📸 Live Webcam Input"],
        horizontal=True
    )

    col_input, col_result = st.columns([1, 1])

    inference_engine = load_engine(model_key, conf_threshold)

    input_image = None
    if "Webcam" in input_mode:
        st.markdown("**Webcam Video Frame Capture:**")
        cam_pic = st.camera_input("Capture live webcam frame for driver state classification")
        if cam_pic is not None:
            input_image = Image.open(cam_pic)
    else:
        uploaded_file = st.file_uploader(
            "Choose a driver image (JPG/PNG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
        else:
            # Provide sample buttons
            st.markdown("**Or test with sample images from dataset:**")
            col_s1, col_s2 = st.columns(2)
            active_dir = resolve_path("0 FaceImages/Active Subjects")
            fatigue_dir = resolve_path("0 FaceImages/Fatigue Subjects")

            with col_s1:
                if st.button("Load Active Subject Sample"):
                    if os.path.exists(active_dir) and len(os.listdir(active_dir)) > 0:
                        sample_p = os.path.join(active_dir, os.listdir(active_dir)[2])
                        input_image = Image.open(sample_p)
            with col_s2:
                if st.button("Load Fatigue Subject Sample"):
                    if os.path.exists(fatigue_dir) and len(os.listdir(fatigue_dir)) > 0:
                        sample_p = os.path.join(fatigue_dir, os.listdir(fatigue_dir)[1])
                        input_image = Image.open(sample_p)

    with col_input:
        st.subheader("Input Frame Preview")
        if input_image is not None:
            render_image(st, input_image, caption=f"Input Image ({input_image.size[0]}x{input_image.size[1]})")
        else:
            st.info("Please capture or upload an image above to run classification.")

    with col_result:
        st.subheader("DriveGuard AI Diagnostic Result")
        if input_image is not None:
            t0 = time.perf_counter()
            result = inference_engine.predict_image(input_image)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            pred_class = result["predicted_class"]
            is_fatigue = result["label_idx"] == 1
            confidence = result["confidence"]
            prob_fatigue = result["prob_fatigue"]
            prob_active = result["prob_active"]

            if is_fatigue:
                st.markdown(
                    f"<div class='metric-card' style='border-left: 6px solid #ff4d4d;'>"
                    f"<h3>⚠️ DROWSINESS DETECTED</h3>"
                    f"<p class='badge-alert'>DRIVER FATIGUE ALERT</p>"
                    f"<p style='margin-top: 10px;'>Status: <b>Yawning / Eyes Drooping / Inattentive</b></p>"
                    f"<p>Fatigue Probability: <b>{prob_fatigue*100:.2f}%</b></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='metric-card' style='border-left: 6px solid #06d6a0;'>"
                    f"<h3>✅ DRIVER IS ATTENTIVE</h3>"
                    f"<p class='badge-safe'>ACTIVE & ALERT</p>"
                    f"<p style='margin-top: 10px;'>Status: <b>Eyes Open / Attentive Driving Posture</b></p>"
                    f"<p>Alert Probability: <b>{prob_active*100:.2f}%</b></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.write(f"**Inference Latency:** `{dt_ms:.2f} ms` | **Architecture:** `{selected_model}`")
            
            # Confidence progress bars
            st.write(f"**Fatigue Probability P(Fatigue):** {prob_fatigue*100:.1f}%")
            st.progress(float(prob_fatigue))
            st.write(f"**Alert Probability P(Active):** {prob_active*100:.1f}%")
            st.progress(float(prob_active))

            if is_fatigue:
                st.error("🚨 AUDIO-VISUAL ALERT: Immediate rest or stop recommended!")
            else:
                st.success("Normal driving status confirmed.")


# -----------------------------------------------------------------------------
# TAB 3: MODEL COMPARISON
# -----------------------------------------------------------------------------
with tabs[2]:
    st.title("⚖️ Comprehensive Architecture Comparison")
    st.markdown(
        "Side-by-side comparison between **DriveGuardCNN** (4-stage CNN trained from scratch) and "
        "**DriveGuardMobileNetV2** (transfer learning with inverted residuals) on held-out test data."
    )

    if "comp" in data_store:
        st.table(pd.DataFrame(data_store["comp"]))

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Confusion Matrices (Test Set N = 1,368)")
        render_image(st, "outputs/plots/6_confusion_matrices_comparison.png")
    with col_m2:
        st.subheader("Per-Class Precision, Recall & F1")
        render_image(st, "outputs/plots/8_per_class_metrics_comparison.png")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Training vs Validation Accuracy")
        render_image(st, "outputs/plots/4_training_validation_accuracy.png")
    with col_c2:
        st.subheader("Training vs Validation Loss")
        render_image(st, "outputs/plots/5_training_validation_loss.png")


# -----------------------------------------------------------------------------
# TAB 4: DATASET EXPLORER
# -----------------------------------------------------------------------------
with tabs[3]:
    st.title("📁 Dataset Explorer & Preprocessing Pipeline")
    st.markdown(
        "Analysis of the **National Tsing Hua University / Kaggle Driver Drowsiness Dataset** "
        "comprising **9,120 balanced images**."
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("Class & Split Distribution")
        render_image(st, "outputs/plots/1_dataset_class_distribution.png")
    with col_d2:
        st.subheader("Data Augmentation & Preprocessing")
        render_image(st, "outputs/plots/3_preprocessing_augmentation_samples.png")

    st.subheader("Representative Dataset Samples")
    render_image(st, "outputs/plots/2_dataset_representative_samples.png")

    if "manifest" in data_store:
        st.markdown("### Split Integrity Verification")
        st.json(data_store["manifest"]["split_summary"])


# -----------------------------------------------------------------------------
# TAB 5: EVALUATION & RESULTS
# -----------------------------------------------------------------------------
with tabs[4]:
    st.title("📈 Detailed Academic Evaluation & Error Analysis")
    st.markdown("Detailed diagnostic evaluation on the held-out test set.")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("Receiver Operating Characteristic (ROC)")
        render_image(st, "outputs/plots/9_roc_curves_comparison.png")
    with col_r2:
        st.subheader("Prediction Confidence Distribution")
        render_image(st, "outputs/plots/11_confidence_distribution.png")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("Representative Misclassification Analysis")
        render_image(st, "outputs/plots/10_error_analysis_misclassifications.png")
    with col_e2:
        st.subheader("Decision Threshold & Error Tradeoff")
        render_image(st, "outputs/plots/12_technique_comparison_charts.png")


# -----------------------------------------------------------------------------
# TAB 6: PROJECT DOCUMENTATION & VIVA PREPARATION
# -----------------------------------------------------------------------------
with tabs[5]:
    st.title("📘 Project Documentation & Academic viva Guide")
    st.markdown("""
    ### Executive Abstract
    **DriveGuard AI** is an intelligent computer vision and deep learning system developed to address vehicular collisions 
    caused by driver fatigue and microsleep. Using a balanced dataset of 9,120 cockpit images across Active and Fatigue subjects, 
    the system evaluates two complementary architectures:
    
    1. **DriveGuardCNN (From Scratch)**: A 4-stage convolutional neural network utilizing Batch Normalization, Dropout, 
       and Global Average Pooling, serving as an interpretable lightweight baseline (422K parameters, 7.24 ms latency).
    2. **DriveGuardMobileNetV2 (Transfer Learning)**: An inverted residual bottleneck architecture fine-tuned from ImageNet weights, 
       achieving **96.20% accuracy, 95.61% fatigue recall, and 0.9933 ROC-AUC**.
    
    ### Computer Vision Techniques Evaluated
    - **Temporal Smoothing**: Reduced transient state oscillations and flickering by **80.0%** using rolling FIFO majority voting and exponential moving averages.
    - **Decision Threshold Calibration**: Analyzed false positive vs false negative safety tradeoffs across thresholds from 0.20 to 0.80.
    - **Alert Cooldown Logic**: Integrated state persistence and alarm cooldown to eliminate repetitive driver alarm fatigue.
    
    ### Academic Deliverables
    - **PDF Academic Report**: Available in `reports/final_report.pdf`
    - **DOCX Technical Report**: Available in `reports/final_report.docx`
    - **Viva Presentation**: Available in `reports/presentation.pptx`
    """)

    st.markdown("---")
    st.markdown("**DriveGuard AI — Academic Capstone Implementation by Antigravity Autonomous Engine**")
