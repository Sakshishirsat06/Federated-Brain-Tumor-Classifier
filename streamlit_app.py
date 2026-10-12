# streamlit_app.py
from pathlib import Path
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================
# TRAINING RESULTS (from the latest FL run)
# ============================================

NUM_CLIENTS = 5
NUM_ROUNDS = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 16
AGGREGATION = "FedAvg"
TRAINING_TIME_SEC = 3048.88

TEST_LOSS = 0.3401
TEST_ACC = 89.98
TEST_SAMPLES = 858

# Client class distributions: [glioma, meningioma, normal, pituitary]
CLIENT_DISTRIBUTIONS = {
    0: [60, 30, 30, 30],
    1: [30, 60, 30, 30],
    2: [30, 30, 60, 30],
    3: [30, 30, 30, 60],
    4: [37, 37, 37, 37],
}

# Global validation accuracy per round
GLOBAL_ACC = [24.13, 84.38, 82.52, 81.24, 89.98]

# Per-client, per-round results (rows = rounds 1..5)
CLIENT_TRAIN_ACC = {
    0: [66.67, 78.67, 81.33, 82.22, 86.67],
    1: [68.44, 75.11, 82.22, 88.89, 85.56],
    2: [71.78, 82.44, 82.89, 86.22, 83.11],
    3: [65.78, 74.00, 83.33, 85.78, 88.44],
    4: [78.15, 77.70, 79.73, 80.41, 91.44],
}
CLIENT_VAL_ACC = {
    0: [74, 56, 76, 62, 56],
    1: [46, 70, 58, 72, 54],
    2: [72, 44, 46, 36, 64],
    3: [62, 66, 62, 62, 66],
    4: [54, 46, 62, 68, 56],
}
CLIENT_SPECIALIZATION = {
    0: "Glioma (40%)",
    1: "Meningioma (40%)",
    2: "Normal Brain (40%)",
    3: "Pituitary (40%)",
    4: "Balanced (25% each)",
}

# Per-class test metrics
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'normal_brain', 'pituitary_tumor']
PER_CLASS_METRICS = {
    'Class': ['Glioma Tumor', 'Meningioma Tumor', 'Normal Brain', 'Pituitary Tumor'],
    'Precision': [0.93, 0.88, 0.94, 0.86],
    'Recall': [0.97, 0.76, 0.94, 0.93],
    'F1-Score': [0.95, 0.81, 0.94, 0.89],
    'Support': [198, 207, 230, 223],
}
MACRO_F1 = 0.90
WEIGHTED_F1 = 0.90

# Paste your real confusion matrix here (rows = true, cols = predicted), e.g.
# from sklearn.metrics import confusion_matrix; confusion_matrix(y_true, y_pred).tolist()
# Leave as None to hide the heatmap rather than show made-up numbers.
CONFUSION_MATRIX = None

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier - Federated Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .info-message {
        padding: 10px;
        border-radius: 5px;
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL DEFINITION (Same as in notebook)
# ============================================

class VisionMAMMALight(nn.Module):
    """Lightweight Vision MAMMA for faster inference"""
    def __init__(self, num_classes=4, img_size=224, pretrained=True):
        super().__init__()

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        self.cnn_backbone = tv_models.resnet18(weights=weights)

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.cnn_backbone.children())[:-1])

        feature_dim = 512

        # Mamba-inspired blocks (simplified)
        self.mamba_blocks = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.flatten(1)
        features = self.mamba_blocks(features)
        return self.classifier(features)

# Image preprocessing
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# LOAD MODEL FUNCTION
# ============================================

def find_model_file():
    """Look for the trained weights next to this script (works for any .pth name)."""
    here = Path(__file__).resolve().parent
    preferred = ["federated_brain_tumor_model.pth", "federated__brain__tumor__model.pth"]
    for name in preferred:
        if (here / name).exists():
            return here / name
    found = sorted(here.glob("*.pth"))
    return found[0] if found else None


@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = find_model_file()
    if model_path is None:
        st.error("No .pth model file found. Put your trained model file in the same "
                 "folder as streamlit_app.py.")
        return None
    try:
        # pretrained=False: we load our own trained weights, no ImageNet download needed
        model = VisionMAMMALight(num_classes=4, img_size=image_size, pretrained=False)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model ({model_path.name}): {e}")
        return None

@st.cache_resource
def count_parameters():
    return sum(p.numel() for p in VisionMAMMALight(pretrained=False).parameters())

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_image(model, image):
    """Predict the class of an image"""
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        all_probs = probabilities[0].cpu().numpy()

    return predicted_class, confidence, all_probs

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_class_color(class_name):
    colors = {
        'glioma_tumor': '#FF6B6B',
        'meningioma_tumor': '#4ECDC4',
        'normal_brain': '#45B7D1',
        'pituitary_tumor': '#96CEB4'
    }
    return colors.get(class_name, '#95A5A6')

def get_class_name_color(class_name):
    """Color lookup for formatted class names (batch prediction)"""
    colors = {
        'Glioma Tumor': '#FF6B6B',
        'Meningioma Tumor': '#4ECDC4',
        'Normal Brain': '#45B7D1',
        'Pituitary Tumor': '#96CEB4'
    }
    clean_name = class_name.replace('_', ' ').title()
    return colors.get(clean_name, '#95A5A6')

def plot_probability_chart(probabilities, class_names):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_list = [get_class_color(name) for name in class_names]
    bars = ax.bar(class_names, probabilities * 100, color=colors_list)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    for bar, prob in zip(bars, probabilities * 100):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig

def create_metrics_dashboard():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{TEST_ACC:.2f}%", f"{TEST_SAMPLES} test images")
    with col2:
        st.metric("Classes", "4", "tumor types")
    with col3:
        st.metric("Federated Rounds", str(NUM_ROUNDS), "FL training")
    with col4:
        st.metric("Clients", str(NUM_CLIENTS), "federated nodes")

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    with st.sidebar:
        st.title("🧠 Brain Tumor Classifier")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["🏠 Home", "📤 Single Prediction", "📁 Batch Prediction", "📊 Model Info", "👥 Federated Learning"]
        )

        st.markdown("---")
        st.info(
            "**About:**\n"
            "- Vision MAMMA Model\n"
            "- Federated Learning\n"
            "- 4 Brain Tumor Classes\n"
            f"- {TEST_ACC:.2f}% Test Accuracy"
        )

        st.markdown("---")
        st.caption("Developed with ❤️ using PyTorch & Streamlit")

    # ---------------- HOME ----------------
    if page == "🏠 Home":
        st.title("🧠 Brain Tumor Classification System")
        st.markdown("### Powered by Vision MAMMA & Federated Learning")
        st.markdown("---")

        st.markdown("""
        <div class="info-message">
        Welcome to the Brain Tumor Classification System! This application uses
        a Vision MAMMA (Mamba-inspired) model trained with Federated Learning to
        classify brain MRI scans into four categories:
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 **Classes**")
            st.markdown("""
            - 🧠 **Glioma Tumor**
            - 🧠 **Meningioma Tumor**
            - ✅ **Normal Brain**
            - 🧠 **Pituitary Tumor**
            """)

            st.markdown("### 🔬 **Model Architecture**")
            st.markdown(f"""
            - ResNet18 Backbone
            - Mamba-inspired Blocks
            - ~{count_parameters() / 1e6:.2f}M Parameters
            - Lightweight Design
            """)

        with col2:
            st.markdown("### 🏛️ **Federated Learning**")
            st.markdown(f"""
            - {NUM_CLIENTS} Federated Clients
            - {NUM_ROUNDS} Communication Rounds
            - {LOCAL_EPOCHS} Local Epochs per Round
            - {AGGREGATION} Aggregation
            """)

            st.markdown("### 📈 **Performance**")
            st.markdown(f"""
            - Test Accuracy: **{TEST_ACC:.2f}%**
            - Test Loss: **{TEST_LOSS:.4f}**
            - Macro Avg F1: **{MACRO_F1:.2f}**
            - Training Time: **{TRAINING_TIME_SEC:,.0f}s**
            """)

        st.markdown("---")
        st.subheader("📊 Model Performance Metrics")
        create_metrics_dashboard()

        st.markdown("---")
        st.subheader("🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("1. Go to **Single Prediction**")
        with col2:
            st.markdown("2. Upload a brain MRI image")
        with col3:
            st.markdown("3. Get instant results!")

        st.markdown("---")

        with st.expander("📖 Per-Class Test Performance"):
            st.markdown("F1-scores measured on the held-out test set:")
            for name, f1 in zip(PER_CLASS_METRICS['Class'], PER_CLASS_METRICS['F1-Score']):
                st.markdown(f"- **{name}**: F1 = {f1:.2f}")

        st.warning("⚠️ This tool is for research and educational purposes only. "
                   "It is not a medical device and must not replace diagnosis by a qualified clinician.")

    # ---------------- SINGLE PREDICTION ----------------
    elif page == "📤 Single Prediction":
        st.title("📤 Single Image Classification")
        st.markdown("Upload a brain MRI scan for instant classification")

        with st.spinner("Loading model..."):
            model = load_model()

        if model is None:
            st.stop()

        uploaded_file = st.file_uploader(
            "Choose a brain MRI image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Supported formats: JPG, PNG, BMP, TIFF"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded MRI Scan", width=420)

            with st.spinner("Analyzing image..."):
                predicted_class_idx, confidence, probabilities = predict_image(model, image)
                predicted_class = CLASS_NAMES[predicted_class_idx]

            with col2:
                st.markdown("### 📋 Prediction Results")

                color = get_class_color(predicted_class)
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                    <h3 style="margin: 0; color: {color};">{predicted_class.replace('_', ' ').title()}</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📊 Probability Distribution")
                fig = plot_probability_chart(probabilities, CLASS_NAMES)
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
            st.subheader("📝 Clinical Interpretation")

            interpretations = {
                'glioma_tumor': "Glioma tumors arise from glial cells in the brain. Further clinical evaluation and consultation with a neurologist is recommended.",
                'meningioma_tumor': "Meningioma tumors are typically slow-growing tumors that arise from the meninges. Medical follow-up is advised.",
                'normal_brain': "The scan appears normal. No tumor detected. Continue regular health check-ups.",
                'pituitary_tumor': "Pituitary tumors are abnormal growths in the pituitary gland. Endocrine consultation recommended."
            }

            st.info(interpretations[predicted_class])

            if confidence < 0.7:
                st.warning("⚠️ Low confidence prediction. Consider re-uploading a clearer image or consulting a specialist.")
            elif confidence > 0.9:
                st.success("✅ High confidence prediction. The model is very certain about this classification.")

            st.caption("Research use only. Not a substitute for professional medical diagnosis.")

    # ---------------- BATCH PREDICTION ----------------
    elif page == "📁 Batch Prediction":
        st.title("📁 Batch Prediction")
        st.markdown("Upload multiple brain MRI scans for batch classification")

        with st.spinner("Loading model..."):
            model = load_model()

        if model is None:
            st.stop()

        uploaded_files = st.file_uploader(
            "Choose multiple brain MRI images...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )

        if uploaded_files:
            st.markdown(f"### 📊 Processing {len(uploaded_files)} images...")

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}...")
                image = Image.open(uploaded_file).convert('RGB')
                pred_class_idx, confidence, _ = predict_image(model, image)
                pred_class = CLASS_NAMES[pred_class_idx]
                results.append({
                    'filename': uploaded_file.name,
                    'prediction': pred_class.replace('_', ' ').title(),
                    'confidence': f"{confidence:.1%}"
                })
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("✅ All images processed!")

            st.markdown("### 📋 Batch Results")
            df = pd.DataFrame(results)
            st.dataframe(df)

            st.markdown("### 📈 Summary Statistics")
            col1, col2, col3 = st.columns(3)

            prediction_counts = df['prediction'].value_counts()

            with col1:
                st.metric("Total Images", len(results))
            with col2:
                st.metric("Unique Classes", len(prediction_counts))
            with col3:
                st.metric("Most Common", prediction_counts.index[0] if len(prediction_counts) > 0 else "N/A")

            fig, ax = plt.subplots(figsize=(10, 6))
            colors_list = [get_class_name_color(pred) for pred in prediction_counts.index]
            prediction_counts.plot(kind='bar', ax=ax, color=colors_list, edgecolor='black')
            ax.set_title('Prediction Distribution Across Batch', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Class', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

    # ---------------- MODEL INFO ----------------
    elif page == "📊 Model Info":
        st.title("📊 Model Information")

        with st.expander("🏗️ Model Architecture", expanded=True):
            st.markdown(f"""
            ### Vision MAMMA (Lightweight Version)

            **Architecture Components:**
            - **Backbone**: ResNet18 (pre-trained on ImageNet)
            - **Feature Extractor**: Extracts 512-dimensional features
            - **Mamba Blocks**:
                - Linear layer (512 → 256) + GELU + Dropout
                - Linear layer (256 → 128) + GELU + Dropout
            - **Classifier**: Linear layer (128 → 4)

            **Model Statistics:**
            - Total Parameters: {count_parameters():,}
            - Input Size: 224×224×3
            - Output Classes: 4
            - Framework: PyTorch
            """)

        with st.expander("🏛️ Federated Learning Configuration", expanded=True):
            dist_rows = []
            for cid, d in CLIENT_DISTRIBUTIONS.items():
                dist_rows.append({
                    'Client': f'Client {cid}',
                    'Glioma': d[0],
                    'Meningioma': d[1],
                    'Normal': d[2],
                    'Pituitary': d[3],
                    'Total': sum(d),
                })

            st.markdown(f"""
            ### Federated Learning Setup

            **Configuration:**
            - Number of Clients: {NUM_CLIENTS}
            - Communication Rounds: {NUM_ROUNDS}
            - Local Epochs per Round: {LOCAL_EPOCHS}
            - Aggregation Method: Federated Averaging ({AGGREGATION})
            - Batch Size: {BATCH_SIZE}
            - Total Training Time: {TRAINING_TIME_SEC:,.2f} seconds

            **Data Distribution (Non-IID, samples per class):**
            """)
            st.dataframe(pd.DataFrame(dist_rows).set_index('Client'))

        with st.expander("🎯 Training Details", expanded=True):
            st.markdown("""
            ### Training Configuration

            **Optimizer:** Adam
            - Learning Rate: 0.001
            - Weight Decay: None

            **Loss Function:** Cross-Entropy Loss

            **Scheduler:** ReduceLROnPlateau
            - Mode: min
            - Patience: 3
            - Factor: 0.5

            **Data Split:**
            - Training: 70% (3,998 images)
            - Validation: 15% (856 images)
            - Testing: 15% (858 images)
            """)

        with st.expander("📈 Performance Metrics", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Per-Class Performance (Test Set)")
                metrics_df = pd.DataFrame(PER_CLASS_METRICS)
                st.dataframe(metrics_df)

            with col2:
                st.markdown("#### Overall Metrics")
                st.metric("Test Accuracy", f"{TEST_ACC:.2f}%")
                st.metric("Test Loss", f"{TEST_LOSS:.4f}")
                st.metric("Macro Avg F1", f"{MACRO_F1:.2f}")
                st.metric("Weighted Avg F1", f"{WEIGHTED_F1:.2f}")

        with st.expander("📊 Confusion Matrix"):
            if CONFUSION_MATRIX is None:
                st.info("Confusion matrix not provided yet. Set `CONFUSION_MATRIX` at the top of "
                        "this file to your real test-set matrix to display it here.")
            else:
                cm_data = np.array(CONFUSION_MATRIX)
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm_data, cmap='Blues')
                ax.set_xticks(range(len(CLASS_NAMES)))
                ax.set_yticks(range(len(CLASS_NAMES)))
                ax.set_xticklabels(CLASS_NAMES)
                ax.set_yticklabels(CLASS_NAMES)
                for i in range(cm_data.shape[0]):
                    for j in range(cm_data.shape[1]):
                        ax.text(j, i, int(cm_data[i, j]), ha='center', va='center')
                fig.colorbar(im, ax=ax)
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('True', fontsize=12)
                ax.set_title('Confusion Matrix', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ---------------- FEDERATED LEARNING ----------------
    elif page == "👥 Federated Learning":
        st.title("👥 Federated Learning Simulation")

        st.markdown("""
        <div class="info-message">
        This section explains the Federated Learning approach used to train the brain tumor classification model.
        Federated Learning enables privacy-preserving collaborative training across multiple hospitals/clients.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 **Why Federated Learning?**")
            st.markdown("""
            - **Privacy Preservation**: Medical data never leaves local hospitals
            - **Regulatory Compliance**: HIPAA and GDPR friendly
            - **Data Diversity**: Leverages diverse patient populations
            - **Reduced Communication**: Only model updates are shared
            - **Scalability**: Easy to add new hospitals/clients
            """)

        with col2:
            st.markdown("### 🔄 **FL Workflow**")
            st.markdown("""
            1. **Initialization**: Global model distributed to all clients
            2. **Local Training**: Each client trains on local data
            3. **Model Upload**: Model updates sent to server
            4. **Aggregation**: Server combines updates (FedAvg)
            5. **Distribution**: Updated global model sent to clients
            6. **Iteration**: Repeat for multiple rounds
            """)

        st.markdown("---")

        st.subheader("⚙️ Simulation Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clients", str(NUM_CLIENTS))
        with col2:
            st.metric("Rounds", str(NUM_ROUNDS))
        with col3:
            st.metric("Local Epochs", str(LOCAL_EPOCHS))
        with col4:
            st.metric("Aggregation", AGGREGATION)

        # Training progress chart
        st.subheader("📈 Training Progress")

        rounds = list(range(1, NUM_ROUNDS + 1))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, GLOBAL_ACC, 'b-o', linewidth=2, markersize=8)
        for r, acc in zip(rounds, GLOBAL_ACC):
            ax.annotate(f'{acc:.2f}%', (r, acc), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)
        ax.set_xlabel('Federated Round', fontsize=12)
        ax.set_ylabel('Global Validation Accuracy (%)', fontsize=12)
        ax.set_title('Federated Learning Convergence', fontsize=14, fontweight='bold')
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Per-client curves
        st.subheader("📉 Client Accuracy by Round")
        tab1, tab2 = st.tabs(["Local Train Accuracy", "Local Validation Accuracy"])

        for tab, data, label in [(tab1, CLIENT_TRAIN_ACC, "Train"), (tab2, CLIENT_VAL_ACC, "Validation")]:
            with tab:
                fig, ax = plt.subplots(figsize=(10, 5))
                for cid, vals in data.items():
                    ax.plot(rounds, vals, marker='o', linewidth=2, label=f'Client {cid}')
                ax.set_xlabel('Federated Round', fontsize=12)
                ax.set_ylabel(f'{label} Accuracy (%)', fontsize=12)
                ax.set_xticks(rounds)
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Client performance (final round)
        st.subheader(f"📊 Client Performance (Round {NUM_ROUNDS})")

        client_df = pd.DataFrame({
            f'Client {cid}': {
                'Train Acc': f"{CLIENT_TRAIN_ACC[cid][-1]:.2f}%",
                'Val Acc': f"{CLIENT_VAL_ACC[cid][-1]:.2f}%",
                'Specialization': CLIENT_SPECIALIZATION[cid],
            } for cid in range(NUM_CLIENTS)
        }).T
        st.dataframe(client_df)

        st.caption("Local validation accuracy is measured on each client's small local split, "
                   "so it is noisier than the global validation/test accuracy.")

        st.markdown("---")
        st.subheader("✅ Advantages of Federated Learning for Medical Imaging")

        adv_col1, adv_col2, adv_col3 = st.columns(3)

        with adv_col1:
            st.markdown("#### 🔒 Privacy-First")
            st.markdown("Medical images never leave hospital servers, ensuring patient data privacy and regulatory compliance.")

        with adv_col2:
            st.markdown("#### 🌍 Diverse Data")
            st.markdown("Model learns from diverse patient populations across different demographics and equipment.")

        with adv_col3:
            st.markdown("#### 📈 Scalable")
            st.markdown("Easily scale to hundreds of hospitals without centralizing sensitive medical data.")


if __name__ == "__main__":
    main()
