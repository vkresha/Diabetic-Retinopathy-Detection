import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import cv2

def add_custom_css():
    st.markdown("""
        <style>
        /* App background */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #e3f2fd, #bbdefb, #90caf9) !important; /* Blue gradient */
            color: #004d80;
        }

        /* Custom container styling */
        .custom-container {
            background-color: white; /* White background */
            border: 2px solid #90caf9; /* Blue border */
            border-radius: 10px; /* Rounded corners */
            padding: 20px;
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            margin-top: 20px;
        }

        /* Summary box styling */
        .summary-box {
            background-color: #e3f2fd; /* Light blue background */
            border: 1px solid #90caf9; /* Light blue border */
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            color: #004d80; /* Dark blue text */
            font-size: 16px;
            margin-bottom: 20px;
            line-height: 1.5;
        }
         .custom-caption {
            background-color: #e3f2fd; /* Light blue background */
            color: #004d80; /* Dark blue text */
            text-align: center;
            padding: 5px;
            border-radius: 5px;
            margin-top: -15px;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #90caf9 !important; /* Blue progress bar */
        }
        </style>
    """, unsafe_allow_html=True)

# Image enhancement using CLAHE
def enhance_image(image, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        enhanced = clahe.apply(gray)
    else:
        enhanced = cv2.equalizeHist(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)

class ImageEnhancementTransform:
    def __init__(self, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def __call__(self, image):
        return enhance_image(image, self.use_clahe, self.clahe_clip_limit, self.clahe_tile_grid_size)

def load_model():
    checkpoint = torch.load('classifier_efficientnetb0.pt', map_location=torch.device('cpu'))
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 5),
            nn.LogSoftmax(dim=1)
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image):
    enhancement_transform = ImageEnhancementTransform(use_clahe=True, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8))
    enhanced_image = enhancement_transform(image)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(enhanced_image).unsqueeze(0)

CLASS_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

SUMMARY_CONTENT = {
    "No DR": """
        The retina shows **no signs** of diabetic retinopathy. Blood vessels appear normal, and there are no visible microaneurysms, hemorrhages, or other signs of damage.

        **Clinical Relevance:** Indicates no diabetic complications in the retina. Routine monitoring may still be necessary.""",
    "Mild": """
        **Early signs** of diabetic retinopathy, characterized by the presence of a few microaneurysms (small bulges in blood vessels caused by weakened vessel walls).

        **Clinical Relevance:** Early stage that requires regular monitoring to prevent progression.
        """,
    "Moderate": """
        **Increased severity** of retinopathy, with more widespread retinal damage including multiple microaneurysms, intraretinal hemorrhages, and possibly hard exudates. 

        **Clinical Relevance:** Progression risk is higher; requires closer follow-up and potential intervention.
        """,
    "Severe": """
        **Significant** retinal damage with a high number of hemorrhages, microaneurysms, and signs of vascular obstruction. "Cotton wool spots" (nerve fiber layer infarctions) may also appear.  

        **Clinical Relevance:** A pre-proliferative stage with high risk of progressing to proliferative DR; requires urgent medical attention.
        """,
    "Proliferative DR": """
        The **most advanced** stage, characterized by abnormal new blood vessel growth (neovascularization) due to retinal ischemia, often accompanied by vitreous hemorrhage and potential retinal detachment.  

        **Clinical Relevance:** A sight-threatening condition requiring immediate treatment, often involving laser therapy, injections, or surgery.
        """
}

# Function to generate a dynamic alert box
def severity_alert(predicted_label):
    severity_colors = {
        "No DR": "#28a745",
        "Mild": "#ffc107",
        "Moderate": "#fd7e14",
        "Severe": "#dc3545",
        "Proliferative DR": "#b71c1c"
    }
    color = severity_colors.get(predicted_label, "#28a745")

    st.markdown(f"""
        <div style="
            background-color: {color};
            border-radius: 5px;
            padding: 10px;
            color: white;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            ">
            Prediction: {predicted_label}
        </div>
    """, unsafe_allow_html=True)

# Function to display the summary
def display_summary(predicted_label):
    summary = SUMMARY_CONTENT.get(predicted_label, "No summary available.")
    st.markdown(f"""
        <div class="summary-box">
            {summary}
        </div>
    """, unsafe_allow_html=True)

def main():
    add_custom_css()
    st.title("🩺 Diabetic Retinopathy Detection")
    st.markdown("#### Upload a retinal image to predict the diabetic retinopathy stage:")
    
    model = load_model()
    probabilities = None  # Initialize probabilities as None

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 2])
        with col1:
            original_image = Image.open(uploaded_file).convert("RGB")
            st.image(original_image, use_container_width=True)
            st.markdown('<div class="custom-caption">Uploaded Image</div>', unsafe_allow_html=True)

        with col2:
            enhanced_image = enhance_image(Image.open(uploaded_file).convert("RGB"))
            st.image(enhanced_image, use_container_width=True)
            st.markdown('<div class="custom-caption">Enhanced Image</div>', unsafe_allow_html=True)
        
        processed_image = preprocess_image(Image.open(uploaded_file).convert("RGB"))
        device = torch.device('cpu')
        processed_image = processed_image.to(device)
        
        with st.spinner("🔍 Analyzing the image..."):
            outputs = model(processed_image)
            probabilities = torch.exp(outputs)  # Assign probabilities
            _, predicted_class = torch.max(outputs, 1)
            predicted_label = CLASS_LABELS[predicted_class.item()]
        
        # Display severity alert
        severity_alert(predicted_label)
        
        # Display summarized prediction
        display_summary(predicted_label)

    # Expander for probabilities
    if probabilities is not None:  # Only display expander if probabilities exist
        with st.expander("🔍 **Click to View Probabilities**", expanded=False):
            for i, prob in enumerate(probabilities.squeeze().tolist()):
                st.progress(int(prob * 100))
                st.write(f"**{CLASS_LABELS[i]}**: {prob:.2%}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Diabetic Retinopathy Detection",
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    main()
