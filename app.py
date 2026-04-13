import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# CBAM
# =======================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.fc(x)
        attn = self.sigmoid(attn)
        return x * attn

# =======================
# DUAL MODEL
# =======================
class DualBranchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone1 = models.efficientnet_b0(weights=None)
        self.backbone2 = models.efficientnet_b0(weights=None)

        self.backbone1.classifier = nn.Identity()
        self.backbone2.classifier = nn.Identity()

        self.cbam1 = CBAM(1280)
        self.cbam2 = CBAM(1280)

        self.fc = nn.Sequential(
            nn.Linear(2560, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x1, x2):
        f1 = self.backbone1(x1)
        f2 = self.backbone2(x2)

        f1 = self.cbam1(f1)
        f2 = self.cbam2(f2)

        fused = torch.cat((f1, f2), dim=1)
        return self.fc(fused)


# =======================
# LOAD MODEL
# =======================
import gdown
import os

@st.cache_resource
def load_model():
    model_path = "Best_model.pth"

    # download from Google Drive if not present
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1yZIfaPk2hlUVDDVyUg1lSmE_Lbe4ujdw"
        gdown.download(url, model_path, quiet=False)

    model = DualBranchModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

model = load_model()


# =======================
# PREPROCESS
# =======================
def preprocess(image):
    image = np.array(image)
    image = cv2.resize(image, (224,224))
    image = image / 255.0
    image = np.transpose(image, (2,0,1))
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32).to(device)

# =======================
# UI
# =======================
st.title("DR Detection with Confidence-Aware AI System")

uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = preprocess(image)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor, img_tensor)
        prob = torch.sigmoid(output).item()

    prediction = "DR Detected" if prob > 0.5 else "No DR"

    # Confidence
    confidence = prob if prob > 0.5 else 1 - prob

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Confidence: {confidence:.2f}")
    st.info("This system provides reliable classification with confidence estimation. Fine-grained lesion visualization is challenging in early DR due to subtle features.")

    # Referral logic
    if confidence < 0.7:
        st.warning("Low Confidence → Refer to Specialist")
    else:
        st.success("High Confidence Prediction")

    