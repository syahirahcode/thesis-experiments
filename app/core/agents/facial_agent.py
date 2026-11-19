"""
app/core/agents/facial_agent.py

Agen yang bertanggung jawab mengeksekusi ekstraksi atribut visual wajah (Skintone, Undertone, Saturation).
Menggunakan arsitektur Model B (EfficientNetV2S Multi-Head) dengan pipeline preprocessing
yang selaras persis dengan bagian 'START FROM HERE' pada notebook eksperimen RQ1.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from app.core.config import get_settings
from app.core.image_preprocessing import preprocess_image
from app.schemas.analysis import (
    FacialAgentOutput,
    VisualAttributes,
    SkinTone,
    Undertone,
    Saturation
)


# --- 1. Definisi Arsitektur Model B ---

class MultiHeadEfficientNetV2S(nn.Module):
    def __init__(self):
        super(MultiHeadEfficientNetV2S, self).__init__()
        self.backbone = models.efficientnet_v2_s(weights=None)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(p=0.3)
        
        self.head_skintone = nn.Linear(in_features, 5)
        self.head_undertone = nn.Linear(in_features, 2)
        self.head_saturation = nn.Linear(in_features, 3)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        return {
            "head_skintone": self.head_skintone(features),
            "head_undertone": self.head_undertone(features),
            "head_saturation": self.head_saturation(features),
        }


# --- 2. Pemetaan Label Kelas (Model B) ---

INV_SKIN_TONE = {0: "Fair", 1: "Medium", 2: "Tanned", 3: "Deep", 4: "Dark"}
INV_UNDERTONE = {0: "Cool", 1: "Warm"}
INV_SATURATION = {0: "Low", 1: "Medium", 2: "High"}

# --- 3. Facial Agent ---

class FacialAgent:
    def __init__(self):
        self.settings = get_settings()
        
        # Penentuan Device
        if self.settings.CNN_DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.settings.CNN_DEVICE)
            
        self._init_model()

    def _init_model(self) -> None:
        """Memuat arsitektur PyTorch dan bobot best_model_b_rgb.pth."""
        self.model = MultiHeadEfficientNetV2S().to(self.device)
        model_path = self.settings.CNN_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Model B tidak ditemukan di path: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _preprocess(self, image_path: str) -> torch.Tensor:
        """
        Pipeline Preprocessing (Sesuai cell START FROM HERE RQ1):
        Baca BGR -> Konversi ke RGB -> Resize ke 224x224 -> Normalisasi [0.0, 1.0] -> PyTorch Tensor.
        """
        return preprocess_image(image_path, self.device)

    def analyze(self, image_path: str) -> FacialAgentOutput:
        """
        Mengeksekusi inferensi model CNN pada citra uji dan mengembalikan atribut terstruktur.
        """
        img_tensor = self._preprocess(image_path)

        with torch.no_grad():
            outputs = self.model(img_tensor)

            # Hitung probabilitas dengan Softmax
            prob_st = F.softmax(outputs["head_skintone"], dim=1)[0].cpu().numpy()
            prob_ut = F.softmax(outputs["head_undertone"], dim=1)[0].cpu().numpy()
            prob_sat = F.softmax(outputs["head_saturation"], dim=1)[0].cpu().numpy()

            # Dapatkan kelas dengan prediksi tertinggi
            pred_st = np.argmax(prob_st)
            pred_ut = np.argmax(prob_ut)
            pred_sat = np.argmax(prob_sat)

        # Petakan indeks ke label teks
        label_st = INV_SKIN_TONE[pred_st]
        label_ut = INV_UNDERTONE[pred_ut]
        label_sat = INV_SATURATION[pred_sat]

        # Susun kamus probabilitas untuk transparansi/debugging
        probabilities = {
            "skintone": {INV_SKIN_TONE[i]: float(prob_st[i]) for i in range(5)},
            "undertone": {INV_UNDERTONE[i]: float(prob_ut[i]) for i in range(2)},
            "saturation": {INV_SATURATION[i]: float(prob_sat[i]) for i in range(3)},
        }

        return FacialAgentOutput(
            attributes=VisualAttributes(
                skintone=SkinTone(label_st),
                undertone=Undertone(label_ut),
                saturation=Saturation(label_sat)
            ),
            probabilities=probabilities,
            model_name="best_model_b_rgb.pth",
            source="cnn",
        )