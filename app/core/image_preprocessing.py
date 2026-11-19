
"""Shared CNN image preprocessing using MediaPipe face ROI extraction."""

from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import torch

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("face_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Landmark indices for lips and eyes.
LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

LEFT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

_FEATURE_INDICES = [LIPS_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX]

# Reuse one MediaPipe instance across image preprocessing calls.
_landmarker = None


def get_landmarker():
    """Load the MediaPipe FaceLandmarker model once."""

    global _landmarker

    if _landmarker is None:
        if not MODEL_PATH.exists():
            print("Downloading MediaPipe FaceLandmarker model...")
            urllib.request.urlretrieve(
                MODEL_URL,
                str(MODEL_PATH)
            )

        base_options = python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.7,
        )

        _landmarker = vision.FaceLandmarker.create_from_options(
            options
        )

    return _landmarker


def extract_pure_skin_roi(image_bgr, landmarker_instance):
    """
    Extract the facial ROI using MediaPipe landmarks.
    Exclude the eyes and lips from the facial mask.

    If no face is detected, return the original image,
    matching the fallback behavior of the original pipeline.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Gambar kosong atau tidak valid.")

    height, width = image_bgr.shape[:2]

    # MediaPipe Tasks requires an RGB image.
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    detection_result = landmarker_instance.detect(mp_image)

    if not detection_result.face_landmarks:
        return image_bgr

    landmarks = detection_result.face_landmarks[0]

    # Convert normalized landmark coordinates to pixel coordinates.
    points = np.array(
        [
            [
                int(np.clip(lm.x * width, 0, width - 1)),
                int(np.clip(lm.y * height, 0, height - 1)),
            ]
            for lm in landmarks
        ],
        dtype=np.int32,
    )

    # Create a mask covering the convex hull of facial landmarks.
    mask = np.zeros((height, width), dtype=np.uint8)

    face_hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, face_hull, 255)

    # Exclude the lips and both eyes.
    for indices in _FEATURE_INDICES:
        feature_points = points[indices]
        feature_hull = cv2.convexHull(feature_points)

        cv2.fillConvexPoly(mask, feature_hull, 0)

    # Keep the masked facial region; background becomes black.
    roi_bgr = cv2.bitwise_and(
        image_bgr,
        image_bgr,
        mask=mask
    )

    return roi_bgr


def preprocess_image(
    image_path: str | Path,
    device: torch.device
) -> torch.Tensor:
    """
    Preprocess one image for CNN inference.

    Steps:
    1. Read image in BGR format.
    2. Extract facial ROI using MediaPipe.
    3. Convert BGR to RGB.
    4. Resize to 224 x 224.
    5. Normalize pixel values to [0, 1].
    6. Convert to a batched PyTorch tensor.
    """

    image_path = Path(image_path)

    # 1. Read the input image.
    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            f"Gambar tidak dapat dibaca atau path salah: {image_path}"
        )

    # 2. Extract facial ROI using MediaPipe.
    landmarker = get_landmarker()

    roi_bgr = extract_pure_skin_roi(
        image_bgr,
        landmarker
    )

    # 3. Convert BGR to RGB.
    image_rgb = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2RGB
    )

    # 4. Resize to 224 x 224.
    image_resized = cv2.resize(
        image_rgb,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    # 5. Normalize pixel values to [0, 1].
    image_normalized = (
        image_resized.astype(np.float32) / 255.0
    )

    # 6. Convert HWC to CHW, add batch dimension, move to device.
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(image_normalized)
    )

    image_tensor = (
        image_tensor
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    return image_tensor


def close_landmarker():
    """Release MediaPipe resources when preprocessing is finished."""

    global _landmarker

    if _landmarker is not None:
        _landmarker.close()
        _landmarker = None
