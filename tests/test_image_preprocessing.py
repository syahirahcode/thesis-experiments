import cv2
import numpy as np
import torch
from types import SimpleNamespace

from app.core import image_preprocessing
from app.core.agents.facial_agent import FacialAgent
from app.core.agents.single_agent import SingleAgent
from app.core.image_preprocessing import IMAGE_SIZE


def test_facial_and_single_agents_use_mediapipe_preprocessing(
    tmp_path, monkeypatch
) -> None:
    image = np.zeros((37, 61, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(61, dtype=np.uint8)
    image[:, :, 1] = np.arange(37, dtype=np.uint8)[:, None]
    image[:, :, 2] = 127
    image_path = tmp_path / "input.png"
    assert cv2.imwrite(str(image_path), image)

    detection_calls = 0

    class NoFaceLandmarker:
        def detect(self, image):
            nonlocal detection_calls
            assert image is not None
            detection_calls += 1
            return SimpleNamespace(face_landmarks=[])

    monkeypatch.setattr(
        image_preprocessing, "get_landmarker", lambda: NoFaceLandmarker()
    )

    facial_agent = FacialAgent.__new__(FacialAgent)
    facial_agent.device = torch.device("cpu")
    single_agent = SingleAgent.__new__(SingleAgent)
    single_agent.device = torch.device("cpu")

    facial_tensor = facial_agent._preprocess(str(image_path))
    single_tensor = single_agent._preprocess_image(str(image_path))
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    expected = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    expected = cv2.resize(expected, IMAGE_SIZE).astype(np.float32) / 255.0
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    expected_tensor = expected_tensor.permute(2, 0, 1).unsqueeze(0)

    assert detection_calls == 2
    assert facial_tensor.shape == (1, 3, 224, 224)
    assert facial_tensor.dtype == torch.float32
    assert torch.equal(facial_tensor, expected_tensor)
    assert torch.equal(facial_tensor, single_tensor)
