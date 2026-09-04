"""Mock checks for the Flask API and the main frontend event bindings.

These checks do not load neural-network weights. They verify request handling,
error responses, threshold forwarding, and that the page controls are wired.
Run from the repository root with the project environment active:
    python -m unittest tests.test_web_mock -v
"""

from __future__ import annotations

import base64
import re
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import ANY, patch

from app import web_app


ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class WebMockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = web_app.app.test_client()

    def post_image(self, endpoint: str, name: str = "sample.png", **fields):
        data = {"file": (BytesIO(ONE_PIXEL_PNG), name)}
        data.update(fields)
        return self.client.post(endpoint, data=data, content_type="multipart/form-data")

    def test_home_page_contains_interactive_controls(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        for element_id in (
            "tab-classify",
            "tab-detect",
            "dropzone",
            "file-input",
            "choose-btn",
            "rerun-btn",
            "clear-btn",
            "threshold",
            "status",
        ):
            self.assertRegex(html, rf'id="{re.escape(element_id)}"')

        for event_binding in (
            "chooseBtn.addEventListener('click'",
            "rerunBtn.addEventListener('click'",
            "clearBtn.addEventListener('click'",
            "threshold.addEventListener('input'",
            "threshold.addEventListener('change'",
            "dropzone.addEventListener('drop'",
            "dropzone.addEventListener('keydown'",
            "fileInput.addEventListener('change'",
        ):
            self.assertIn(event_binding, html)

    def test_classification_request_returns_mocked_results(self) -> None:
        mocked_results = [{"label": "cat", "score": 91.25}]
        with patch.object(web_app, "predict_image", return_value=mocked_results):
            response = self.post_image("/predict")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["results"], mocked_results)
        self.assertTrue(response.get_json()["image_url"].startswith("/static/uploads/"))

    def test_unicode_filename_is_accepted(self) -> None:
        with patch.object(web_app, "predict_image", return_value=[]):
            response = self.post_image("/predict", name="金渐层.png")

        self.assertEqual(response.status_code, 200)
        uploaded_name = Path(response.get_json()["image_url"]).name
        uploaded_path = web_app.UPLOAD_FOLDER / uploaded_name
        self.assertTrue(uploaded_path.is_file())
        uploaded_path.unlink(missing_ok=True)

    def test_detection_request_forwards_threshold_without_loading_model(self) -> None:
        mocked_detections = [
            {
                "label": "cat",
                "confidence": 88.0,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 10.0,
                "y2": 12.0,
            }
        ]
        rendered_path = web_app.UPLOAD_FOLDER / "mock_rendered.png"
        with patch.object(
            web_app,
            "detect_image",
            return_value=(mocked_detections, rendered_path),
        ) as mocked_detect:
            response = self.post_image("/detect", conf_threshold="0.3")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["count"], 1)
        self.assertEqual(response.get_json()["conf_threshold"], 0.3)
        mocked_detect.assert_called_once_with(ANY, 0.3)

    def test_invalid_threshold_returns_bad_request(self) -> None:
        response = self.post_image("/detect", conf_threshold="1.5")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_corrupt_image_returns_bad_request(self) -> None:
        response = self.client.post(
            "/predict",
            data={"file": (BytesIO(b"not-an-image"), "fake.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_oversized_request_returns_413(self) -> None:
        original_limit = web_app.app.config["MAX_CONTENT_LENGTH"]
        web_app.app.config["MAX_CONTENT_LENGTH"] = 10
        try:
            response = self.client.post(
                "/predict",
                data={"file": (BytesIO(b"12345678901"), "sample.jpg")},
                content_type="multipart/form-data",
            )
        finally:
            web_app.app.config["MAX_CONTENT_LENGTH"] = original_limit

        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
