"""
Streamlit UI: upload a road scene image, call the FastAPI /predict
endpoint, and show the original image next to the colorized
segmentation overlay, plus a per-class pixel-coverage breakdown.
"""
import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Road Scene Segmentation", layout="wide")
st.title("Semantic Segmentation for Autonomous Driving")
st.caption("CamVid, 11-class SegNet taxonomy · DeepLabV3 / U-Net")

uploaded = st.file_uploader("Upload a road scene image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image_bytes = uploaded.getvalue()
    st.image(Image.open(io.BytesIO(image_bytes)), caption="Input", width=500)

    if st.button("Run segmentation"):
        with st.spinner("Running inference..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/predict",
                    files={"file": (uploaded.name, image_bytes, uploaded.type)},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as e:
                st.error(f"Request to backend failed: {e}")
                st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Overlay")
            overlay = Image.open(io.BytesIO(base64.b64decode(result["overlay_image_base64"])))
            st.image(overlay, width=500)
        with col2:
            st.subheader("Segmentation mask")
            mask = Image.open(io.BytesIO(base64.b64decode(result["mask_image_base64"])))
            st.image(mask, width=500)

        st.subheader("Class coverage")
        for entry in result["class_coverage"]:
            st.write(f"**{entry['label']}** — {entry['pixel_percentage']}%")
            st.progress(min(entry["pixel_percentage"] / 100, 1.0))

        st.caption(
            f"Inference time: {result['inference_time_ms']} ms · model v{result['model_version']}"
        )
else:
    st.info("Upload an image to run segmentation.")
