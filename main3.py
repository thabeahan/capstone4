import pandas as pd
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import supervision as sv
from io import BytesIO
import tempfile


# === MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(page_title="Vehicle Detection App 🚘", layout="centered")

# === STYLE & UI ENHANCEMENTS ===
st.markdown( """
    <style>

    /* === White semi-transparent container for all content === */
    .block-container {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem 3rem;
        box-shadow: 0 0 25px rgba(0,0,0,0.25);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* === Headings and text colors === */
    h1, p, label{
        color: #000000 !important;
    }

    /* === Tabs === */
    [data-testid="stTabs"], [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Main uploader container */
        div[data-testid="stFileUploader"] {
            background: #262730 !important;                 /* clean black background */
            border-radius: 14px !important;
            border: 1.5px solid #d0d0d0 !important;        /* soft gray border */
            padding: 18px !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            transition: all 0.25s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# === NEW: Hover effect for the Browse files button inside uploader ===
st.markdown("""
    <style>
    [data-testid="stFileUploader"] button:hover {
        background-color: #0056cc !important;
        transform: scale(1.03);
        transition: 0.15s ease-in-out;
        color: white !important;  /* Optional: white text on hover */
    }
    [data-testid="stFileUploader"] button {
        transition: 0.15s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown( """
    <style>
    /* === System toolbar buttons (Rerun, Deploy, etc.) === */
    button[kind="secondary"], button[data-testid="stToolbarButton"] {
        background-color: rgba(255,255,255,0.95) !important;
        color: #000 !important;
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* === Clean table style with soft gray header === */
    table {
        background-color: #ffffff;      /* white background */
        border-radius: 10px;            /* rounded corners */
        width: 100%;
        border-collapse: collapse;      /* remove double lines */
        color: #000000;                 /* black text */
        font-weight: 500;
        font-size: 1rem;
    }

    /* === Header row === */
    thead th {
        background-color: #f0f0f0;      /* soft gray header */
        text-align: center;
        padding: 10px;
        color: #000000;                 /* black text */
        font-weight: 700;               /* bold header */
    }

    /* === Body cells === */
    tbody td {
        text-align: center;
        padding: 8px;
        color: #000000;
        border: none;                   /* remove all lines */
    }

    /* Optional: add spacing between rows */
    tbody tr {
        border-bottom: 1px solid #eaeaea; /* subtle row separator (very light) */
    }
</style>
""", unsafe_allow_html=True)

# === LOAD YOLO MODEL ===
MODEL_PATH = "best_3c_25p100.pt"

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    model = YOLO(MODEL_PATH)
    try:
        model.fuse()
    except Exception:
        pass
    return model

model = load_model()

# === STREAMLIT APP UI ===
st.title("🚗 Vehicle Detection App")
st.markdown("Upload an **image**  — the app detects **cars**, **buses**, and **vans**, then counts them automatically.")

tab_img = st.tabs(["📸 Image Detection"])[0]

# ======================================
# 📸 IMAGE DETECTION TAB
# ======================================
with tab_img:  
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"],key="img_uploader")
    conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.25, 0.05, key="img_conf")

    FRAME_WIDTH, FRAME_HEIGHT = 624, 400
    FRAME_COLOR = (200, 200, 200)

    def resize_with_padding(image: Image.Image, target_w: int, target_h: int, color=(200, 200, 200)):
        img_w, img_h = image.size
        scale = min(target_w / img_w, target_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = image.resize((new_w, new_h))
        new_img = Image.new("RGB", (target_w, target_h), color)
        new_img.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return new_img

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        preview_img = resize_with_padding(image, FRAME_WIDTH, FRAME_HEIGHT, FRAME_COLOR)
        st.image(preview_img, caption="📥 Uploaded Image (scaled with padding)", width=FRAME_WIDTH)

        if st.button("🚀 Detect Vehicles", key="detect_img"):
            progress_text = st.empty()
            progress_text.markdown("🚗 **Detecting vehicles... please wait.**")

            try:
                image_np = np.array(image)
                results = model(image_np, conf=conf_threshold, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results).with_nms(threshold=0.3, class_agnostic=True)

                h, w = image_np.shape[:2]
                scale_factor = np.sqrt((h * w) / (1280 * 720))
                box_thickness = max(1, int(4 * scale_factor))
                text_scale = max(0.3, min(0.3, 0.45 * scale_factor))
                text_thickness = max(1, int(1.1 * scale_factor))

                box_annotator = sv.BoxAnnotator(thickness=box_thickness)
                label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=text_thickness)

                labels = [f"{model.names[int(c)]} {float(conf):.2f}" for c, conf in zip(detections.class_id, detections.confidence)]
                annotated = box_annotator.annotate(scene=image_np.copy(), detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

                annotated_display = Image.fromarray(annotated)
                annotated_display = resize_with_padding(annotated_display, FRAME_WIDTH, FRAME_HEIGHT, FRAME_COLOR)
                st.image(annotated_display, caption="📊 Detected Vehicles (scaled with padding)", width=FRAME_WIDTH, clamp=True)

                # === Vehicle Count Table
                class_names = model.names
                counts = {}
                for c in detections.class_id:
                    label = class_names[int(c)]
                    counts[label] = counts.get(label, 0) + 1
                target_classes = ["car", "bus", "van"]
                filtered_counts = {k: v for k, v in counts.items() if k.lower() in target_classes}
                st.markdown("<h3 style='color: black;'>🔢 Detection Summary:</h3>", unsafe_allow_html=True)
                if filtered_counts:
                    df = pd.DataFrame({
                        "Vehicle Type": list(filtered_counts.keys()),
                        "Count": list(filtered_counts.values())
                    })
                    df.index = np.arange(1, len(df) + 1)
                    st.markdown(df.to_html(index=True, justify="center")
                        .replace('<th>', '<th style="text-align:center;">')
                        .replace('<td>', '<td style="text-align:center;">'),
                        unsafe_allow_html=True)
                else:
                    st.warning("No vehicles detected.")

                buf = BytesIO()
                Image.fromarray(annotated).save(buf, format="PNG")
                st.download_button("💾 Download Annotated Image", buf.getvalue(), "detected_vehicles.png", "image/png", key="download_img")
                progress_text.markdown("✅ **Detection complete!**")

            except Exception as e:
                progress_text.markdown(f"❌ **Error:** {e}")
    else:
        st.info("👆 Upload an image to start detection.")
