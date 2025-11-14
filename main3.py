import pandas as pd
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio
import supervision as sv
from io import BytesIO
import tempfile
import math

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
st.markdown("Upload an **image** or **video** — the app detects **cars**, **buses**, and **vans**, then counts them automatically.")

tab_img, tab_vid = st.tabs(["📸 Image Detection", "🎞️ Video Detection"])

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
                detections = sv.Detections.from_ultralytics(results).with_nms()

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

# === HELPER FUNCTIONS FOR VIDEO ANNOTATION ===

# === CUSTOM COLORS FOR VEHICLE CLASSES ===
CLASS_COLORS = {
    "car": (64, 64, 255),      # #4040FF
    "bus": (251, 81, 162),     # #FB51A2
    "van": (160, 161, 253),    # #A0A1FD
}

def draw_box_and_label(image_pil, xyxy, label, color=(255, 0, 0)):
    """
    Draw bounding box + label with auto-scaling based on image size.
    """
    draw = ImageDraw.Draw(image_pil)
    x1, y1, x2, y2 = map(int, xyxy)

    # ===== AUTO SCALE BASED ON IMAGE SIZE =====
    W, H = image_pil.size
    diag = math.sqrt(W * H)

    thickness = max(2, int(diag * 0.0025))          # Scales with image size
    font_size = max(12, int(diag * 0.015))          # Scales with size
    padding = max(4, int(diag * 0.005))             # Padding around label

    # Load scalable font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text = label

    # Compute text size safely
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        text_w, text_h = font.getsize(text)

    # ===== DRAW BOX WITH SCALED THICKNESS =====
    for i in range(thickness):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)

    # ===== DRAW LABEL BACKGROUND =====
    draw.rectangle(
        [x1,
         y1 - text_h - padding*2,
         x1 + text_w + padding*2,
         y1],
        fill=color
    )

    # ===== DRAW TEXT =====
    draw.text((x1 + padding, y1 - text_h - padding), text, fill="white", font=font)

    return image_pil


def draw_overlay_counts(image_pil, counts):
    """
    Auto-scaling overlay showing vehicle counts.
    """
    W, H = image_pil.size
    diag = math.sqrt(W * H)

    # ===== AUTO SCALE =====
    panel_w = int(W * 0.22)               # 22% of width
    panel_h = int(H * 0.18)               # 18% of height
    padding = max(6, int(diag * 0.006))
    title_font_size = max(14, int(diag * 0.02))
    text_font_size  = max(12, int(diag * 0.018))

    # Fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", title_font_size)
        font_text = ImageFont.truetype("arial.ttf", text_font_size)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw panel
    draw.rectangle([0, 0, panel_w, panel_h], fill=(0, 0, 0, 150))

    # Panel text
    draw.text((padding, padding), "Vehicle Counts", fill="white", font=font_title)
    draw.text((padding, padding + title_font_size + 4),
              f"Car : {counts['car']}", fill="white", font=font_text)
    draw.text((padding, padding + title_font_size + text_font_size + 10),
              f"Bus : {counts['bus']}", fill="white", font=font_text)
    draw.text((padding, padding + title_font_size + text_font_size*2 + 16),
              f"Van : {counts['van']}", fill="white", font=font_text)

    return Image.alpha_composite(image_pil.convert("RGBA"), overlay).convert("RGB")


# ==================================================================
# === VIDEO DETECTION ===
with tab_vid:
    uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"])

    conf_vid = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

    if uploaded_video:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
            tmp_video_file.write(uploaded_video.read())
            tmp_video_path = tmp_video_file.name
        
        # Display the video player
        st.video(tmp_video_path)

        if st.button("🚀 Detect Vehicles in Video"):
            progress = st.empty()

            # Read video with imageio from the temporary file
            reader = imageio.get_reader(tmp_video_path)
            meta = reader.get_meta_data()
            fps = meta.get("fps", None)
            
            # If fps missing, manually calculate or default to 30 FPS
            if fps is None:
                try:
                    # Some formats store fps as: 1 / time_base
                    time_base = meta.get("fps_time_base", None)
                    if time_base:
                        fps = 1 / time_base
                    else:
                        fps = 30  # <- safe fallback
                except:
                    fps = 30  # final fallback
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            writer = imageio.get_writer(output_path, fps=fps)

            last_counts = {"car": 0, "bus": 0, "van": 0}
            frame_id = 0

            for frame in reader:
                frame_id += 1
                img = Image.fromarray(frame)

                results = model(frame, conf=conf_vid, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Update counts every 3 frames
                if frame_id % 1 == 0:
                    counts = {"car": 0, "bus": 0, "van": 0}
                    for cid in detections.class_id:
                        cls = model.names[int(cid)]
                        if cls in counts:
                            counts[cls] += 1
                    last_counts = counts

                # Annotate frame (PIL)
                annotated = img.copy()
                for xyxy, cid in zip(detections.xyxy, detections.class_id):
                    cls = model.names[int(cid)]

                    # <<<< APPLY CUSTOM COLORS HERE >>>>
                    color = CLASS_COLORS.get(cls, (255, 0, 0))

                    annotated = draw_box_and_label(
                        annotated,
                        xyxy.tolist(),
                        cls,
                        color=color
                    )

                # Overlay vehicle count
                annotated = draw_overlay_counts(annotated, last_counts)

                # Write video frame
                writer.append_data(np.array(annotated))

                # Update progress every 20 frames
                if frame_id % 20 == 0:
                    progress.markdown(f"🎞️ Processing frame {frame_id} ...")

            reader.close()
            writer.close()
            progress.markdown("✅ Video processing complete!")

            # Display the processed video
            st.video(output_path)

            # Provide download button for processed video
            with open(output_path, "rb") as f:
                st.download_button("💾 Download Processed Video", f,
                                   "detected_vehicles.mp4", "video/mp4")
