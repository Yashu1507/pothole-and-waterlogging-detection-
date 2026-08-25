import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart City AI",
    page_icon="🚁",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

POTHOLE_MODEL_PATH = "models/pothole_best.pt"
WATERLOGGING_MODEL_PATH = "models/waterlogging_best.pt"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

Path("outputs/images").mkdir(parents=True, exist_ok=True)
Path("outputs/videos").mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    pothole_model = YOLO(POTHOLE_MODEL_PATH)
    waterlogging_model = YOLO(WATERLOGGING_MODEL_PATH)

    return pothole_model, waterlogging_model


try:

    pothole_model, waterlogging_model = load_models()

except Exception as e:

    st.error("❌ Could not load YOLO models.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("⚙️ Detection Settings")

st.sidebar.markdown("### 🕳️ Pothole")

pothole_confidence = st.sidebar.slider(
    "Pothole confidence",
    0.10,
    0.95,
    0.40,
    0.01
)


st.sidebar.markdown("### 💧 Waterlogging")

water_confidence = st.sidebar.slider(
    "Waterlogging confidence",
    0.10,
    0.95,
    0.45,
    0.01
)


st.sidebar.markdown("### 🎯 Detection Quality")

iou_threshold = st.sidebar.slider(
    "IoU / duplicate suppression",
    0.20,
    0.80,
    0.50,
    0.01
)


image_size = st.sidebar.selectbox(
    "Inference image size",
    [640, 768, 960, 1280],
    index=2
)


st.sidebar.markdown("---")

st.sidebar.success("🕳️ Pothole model loaded")
st.sidebar.success("💧 Waterlogging model loaded")


# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_frame(
    frame,
    pothole_conf=0.40,
    water_conf=0.45,
    iou=0.50,
    imgsz=960
):

    # Work on a copy
    output = frame.copy()

    pothole_count = 0
    waterlogging_count = 0

    # ========================================================
    # POTHOLE DETECTION
    # ========================================================

    pothole_results = pothole_model.predict(
        source=frame,
        conf=pothole_conf,
        iou=iou,
        imgsz=imgsz,
        max_det=100,
        verbose=False
    )

    for result in pothole_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < pothole_conf:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Ignore extremely tiny detections
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 10 or box_height < 10:
                continue

            pothole_count += 1

            label = f"Pothole {confidence:.2f}"

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.rectangle(
                output,
                (x1, max(0, y1 - 32)),
                (x2, y1),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                output,
                label,
                (x1 + 5, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


    # ========================================================
    # WATERLOGGING DETECTION
    # ========================================================

    water_results = waterlogging_model.predict(
        source=frame,
        conf=water_conf,
        iou=iou,
        imgsz=imgsz,
        max_det=100,
        verbose=False
    )

    for result in water_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = waterlogging_model.names.get(
                class_id,
                str(class_id)
            )

            # =================================================
            # ONLY ACCEPT WATERLOGGING CLASS
            # =================================================

            if class_name.lower().strip() != "waterlogging":
                continue

            if confidence < water_conf:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 10 or box_height < 10:
                continue

            waterlogging_count += 1

            label = f"Waterlogging {confidence:.2f}"

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            cv2.rectangle(
                output,
                (x1, max(0, y1 - 32)),
                (x2, y1),
                (255, 0, 0),
                -1
            )

            cv2.putText(
                output,
                label,
                (x1 + 5, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    panel_height = 125

    cv2.rectangle(
        output,
        (10, 10),
        (410, panel_height),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        output,
        f"Potholes: {pothole_count}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"Waterlogging: {waterlogging_count}",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 0),
        2
    )

    cv2.putText(
        output,
        f"Confidence: P {pothole_conf:.2f} / W {water_conf:.2f}",
        (25, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    return (
        output,
        pothole_count,
        waterlogging_count
    )


# ============================================================
# HEADER
# ============================================================

st.title("🚁 Smart City AI")

st.subheader(
    "🕳️ Pothole + 💧 Waterlogging Detection System"
)

st.write(
    "AI-powered road-condition monitoring using two YOLO models."
)


# ============================================================
# TABS
# ============================================================

image_tab, video_tab, live_tab = st.tabs(
    [
        "📷 Images",
        "🎥 Videos",
        "🚁 Live Camera"
    ]
)


# ============================================================
# IMAGE TAB
# ============================================================

with image_tab:

    st.header("📷 Multiple Image Detection")

    st.write(
        "Upload as many images as your computer can handle."
    )

    uploaded_images = st.file_uploader(
        "Choose images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ],
        accept_multiple_files=True,
        key="image_upload"
    )

    if uploaded_images:

        st.success(
            f"✅ {len(uploaded_images)} image(s) selected."
        )

        if st.button(
            "🔍 Detect All Images",
            type="primary",
            key="detect_images"
        ):

            total_potholes = 0
            total_waterlogging = 0

            progress = st.progress(0)

            start_time = time.time()

            for index, uploaded_file in enumerate(
                uploaded_images
            ):

                st.write(
                    f"### 📷 {uploaded_file.name}"
                )

                file_bytes = np.asarray(
                    bytearray(
                        uploaded_file.read()
                    ),
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_COLOR
                )

                if frame is None:

                    st.error(
                        f"❌ Could not read {uploaded_file.name}"
                    )

                    continue

                result, potholes, waterlogging = detect_frame(
                    frame,
                    pothole_confidence,
                    water_confidence,
                    iou_threshold,
                    image_size
                )

                total_potholes += potholes

                total_waterlogging += waterlogging

                result_rgb = cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    result_rgb,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

                output_name = (
                    Path(uploaded_file.name).stem
                    + "_detected.jpg"
                )

                output_path = (
                    Path("outputs/images")
                    / output_name
                )

                cv2.imwrite(
                    str(output_path),
                    result
                )

                progress.progress(
                    (index + 1) / len(uploaded_images)
                )

            elapsed = time.time() - start_time

            st.success(
                "✅ All images processed."
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Images",
                    len(uploaded_images)
                )

            with col2:

                st.metric(
                    "Potholes",
                    total_potholes
                )

            with col3:

                st.metric(
                    "Waterlogging",
                    total_waterlogging
                )

            with col4:

                st.metric(
                    "Time",
                    f"{elapsed:.1f}s"
                )


# ============================================================
# VIDEO TAB
# ============================================================

with video_tab:

    st.header("🎥 Video Detection")

    uploaded_videos = st.file_uploader(
        "Choose videos",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm"
        ],
        accept_multiple_files=True,
        key="video_upload"
    )

    if uploaded_videos:

        st.success(
            f"✅ {len(uploaded_videos)} video(s) selected."
        )

        if st.button(
            "🎥 Process All Videos",
            type="primary",
            key="process_videos"
        ):

            for video_index, uploaded_video in enumerate(
                uploaded_videos
            ):

                st.write(
                    f"## 🎥 {uploaded_video.name}"
                )

                temp_input = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                temp_input.write(
                    uploaded_video.read()
                )

                temp_input.close()

                cap = cv2.VideoCapture(
                    temp_input.name
                )

                if not cap.isOpened():

                    st.error(
                        "❌ Could not open video."
                    )

                    continue

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if fps <= 0:
                    fps = 25

                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                output_name = (
                    Path(uploaded_video.name).stem
                    + "_detected.mp4"
                )

                output_path = (
                    Path("outputs/videos")
                    / output_name
                )

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height)
                )

                frame_placeholder = st.empty()

                progress = st.progress(0)

                frame_number = 0

                max_potholes = 0
                max_waterlogging = 0

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    result, potholes, waterlogging = detect_frame(
                        frame,
                        pothole_confidence,
                        water_confidence,
                        iou_threshold,
                        image_size
                    )

                    max_potholes = max(
                        max_potholes,
                        potholes
                    )

                    max_waterlogging = max(
                        max_waterlogging,
                        waterlogging
                    )

                    writer.write(result)

                    rgb = cv2.cvtColor(
                        result,
                        cv2.COLOR_BGR2RGB
                    )

                    frame_placeholder.image(
                        rgb,
                        channels="RGB",
                        use_container_width=True
                    )

                    frame_number += 1

                    if total_frames > 0:

                        progress.progress(
                            min(
                                frame_number / total_frames,
                                1.0
                            )
                        )

                cap.release()

                writer.release()

                try:
                    os.unlink(temp_input.name)
                except:
                    pass

                st.success(
                    f"✅ Finished {uploaded_video.name}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Maximum potholes in frame",
                        max_potholes
                    )

                with col2:

                    st.metric(
                        "Maximum waterlogging in frame",
                        max_waterlogging
                    )

                st.info(
                    f"Output: {output_path}"
                )


# ============================================================
# LIVE CAMERA
# ============================================================

with live_tab:

    st.header("🚁 Live Camera / Drone Stream")

    st.info(
        "For best live performance, use 640 or 768 image size."
    )

    source_type = st.radio(
        "Camera Source",
        [
            "Laptop Webcam",
            "IP / RTSP Camera"
        ]
    )

    if source_type == "Laptop Webcam":

        camera_source = 0

    else:

        camera_source = st.text_input(
            "RTSP / HTTP URL",
            placeholder="rtsp://camera-address/stream"
        )

    start_camera = st.checkbox(
        "🚁 START LIVE DETECTION"
    )

    if start_camera:

        if source_type == "IP / RTSP Camera":

            if not camera_source:

                st.warning(
                    "Enter your camera URL."
                )

                st.stop()

        cap = cv2.VideoCapture(
            camera_source
        )

        if not cap.isOpened():

            st.error(
                "❌ Could not open camera."
            )

            st.stop()

        st.success(
            "🟢 Live detection started."
        )

        live_placeholder = st.empty()
        stats_placeholder = st.empty()

        frame_counter = 0

        start_time = time.time()

        while True:

            ret, frame = cap.read()

            if not ret:

                st.error(
                    "Camera stream stopped."
                )

                break

            result, potholes, waterlogging = detect_frame(
                frame,
                pothole_confidence,
                water_confidence,
                iou_threshold,
                640
            )

            rgb = cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            )

            live_placeholder.image(
                rgb,
                channels="RGB",
                use_container_width=True
            )

            frame_counter += 1

            elapsed = time.time() - start_time

            fps = (
                frame_counter / elapsed
                if elapsed > 0
                else 0
            )

            stats_placeholder.markdown(
                f"""
                ## 🚁 LIVE AI STATUS

                🔴 **Potholes:** {potholes}

                🔵 **Waterlogging:** {waterlogging}

                ⚡ **FPS:** {fps:.2f}

                🎯 **Pothole confidence:** {pothole_confidence:.2f}

                🎯 **Waterlogging confidence:** {water_confidence:.2f}
                """
            )

        cap.release()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Smart City AI | Pothole + Waterlogging Detection"
)