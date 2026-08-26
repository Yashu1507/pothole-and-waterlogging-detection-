import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pothole & Waterlogging Detection",
    page_icon="🚧",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

POTHOLE_MODEL_PATH = "models/pothole_best.pt"
WATERLOGGING_MODEL_PATH = "models/waterlogging_best.pt"


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
    st.error(str(e))
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0

if "results" not in st.session_state:
    st.session_state.results = []

if "total_potholes" not in st.session_state:
    st.session_state.total_potholes = 0

if "total_waterlogging" not in st.session_state:
    st.session_state.total_waterlogging = 0

if "images_processed" not in st.session_state:
    st.session_state.images_processed = 0


# ============================================================
# RESET FUNCTION
# ============================================================

def start_new_upload():

    st.session_state.upload_version += 1

    st.session_state.results = []

    st.session_state.total_potholes = 0

    st.session_state.total_waterlogging = 0

    st.session_state.images_processed = 0


# ============================================================
# IMAGE DETECTION
# ============================================================

def detect_image(image, confidence):

    image_np = np.array(image)

    if image_np.ndim == 2:

        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_GRAY2RGB
        )

    if image_np.shape[-1] == 4:

        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGBA2RGB
        )

    output = image_np.copy()

    pothole_count = 0

    waterlogging_count = 0

    # --------------------------------------------------------
    # POTHOLE MODEL
    # --------------------------------------------------------

    pothole_results = pothole_model.predict(
        image_np,
        conf=confidence,
        verbose=False
    )

    for result in pothole_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(box.conf[0])

            if conf < confidence:
                continue

            pothole_count += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

            cv2.putText(
                output,
                f"POTHOLE {conf:.0%}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    # --------------------------------------------------------
    # WATERLOGGING MODEL
    # --------------------------------------------------------

    water_results = waterlogging_model.predict(
        image_np,
        conf=confidence,
        verbose=False
    )

    for result in water_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(box.conf[0])

            if conf < confidence:
                continue

            waterlogging_count += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            cv2.putText(
                output,
                f"WATERLOGGING {conf:.0%}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
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

st.title(
    "🚧 Pothole & Waterlogging Detection System"
)

st.markdown(
    """
    ### AI-Based Road Condition Monitoring

    Upload road images or videos and automatically detect:

    🟨 **Potholes**

    🟦 **Waterlogging**
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.write(
    f"Current threshold: **{confidence:.0%}**"
)

st.sidebar.info(
    "Lower values detect more objects but may produce more false detections."
)


# ============================================================
# MAIN MODE
# ============================================================

mode = st.selectbox(
    "Select Detection Type",
    [
        "📷 Image Detection",
        "🎥 Video Detection",
        "📹 Camera Detection"
    ]
)

st.divider()


# ============================================================
# IMAGE DETECTION
# ============================================================

if mode == "📷 Image Detection":

    st.header("📷 Multiple Image Detection")

    st.write(
        "Select as many images as you want."
    )

    uploaded_files = st.file_uploader(
        "Upload road images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        accept_multiple_files=True,
        key=f"image_uploader_{st.session_state.upload_version}"
    )

    if uploaded_files:

        st.success(
            f"✅ {len(uploaded_files)} image(s) selected."
        )

        st.write("")

        detect_button = st.button(
            "🔍 DETECT ALL IMAGES",
            type="primary",
            use_container_width=True
        )

        if detect_button:

            st.session_state.results = []

            total_potholes = 0

            total_waterlogging = 0

            progress = st.progress(0)

            status = st.empty()

            # ------------------------------------------------
            # PROCESS EACH IMAGE
            # ------------------------------------------------

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                status.write(
                    f"Processing image {index + 1} "
                    f"of {len(uploaded_files)}: "
                    f"{uploaded_file.name}"
                )

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                output, potholes, waterlogging = detect_image(
                    image,
                    confidence
                )

                total_potholes += potholes

                total_waterlogging += waterlogging

                # Save result information

                st.session_state.results.append(
                    {
                        "Image": uploaded_file.name,
                        "Potholes": potholes,
                        "Waterlogging": waterlogging
                    }
                )

                # ------------------------------------------------
                # DISPLAY
                # ------------------------------------------------

                st.subheader(
                    f"Image {index + 1}: {uploaded_file.name}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        image,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        output,
                        caption="Detection Result",
                        use_container_width=True
                    )

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "🟨 Potholes",
                        potholes
                    )

                with c2:

                    st.metric(
                        "🟦 Waterlogging",
                        waterlogging
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

                st.divider()

            status.success(
                "✅ All images processed!"
            )

            st.session_state.total_potholes = total_potholes

            st.session_state.total_waterlogging = (
                total_waterlogging
            )

            st.session_state.images_processed = (
                len(uploaded_files)
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        if st.session_state.results:

            st.header("📊 Overall Detection Calculation")

            total_potholes = (
                st.session_state.total_potholes
            )

            total_waterlogging = (
                st.session_state.total_waterlogging
            )

            total_detections = (
                total_potholes +
                total_waterlogging
            )

            images_processed = (
                st.session_state.images_processed
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "📷 Images",
                    images_processed
                )

            with c2:

                st.metric(
                    "🟨 Total Potholes",
                    total_potholes
                )

            with c3:

                st.metric(
                    "🟦 Total Waterlogging",
                    total_waterlogging
                )

            with c4:

                st.metric(
                    "🚧 Total Problems",
                    total_detections
                )

            st.divider()

            # ------------------------------------------------
            # TABLE
            # ------------------------------------------------

            st.subheader(
                "📋 Image-by-Image Calculation"
            )

            df = pd.DataFrame(
                st.session_state.results
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # AVERAGES
            # ------------------------------------------------

            if images_processed > 0:

                average_potholes = (
                    total_potholes /
                    images_processed
                )

                average_waterlogging = (
                    total_waterlogging /
                    images_processed
                )

                st.subheader(
                    "📈 Average Detection"
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "Average Potholes / Image",
                        f"{average_potholes:.2f}"
                    )

                with a2:

                    st.metric(
                        "Average Waterlogging / Image",
                        f"{average_waterlogging:.2f}"
                    )

            # ------------------------------------------------
            # DOWNLOAD CALCULATION
            # ------------------------------------------------

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 Download Detection Report",
                data=csv_data,
                file_name="detection_report.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.divider()

            # ------------------------------------------------
            # NEW UPLOAD
            # ------------------------------------------------

            if st.button(
                "🔄 START NEW UPLOAD",
                type="primary",
                use_container_width=True
            ):

                start_new_upload()

                st.rerun()


# ============================================================
# VIDEO DETECTION
# ============================================================

elif mode == "🎥 Video Detection":

    st.header("🎥 Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key="video_uploader"
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button(
            "🎬 START VIDEO DETECTION",
            type="primary",
            use_container_width=True
        ):

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

                os.unlink(
                    temp_input.name
                )

                st.stop()

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

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_file.name,
                fourcc,
                fps,
                (width, height)
            )

            video_display = st.empty()

            progress = st.progress(0)

            video_potholes = 0

            video_waterlogging = 0

            frame_number = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                output, potholes, waterlogging = detect_image(
                    Image.fromarray(rgb),
                    confidence
                )

                video_potholes += potholes

                video_waterlogging += waterlogging

                output_bgr = cv2.cvtColor(
                    output,
                    cv2.COLOR_RGB2BGR
                )

                writer.write(
                    output_bgr
                )

                video_display.image(
                    output,
                    channels="RGB",
                    use_container_width=True
                )

                frame_number += 1

                if total_frames > 0:

                    progress.progress(
                        min(
                            frame_number /
                            total_frames,
                            1.0
                        )
                    )

            cap.release()

            writer.release()

            os.unlink(
                temp_input.name
            )

            progress.progress(1.0)

            st.success(
                "✅ Video detection completed!"
            )

            st.header(
                "📊 Video Detection Calculation"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Frames Processed",
                    frame_number
                )

            with c2:

                st.metric(
                    "🟨 Pothole Detections",
                    video_potholes
                )

            with c3:

                st.metric(
                    "🟦 Waterlogging Detections",
                    video_waterlogging
                )

            with open(
                output_file.name,
                "rb"
            ) as f:

                video_bytes = f.read()

            st.download_button(
                "📥 DOWNLOAD DETECTED VIDEO",
                data=video_bytes,
                file_name="pothole_waterlogging_detected.mp4",
                mime="video/mp4",
                use_container_width=True
            )

            os.unlink(
                output_file.name
            )


# ============================================================
# CAMERA
# ============================================================

elif mode == "📹 Camera Detection":

    st.header("📹 Camera Detection")

    camera_image = st.camera_input(
        "Take a road image"
    )

    if camera_image:

        image = Image.open(
            camera_image
        ).convert("RGB")

        output, potholes, waterlogging = detect_image(
            image,
            confidence
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Camera Image",
                use_container_width=True
            )

        with col2:

            st.image(
                output,
                caption="AI Detection",
                use_container_width=True
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "🟨 Potholes",
                potholes
            )

        with c2:

            st.metric(
                "🟦 Waterlogging",
                waterlogging
            )

        with c3:

            st.metric(
                "🚧 Total Problems",
                potholes + waterlogging
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚧 Pothole & Waterlogging Detection System | "
    "YOLO + Streamlit"
)