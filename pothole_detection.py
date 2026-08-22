from ultralytics import YOLO
import cv2
import os
import csv
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"C:\Users\M.N THUSHAR\runs\detect\train-2\weights\best.pt"

CONFIDENCE = 0.25

PHOTO_OUTPUT = "photo_results"
VIDEO_OUTPUT = "video_results"
WEBCAM_OUTPUT = "webcam_results"

os.makedirs(PHOTO_OUTPUT, exist_ok=True)
os.makedirs(VIDEO_OUTPUT, exist_ok=True)
os.makedirs(WEBCAM_OUTPUT, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO pothole model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print("GPU: NVIDIA GeForce RTX 4050")
print("CUDA device: 0")


# ============================================================
# SEVERITY ESTIMATION
# ============================================================

def calculate_severity(box, frame_width, frame_height):

    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    box_area = box_width * box_height
    frame_area = frame_width * frame_height

    percentage = (box_area / frame_area) * 100

    if percentage < 2:
        severity = "LOW"

    elif percentage < 7:
        severity = "MEDIUM"

    else:
        severity = "HIGH"

    return severity, percentage


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_detection(
    frame,
    box,
    confidence,
    severity,
    area_percentage
):

    x1, y1, x2, y2 = map(int, box)

    # Color based on severity
    if severity == "LOW":
        color = (0, 255, 0)

    elif severity == "MEDIUM":
        color = (0, 165, 255)

    else:
        color = (0, 0, 255)

    # Bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    # Label
    label = (
        f"POTHOLE | "
        f"{confidence * 100:.1f}% | "
        f"{severity}"
    )

    # Background for label
    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    cv2.rectangle(
        frame,
        (x1, max(0, y1 - th - 12)),
        (x1 + tw + 8, y1),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Area percentage
    cv2.putText(
        frame,
        f"Area: {area_percentage:.2f}%",
        (x1, y2 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


# ============================================================
# PHOTO DETECTION
# ============================================================

def detect_photo():

    print("\n====================================")
    print("        PHOTO POTHOLE DETECTION")
    print("====================================")

    image_path = input(
        "\nEnter the complete path of your image:\n"
    ).strip('"')

    if not os.path.exists(image_path):

        print("\nERROR: Image not found!")
        return

    frame = cv2.imread(image_path)

    if frame is None:

        print("\nERROR: Could not read image!")
        return

    height, width = frame.shape[:2]

    print("\nDetecting potholes...")

    start_time = time.time()

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    result = results[0]

    detections = []

    for box, confidence in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy()
    ):

        severity, area_percentage = calculate_severity(
            box,
            width,
            height
        )

        detections.append(
            (
                box,
                float(confidence),
                severity,
                area_percentage
            )
        )

        draw_detection(
            frame,
            box,
            float(confidence),
            severity,
            area_percentage
        )

    pothole_count = len(detections)

    processing_time = time.time() - start_time

    # Information panel
    cv2.rectangle(
        frame,
        (10, 10),
        (390, 125),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "AI POTHOLE DETECTION",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Potholes: {pothole_count}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Processing: {processing_time:.2f}s",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        PHOTO_OUTPUT,
        "detected_" + filename
    )

    cv2.imwrite(
        output_path,
        frame
    )

    # CSV
    csv_path = os.path.join(
        PHOTO_OUTPUT,
        "photo_report.csv"
    )

    write_csv_header = not os.path.exists(csv_path)

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if write_csv_header:

            writer.writerow([
                "Date",
                "Image",
                "Potholes",
                "Processing Time"
            ])

        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            filename,
            pothole_count,
            f"{processing_time:.2f}"
        ])

    print("\n====================================")
    print("PHOTO DETECTION COMPLETED")
    print("====================================")

    print(f"Potholes detected: {pothole_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Result: {output_path}")
    print(f"Report: {csv_path}")

    cv2.imshow(
        "AI Pothole Photo Detection",
        frame
    )

    print("\nPress any key to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# VIDEO DETECTION
# ============================================================

def detect_video():

    print("\n====================================")
    print("        VIDEO POTHOLE DETECTION")
    print("====================================")

    video_path = input(
        "\nEnter the complete path of your video:\n"
    ).strip('"')

    if not os.path.exists(video_path):

        print("\nERROR: Video not found!")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("\nERROR: Could not open video!")
        return

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    filename = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    output_path = os.path.join(
        VIDEO_OUTPUT,
        filename + "_detected.mp4"
    )

    csv_path = os.path.join(
        VIDEO_OUTPUT,
        filename + "_report.csv"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0

    total_detections = 0

    low_count = 0
    medium_count = 0
    high_count = 0

    max_potholes_frame = 0

    start_time = time.time()

    # CSV
    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(csv_file)

    writer.writerow([
        "Timestamp",
        "Frame",
        "Potholes",
        "Confidence",
        "Severity",
        "Area_Percentage"
    ])

    print("\nStarting detection...")
    print("Press Q to stop.\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            device=0,
            verbose=False
        )

        result = results[0]

        frame_potholes = len(result.boxes)

        if frame_potholes > max_potholes_frame:

            max_potholes_frame = frame_potholes

        total_detections += frame_potholes

        frame_time = (
            frame_number / fps
            if fps > 0
            else 0
        )

        minutes = int(frame_time // 60)

        seconds = int(frame_time % 60)

        timestamp = (
            f"{minutes:02d}:{seconds:02d}"
        )

        for box, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):

            confidence = float(confidence)

            severity, area_percentage = (
                calculate_severity(
                    box,
                    width,
                    height
                )
            )

            if severity == "LOW":
                low_count += 1

            elif severity == "MEDIUM":
                medium_count += 1

            else:
                high_count += 1

            draw_detection(
                frame,
                box,
                confidence,
                severity,
                area_percentage
            )

            writer.writerow([
                timestamp,
                frame_number,
                frame_potholes,
                f"{confidence:.3f}",
                severity,
                f"{area_percentage:.3f}"
            ])

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (410, 155),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            "AI POTHOLE DETECTION",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Potholes in frame: {frame_potholes}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"LOW: {low_count}  MED: {medium_count}  HIGH: {high_count}",
            (20, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Time: {timestamp}",
            (20, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        out.write(frame)

        cv2.imshow(
            "AI Pothole Video Detection - RTX 4050",
            frame
        )

        if frame_number % 30 == 0:

            progress = (
                frame_number / total_frames
            ) * 100

            print(
                f"Progress: {progress:.1f}%"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("\nDetection stopped by user.")

            break

    cap.release()
    out.release()
    csv_file.close()

    cv2.destroyAllWindows()

    processing_time = (
        time.time() - start_time
    )

    print("\n====================================")
    print("VIDEO DETECTION COMPLETED")
    print("====================================")

    print(f"Frames processed: {frame_number}")
    print(f"Maximum potholes in one frame: {max_potholes_frame}")
    print(f"Low detections: {low_count}")
    print(f"Medium detections: {medium_count}")
    print(f"High detections: {high_count}")
    print(f"Total detection events: {total_detections}")
    print(f"Processing time: {processing_time:.2f} seconds")

    print("\nOutput video:")
    print(output_path)

    print("\nCSV report:")
    print(csv_path)


# ============================================================
# WEBCAM DETECTION
# ============================================================

def detect_webcam():

    print("\n====================================")
    print("        LIVE POTHOLE DETECTION")
    print("====================================")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("\nERROR: Webcam could not be opened.")
        return

    print("\nWebcam started.")
    print("Press Q to stop.")

    total_detections = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        height, width = frame.shape[:2]

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            device=0,
            verbose=False
        )

        result = results[0]

        frame_potholes = len(
            result.boxes
        )

        total_detections += frame_potholes

        for box, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):

            severity, area_percentage = (
                calculate_severity(
                    box,
                    width,
                    height
                )
            )

            draw_detection(
                frame,
                box,
                float(confidence),
                severity,
                area_percentage
            )

        cv2.rectangle(
            frame,
            (10, 10),
            (390, 100),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            "LIVE AI POTHOLE DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Potholes: {frame_potholes}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Live Pothole Detection - RTX 4050",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()

    print("\nLive detection stopped.")
    print(
        f"Total detection events: {total_detections}"
    )


# ============================================================
# MAIN MENU
# ============================================================

while True:

    print("\n")
    print("========================================")
    print("       AI POTHOLE DETECTION SYSTEM")
    print("========================================")

    print()
    print("GPU: NVIDIA RTX 4050")
    print("CUDA: Enabled")
    print()

    print("1. Detect potholes in PHOTO")
    print("2. Detect potholes in VIDEO")
    print("3. Detect potholes using WEBCAM")
    print("4. Exit")
    print()

    choice = input(
        "Enter your choice (1/2/3/4): "
    )

    if choice == "1":

        detect_photo()

    elif choice == "2":

        detect_video()

    elif choice == "3":

        detect_webcam()

    elif choice == "4":

        print("\nExiting...")
        break

    else:

        print("\nInvalid choice!")
        print("Please select 1, 2, 3, or 4.")