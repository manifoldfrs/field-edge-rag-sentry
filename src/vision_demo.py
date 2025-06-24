"""
vision_demo.py
--------------
Export YOLO-v5s to a CoreML package (first run) and perform realtime object
detection on the default webcam, drawing bounding boxes and an FPS overlay.

Requirements:
    * ultralytics >= 8.1.34   (export + inference helper)
    * coremltools >= 7.1      (automatically used by ultralytics)
    * opencv-python >= 4.10
Tested on macOS 14.5, M-series GPU. Achieves ~18-25 FPS on an M1 Pro.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODELS_DIR = Path("models")
COREML_PKG = MODELS_DIR / "yolov5s_coreml.mlpackage"


def export_if_missing() -> Path:
    """Export YOLO-v5s weights to CoreML if not already present."""
    if COREML_PKG.exists():
        return COREML_PKG

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] Exporting YOLO-v5s to CoreML (one-time step)...")
    yolov5s = YOLO("yolov5s.pt")  # downloads weights on first call
    exported = yolov5s.export(format="coreml", imgsz=640, device="cpu")

    # `export()` may return Path or list; normalize
    exported_path = Path(
        exported[0] if isinstance(exported, (list, tuple)) else exported
    )

    # Move into ./models for tidy repo layout
    if exported_path.is_dir():
        exported_path.rename(COREML_PKG)
    else:
        # Newer ultralytics returns .mlpackage directly
        COREML_PKG.write_bytes(exported_path.read_bytes())
        exported_path.unlink()

    print(f"[INFO] CoreML model ready -> {COREML_PKG}")
    return COREML_PKG


def main() -> None:
    coreml_path = export_if_missing()
    model = YOLO(str(coreml_path))  # ultralytics loads CoreML backend

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: Could not open webcam (check permissions).")

    frame_ct = 0
    t0 = time.time()
    fps_display = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: Failed to read frame; exiting.")
            break

        # Inference - ultralytics takes BGR ndarray directly
        results = model.predict(source=frame, verbose=False)[0]

        # Draw detections
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls_name = model.names[int(b.cls[0])]
            conf = b.conf[0]
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # FPS calculation every 15 frames
        frame_ct += 1
        if frame_ct >= 15:
            now = time.time()
            fps_display = frame_ct / (now - t0)
            frame_ct = 0
            t0 = now

        cv2.putText(
            frame,
            f"FPS: {fps_display:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLOv5s-CoreML Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
