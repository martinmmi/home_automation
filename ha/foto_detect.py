import cv2
import subprocess


# ============================================================
# Model
# ============================================================

MODEL = "/root/model/mobilenet_iter_73000.caffemodel"
CONFIG = "/root/model/deploy.prototxt"

IMAGE = "/tmp/detect.jpg"
RESULT = "/tmp/detect_result.jpg"


# ============================================================
# Object classes
# ============================================================

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


# ============================================================
# Load model
# ============================================================

print("Loading model...")

net = cv2.dnn.readNetFromCaffe(
    CONFIG,
    MODEL
)

print("Model loaded.")
print()


# ============================================================
# Take photo
# ============================================================

print("Taking photo...")

subprocess.run([
    "rpicam-still",
    "--width", "1280",
    "--height", "720",
    "--timeout", "1000",
    "--nopreview",
    "-o", IMAGE
], check=True)

print("Photo captured.")
print()


# ============================================================
# Load image
# ============================================================

image = cv2.imread(IMAGE)

if image is None:
    print("ERROR: Could not load image.")
    exit(1)

height, width = image.shape[:2]

print(f"Image: {width}x{height}")
print("Detecting objects...")
print()


# ============================================================
# Prepare image for neural network
# ============================================================

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    0.007843,
    (300, 300),
    127.5
)

net.setInput(blob)

detections = net.forward()


# ============================================================
# Draw detections
# ============================================================

found = False

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    # Minimum confidence
    if confidence < 0.40:
        continue

    class_id = int(
        detections[0, 0, i, 1]
    )

    if class_id >= len(CLASSES):
        continue

    name = CLASSES[class_id]

    # Detection coordinates
    x1 = int(
        detections[0, 0, i, 3] * width
    )

    y1 = int(
        detections[0, 0, i, 4] * height
    )

    x2 = int(
        detections[0, 0, i, 5] * width
    )

    y2 = int(
        detections[0, 0, i, 6] * height
    )

    # Keep coordinates inside image
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # Yellow bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        3
    )

    # Label
    label = f"{name} {confidence * 100:.1f}%"

    # Text size
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )

    # Label background
    label_y1 = max(
        0,
        y1 - text_height - baseline - 8
    )

    label_y2 = y1

    cv2.rectangle(
        image,
        (x1, label_y1),
        (x1 + text_width + 10, label_y2),
        (0, 255, 255),
        -1
    )

    # Black text
    cv2.putText(
        image,
        label,
        (x1 + 5, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    print(
        f"{name}: {confidence * 100:.1f}%"
    )

    found = True


# ============================================================
# No objects found
# ============================================================

if not found:

    print("No objects detected.")


# ============================================================
# Save result
# ============================================================

cv2.imwrite(
    RESULT,
    image
)

print()
print("==========================================")
print("Detection finished.")
print(f"Original : {IMAGE}")
print(f"Result   : {RESULT}")
print("==========================================")