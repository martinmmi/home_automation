from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import socket
import time
import cv2
import numpy as np
import threading


# ============================================================
# Settings
# ============================================================

WIDTH = 1280
HEIGHT = 720
FPS = 5
QUALITY = 25

DETECT_INTERVAL = 5.0
CONFIDENCE = 0.40

# AI arbeitet nur mit diesem kleinen Bild
AI_WIDTH = 640
AI_HEIGHT = 360

MODEL = "/root/model/mobilenet_iter_73000.caffemodel"
CONFIG = "/root/model/deploy.prototxt"


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
# Load AI model
# ============================================================

print("Loading object detection model...")

net = cv2.dnn.readNetFromCaffe(
    CONFIG,
    MODEL
)

print("Model loaded.")
print()


# ============================================================
# Shared AI data
# ============================================================

latest_detections = []
latest_frame = None

ai_lock = threading.Lock()
frame_lock = threading.Lock()

ai_running = True


# ============================================================
# Object detection
# ============================================================

def detect_objects(image):

    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(
            image,
            (300, 300)
        ),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)

    detections = net.forward()

    results = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < CONFIDENCE:
            continue

        class_id = int(
            detections[0, 0, i, 1]
        )

        if class_id >= len(CLASSES):
            continue

        name = CLASSES[class_id]

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

        x1 = max(
            0,
            min(x1, width - 1)
        )

        y1 = max(
            0,
            min(y1, height - 1)
        )

        x2 = max(
            0,
            min(x2, width - 1)
        )

        y2 = max(
            0,
            min(y2, height - 1)
        )

        results.append(
            (
                name,
                confidence,
                x1,
                y1,
                x2,
                y2
            )
        )

    return results


# ============================================================
# AI worker
# ============================================================

def ai_worker():

    global latest_detections
    global latest_frame

    print("AI worker started.")

    while ai_running:

        time.sleep(DETECT_INTERVAL)

        with frame_lock:

            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        # ----------------------------------------------------
        # Resize image for AI
        # ----------------------------------------------------

        small = cv2.resize(
            frame,
            (AI_WIDTH, AI_HEIGHT)
        )

        # ----------------------------------------------------
        # Run detection
        # ----------------------------------------------------

        try:

            detections = detect_objects(
                small
            )

        except Exception as e:

            print(
                f"\nAI error: {e}"
            )

            continue

        # ----------------------------------------------------
        # Scale coordinates back to 1280x720
        # ----------------------------------------------------

        scale_x = WIDTH / AI_WIDTH
        scale_y = HEIGHT / AI_HEIGHT

        scaled = []

        for (
            name,
            confidence,
            x1,
            y1,
            x2,
            y2
        ) in detections:

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            scaled.append(
                (
                    name,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

        with ai_lock:

            latest_detections = scaled

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print()

        if scaled:

            for (
                name,
                confidence,
                x1,
                y1,
                x2,
                y2
            ) in scaled:

                print(
                    f"{name}: "
                    f"{confidence * 100:.1f}%"
                )

        else:

            print(
                "No objects detected."
            )


# ============================================================
# Draw detections
# ============================================================

def draw_detections(
    image,
    detections
):

    for (
        name,
        confidence,
        x1,
        y1,
        x2,
        y2
    ) in detections:

        # Yellow rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3
        )

        label = (
            f"{name} "
            f"{confidence * 100:.1f}%"
        )

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        label_y1 = max(
            0,
            y1 - text_height - baseline - 8
        )

        label_y2 = y1

        # Yellow label background
        cv2.rectangle(
            image,
            (x1, label_y1),
            (
                x1 + text_width + 10,
                label_y2
            ),
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

    return image


# ============================================================
# HTTP handler
# ============================================================

class CameraHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        global latest_frame

        if self.path != "/":
            self.send_error(404)
            return

        self.connection.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )

        self.send_header(
            "Cache-Control",
            "no-cache, no-store"
        )

        self.send_header(
            "Connection",
            "close"
        )

        self.end_headers()

        camera = None

        try:

            print()
            print("Starting camera...")

            camera = subprocess.Popen(
                [
                    "rpicam-vid",
                    "-t", "0",

                    "--width",
                    str(WIDTH),

                    "--height",
                    str(HEIGHT),

                    "--framerate",
                    str(FPS),

                    "--codec",
                    "mjpeg",

                    "--quality",
                    str(QUALITY),

                    "--nopreview",

                    "--denoise",
                    "cdn_off",

                    "-o",
                    "-"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )

            buffer = bytearray()

            frame_count = 0
            fps_start = time.time()

            print()
            print()

            while True:

                data = camera.stdout.read(
                    65536
                )

                if not data:
                    break

                buffer.extend(data)

                while True:

                    start = buffer.find(
                        b"\xff\xd8"
                    )

                    if start < 0:

                        buffer.clear()
                        break

                    end = buffer.find(
                        b"\xff\xd9",
                        start + 2
                    )

                    if end < 0:

                        if start > 0:
                            del buffer[:start]

                        break

                    end += 2

                    jpeg = bytes(
                        buffer[start:end]
                    )

                    del buffer[:end]

                    # ====================================================
                    # Decode JPEG
                    # ====================================================

                    array = np.frombuffer(
                        jpeg,
                        dtype=np.uint8
                    )

                    image = cv2.imdecode(
                        array,
                        cv2.IMREAD_COLOR
                    )

                    if image is None:
                        continue

                    # ====================================================
                    # Give latest frame to AI thread
                    # ====================================================

                    with frame_lock:

                        latest_frame = image.copy()

                    # ====================================================
                    # Get latest detections
                    # ====================================================

                    with ai_lock:

                        detections = list(
                            latest_detections
                        )

                    # ====================================================
                    # Draw detections
                    # ====================================================

                    image = draw_detections(
                        image,
                        detections
                    )

                    # ====================================================
                    # Encode JPEG
                    # ====================================================

                    success, encoded = cv2.imencode(
                        ".jpg",
                        image,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            QUALITY
                        ]
                    )

                    if not success:
                        continue

                    frame = encoded.tobytes()

                    # ====================================================
                    # Send frame
                    # ====================================================

                    self.wfile.write(
                        b"--frame\r\n"
                    )

                    self.wfile.write(
                        b"Content-Type: image/jpeg\r\n"
                    )

                    self.wfile.write(
                        f"Content-Length: "
                        f"{len(frame)}\r\n\r\n".encode()
                    )

                    self.wfile.write(
                        frame
                    )

                    self.wfile.write(
                        b"\r\n"
                    )

                    # ====================================================
                    # FPS display
                    # ====================================================

                    frame_count += 1

                    elapsed = (
                        time.time()
                        - fps_start
                    )

                    if elapsed >= 2:

                        actual_fps = (
                            frame_count
                            / elapsed
                        )

                        print(
                            f"\rStream FPS: "
                            f"{actual_fps:.1f} | "
                            f"{WIDTH}x{HEIGHT} | "
                            f"JPEG {QUALITY}",
                            end="",
                            flush=True
                        )

                        frame_count = 0
                        fps_start = time.time()

        except (
            BrokenPipeError,
            ConnectionResetError
        ):

            print(
                "\nBrowser disconnected."
            )

        except Exception as e:

            print(
                f"\nStream error: {e}"
            )

        finally:

            if camera is not None:

                try:

                    camera.terminate()

                    camera.wait(
                        timeout=2
                    )

                except Exception:

                    try:
                        camera.kill()
                    except Exception:
                        pass


# ============================================================
# Start AI thread
# ============================================================

ai_thread = threading.Thread(
    target=ai_worker,
    daemon=True
)

ai_thread.start()


# ============================================================
# Determine IP address
# ============================================================

s = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

try:

    s.connect(
        ("8.8.8.8", 80)
    )

    ip = s.getsockname()[0]

except Exception:

    ip = "unknown"

finally:

    s.close()


# ============================================================
# Start web server
# ============================================================

server = ThreadingHTTPServer(
    ("0.0.0.0", 8080),
    CameraHandler
)


# ============================================================
# Console information
# ============================================================

print()
print("==========================================")
print("     Raspberry Pi AI Camera Stream")
print("==========================================")
print()

print(f"Resolution      : {WIDTH}x{HEIGHT}")
print(f"FPS target      : {FPS}")
print(f"JPEG            : {QUALITY}%")
print(f"Browser         : http://{ip}:8080")
print(f"AI interval     : {DETECT_INTERVAL}s")
print(f"AI resolution   : {AI_WIDTH}x{AI_HEIGHT}")
print(f"Confidence      : {CONFIDENCE * 100:.0f}%")
print("Stream started...")

# ============================================================
# Run server
# ============================================================

server.serve_forever()