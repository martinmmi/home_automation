import os
import json
import time
import socket
import subprocess
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import glob

# ============================================================
# Settings
# ============================================================

WIDTH = 1280
HEIGHT = 720
FPS = 5
QUALITY = 25

DETECT_INTERVAL = 2.0
CONFIDENCE = 0.40

AI_WIDTH = 300
AI_HEIGHT = 300

MODEL = "/root/model/mobilenet_iter_73000.caffemodel"
CONFIG = "/root/model/deploy.prototxt"
DETECT_PREFIX = "/tmp/detect_result_"

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080

MQTT_HOST = "192.168.2.160"
MQTT_PORT = 1883
MQTT_USER = "root"
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "Hallo16727451")
MQTT_PREFIX = "dietpi_camera"
MQTT_SHUTDOWN_TOPIC = f"{MQTT_PREFIX}/command/shutdown"
HA_DISCOVERY_PREFIX = "homeassistant"

DEVICE_ID = "dietpi_ai_camera"
DEVICE_NAME = "DietPi AI Kamera"

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
    "tvmonitor",
]

OBJECT_CLASSES = [name for name in CLASSES if name != "background"]

# ============================================================
# Shared data
# ============================================================

latest_detections = []
latest_frame = None
latest_jpeg = None
frame_id = 0
detect_number = 0

last_published_classes = None

ai_lock = threading.Lock()
frame_lock = threading.Lock()
stream_condition = threading.Condition()

stop_event = threading.Event()
mqtt_connected = threading.Event()
mqtt_client = None
net = None

# ============================================================
# Clear Old Images
# ============================================================

def clear_old_detect_images():
    for filename in glob.glob(f"{DETECT_PREFIX}*.jpg"):
        try:
            os.remove(filename)
        except OSError as exc:
            print(f"Could not delete {filename}: {exc}")

    print("Old detection images cleared.")

# ============================================================
# MQTT and Home Assistant discovery
# ============================================================

def mqtt_publish(topic, payload, retain=True):
    if mqtt_client is None or not mqtt_connected.is_set():
        return

    try:
        mqtt_client.publish(topic, payload, qos=0, retain=retain)
    except Exception as exc:
        print(f"MQTT publish error: {exc}")


def publish_discovery():
    device = {
        "identifiers": [DEVICE_ID],
        "name": DEVICE_NAME,
        "manufacturer": "OpenCV / MobileNet SSD",
        "model": "Raspberry Pi",
    }

    for name in OBJECT_CLASSES:
        slug = name.replace(" ", "_")
        state_topic = f"{MQTT_PREFIX}/objects/{slug}/state"
        confidence_topic = f"{MQTT_PREFIX}/objects/{slug}/confidence"

        binary_config = {
            "name": name,
            "unique_id": f"{DEVICE_ID}_{slug}",
            "state_topic": state_topic,
            "payload_on": "ON",
            "payload_off": "OFF",
            "device": device,
        }

        mqtt_publish(
            f"{HA_DISCOVERY_PREFIX}/binary_sensor/{DEVICE_ID}/{slug}/config",
            json.dumps(binary_config),
        )

        confidence_config = {
            "name": f"{name} Konfidenz",
            "unique_id": f"{DEVICE_ID}_{slug}_confidence",
            "state_topic": confidence_topic,
            "unit_of_measurement": "%",
            "icon": "mdi:percent",
            "device": device,
        }

        mqtt_publish(
            f"{HA_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/{slug}_confidence/config",
            json.dumps(confidence_config),
        )

    # Additional sensor: last detected AI classes
    mqtt_publish(
        f"{HA_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/last_classes/config",
        json.dumps({
            "name": "Letzte KI-Klassen",
            "unique_id": f"{DEVICE_ID}_last_classes",
            "state_topic": f"{MQTT_PREFIX}/last_classes/state",
            "icon": "mdi:tag-multiple",
            "device": device,
        }),
    )

    # Additional sensor: timestamp of the last class-list change
    mqtt_publish(
        f"{HA_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/last_class_change/config",
        json.dumps({
            "name": "Letzter Klassenwechsel",
            "unique_id": f"{DEVICE_ID}_last_class_change",
            "state_topic": f"{MQTT_PREFIX}/last_class_change/state",
            "device_class": "timestamp",
            "icon": "mdi:clock-outline",
            "device": device,
        }),
    )


def on_mqtt_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        mqtt_connected.set()
        print("MQTT connected.")
        publish_discovery()

        client.subscribe(MQTT_SHUTDOWN_TOPIC)
        print(f"Shutdown-Topic abonniert: {MQTT_SHUTDOWN_TOPIC}")

    else:
        mqtt_connected.clear()
        print(f"MQTT connection failed. Reason: {reason_code}")


def on_mqtt_message(client, userdata, msg):
    print(f"MQTT-Nachricht empfangen: Topic={msg.topic}, Payload={msg.payload!r}", flush=True)

    if msg.topic != MQTT_SHUTDOWN_TOPIC:
        return

    if msg.payload.decode("utf-8", errors="ignore").strip() != "SHUTDOWN":
        print("Ungueltiger Shutdown-Befehl ignoriert.")
        return

    print("Shutdown-Befehl von Home Assistant empfangen.")

    try:
        subprocess.Popen([
            "/usr/sbin/shutdown", "-h", "now"
        ])
    except Exception as exc:
        print(f"Shutdown-Fehler: {exc}")


def on_mqtt_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    mqtt_connected.clear()
    print(f"MQTT disconnected. Reason: {reason_code}")


def start_mqtt():
    global mqtt_client

    if not MQTT_PASSWORD:
        raise RuntimeError("MQTT password is missing.")

    mqtt_client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=DEVICE_ID,
        protocol=mqtt.MQTTv311,
    )

    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.on_disconnect = on_mqtt_disconnect

    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

    mqtt_client.connect_async(
        MQTT_HOST,
        MQTT_PORT,
        keepalive=60,
    )

    mqtt_client.loop_start()


def publish_detections(detections):
    global last_published_classes

    best_confidence = {name: 0.0 for name in OBJECT_CLASSES}

    for name, confidence, *_coords in detections:
        if name in best_confidence:
            best_confidence[name] = max(
                best_confidence[name],
                float(confidence),
            )

    for name, confidence in best_confidence.items():
        slug = name.replace(" ", "_")
        state = "ON" if confidence >= CONFIDENCE else "OFF"

        mqtt_publish(
            f"{MQTT_PREFIX}/objects/{slug}/state",
            state,
        )

        mqtt_publish(
            f"{MQTT_PREFIX}/objects/{slug}/confidence",
            f"{confidence * 100:.1f}",
        )

    # Publish the current list of detected classes
    current_classes = [
        name
        for name in OBJECT_CLASSES
        if best_confidence[name] >= CONFIDENCE
    ]

    classes_text = ", ".join(current_classes) if current_classes else "Keine"

    mqtt_publish(
        f"{MQTT_PREFIX}/last_classes/state",
        classes_text,
    )

    # Publish timestamp only when the class list changes
    if current_classes != last_published_classes:
        timestamp = datetime.now(timezone.utc).isoformat()

        mqtt_publish(
            f"{MQTT_PREFIX}/last_class_change/state",
            timestamp,
        )

        last_published_classes = current_classes.copy()


# ============================================================
# Object detection
# ============================================================

def detect_objects(image):
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        0.007843,
        (AI_WIDTH, AI_HEIGHT),
        127.5,
    )

    net.setInput(blob)
    results = net.forward()

    detections = []

    for i in range(results.shape[2]):
        confidence = float(results[0, 0, i, 2])

        if confidence < CONFIDENCE:
            continue

        class_id = int(results[0, 0, i, 1])

        if class_id <= 0 or class_id >= len(CLASSES):
            continue

        name = CLASSES[class_id]

        x1 = int(results[0, 0, i, 3] * width)
        y1 = int(results[0, 0, i, 4] * height)
        x2 = int(results[0, 0, i, 5] * width)
        y2 = int(results[0, 0, i, 6] * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        detections.append((name, confidence, x1, y1, x2, y2))

    return detections


# ============================================================
# Draw detections
# ============================================================

def draw_detections(image, detections):
    for name, confidence, x1, y1, x2, y2 in detections:
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3,
        )

        label = f"{name} {confidence * 100:.1f}%"

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )

        label_y1 = max(0, y1 - text_height - baseline - 8)

        cv2.rectangle(
            image,
            (x1, label_y1),
            (x1 + text_width + 10, y1),
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            image,
            label,
            (x1 + 5, max(text_height, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return image


# ============================================================
# Persistent camera worker
# ============================================================

def camera_worker():
    global latest_frame, latest_jpeg, frame_id

    print("Camera worker started.")

    command = [
        "rpicam-vid",
        "-t", "0",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FPS),
        "--codec", "mjpeg",
        "--quality", str(QUALITY),
        "--nopreview",
        "--denoise", "cdn_off",
        "-o", "-",
    ]

    while not stop_event.is_set():
        camera = None

        try:
            print("\nStarting camera...")

            camera = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )

            buffer = bytearray()
            frame_count = 0
            fps_start = time.time()

            while not stop_event.is_set():
                data = camera.stdout.read(4096)

                if not data:
                    break

                buffer.extend(data)

                while True:
                    start = buffer.find(b"\xff\xd8")

                    if start < 0:
                        if len(buffer) > 2:
                            del buffer[:-2]
                        break

                    end = buffer.find(b"\xff\xd9", start + 2)

                    if end < 0:
                        if start > 0:
                            del buffer[:start]
                        break

                    end += 2
                    jpeg_data = bytes(buffer[start:end])
                    del buffer[:end]

                    array = np.frombuffer(jpeg_data, dtype=np.uint8)
                    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

                    if image is None:
                        continue

                    with frame_lock:
                        latest_frame = image.copy()

                    with ai_lock:
                        detections = list(latest_detections)

                    display = draw_detections(image.copy(), detections)

                    success, encoded = cv2.imencode(
                        ".jpg",
                        display,
                        [cv2.IMWRITE_JPEG_QUALITY, QUALITY],
                    )

                    if not success:
                        continue

                    with stream_condition:
                        latest_jpeg = encoded.tobytes()
                        frame_id += 1
                        stream_condition.notify_all()

                    frame_count += 1
                    elapsed = time.time() - fps_start

                    if elapsed >= 2:
                        print(
                            f"\rStream FPS: {frame_count / elapsed:.1f} | "
                            f"{WIDTH}x{HEIGHT} | JPEG {QUALITY}",
                            end="",
                            flush=True,
                        )
                        frame_count = 0
                        fps_start = time.time()

        except Exception as exc:
            if not stop_event.is_set():
                print(f"\nCamera error: {exc}")

        finally:
            if camera is not None:
                try:
                    camera.terminate()
                    camera.wait(timeout=2)
                except Exception:
                    try:
                        camera.kill()
                        camera.wait(timeout=2)
                    except Exception:
                        pass

        if not stop_event.is_set():
            print("\nCamera stopped. Restarting in 2 seconds...")
            stop_event.wait(2)


# ============================================================
# AI worker
# ============================================================

def ai_worker():
    global latest_detections, detect_number

    print("AI worker started.")

    while not stop_event.is_set():
        if stop_event.wait(DETECT_INTERVAL):
            break

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        small = cv2.resize(frame, (AI_WIDTH, AI_HEIGHT))

        try:
            detections = detect_objects(small)
        except Exception as exc:
            print(f"\nAI error: {exc}")
            continue

        scale_x = WIDTH / AI_WIDTH
        scale_y = HEIGHT / AI_HEIGHT

        scaled = []

        for name, confidence, x1, y1, x2, y2 in detections:
            scaled.append(
                (
                    name,
                    confidence,
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                )
            )

        with ai_lock:
            latest_detections = scaled

        publish_detections(scaled)

        result_image = draw_detections(frame.copy(), scaled)

        detect_number += 1
        filename = f"{DETECT_PREFIX}{detect_number:04d}.jpg"

        success = cv2.imwrite(
            filename,
            result_image,
            [cv2.IMWRITE_JPEG_QUALITY, QUALITY],
        )

        if success:
            print(f"\nAI image saved: {filename}")

        if scaled:
            for name, confidence, *_coords in scaled:
                print(f"{name}: {confidence * 100:.1f}%")
        else:
            print("No objects detected.")


# ============================================================
# HTTP MJPEG stream
# ============================================================


class CameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        
        if self.path == "/photos":
            files = sorted(
                glob.glob(f"{DETECT_PREFIX}*.jpg"),
                reverse=True,
            )

            items = []
            for filename in files:
                name = os.path.basename(filename)
                items.append(
                    f'<a href="/photos/{name}">'
                    f'<img src="/photos/{name}" loading="lazy" alt="{name}">'
                    f'<p>{name}</p></a>'
                )

            html = (
                '<!doctype html><html><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>Pictures AI</title>'
                '<style>'
                'body{font-family:sans-serif;background:#181818;color:white;margin:16px}'
                '.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}'
                'a{color:white;text-decoration:none;overflow-wrap:anywhere}'
                'img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#303030}'
                '</style></head><body><div class="gallery">'
                + "".join(items)
                + '</div></body></html>'
            )

            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/photos/"):
            name = self.path[len("/photos/"):]

            if (
                "/" in name
                or not name.startswith("detect_result_")
                or not name.endswith(".jpg")
            ):
                self.send_error(404)
                return

            try:
                with open(os.path.join("/tmp", name), "rb") as image_file:
                    body = image_file.read()
            except OSError:
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return

        self.connection.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        last_frame_id = -1

        try:
            print("\nStream client connected.")

            while not stop_event.is_set():
                with stream_condition:
                    stream_condition.wait_for(
                        lambda: stop_event.is_set()
                        or (
                            latest_jpeg is not None
                            and frame_id != last_frame_id
                        ),
                        timeout=5,
                    )

                    if stop_event.is_set():
                        break

                    jpeg = latest_jpeg
                    current_id = frame_id

                if jpeg is None or current_id == last_frame_id:
                    continue

                last_frame_id = current_id

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("\nStream client disconnected.")

        except Exception as exc:
            print(f"\nStream error: {exc}")

    def log_message(self, format, *args):
        return


# ============================================================
# Main
# ============================================================

def main():
    global net

    print("Loading object detection model...")

    net = cv2.dnn.readNetFromCaffe(CONFIG, MODEL)

    print("Model loaded.")

    start_mqtt()

    clear_old_detect_images()

    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True,
    )
    camera_thread.start()

    ai_thread = threading.Thread(
        target=ai_worker,
        daemon=True,
    )
    ai_thread.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "unknown"
    finally:
        sock.close()

    server = ThreadingHTTPServer(
        (HTTP_HOST, HTTP_PORT),
        CameraHandler,
    )
    server.daemon_threads = True
    server.timeout = 1

    print()
    print("==========================================")
    print("       Raspberry Pi AI Camera")
    print("==========================================")
    print(f"Resolution      : {WIDTH}x{HEIGHT}")
    print(f"FPS target      : {FPS}")
    print(f"AI interval     : {DETECT_INTERVAL}s")
    print(f"AI resolution   : {AI_WIDTH}x{AI_HEIGHT}")
    print(f"Confidence      : {CONFIDENCE * 100:.0f}%")
    print(f"Stream          : http://{ip}:{HTTP_PORT}/")
    print(f"MQTT broker     : http://{MQTT_HOST}:{MQTT_PORT}/")
    print("Starting HTTP stream server...")

    try:
        while not stop_event.is_set():
            server.handle_request()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()

        with stream_condition:
            stream_condition.notify_all()

        server.server_close()

        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
