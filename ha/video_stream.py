from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import socket
import time


# ============================================================
# Camera settings
# ============================================================

WIDTH = 1280
HEIGHT = 720
FPS = 5
QUALITY = 25


# ============================================================
# Camera handler
# ============================================================

class CameraHandler(BaseHTTPRequestHandler):

    def do_GET(self):

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

        frame_count = 0
        fps_start = time.time()

        # Empty lines before FPS display
        print()
        print()

        while True:

            camera = None

            try:

                print("\nStarting camera...", flush=True)

                camera = subprocess.Popen(
                    [
                        "rpicam-vid",
                        "-t", "0",

                        "--width", str(WIDTH),
                        "--height", str(HEIGHT),

                        "--framerate", str(FPS),

                        "--codec", "mjpeg",
                        "--quality", str(QUALITY),

                        "--nopreview",

                        "--denoise", "cdn_off",

                        "-o", "-"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0
                )

                buffer = bytearray()

                while True:

                    data = camera.stdout.read(65536)

                    if not data:
                        break

                    buffer.extend(data)

                    while True:

                        # JPEG start
                        start = buffer.find(b"\xff\xd8")

                        if start < 0:
                            buffer.clear()
                            break

                        # JPEG end
                        end = buffer.find(
                            b"\xff\xd9",
                            start + 2
                        )

                        if end < 0:

                            if start > 0:
                                del buffer[:start]

                            break

                        end += 2

                        frame = bytes(buffer[start:end])

                        del buffer[:end]

                        # Send MJPEG frame
                        self.wfile.write(
                            b"--frame\r\n"
                        )

                        self.wfile.write(
                            b"Content-Type: image/jpeg\r\n"
                        )

                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )

                        self.wfile.write(frame)

                        self.wfile.write(
                            b"\r\n"
                        )

                        # FPS
                        frame_count += 1

                        elapsed = time.time() - fps_start

                        if elapsed >= 2:

                            actual_fps = frame_count / elapsed

                            print(
                                f"\rFPS: {actual_fps:.1f} | "
                                f"{WIDTH}x{HEIGHT} | "
                                f"JPEG {QUALITY}",
                                end="",
                                flush=True
                            )

                            frame_count = 0
                            fps_start = time.time()

                # Camera process stopped
                print(
                    "\nCamera stopped - restarting...",
                    flush=True
                )

            except (
                BrokenPipeError,
                ConnectionResetError
            ):

                print(
                    "\nBrowser disconnected.",
                    flush=True
                )

                break

            except Exception as e:

                print(
                    f"\nStream error: {e}",
                    flush=True
                )

            finally:

                if camera is not None:

                    try:
                        camera.terminate()
                        camera.wait(timeout=2)

                    except Exception:

                        try:
                            camera.kill()
                        except Exception:
                            pass

            # Short pause before restart
            time.sleep(1)


# ============================================================
# Determine Pi IP address
# ============================================================

s = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

try:

    s.connect(("8.8.8.8", 80))
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
print("      Raspberry Pi Camera Stream")
print("==========================================")
print()
print(f"Resolution : {WIDTH}x{HEIGHT}")
print(f"FPS target : {FPS}")
print(f"JPEG       : {QUALITY}%")
print(f"Browser    : http://{ip}:8080")
print("Stream started...")


# ============================================================
# Run server
# ============================================================

server.serve_forever()