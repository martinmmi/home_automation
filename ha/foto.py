import subprocess

IMAGE = "/tmp/foto.jpg"


print("Taking photo...")

subprocess.run([
    "rpicam-still",
    "--width", "3280",
    "--height", "2464",
    "--timeout", "1000",
    "--nopreview",
    "-o", IMAGE 
], check=True)

print(f"Photo captured: {IMAGE}")
