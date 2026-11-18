import os
import urllib.request
import cv2
import numpy as np

def prepare_assets():
    os.makedirs("assets", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("code", exist_ok=True)
    
    # Download sample images
    # We need:
    # 1. Standard scene / landscape / building image
    # 2. Face / portrait image with clear face, eyes, smile
    # 3. Multiple faces image
    # 4. Watch image
    # 5. Watermark logo
    # 6. Car / vehicle image
    # 7. Text image
    
    urls = {
        "assets/scene.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "assets/building.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
        "assets/face.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "assets/fruits.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
        "assets/watermark.png": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/opencv-logo-white.png",
        "assets/car.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/detect_blob.png"
    }

    # Also download a natural watch image and group face image from wikipedia / wikimedia commons
    extra_urls = {
        "assets/watch.jpg": "https://images.unsplash.com/photo-1524805444758-089113d48a6d?w=600&auto=format&fit=crop&q=80",
        "assets/multi_faces.jpg": "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=600&auto=format&fit=crop&q=80",
        "assets/traffic.jpg": "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=600&auto=format&fit=crop&q=80",
        "assets/nature.jpg": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=600&auto=format&fit=crop&q=80"
    }

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for path, url in {**urls, **extra_urls}.items():
        if not os.path.exists(path):
            try:
                print(f"Downloading {path}...")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    with open(path, 'wb') as f:
                        f.write(resp.read())
            except Exception as e:
                print(f"Failed {url}: {e}, creating synthetic asset")

    # If any asset failed or is needed synthetically, make sure valid images exist
    if not os.path.exists("assets/scene.jpg") or os.path.getsize("assets/scene.jpg") < 100:
        img = np.zeros((400, 500, 3), dtype=np.uint8)
        cv2.circle(img, (250, 200), 100, (0, 200, 255), -1)
        cv2.rectangle(img, (100, 250), (400, 350), (50, 180, 50), -1)
        cv2.imwrite("assets/scene.jpg", img)

    # Synthetic sample video for experiments 7, 26, 28, 39, 40
    # Create video with moving ball/car and moving text
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("assets/sample_video.mp4", fourcc, 20.0, (400, 300))
    for i in range(60):
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 230
        # Draw road
        cv2.rectangle(frame, (0, 180), (400, 300), (80, 80, 80), -1)
        # Draw moving car-like block
        cx = (i * 6) % 360
        cv2.rectangle(frame, (cx, 200), (cx + 50, 230), (0, 0, 220), -1)
        cv2.circle(frame, (cx + 12, 235), 8, (20, 20, 20), -1)
        cv2.circle(frame, (cx + 38, 235), 8, (20, 20, 20), -1)
        # Add frame text
        cv2.putText(frame, f"FRAME {i+1} : OPENCV DEMO", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)
        out.write(frame)
    out.release()
    print("Sample video generated successfully.")

if __name__ == "__main__":
    prepare_assets()
