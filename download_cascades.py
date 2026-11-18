import os
import urllib.request

cascades = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml",
    "haarcascade_smile.xml",
    "haarcascade_fullbody.xml",
    "haarcascade_russian_plate_number.xml"
]

base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
os.makedirs("assets/cascades", exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0'}
for c in cascades:
    target = os.path.join("assets/cascades", c)
    if not os.path.exists(target):
        url = base_url + c
        print(f"Downloading {c}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                with open(target, 'wb') as f:
                    f.write(resp.read())
            print(f"Saved {c} ({os.path.getsize(target)} bytes)")
        except Exception as e:
            print(f"Error {c}: {e}")
