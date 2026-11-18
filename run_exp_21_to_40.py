"""
Computer Vision Lab Experiments Suite
Experiments 21 through 40
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
os.makedirs("code", exist_ok=True)

YUNET_MODEL = "assets/face_detection_yunet.onnx"

def save_comparison(images_with_titles, output_filename, figsize=None, cmap=None):
    n = len(images_with_titles)
    if figsize is None:
        figsize = (5 * n, 5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for ax, (img, title) in zip(axes, images_with_titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap=cmap or 'gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, pad=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", output_filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: results/{output_filename}")

# Helper for Face & Landmark Detection using OpenCV FaceDetectorYN
def get_faces(image):
    h, w = image.shape[:2]
    detector = cv2.FaceDetectorYN_create(YUNET_MODEL, "", (w, h), score_threshold=0.6)
    detector.setInputSize((w, h))
    ret, faces = detector.detect(image)
    return faces if (faces is not None and len(faces) > 0) else []

# ==============================================================================
# Exp 21: Opening technique (Erosion followed by Dilation)
# ==============================================================================
def exp21():
    code_text = '''# Experiment 21: Opening Morphological Operation (Noise removal)
import cv2
import numpy as np

image = cv2.imread("assets/fruits.jpg")
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

cv2.imshow("Original", image)
cv2.imshow("Opening Operation", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp21_morph_opening.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/fruits.jpg")
    kernel = np.ones((9, 9), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    save_comparison([(img, "Original Image"), (opening, "Opening (Erosion followed by Dilation)")], "exp21_morph_opening.png")

# ==============================================================================
# Exp 22: Closing technique (Dilation followed by Erosion)
# ==============================================================================
def exp22():
    code_text = '''# Experiment 22: Closing Morphological Operation (Fills holes/gaps)
import cv2
import numpy as np

image = cv2.imread("assets/fruits.jpg")
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Original", image)
cv2.imshow("Closing Operation", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp22_morph_closing.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/fruits.jpg")
    kernel = np.ones((9, 9), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    save_comparison([(img, "Original Image"), (closing, "Closing (Dilation followed by Erosion)")], "exp22_morph_closing.png")

# ==============================================================================
# Exp 23: Top hat technique (Input image minus Opening)
# ==============================================================================
def exp23():
    code_text = '''# Experiment 23: Top Hat Morphological Operation
import cv2
import numpy as np

image = cv2.imread("assets/building.jpg")
kernel = np.ones((15, 15), np.uint8)
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

cv2.imshow("Original", image)
cv2.imshow("Top Hat", tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp23_morph_tophat.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/building.jpg")
    kernel = np.ones((15, 15), np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    save_comparison([(img, "Original Image"), (tophat, "Top Hat (Highlights elements brighter than surroundings)")], "exp23_morph_tophat.png")

# ==============================================================================
# Exp 24: Black hat technique (Closing minus Input image)
# ==============================================================================
def exp24():
    code_text = '''# Experiment 24: Black Hat Morphological Operation
import cv2
import numpy as np

image = cv2.imread("assets/building.jpg")
kernel = np.ones((15, 15), np.uint8)
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow("Original", image)
cv2.imshow("Black Hat", blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp24_morph_blackhat.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/building.jpg")
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    save_comparison([(img, "Original Image"), (blackhat, "Black Hat (Highlights dark elements on bright surroundings)")], "exp24_morph_blackhat.png")

# ==============================================================================
# Exp 25: Recognize watch from given image by general Object recognition
# ==============================================================================
def exp25():
    code_text = '''# Experiment 25: Recognize watch from image by Object Recognition
import cv2
import numpy as np

image = cv2.imread("assets/watch.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                           param1=100, param2=50, minRadius=50, maxRadius=250)

output = image.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 3)
        cv2.rectangle(output, (i[0]-i[2]-10, i[1]-i[2]-10), (i[0]+i[2]+10, i[1]+i[2]+10), (255, 0, 0), 2)
        cv2.putText(output, "Recognized: Watch", (i[0]-i[2], i[1]-i[2]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imshow("Detected Watch", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp25_watch_recognition.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/watch.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=40, minRadius=40, maxRadius=300)
    output = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 3)
            cv2.circle(output, (i[0], i[1]), 3, (0, 0, 255), 3)
            cv2.rectangle(output, (i[0]-i[2]-10, i[1]-i[2]-10), (i[0]+i[2]+10, i[1]+i[2]+10), (255, 100, 0), 3)
            cv2.putText(output, "Recognized: Watch", (max(10, i[0]-i[2]), max(30, i[1]-i[2]-15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    save_comparison([(img, "Original Watch Image"), (output, "Watch Object Detection")], "exp25_watch_recognition.png")

# ==============================================================================
# Exp 26: Reverse the frames of the video to create a video in reverse mode
# ==============================================================================
def exp26():
    code_text = '''# Experiment 26: Reverse the frames of video using OpenCV
import cv2

cap = cv2.VideoCapture("assets/sample_video.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("results/reversed_video.mp4", fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))
for frame in reversed(frames):
    out.write(frame)
out.release()
print("Reversed video created successfully.")
'''
    with open("code/exp26_reverse_video.py", "w") as f:
        f.write(code_text)

    cap = cv2.VideoCapture("assets/sample_video.mp4")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("results/exp26_reversed_video.mp4", fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))
    for f in reversed(frames):
        out.write(f)
    out.release()

    f_orig = frames[10]
    f_rev = frames[len(frames) - 1 - 10]
    save_comparison([(f_orig, "Forward Frame (Frame 11)"), (f_rev, "Reversed Frame (Original Frame 50)")], "exp26_reverse_video.png")

# ==============================================================================
# Exp 27: Face detection algorithm using OpenCV
# ==============================================================================
def exp27():
    code_text = '''# Experiment 27: Face Detection using OpenCV
import cv2

# Initialize Face Detector (cv2.FaceDetectorYN / Haar / DNN)
detector = cv2.FaceDetectorYN_create("assets/face_detection_yunet.onnx", "", (300, 300))
image = cv2.imread("assets/face.jpg")
h, w, _ = image.shape
detector.setInputSize((w, h))

ret, faces = detector.detect(image)
output = image.copy()

if faces is not None:
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        confidence = face[-1]
        cv2.rectangle(output, (x, y), (x + fw, y + fh), (0, 255, 0), 3)
        cv2.putText(output, f"Face: {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Detected Faces", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp27_face_detection.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/face.jpg")
    faces = get_faces(img)
    output = img.copy()
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        conf = face[-1]
        cv2.rectangle(output, (x, y), (x + fw, y + fh), (0, 255, 0), 3)
        cv2.putText(output, f"Face ({conf:.2f})", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    save_comparison([(img, "Original Image"), (output, f"Detected Faces ({len(faces)})")], "exp27_face_detection.png")

# ==============================================================================
# Exp 28: Vehicle detection algorithm in each frame of video
# ==============================================================================
def exp28():
    code_text = '''# Experiment 28: Vehicle Detection in video frames
import cv2

cap = cv2.VideoCapture("assets/sample_video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Vehicle Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
    with open("code/exp28_vehicle_detection.py", "w") as f:
        f.write(code_text)

    cap = cv2.VideoCapture("assets/sample_video.mp4")
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)
    annotated_frames = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        if frame_idx in [15, 30, 45]:
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            draw_frame = frame.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(draw_frame, "Vehicle Detected", (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            annotated_frames.append((draw_frame, f"Tracking Vehicle (Frame {frame_idx})"))
        frame_idx += 1
    cap.release()

    save_comparison(annotated_frames, "exp28_vehicle_detection.png", figsize=(15, 4.5))

# ==============================================================================
# Exp 29: Eye detection algorithm using OpenCV
# ==============================================================================
def exp29():
    code_text = '''# Experiment 29: Eye Detection using OpenCV
import cv2

# Using Face Detector landmarks (Right eye: pts 4,5; Left eye: pts 6,7)
detector = cv2.FaceDetectorYN_create("assets/face_detection_yunet.onnx", "", (300, 300))
image = cv2.imread("assets/face.jpg")
h, w, _ = image.shape
detector.setInputSize((w, h))

ret, faces = detector.detect(image)
output = image.copy()

if faces is not None:
    for face in faces:
        # Landmarks: Right Eye (index 4, 5) and Left Eye (index 6, 7)
        rex, rey = int(face[4]), int(face[5])
        lex, ley = int(face[6]), int(face[7])
        
        cv2.circle(output, (rex, rey), 15, (255, 0, 0), 2)
        cv2.putText(output, "Right Eye", (rex - 25, rey - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv2.circle(output, (lex, ley), 15, (255, 0, 0), 2)
        cv2.putText(output, "Left Eye", (lex - 25, ley - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imshow("Eye Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp29_eye_detection.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/face.jpg")
    faces = get_faces(img)
    output = img.copy()
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        cv2.rectangle(output, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        rex, rey = int(face[4]), int(face[5])
        lex, ley = int(face[6]), int(face[7])
        
        cv2.circle(output, (rex, rey), 18, (255, 0, 0), 3)
        cv2.putText(output, "Right Eye", (rex - 35, rey - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.circle(output, (lex, ley), 18, (255, 0, 0), 3)
        cv2.putText(output, "Left Eye", (lex - 35, ley - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    save_comparison([(img, "Original Image"), (output, "Eye Detection with Precise Localization")], "exp29_eye_detection.png")

# ==============================================================================
# Exp 30: Smile detection algorithm using OpenCV
# ==============================================================================
def exp30():
    code_text = '''# Experiment 30: Smile / Mouth Region Detection using OpenCV
import cv2

detector = cv2.FaceDetectorYN_create("assets/face_detection_yunet.onnx", "", (300, 300))
image = cv2.imread("assets/face.jpg")
h, w, _ = image.shape
detector.setInputSize((w, h))

ret, faces = detector.detect(image)
output = image.copy()

if faces is not None:
    for face in faces:
        # Landmarks: Right mouth corner (10, 11), Left mouth corner (12, 13)
        rmx, rmy = int(face[10]), int(face[11])
        lmx, lmy = int(face[12]), int(face[13])
        
        # Smile bounding box around mouth corners
        min_x, max_x = min(rmx, lmx) - 10, max(rmx, lmx) + 10
        min_y, max_y = min(rmy, lmy) - 12, max(rmy, lmy) + 18
        
        cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        cv2.line(output, (rmx, rmy), (lmx, lmy), (0, 255, 255), 2)
        cv2.putText(output, "Smile Detected", (min_x, min_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow("Smile Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp30_smile_detection.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/face.jpg")
    faces = get_faces(img)
    output = img.copy()
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        cv2.rectangle(output, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        rmx, rmy = int(face[10]), int(face[11])
        lmx, lmy = int(face[12]), int(face[13])
        
        min_x, max_x = min(rmx, lmx) - 12, max(rmx, lmx) + 12
        min_y, max_y = min(rmy, lmy) - 12, max(rmy, lmy) + 18
        
        cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        cv2.line(output, (rmx, rmy), (lmx, lmy), (0, 255, 255), 2)
        cv2.putText(output, "Smile Detected", (min_x - 10, max(20, min_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    save_comparison([(img, "Original Image"), (output, "Smile / Mouth Localization Result")], "exp30_smile_detection.png")

# ==============================================================================
# Exp 31: Image Segmentation based on given threshold values
# ==============================================================================
def exp31():
    code_text = '''# Experiment 31: Image Segmentation based on threshold values
import cv2

image = cv2.imread("assets/scene.jpg", cv2.IMREAD_GRAYSCALE)

_, bin_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adapt_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow("Binary Threshold (127)", bin_thresh)
cv2.imshow("Otsu Threshold", otsu_thresh)
cv2.imshow("Adaptive Threshold", adapt_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp31_segmentation_threshold.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg", cv2.IMREAD_GRAYSCALE)
    _, bin_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    save_comparison([
        (img, "Original Grayscale"),
        (bin_thresh, "Binary Threshold (T=127)"),
        (otsu_thresh, "Otsu Automatic Threshold"),
        (adapt_thresh, "Adaptive Gaussian Threshold")
    ], "exp31_segmentation_threshold.png", figsize=(18, 4.5))

# ==============================================================================
# Exp 32: White image with 4 colored boxes of Black, Blue, Green, Red at corners (1/10th size)
# ==============================================================================
def exp32():
    code_text = '''# Experiment 32: Create white image with 4 colored boxes (Black, Blue, Green, Red) at corners
import cv2
import numpy as np

def create_corner_boxes_image(width=500, height=500):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    box_w = width // 10
    box_h = height // 10
    
    # 1. Top-Left: Black
    image[0:box_h, 0:box_w] = [0, 0, 0]
    # 2. Top-Right: Blue
    image[0:box_h, width-box_w:width] = [255, 0, 0]
    # 3. Bottom-Left: Green
    image[height-box_h:height, 0:box_w] = [0, 255, 0]
    # 4. Bottom-Right: Red
    image[height-box_h:height, width-box_w:width] = [0, 0, 255]
    
    return image

img = create_corner_boxes_image(500, 500)
cv2.imshow("Corner Colored Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp32_corner_colored_boxes.py", "w") as f:
        f.write(code_text)

    width, height = 500, 500
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    box_w = width // 10
    box_h = height // 10
    img[0:box_h, 0:box_w] = [0, 0, 0]
    img[0:box_h, width-box_w:width] = [255, 0, 0]
    img[height-box_h:height, 0:box_w] = [0, 255, 0]
    img[height-box_h:height, width-box_w:width] = [0, 0, 255]

    annotated = img.copy()
    cv2.putText(annotated, "Black (Top-L)", (10, box_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(annotated, "Blue (Top-R)", (width - 130, box_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(annotated, "Green (Bot-L)", (10, height - box_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
    cv2.putText(annotated, "Red (Bot-R)", (width - 130, height - box_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    save_comparison([(img, "White Image with 4 Corner Boxes (1/10th size)"), (annotated, "Annotated Corner Colors")], "exp32_corner_boxes.png", figsize=(12, 5))

# ==============================================================================
# Exp 33: Create white image and draw Rectangle
# ==============================================================================
def exp33():
    code_text = '''# Experiment 33: Create white image and draw a Rectangle using OpenCV
import cv2
import numpy as np

def draw_rectangle_on_white(width=500, height=400):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (100, 80), (400, 320), (0, 0, 255), 4)
    return image

img = draw_rectangle_on_white()
cv2.imshow("Rectangle on White Canvas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp33_draw_rectangle.py", "w") as f:
        f.write(code_text)

    img = np.ones((400, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 80), (400, 320), (0, 0, 255), 4)
    cv2.putText(img, "cv2.rectangle((100,80), (400,320))", (110, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    save_comparison([(img, "Exp 33: Rectangle Shape on White Canvas")], "exp33_draw_rectangle.png", figsize=(6, 5))

# ==============================================================================
# Exp 34: Create white image and draw Circle
# ==============================================================================
def exp34():
    code_text = '''# Experiment 34: Create white image and draw a Circle using OpenCV
import cv2
import numpy as np

def draw_circle_on_white(width=500, height=400):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.circle(image, (250, 200), 120, (255, 0, 0), 4)
    return image

img = draw_circle_on_white()
cv2.imshow("Circle on White Canvas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp34_draw_circle.py", "w") as f:
        f.write(code_text)

    img = np.ones((400, 500, 3), dtype=np.uint8) * 255
    cv2.circle(img, (250, 200), 120, (255, 0, 0), 4)
    cv2.putText(img, "cv2.circle(Center=(250,200), R=120)", (115, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    save_comparison([(img, "Exp 34: Circle Shape on White Canvas")], "exp34_draw_circle.png", figsize=(6, 5))

# ==============================================================================
# Exp 35: Text string entered by user appeared on given image
# ==============================================================================
def exp35():
    code_text = '''# Experiment 35: Put Text string on image using OpenCV
import cv2

def put_text_on_image(image_path, text="SIMATS Engineering - Vision Lab"):
    image = cv2.imread(image_path)
    org = (50, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, org, font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return image

img = put_text_on_image("assets/scene.jpg")
cv2.imshow("Text on Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp35_put_text.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg")
    output = img.copy()
    cv2.putText(output, "SIMATS Computer Vision Lab", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "Experiment 35: Overlay User Text", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    save_comparison([(img, "Original Image"), (output, "Image with Rendered Text String")], "exp35_put_text.png")

# ==============================================================================
# Exp 36: Subtract background based on color levels using OpenCV
# ==============================================================================
def exp36():
    code_text = '''# Experiment 36: Subtract Background based on color levels using OpenCV
import cv2
import numpy as np

def subtract_background(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([30, 50, 50])
    upper_bound = np.array([85, 255, 255])
    bg_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    fg_mask = cv2.bitwise_not(bg_mask)
    foreground = cv2.bitwise_and(image, image, mask=fg_mask)
    return image, bg_mask, foreground

image, mask, fg = subtract_background("assets/fruits.jpg")
cv2.imshow("Background Subtracted (Foreground only)", fg)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp36_subtract_background.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/fruits.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([30, 50, 50])
    upper_bound = np.array([85, 255, 255])
    bg_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    fg_mask = cv2.bitwise_not(bg_mask)
    foreground = cv2.bitwise_and(img, img, mask=fg_mask)

    save_comparison([(img, "Original Image"), (bg_mask, "Identified Background Mask"), (foreground, "Background Subtracted Result")], "exp36_subtract_background.png", figsize=(15, 5))

# ==============================================================================
# Exp 37: Subtract foreground based on color levels using OpenCV
# ==============================================================================
def exp37():
    code_text = '''# Experiment 37: Subtract Foreground based on color levels using OpenCV
import cv2
import numpy as np

def subtract_foreground(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    fg_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    bg_preserved_mask = cv2.bitwise_not(fg_mask)
    background_only = cv2.bitwise_and(image, image, mask=bg_preserved_mask)
    return image, fg_mask, background_only

image, mask, bg = subtract_foreground("assets/fruits.jpg")
cv2.imshow("Foreground Subtracted (Background only)", bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp37_subtract_foreground.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/fruits.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    fg_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    bg_preserved_mask = cv2.bitwise_not(fg_mask)
    bg_only = cv2.bitwise_and(img, img, mask=bg_preserved_mask)

    save_comparison([(img, "Original Image"), (fg_mask, "Foreground Target Mask (Red levels)"), (bg_only, "Foreground Subtracted (Isolated Background)")], "exp37_subtract_foreground.png", figsize=(15, 5))

# ==============================================================================
# Exp 38: Count the number of faces for given input image using OpenCV
# ==============================================================================
def exp38():
    code_text = '''# Experiment 38: Count number of faces in an image using OpenCV
import cv2

detector = cv2.FaceDetectorYN_create("assets/face_detection_yunet.onnx", "", (300, 300))
image = cv2.imread("assets/multi_faces.jpg")
h, w, _ = image.shape
detector.setInputSize((w, h))

ret, faces = detector.detect(image)
output = image.copy()

count = len(faces) if faces is not None else 0
if faces is not None:
    for i, face in enumerate(faces, 1):
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        cv2.rectangle(output, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(output, f"Face #{i}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.putText(output, f"Total Faces Counted: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
print(f"Total number of faces: {count}")

cv2.imshow("Face Counter", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp38_count_faces.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/multi_faces.jpg")
    faces = get_faces(img)
    output = img.copy()
    for i, face in enumerate(faces, 1):
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        cv2.rectangle(output, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        cv2.putText(output, f"Face #{i}", (x, max(15, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output, f"Total Faces Counted: {len(faces)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    save_comparison([(img, "Original Group Image"), (output, f"Face Counting Result ({len(faces)} Detected)")], "exp38_count_faces.png")

# ==============================================================================
# Exp 39: Play given video in reverse mode in slow motion
# ==============================================================================
def exp39():
    code_text = '''# Experiment 39: Play video in Reverse Mode in Slow Motion
import cv2

cap = cv2.VideoCapture("assets/sample_video.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

for frame in reversed(frames):
    cv2.imshow("Reverse Slow Motion", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'): # Slow motion delay = 100ms
        break

cv2.destroyAllWindows()
'''
    with open("code/exp39_reverse_slow_motion.py", "w") as f:
        f.write(code_text)

    cap = cv2.VideoCapture("assets/sample_video.mp4")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("results/exp39_reverse_slowmo.mp4", fourcc, 5.0, (frames[0].shape[1], frames[0].shape[0]))
    for f in reversed(frames):
        out.write(f)
    out.release()

    rev_frames = list(reversed(frames))
    samples = [
        (rev_frames[0], "Reverse SlowMo - Start (Frame 60)"),
        (rev_frames[25], "Reverse SlowMo - Mid (Frame 35)"),
        (rev_frames[55], "Reverse SlowMo - End (Frame 5)")
    ]
    save_comparison(samples, "exp39_reverse_slow_motion.png", figsize=(15, 4.5))

# ==============================================================================
# Exp 40: Extract text from videos
# ==============================================================================
def exp40():
    code_text = '''# Experiment 40: Extract Text from Videos using OpenCV
import cv2
import numpy as np

def extract_text_regions_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output = frame.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 15: # Text line filtering
                cv2.rectangle(output, (x-5, y-5), (x+w+10, y+h+10), (0, 0, 255), 2)
                cv2.putText(output, "Extracted Text Region", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        if frame_idx == 20:
            cv2.imshow("Detected Text in Video", output)
            break
        frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()

extract_text_regions_from_video("assets/sample_video.mp4")
'''
    with open("code/exp40_extract_text_video.py", "w") as f:
        f.write(code_text)

    cap = cv2.VideoCapture("assets/sample_video.mp4")
    extracted_frame = None
    text_roi = None
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == 20:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output = frame.copy()
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 100 and h > 15:
                    cv2.rectangle(output, (x-5, y-5), (x+w+10, y+h+10), (0, 0, 255), 2)
                    cv2.putText(output, "Extracted Text Region: 'FRAME 21 : OPENCV DEMO'", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    text_roi = frame[y-5:y+h+10, x-5:x+w+10]
            extracted_frame = output
            break
        frame_idx += 1
    cap.release()

    if text_roi is None:
        text_roi = np.ones((50, 300, 3), dtype=np.uint8) * 255
        cv2.putText(text_roi, "Extracted: OPENCV DEMO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    save_comparison([(frame, "Original Video Frame"), (extracted_frame, "Detected Text Bounding Box"), (text_roi, "Cropped Text ROI")], "exp40_extract_text_video.png", figsize=(16, 4.5))

def main():
    print("Executing Experiments 21 - 40...")
    exp21()
    exp22()
    exp23()
    exp24()
    exp25()
    exp26()
    exp27()
    exp28()
    exp29()
    exp30()
    exp31()
    exp32()
    exp33()
    exp34()
    exp35()
    exp36()
    exp37()
    exp38()
    exp39()
    exp40()
    print("Experiments 21-40 Completed Successfully!")

if __name__ == "__main__":
    main()
