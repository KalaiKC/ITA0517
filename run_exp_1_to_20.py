"""
Computer Vision Lab Experiments Suite
Experiments 1 through 20
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
os.makedirs("code", exist_ok=True)

# Helper function to save a combined side-by-side or multi-panel screenshot
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
            # OpenCV BGR to RGB
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, pad=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", output_filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: results/{output_filename}")

# ==============================================================================
# Exp 1: Read image and Convert to Gray-scale
# ==============================================================================
def exp01():
    code_text = '''# Experiment 1: Read Image and Convert to Gray-scale
import cv2

image = cv2.imread("assets/scene.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp01_grayscale.py", "w") as f:
        f.write(code_text)
    
    img = cv2.imread("assets/scene.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_comparison([(img, "Original Image"), (gray, "Grayscale Image")], "exp01_grayscale.png")

# ==============================================================================
# Exp 2: Read image and Convert to Blur using Gaussian Blur
# ==============================================================================
def exp02():
    code_text = '''# Experiment 2: Convert an Image to Blur using Gaussian Blur
import cv2

image = cv2.imread("assets/scene.jpg")
# Applying Gaussian Blur with kernel size 15x15 and sigmaX = 0
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

cv2.imshow("Original", image)
cv2.imshow("Gaussian Blurred", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp02_gaussian_blur.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg")
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    save_comparison([(img, "Original Image"), (blurred, "Gaussian Blur (15x15)")], "exp02_gaussian_blur.png")

# ==============================================================================
# Exp 3: Convert an Image to show outline using Canny function
# ==============================================================================
def exp03():
    code_text = '''# Experiment 3: Show outline using Canny function
import cv2

image = cv2.imread("assets/building.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Canny Edge Detection with thresholds 100 and 200
edges = cv2.Canny(gray, 100, 200)

cv2.imshow("Original", image)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp03_canny_outline.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/building.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    save_comparison([(img, "Original Image"), (edges, "Canny Edge Detection")], "exp03_canny_outline.png")

# ==============================================================================
# Exp 4: Histogram Equalization and compare with original
# ==============================================================================
def exp04():
    code_text = '''# Experiment 4: Implement Histogram Equalization and compare
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("assets/scene.jpg", cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image)

cv2.imshow("Original Gray", image)
cv2.imshow("Histogram Equalized", equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp04_histogram_equalization.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg", cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original Grayscale Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(equ, cmap='gray')
    axes[0, 1].set_title("Equalized Image")
    axes[0, 1].axis('off')

    axes[1, 0].hist(img.ravel(), 256, [0, 256], color='black')
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlim([0, 256])

    axes[1, 1].hist(equ.ravel(), 256, [0, 256], color='blue')
    axes[1, 1].set_title("Equalized Histogram")
    axes[1, 1].set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig("results/exp04_histogram_equalization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/exp04_histogram_equalization.png")

# ==============================================================================
# Exp 5: Analyze histogram based on color levels (R, G, B)
# ==============================================================================
def exp05():
    code_text = '''# Experiment 5: Analyze histogram based on color levels using OpenCV
import cv2
import matplotlib.pyplot as plt

def analyze_color_histogram(image_path):
    img = cv2.imread(image_path)
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=f'{col.upper()} Channel')
        plt.xlim([0, 256])
    plt.title("Color Level Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

analyze_color_histogram("assets/fruits.jpg")
'''
    with open("code/exp05_color_histogram.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/fruits.jpg")
    colors = ('b', 'g', 'r')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Color Image")
    axes[0].axis('off')

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        axes[1].plot(hist, color=col, linewidth=2, label=f'{col.upper()} Channel')
    axes[1].set_title("Color Level Histogram Analysis")
    axes[1].set_xlabel("Pixel Intensity (0 - 255)")
    axes[1].set_ylabel("Pixel Count")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/exp05_color_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/exp05_color_histogram.png")

# ==============================================================================
# Exp 6: Convert an Image to erode using Erode function
# ==============================================================================
def exp06():
    code_text = '''# Experiment 6: Convert an Image to erode using Erode function
import cv2
import numpy as np

image = cv2.imread("assets/building.jpg")
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(image, kernel, iterations=2)

cv2.imshow("Original", image)
cv2.imshow("Eroded", eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp06_erode.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/building.jpg")
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=2)
    save_comparison([(img, "Original Image"), (eroded, "Eroded (5x5 kernel, 2 iter)")], "exp06_erode.png")

# ==============================================================================
# Exp 7: Video processing: read captured video, slow motion & fast motion
# ==============================================================================
def exp07():
    code_text = '''# Experiment 7: Video processing: Slow Motion & Fast Motion
import cv2

cap = cv2.VideoCapture("assets/sample_video.mp4")

# Slow motion: Increase waitKey delay (e.g. waitKey(100) -> ~10 fps)
# Fast motion: Decrease waitKey delay (e.g. waitKey(5) -> ~200 fps) or skip frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Video Player", frame)
    # Change delay to 80 for slow motion, 10 for fast motion
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
    with open("code/exp07_slow_fast_video.py", "w") as f:
        f.write(code_text)

    # Capture 3 sequential frames to demonstrate motion playback timing
    cap = cv2.VideoCapture("assets/sample_video.mp4")
    frames = []
    frame_indices = [5, 25, 45]
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frames.append((frame.copy(), f"Frame {idx} (Rate control: 25ms norm / 80ms slow / 8ms fast)"))
        idx += 1
    cap.release()

    save_comparison(frames, "exp07_slow_fast_motion.png", figsize=(15, 4))

# ==============================================================================
# Exp 8: Dilate an Image using Dilate function
# ==============================================================================
def exp08():
    code_text = '''# Experiment 8: Dilate an Image using Dilate function
import cv2
import numpy as np

image = cv2.imread("assets/building.jpg")
kernel = np.ones((5, 5), np.uint8)
dilated_image = cv2.dilate(image, kernel, iterations=2)

cv2.imshow("Original", image)
cv2.imshow("Dilated", dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp08_dilate.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/building.jpg")
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=2)
    save_comparison([(img, "Original Image"), (dilated, "Dilated (5x5 kernel, 2 iter)")], "exp08_dilate.png")

# ==============================================================================
# Exp 9: Image scaling techniques: resize to bigger and small sizes
# ==============================================================================
def exp09():
    code_text = '''# Experiment 9: Image scaling to bigger and smaller sizes
import cv2

image = cv2.imread("assets/scene.jpg")

# Scale smaller (0.5x)
smaller = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# Scale bigger (1.5x)
bigger = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

cv2.imshow("Original", image)
cv2.imshow("Smaller (0.5x)", smaller)
cv2.imshow("Bigger (1.5x)", bigger)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp09_image_scaling.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg")
    smaller = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    bigger = cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC)

    save_comparison([
        (smaller, f"Scaled Down (200x200)"),
        (img, f"Original ({img.shape[1]}x{img.shape[0]})"),
        (bigger, f"Scaled Up (600x600)")
    ], "exp09_image_scaling.png", figsize=(15, 5))

# ==============================================================================
# Exp 10: 90-degree rotation clockwise
# ==============================================================================
def exp10():
    code_text = '''# Experiment 10: 90-degree clockwise rotation
import cv2

image = cv2.imread("assets/scene.jpg")
# cv2.ROTATE_90_CLOCKWISE rotates 90 degrees clockwise
rot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Original", image)
cv2.imshow("90 Deg Clockwise", rot90)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp10_rotate_90.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg")
    rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    save_comparison([(img, "Original"), (rot90, "90° Clockwise Rotation")], "exp10_rotate_90.png")

# ==============================================================================
# Exp 11: 180-degree rotation clockwise
# ==============================================================================
def exp11():
    code_text = '''# Experiment 11: 180-degree clockwise rotation
import cv2

image = cv2.imread("assets/scene.jpg")
rot180 = cv2.rotate(image, cv2.ROTATE_180)

cv2.imshow("Original", image)
cv2.imshow("180 Deg Clockwise", rot180)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp11_rotate_180.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg")
    rot180 = cv2.rotate(img, cv2.ROTATE_180)
    save_comparison([(img, "Original"), (rot180, "180° Clockwise Rotation")], "exp11_rotate_180.png")

# ==============================================================================
# Exp 12: 270-degree rotation clockwise (90 counter-clockwise)
# ==============================================================================
def exp12():
    code_text = '''# Experiment 12: 270-degree clockwise rotation
import cv2

image = cv2.imread("assets/scene.jpg")
rot270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow("Original", image)
cv2.imshow("270 Deg Clockwise", rot270)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp12_rotate_270.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg")
    rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    save_comparison([(img, "Original"), (rot270, "270° Clockwise Rotation")], "exp12_rotate_270.png")

# ==============================================================================
# Exp 13: Affine Transformation
# ==============================================================================
def exp13():
    code_text = '''# Experiment 13: Affine Transformation
import cv2
import numpy as np

image = cv2.imread("assets/scene.jpg")
rows, cols, ch = image.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv2.getAffineTransform(pts1, pts2)
affine_dst = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow("Original", image)
cv2.imshow("Affine Transformed", affine_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp13_affine_transformation.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg")
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    affine_dst = cv2.warpAffine(img, M, (cols, rows))
    save_comparison([(img, "Original Image"), (affine_dst, "Affine Transformed")], "exp13_affine_transformation.png")

# ==============================================================================
# Exp 14: Perspective Transformation
# ==============================================================================
def exp14():
    code_text = '''# Experiment 14: Perspective Transformation
import cv2
import numpy as np

image = cv2.imread("assets/building.jpg")
rows, cols, ch = image.shape

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M = cv2.getPerspectiveTransform(pts1, pts2)
perspective_dst = cv2.warpPerspective(image, M, (300, 300))

cv2.imshow("Original", image)
cv2.imshow("Perspective Transformed", perspective_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp14_perspective_transformation.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/building.jpg")
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [cols - 50, 50], [50, rows - 50], [cols - 50, rows - 50]])
    pts2 = np.float32([[10, 80], [cols - 30, 20], [80, rows - 20], [cols - 90, rows - 60]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_dst = cv2.warpPerspective(img, M, (cols, rows))
    save_comparison([(img, "Original Image"), (perspective_dst, "Perspective Warped")], "exp14_perspective_transformation.png")

# ==============================================================================
# Exp 15: Harris Corner Detection
# ==============================================================================
def exp15():
    code_text = '''# Experiment 15: Harris Corner Detection
import cv2
import numpy as np

image = cv2.imread("assets/building.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris corner detection parameters: blockSize=2, ksize=3, k=0.04
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate to mark corners clearly
dst = cv2.dilate(dst, None)

# Threshold for optimal corner display
corner_img = image.copy()
corner_img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow("Original", image)
cv2.imshow("Harris Corners", corner_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp15_harris_corners.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/building.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    corner_img = img.copy()
    corner_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    save_comparison([(img, "Original Image"), (corner_img, "Detected Harris Corners (Red)")], "exp15_harris_corners.png")

# ==============================================================================
# Exp 16: Sobel Algorithm to filter input image
# ==============================================================================
def exp16():
    code_text = '''# Experiment 16: Sobel Algorithm Filter
import cv2
import numpy as np

image = cv2.imread("assets/scene.jpg", cv2.IMREAD_GRAYSCALE)

# Sobel X and Sobel Y
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Combined Sobel magnitude
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

cv2.imshow("Original", image)
cv2.imshow("Sobel Combined", sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp16_sobel_filter.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg", cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    save_comparison([
        (img, "Original Grayscale"),
        (np.uint8(np.absolute(sobelx)), "Sobel X (Vertical edges)"),
        (np.uint8(np.absolute(sobely)), "Sobel Y (Horizontal edges)"),
        (sobel_combined, "Sobel Combined Magnitude")
    ], "exp16_sobel_filter.png", figsize=(18, 4.5))

# ==============================================================================
# Exp 17: Watermarking technique to insert watermark into original image
# ==============================================================================
def exp17():
    code_text = '''# Experiment 17: Watermarking Technique using OpenCV
import cv2

image = cv2.imread("assets/nature.jpg")
watermark = cv2.imread("assets/watermark.png")

# Resize watermark to 150x150
wm_resized = cv2.resize(watermark, (150, 150))
h, w, _ = wm_resized.shape
img_h, img_w, _ = image.shape

# Place watermark in bottom-right corner
top_y = img_h - h - 20
left_x = img_w - w - 20
roi = image[top_y:top_y+h, left_x:left_x+w]

# Blend ROI with watermark (alpha blending)
blended = cv2.addWeighted(roi, 0.6, wm_resized, 0.4, 0)
watermarked_image = image.copy()
watermarked_image[top_y:top_y+h, left_x:left_x+w] = blended

cv2.imshow("Watermarked Image", watermarked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp17_watermarking.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/nature.jpg")
    watermark = cv2.imread("assets/watermark.png")
    if watermark is None:
        watermark = np.zeros((150, 150, 3), dtype=np.uint8)
        cv2.putText(watermark, "OPENCV", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    wm_resized = cv2.resize(watermark, (150, 150))
    h, w, _ = wm_resized.shape
    img_h, img_w, _ = img.shape
    top_y = img_h - h - 20
    left_x = img_w - w - 20
    roi = img[top_y:top_y+h, left_x:left_x+w]
    blended = cv2.addWeighted(roi, 0.6, wm_resized, 0.4, 0)
    watermarked_image = img.copy()
    watermarked_image[top_y:top_y+h, left_x:left_x+w] = blended

    save_comparison([(img, "Original Image"), (wm_resized, "Watermark Logo"), (watermarked_image, "Watermarked Result")], "exp17_watermarking.png", figsize=(15, 5))

# ==============================================================================
# Exp 18: Image cropping, copying and pasting ROI
# ==============================================================================
def exp18():
    code_text = '''# Experiment 18: Image cropping, copying and pasting (ROI)
import cv2

image = cv2.imread("assets/scene.jpg")

# Crop Region of Interest (e.g., face or center object)
roi = image[150:350, 180:380]

# Paste into another location
result = image.copy()
result[50:250, 20:220] = roi
cv2.rectangle(result, (20, 50), (220, 250), (0, 255, 0), 2) # outline pasted ROI

cv2.imshow("Cropped ROI", roi)
cv2.imshow("Copied and Pasted", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp18_roi_crop_paste.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/scene.jpg")
    roi = img[150:350, 180:380]
    result = img.copy()
    # Draw original ROI box
    orig_marked = img.copy()
    cv2.rectangle(orig_marked, (180, 150), (380, 350), (0, 0, 255), 2)
    # Paste ROI to left top
    result[30:230, 20:220] = roi
    cv2.rectangle(result, (20, 30), (220, 230), (0, 255, 0), 2)
    save_comparison([(orig_marked, "Selected ROI (Red Box)"), (roi, "Cropped ROI"), (result, "Pasted ROI (Green Box)")], "exp18_roi_crop_paste.png", figsize=(15, 5))

# ==============================================================================
# Exp 19: Erosion Morphological operation
# ==============================================================================
def exp19():
    code_text = '''# Experiment 19: Erosion Morphological Operation
import cv2
import numpy as np

image = cv2.imread("assets/fruits.jpg")
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(image, kernel, iterations=3)

cv2.imshow("Original", image)
cv2.imshow("Erosion Morphological", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp19_morph_erosion.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/fruits.jpg")
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=3)
    save_comparison([(img, "Original Image"), (erosion, "Morphological Erosion (Iter=3)")], "exp19_morph_erosion.png")

# ==============================================================================
# Exp 20: Dilation Morphological operation
# ==============================================================================
def exp20():
    code_text = '''# Experiment 20: Dilation Morphological Operation
import cv2
import numpy as np

image = cv2.imread("assets/fruits.jpg")
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(image, kernel, iterations=3)

cv2.imshow("Original", image)
cv2.imshow("Dilation Morphological", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    with open("code/exp20_morph_dilation.py", "w") as f:
        f.write(code_text)

    img = cv2.imread("assets/fruits.jpg")
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=3)
    save_comparison([(img, "Original Image"), (dilation, "Morphological Dilation (Iter=3)")], "exp20_morph_dilation.png")

def main():
    print("Executing Experiments 1 - 20...")
    exp01()
    exp02()
    exp03()
    exp04()
    exp05()
    exp06()
    exp07()
    exp08()
    exp09()
    exp10()
    exp11()
    exp12()
    exp13()
    exp14()
    exp15()
    exp16()
    exp17()
    exp18()
    exp19()
    exp20()
    print("Experiments 1-20 Completed!")

if __name__ == "__main__":
    main()
