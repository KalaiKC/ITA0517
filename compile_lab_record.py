import os
import glob
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

EXPERIMENTS = [
    (1, "Basic Image Handling: Convert an Image to Gray-scale", "code/exp01_grayscale.py", "results/exp01_grayscale.png",
     "Reads an input image in color format and converts it to a single-channel grayscale image using cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)."),
    (2, "Convert an Image to Blur using Gaussian Blur", "code/exp02_gaussian_blur.py", "results/exp02_gaussian_blur.png",
     "Applies Gaussian smoothing using a 2D Gaussian kernel via cv2.GaussianBlur() to suppress high-frequency noise."),
    (3, "Convert an Image to Show Outline using Canny Function", "code/exp03_canny_outline.py", "results/exp03_canny_outline.png",
     "Applies the Canny multi-stage edge detection algorithm to isolate sharp structural boundaries."),
    (4, "Histogram Equalization and Comparison", "code/exp04_histogram_equalization.py", "results/exp04_histogram_equalization.png",
     "Performs contrast enhancement by stretching the intensity distribution across the dynamic range using cv2.equalizeHist()."),
    (5, "Analyze Histogram Based on Color Levels (RGB)", "code/exp05_color_histogram.py", "results/exp05_color_histogram.png",
     "Computes and plots separate histograms for Blue, Green, and Red color channels using cv2.calcHist()."),
    (6, "Image Erosion using Erode Function", "code/exp06_erode.py", "results/exp06_erode.png",
     "Shrinks foreground boundaries and eliminates isolated speckle noise using cv2.erode()."),
    (7, "Video Processing: Slow Motion and Fast Motion", "code/exp07_slow_fast_video.py", "results/exp07_slow_fast_motion.png",
     "Processes captured video frames with inter-frame delay scaling to demonstrate slow and accelerated motion."),
    (8, "Dilate an Image using Dilate Function", "code/exp08_dilate.py", "results/exp08_dilate.png",
     "Expands bright regions and bridges broken outlines using cv2.dilate()."),
    (9, "Image Scaling Techniques (Resize Bigger & Smaller)", "code/exp09_image_scaling.py", "results/exp09_image_scaling.png",
     "Resizes images to smaller and larger dimensions using interpolation techniques (cv2.INTER_AREA and cv2.INTER_CUBIC)."),
    (10, "90-Degree Clockwise Rotation", "code/exp10_rotate_90.py", "results/exp10_rotate_90.png",
     "Rotates the image 90 degrees clockwise using cv2.rotate(..., cv2.ROTATE_90_CLOCKWISE)."),
    (11, "180-Degree Clockwise Rotation", "code/exp11_rotate_180.py", "results/exp11_rotate_180.png",
     "Rotates the image 180 degrees using cv2.rotate(..., cv2.ROTATE_180)."),
    (12, "270-Degree Clockwise Rotation", "code/exp12_rotate_270.py", "results/exp12_rotate_270.png",
     "Rotates the image 270 degrees clockwise using cv2.rotate(..., cv2.ROTATE_90_COUNTERCLOCKWISE)."),
    (13, "Affine Transformation on Given Image", "code/exp13_affine_transformation.py", "results/exp13_affine_transformation.png",
     "Applies affine geometric warping mapping 3 control points with cv2.getAffineTransform() and cv2.warpAffine()."),
    (14, "Perspective Transformation on Given Image", "code/exp14_perspective_transformation.py", "results/exp14_perspective_transformation.png",
     "Applies 4-point projective transformation using cv2.getPerspectiveTransform() and cv2.warpPerspective()."),
    (15, "Harris Corner Detection", "code/exp15_harris_corners.py", "results/exp15_harris_corners.png",
     "Detects high-curvature corner points based on local intensity gradients using cv2.cornerHarris()."),
    (16, "Sobel Algorithm Filtering (X, Y, Combined)", "code/exp16_sobel_filter.py", "results/exp16_sobel_filter.png",
     "Calculates first-order horizontal and vertical spatial image gradients with cv2.Sobel()."),
    (17, "Watermarking Technique using OpenCV", "code/exp17_watermarking.py", "results/exp17_watermarking.png",
     "Inserts transparent digital watermark logo into the primary image using alpha blending with cv2.addWeighted()."),
    (18, "Image Cropping, Copying, and Pasting (ROI)", "code/exp18_roi_crop_paste.py", "results/exp18_roi_crop_paste.png",
     "Extracts a Region of Interest (ROI) via array slicing and pastes it into another target location."),
    (19, "Erosion Morphological Operations Technique", "code/exp19_morph_erosion.py", "results/exp19_morph_erosion.png",
     "Implements morphological erosion with structured kernels to strip outer boundary layers."),
    (20, "Dilation Morphological Operations Technique", "code/exp20_morph_dilation.py", "results/exp20_morph_dilation.png",
     "Implements morphological dilation to thicken bright features and bridge gaps."),
    (21, "Opening Morphological Operation", "code/exp21_morph_opening.py", "results/exp21_morph_opening.png",
     "Applies erosion followed by dilation using cv2.morphologyEx(..., cv2.MORPH_OPEN) to eliminate foreground noise."),
    (22, "Closing Morphological Operation", "code/exp22_morph_closing.py", "results/exp22_morph_closing.png",
     "Applies dilation followed by erosion using cv2.morphologyEx(..., cv2.MORPH_CLOSE) to fill holes and gaps."),
    (23, "Top Hat Morphological Operation", "code/exp23_morph_tophat.py", "results/exp23_morph_tophat.png",
     "Isolates elements and details brighter than their surrounding neighborhood (Input - Opening)."),
    (24, "Black Hat Morphological Operation", "code/exp24_morph_blackhat.py", "results/exp24_morph_blackhat.png",
     "Isolates dark elements that are darker than their local neighborhood (Closing - Input)."),
    (25, "Recognize Watch from Given Image", "code/exp25_watch_recognition.py", "results/exp25_watch_recognition.png",
     "Performs object recognition to detect circular watch dial geometry using Hough Circle transform."),
    (26, "Reverse Video Frames into Reverse Video", "code/exp26_reverse_video.py", "results/exp26_reverse_video.png",
     "Buffers incoming video frames into memory and writes them in backwards order using cv2.VideoWriter."),
    (27, "Face Detection Algorithm using OpenCV", "code/exp27_face_detection.py", "results/exp27_face_detection.png",
     "Detects and localizes human faces in images with bounding box coordinates and confidence scores."),
    (28, "Vehicle Detection Algorithm in Video Frames", "code/exp28_vehicle_detection.py", "results/exp28_vehicle_detection.png",
     "Detects moving vehicles across video frames using background subtraction (MOG2) and bounding contour filters."),
    (29, "Eye Detection Algorithm using OpenCV", "code/exp29_eye_detection.py", "results/exp29_eye_detection.png",
     "Locates precise left and right eye coordinates and annotates them on the detected human face."),
    (30, "Smile Detection Algorithm using OpenCV", "code/exp30_smile_detection.py", "results/exp30_smile_detection.png",
     "Detects the presence and bounds of human smiles via mouth corner landmark geometry."),
    (31, "Image Segmentation Based on Threshold Values", "code/exp31_segmentation_threshold.py", "results/exp31_segmentation_threshold.png",
     "Segments images into foreground and background using Binary, Otsu, and Adaptive Gaussian thresholding."),
    (32, "Create White Image with 4 Corner Boxes (1/10th size)", "code/exp32_corner_boxes.py", "results/exp32_corner_boxes.png",
     "Generates a pure white canvas and constructs 1/10th scale colored boxes (Black, Blue, Green, Red) at all 4 corners."),
    (33, "Draw Rectangle Shape on White Image", "code/exp33_draw_rectangle.py", "results/exp33_draw_rectangle.png",
     "Creates a blank white image canvas and renders a geometric rectangle with specified coordinates."),
    (34, "Draw Circle Shape on White Image", "code/exp34_draw_circle.py", "results/exp34_draw_circle.png",
     "Creates a blank white canvas and draws an anti-aliased circle with defined center and radius."),
    (35, "Render User-Specified Text String on Image", "code/exp35_put_text.py", "results/exp35_put_text.png",
     "Draws custom text strings onto an image canvas with font style, scale, color, and thickness settings."),
    (36, "Subtract Background Based on Color Levels", "code/exp36_subtract_background.py", "results/exp36_subtract_background.png",
     "Converts to HSV color space, masks out specific background color ranges, and isolates foreground objects."),
    (37, "Subtract Foreground Based on Color Levels", "code/exp37_subtract_foreground.py", "results/exp37_subtract_foreground.png",
     "Masks out distinct foreground color ranges to subtract foreground objects while preserving the background."),
    (38, "Count Number of Faces in Input Image", "code/exp38_count_faces.py", "results/exp38_count_faces.png",
     "Detects multiple faces in crowd/group images, labels each face sequentially, and tallies the total count."),
    (39, "Play Video in Reverse Mode in Slow Motion", "code/exp39_reverse_slow_motion.py", "results/exp39_reverse_slow_motion.png",
     "Renders video frames in reverse chronological sequence while reducing playback frame rate for slow motion."),
    (40, "Extract Text Regions from Videos", "code/exp40_extract_text_video.py", "results/exp40_extract_text_video.png",
     "Applies morphological gradient and edge analysis to locate and isolate text banners within video frames.")
]

def generate_docx():
    doc = Document()
    
    # Page setup - 0.75 in margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Document Header / Title
    title = doc.add_paragraph()
    r_title = title.add_run("SIMATS ENGINEERING\nDEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\nCOMPUTER VISION LAB EXPERIMENTS RECORD")
    r_title.bold = True
    r_title.font.size = Pt(16)
    r_title.font.color.rgb = RGBColor(16, 44, 87)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    r_sub = sub.add_run("Complete 40 Lab Experiments: Source Code, Methodologies, and Execution Results")
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(80, 80, 80)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    for exp_num, exp_title, code_path, result_img_path, desc in EXPERIMENTS:
        h = doc.add_heading(level=1)
        r_h = h.add_run(f"Experiment {exp_num}: {exp_title}")
        r_h.font.size = Pt(13)
        r_h.font.color.rgb = RGBColor(26, 77, 144)

        # Description
        p_desc = doc.add_paragraph()
        r_desc_label = p_desc.add_run("Objective & Methodology: ")
        r_desc_label.bold = True
        p_desc.add_run(desc)

        # Source code
        doc.add_heading("Python Source Code:", level=2)
        p_code = doc.add_paragraph()
        p_code.paragraph_format.left_indent = Inches(0.2)
        if os.path.exists(code_path):
            with open(code_path, "r", encoding="utf-8") as f:
                code_content = f.read().strip()
        else:
            code_content = "# Source code"
        r_code = p_code.add_run(code_content)
        r_code.font.name = "Consolas"
        r_code.font.size = Pt(9)
        r_code.font.color.rgb = RGBColor(30, 30, 30)

        # Execution Screenshot
        doc.add_heading("Execution Output Screenshot:", level=2)
        if os.path.exists(result_img_path):
            doc.add_picture(result_img_path, width=Inches(6.5))
        
        doc.add_page_break()

    output_path = "Computer_Vision_Lab_Record_All_40_Experiments.docx"
    doc.save(output_path)
    print(f"Document saved successfully: {output_path}")

if __name__ == "__main__":
    generate_docx()
