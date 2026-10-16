import cv2
import numpy as np
import pytesseract
import os
import re
import time

# =========================================================
# CONFIGURATION
# =========================================================
INPUT_FOLDER = 'total_pages'         # Folder containing your PNGs
OUTPUT_FILE = 'combined_voter_data_HIN_WI.txt' # The master output file
OCR_CONFIG = r'--oem 3 --psm 6 -l hin+eng'
PADDING = 15

# Create input folder if it doesn't exist (safety check)
if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)
    print(f"Created folder '{INPUT_FOLDER}'. Please paste your PNG images inside it.")

def natural_sort_key(s):
    """
    Sorts strings containing numbers naturally.
    e.g., page_2.png comes before page_10.png
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    
    final_cnts = []
    row_group = []
    prev_y = boundingBoxes[0][1]
    
    for c, bbox in zip(cnts, boundingBoxes):
        x, y, w, h = bbox
        if abs(y - prev_y) > 50: 
            row_group.sort(key=lambda k: cv2.boundingRect(k)[0])
            final_cnts.extend(row_group)
            row_group = []
        row_group.append(c)
        prev_y = y
        
    if row_group:
        row_group.sort(key=lambda k: cv2.boundingRect(k)[0])
        final_cnts.extend(row_group)
        
    return final_cnts

def preprocess_roi_raw(roi):
    # 1. Upscale
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = roi.shape[:2]
    
    # 2. Convert to Gray
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 3. Mask the Photo Area
    mask_x_start = int(w * 0.72) 
    mask_y_start = int(h * 0.25)
    cv2.rectangle(gray, (mask_x_start, mask_y_start), (w, h), (255, 255, 255), -1)

    # 4. Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_grid_blocks(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, []
    
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) 

    # Grid Detection
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 40, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 40))
    image_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel)
    image_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel)

    grid_mask = cv2.addWeighted(image_horizontal, 1, image_vertical, 1, 0)
    
    # Close and Dilate
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, close_kernel)
    grid_mask = cv2.dilate(grid_mask, close_kernel, iterations=1)

    contours, _ = cv2.findContours(grid_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_blocks = []
    expected_area = (w_img / 3) * (h_img / 10)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if (expected_area * 0.3) < area < (expected_area * 1.5):
            aspect = w / float(h)
            if 1.5 < aspect < 5.0:
                valid_blocks.append(cnt)

    if valid_blocks:
        return img, sort_contours(valid_blocks, method="top-to-bottom")
    else:
        return img, []

def process_batch():
    # 1. Get List of Images
    all_files = os.listdir(INPUT_FOLDER)
    image_files = [f for f in all_files if f.lower().endswith('.png')]
    
    if not image_files:
        print(f"No PNG images found in '{INPUT_FOLDER}'.")
        return

    # 2. Sort them naturally (page_1, page_2, page_10...)
    image_files.sort(key=natural_sort_key)
    
    total_files = len(image_files)
    print(f"Found {total_files} images to process.")
    
    start_time = time.time()

    # 3. Open Output File ONCE
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for i, filename in enumerate(image_files):
            full_path = os.path.join(INPUT_FOLDER, filename)
            print(f"[{i+1}/{total_files}] Processing: {filename}...", end='\r')
            
            # Extract
            original_img, blocks = extract_grid_blocks(full_path)
            
            # --- FILE HEADER ---
            f_out.write(f"\n{'='*50}\n")
            f_out.write(f"FILENAME: {filename}\n")
            f_out.write(f"{'='*50}\n")

            if not blocks:
                f_out.write("ERROR: No blocks found or image corrupt.\n")
                continue

            h_img, w_img = original_img.shape[:2]

            # Process Blocks
            for idx, block in enumerate(blocks):
                x, y, w, h = cv2.boundingRect(block)
                
                # Expand Box
                x_new = max(0, x - PADDING)
                y_new = max(0, y - PADDING)
                w_new = min(w_img - x_new, w + (2 * PADDING))
                h_new = min(h_img - y_new, h + (2 * PADDING))
                
                roi = original_img[y_new : y_new + h_new, x_new : x_new + w_new]
                processed_roi = preprocess_roi_raw(roi)
                
                # OCR
                raw_text = pytesseract.image_to_string(processed_roi, config=OCR_CONFIG)
                
                # Cleanup Text
                clean_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
                final_text = "\n".join(clean_lines)

                # Write Block Data
                f_out.write(f"--- BLOCK {idx+1} ---\n")
                f_out.write(final_text)
                f_out.write(f"\n{'-'*30}\n")
            
            # Force write to disk occasionally (optional, prevents data loss on crash)
            if i % 10 == 0:
                f_out.flush()

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n\nDone! Processed {total_files} images in {duration:.2f} seconds.")
    print(f"All data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_batch()