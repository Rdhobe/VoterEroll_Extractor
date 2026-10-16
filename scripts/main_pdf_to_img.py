import fitz  # PyMuPDF
import os
import sys
 
# =========================================================
# CONFIGURATION
# =========================================================
PDF_PATH = "data/01_input/2026-EROLLGEN-S24-86-SIR-FinalRoll-Revision1-HIN-1-WI.pdf"
OUTPUT_FOLDER = "total_pages"      # Folder name as requested
START_PAGE = 14
END_PAGE = 6905

# RECOMMENDED FOR OCR: 3.0
# 2.0 = 144 DPI (Okayish, smaller files)
# 3.0 = 216 DPI (Best balance for OCR accuracy vs Speed)
# 4.0 = 288 DPI (High quality, but very large files)
ZOOM_FACTOR = 3.0 

def convert_pdf_to_images():
    # 1. Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: File '{PDF_PATH}' not found. Please put it in the same folder as this script.")
        return

    # 2. Create the 'total_pages' folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: '{OUTPUT_FOLDER}'")
    else:
        print(f"Saving to existing folder: '{OUTPUT_FOLDER}'")

    # 3. Open PDF
    try:
        doc = fitz.open(PDF_PATH)
        total_pages_in_pdf = doc.page_count
        print(f"PDF Loaded. Total Pages available: {total_pages_in_pdf}")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    # 4. Validate Page Range
    # Adjust logic because PyMuPDF is 0-indexed (Page 1 is index 0)
    start_index = START_PAGE - 1
    end_index = min(END_PAGE, total_pages_in_pdf)

    if start_index >= total_pages_in_pdf:
        print(f"Error: Start page {START_PAGE} is beyond the end of the PDF.")
        return

    print(f"Starting conversion from Page {START_PAGE} to {end_index}...")
    print(f"Zoom Factor: {ZOOM_FACTOR} (Higher resolution for OCR)")

    # 5. Set Matrix for High Resolution
    mat = fitz.Matrix(ZOOM_FACTOR, ZOOM_FACTOR)

    count = 0
    
    # 6. Loop through pages
    for page_num in range(start_index, end_index):
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            
            # Filename formatting: page_0014.png (Ensures correct sorting order)
            # We use the actual page number (page_num + 1) for the filename
            filename = f"page_{page_num + 1:04d}.png"
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            
            pix.save(output_path)
            
            count += 1
            # Print progress every 50 pages to keep console clean
            if count % 50 == 0:
                print(f"Processed {count} pages... (Current: {filename})")

        except KeyboardInterrupt:
            print("\nStopping process manually...")
            sys.exit()
        except Exception as e:
            print(f"Failed to convert Page {page_num + 1}: {e}")

    print("-" * 30)
    print(f"Conversion Complete!")
    print(f"Total images saved: {count}")
    print(f"Location: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    convert_pdf_to_images()