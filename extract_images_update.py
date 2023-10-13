from PyPDF2 import PdfFileReader
import fitz  # PyMuPDF
import os

# Path to save the extracted images
img_folder = "/mnt/data/extracted_images"
os.makedirs(img_folder, exist_ok=True)

# Open the PDF using PyMuPDF
pdf_path = "/mnt/data/joi210144supp1_prod_1643049438.9495.pdf"
doc = fitz.open(pdf_path)
img_files = []

# Extract images from each page
for page_num in range(doc.pageCount):
    page = doc.loadPage(page_num)
    image_list = page.getImageList(full=True)
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extractImage(xref)
        image_bytes = base_image["image"]
        
        # Save the image to a file
        img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
        img_path = os.path.join(img_folder, img_filename)
        with open(img_path, "wb") as img_file:
            img_file.write(image_bytes)
        img_files.append(img_path)

img_files