import fitz 
import os
import re
from pathlib import Path
from PIL import Image
import io

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TEXT_OUT = PROCESSED_DIR / "text"
IMG_OUT = PROCESSED_DIR / "images"

MIN_IMG_SIZE_KB = 15
MIN_IMG_DIM = 200

def setup_directories():
    TEXT_OUT.mkdir(parents=True, exist_ok=True)
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Directories checked:\n - {TEXT_OUT}\n - {IMG_OUT}")

def clean_text_for_rag(text):
    if not text:
        return ""

    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = text.replace('\x0c', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()

def process_pdf(pdf_path):
    doc_name = pdf_path.stem
    print(f"Processing: {pdf_path.name}")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path.name}: {e}")
        return

    extracted_images_count = 0
    
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        
        raw_text = page.get_text("text")
        cleaned_text = clean_text_for_rag(raw_text)

        if len(cleaned_text) > 50:
            text_filename = f"{doc_name}_page_{page_num}.txt"
            with open(TEXT_OUT / text_filename, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            try:
                rects = page.get_image_rects(xref)
                
                if not rects:
                    continue
                    
                rect = rects[0]

                if rect.width < 100 or rect.height < 100:
                    continue

                zoom = 3 
                mat = fitz.Matrix(zoom, zoom)

                pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)

                if pix.size < MIN_IMG_SIZE_KB * 1024:
                    continue

                img_filename = f"{doc_name}_page_{page_num}_img_{img_index}.png"
                img_path = IMG_OUT / img_filename
                
                pix.save(img_path)
                
                extracted_images_count += 1
                
            except Exception as e:
                print(f"Error processing image on page {page_num}: {e}")
                continue
    print(f"Completed {doc_name}: Found {extracted_images_count} images.")

def main():
    setup_directories()
    
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {RAW_DIR}!")
        return

    print(f"Found files: {len(pdf_files)}")
    
    for pdf in pdf_files:
        process_pdf(pdf)
    
    print("\nFull processing completed.")

if __name__ == "__main__":
    main()