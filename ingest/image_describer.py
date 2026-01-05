import os
import json
import time
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API Key not found! Check .env file.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "data" / "processed" / "images"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "image_summaries.json"

def analyze_image_with_gemini(img_path):
    """
    Sends image to Gemini for detailed analysis.
    """

    prompt = """
    You are an expert Technical Documentation Engineer. Analyze this image from a US Army Generator Manual.
    
    Perform two tasks:
    1. **General Caption**: Briefly describe what this image is (e.g., "Wiring diagram for the Fuel Pump Control circuit").
    2. **Detailed Transcription**: Extract ALL technical text, labels, part numbers, and values visible. 
       - If it is a diagram, describe the connections (e.g., "K1 relay connects to Pin 4").
       - If it is a table, transcribe the key rows/columns.
       - Ignore artifacts or noise. If text is unreadable, ignore it.
    
    Format the output strictly as plain text, separating the two sections clearly.
    """
    
    try:
        img = Image.open(img_path)
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        print(f"Error analyzing {img_path.name}: {e}")
        if "429" in str(e):
            print("API limit reached. Waiting 60 seconds...")
            time.sleep(60)
        return None

def main():
    if not IMG_DIR.exists():
        print(f"Directory {IMG_DIR} does not exist. Run pdf_parser.py first.")
        return

    image_files = list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.jpg"))
    image_files.sort()
    
    print(f"Found {len(image_files)} images.")

    processed_data = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                processed_data = json.load(f)
                print(f"Loaded {len(processed_data)} existing records.")
            except json.JSONDecodeError:
                print("JSON file corrupted, starting from scratch.")

    processed_filenames = {item['image_filename'] for item in processed_data}

    for i, img_path in enumerate(image_files):
        if img_path.name in processed_filenames:
            continue 

        print(f"[{i+1}/{len(image_files)}] Analyzing: {img_path.name}...")
        
        description = analyze_image_with_gemini(img_path)
        
        if description:
            record = {
                "image_filename": img_path.name,
                "image_path": str(img_path.relative_to(BASE_DIR)),
                "source_doc": img_path.stem.split("_page_")[0],
                "page_num": img_path.stem.split("_page_")[1].split("_")[0],
                "description": description
            }
            processed_data.append(record)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            
            print("   Saved.")
        time.sleep(6) 

    print(f"\nDone! All summaries saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()