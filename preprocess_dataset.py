import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = "data/BinaryBrainTumorDataset"
OUTPUT_DIR = "data/PreprocessedBinaryBrainTumorDataset"
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff'}

def auto_crop(img):
    """Automatically removes the black border around the brain region."""
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]

def apply_clahe(img):
    """Improved method for contrast enhancement using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img)

def preprocess_image(img_path):
    """Main function for preprocessing a single image."""
    try:
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        # Apply all preprocessing steps
        img = auto_crop(img)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = apply_clahe(img)
        img = cv2.medianBlur(img, 3)
        img = (img / 255.0).astype(np.float32)  # Normalize
        
        return (img * 255).astype(np.uint8)  # Convert back for saving
    
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def process_batch(args):
    """Processes a batch of images in parallel."""
    input_subdir, output_subdir, files = args
    os.makedirs(output_subdir, exist_ok=True)
    
    for fname in files:
        in_path = os.path.join(input_subdir, fname)
        out_path = os.path.join(output_subdir, fname)
        img = preprocess_image(in_path)
        if img is not None:
            cv2.imwrite(out_path, img)

def process_and_save_images():
    """Main function to run the entire preprocessing pipeline."""
    for split in ["Training", "Testing"]:
        for category in ["yes", "no"]:
            input_subdir = os.path.join(INPUT_DIR, split, category)
            output_subdir = os.path.join(OUTPUT_DIR, split, category)
            
            # Collect all valid files
            files = [
                f for f in os.listdir(input_subdir) 
                if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
            ]
            
            # Split into batches for parallel processing
            batch_size = 100
            batches = [
                (input_subdir, output_subdir, files[i:i + batch_size]) 
                for i in range(0, len(files), batch_size)
            ]
            
            # Run parallel processing with a progress bar
            with ThreadPoolExecutor() as executor:
                list(tqdm(
                    executor.map(process_batch, batches),
                    total=len(batches),
                    desc=f"Processing {split}/{category}"
                ))

if __name__ == "__main__":
    process_and_save_images()
    print("Preprocessed images saved in:", OUTPUT_DIR)
