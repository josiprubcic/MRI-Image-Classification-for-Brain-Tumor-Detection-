import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = "data/BinaryBrainTumorDataset"
OUTPUT_DIR = "data/PreprocessedBinaryBrainTumorDataset2"
IMG_SIZE = (384, 384)
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff'}

def auto_crop(img):
    """Automatically removes the black border around the brain region."""
    # Convert to grayscale only for thresholding
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Add padding
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    return img[y:y+h, x:x+w]

def apply_clahe_rgb(img):
    """Apply CLAHE to RGB image."""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    # Convert back to BGR
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_image(img_path):
    """Main function for preprocessing a single RGB image."""
    try:
        # Load RGB image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        # Apply all preprocessing steps
        img = auto_crop(img)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = apply_clahe_rgb(img)
        img = cv2.medianBlur(img, 3)
       
       
       
        
        # Normalize to [0, 1] and convert back for saving
        img = (img / 255.0).astype(np.float32)
        return (img * 255).astype(np.uint8)
    
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
