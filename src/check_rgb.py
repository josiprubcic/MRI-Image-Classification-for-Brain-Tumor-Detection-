import cv2
import os
import numpy as np

DATASET_DIR = "../data/PreprocessedBinaryBrainTumorDataset2"
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff')
OUTPUT_FILE = "image_analysis.txt"

def analyze_color_channels(img, filename):
    """Analyze color channels to determine if RGB or BGR"""
    if len(img.shape) == 2:
        return "GRAYSCALE"
    
    if img.shape[2] != 3:
        return f"UNKNOWN ({img.shape[2]} channels)"
    
    # Calculate mean values for each channel
    b_mean = np.mean(img[:,:,0])
    g_mean = np.mean(img[:,:,1]) 
    r_mean = np.mean(img[:,:,2])
    
    # Check if all channels are identical (grayscale stored as RGB)
    if np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]):
        return "GRAYSCALE_AS_RGB"
    
    # Check channel statistics
    channel_diff = np.max([b_mean, g_mean, r_mean]) - np.min([b_mean, g_mean, r_mean])
    
    return f"3-channel (B:{b_mean:.1f}, G:{g_mean:.1f}, R:{r_mean:.1f}, diff:{channel_diff:.1f})"

def check_opencv_default():
    """Check OpenCV's default color space"""
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[:, :, 0] = 255  # Set first channel to red
    
    cv2.imwrite("test_color.jpg", test_img)
    reloaded = cv2.imread("test_color.jpg")
    
    if np.mean(reloaded[:,:,2]) > np.mean(reloaded[:,:,0]):
        color_order = "BGR (OpenCV default)"
    else:
        color_order = "RGB"
    
    os.remove("test_color.jpg")
    return color_order

# Initialize counters
stats = {
    'total_images': 0,
    'grayscale': 0,
    'grayscale_as_rgb': 0,
    'true_rgb': 0,
    'unreadable': 0,
    'shapes': {},
    'formats': {},
    'by_category': {}
}

with open(OUTPUT_FILE, "w") as out:
    out.write(f"OpenCV default color order: {check_opencv_default()}\n")
    out.write("="*70 + "\n")
    out.write("COMPREHENSIVE IMAGE ANALYSIS\n")
    out.write("="*70 + "\n")
    
    for split in ["Training", "Testing"]:
        for category in ["yes", "no"]:
            folder = os.path.join(DATASET_DIR, split, category)
            if not os.path.exists(folder):
                out.write(f"Folder not found: {folder}\n")
                continue
                
            out.write(f"\n{'='*50}\n")
            out.write(f"ANALYZING: {split}/{category}\n")
            out.write(f"{'='*50}\n")
            
            category_stats = {
                'total': 0,
                'grayscale': 0,
                'grayscale_as_rgb': 0,
                'true_rgb': 0,
                'unreadable': 0
            }
            
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(ALLOWED_EXTENSIONS):
                    img_path = os.path.join(folder, fname)
                    img = cv2.imread(img_path)
                    
                    stats['total_images'] += 1
                    category_stats['total'] += 1
                    
                    if img is not None:
                        color_info = analyze_color_channels(img, fname)
                        shape_str = f"{img.shape}"
                        
                        # Update statistics
                        if shape_str not in stats['shapes']:
                            stats['shapes'][shape_str] = 0
                        stats['shapes'][shape_str] += 1
                        
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in stats['formats']:
                            stats['formats'][ext] = 0
                        stats['formats'][ext] += 1
                        
                        if "GRAYSCALE_AS_RGB" in color_info:
                            stats['grayscale_as_rgb'] += 1
                            category_stats['grayscale_as_rgb'] += 1
                        elif "GRAYSCALE" in color_info:
                            stats['grayscale'] += 1
                            category_stats['grayscale'] += 1
                        elif "3-channel" in color_info:
                            stats['true_rgb'] += 1
                            category_stats['true_rgb'] += 1
                        
                        out.write(f"{fname}: {shape_str}, {color_info}\n")
                    else:
                        stats['unreadable'] += 1
                        category_stats['unreadable'] += 1
                        out.write(f"{fname}: COULD NOT BE READ\n")
            
            # Category summary
            out.write(f"\nSUMMARY for {split}/{category}:\n")
            out.write(f"  Total files: {category_stats['total']}\n")
            out.write(f"  Grayscale: {category_stats['grayscale']}\n")
            out.write(f"  Grayscale as RGB: {category_stats['grayscale_as_rgb']}\n")
            out.write(f"  True RGB: {category_stats['true_rgb']}\n")
            out.write(f"  Unreadable: {category_stats['unreadable']}\n")
            
            stats['by_category'][f"{split}/{category}"] = category_stats

    # Overall summary
    out.write(f"\n{'='*70}\n")
    out.write("OVERALL ANALYSIS SUMMARY\n")
    out.write(f"{'='*70}\n")
    out.write(f"Total images processed: {stats['total_images']}\n")
    out.write(f"Grayscale images: {stats['grayscale']} ({stats['grayscale']/stats['total_images']*100:.1f}%)\n")
    out.write(f"Grayscale as RGB: {stats['grayscale_as_rgb']} ({stats['grayscale_as_rgb']/stats['total_images']*100:.1f}%)\n")
    out.write(f"True RGB images: {stats['true_rgb']} ({stats['true_rgb']/stats['total_images']*100:.1f}%)\n")
    out.write(f"Unreadable images: {stats['unreadable']} ({stats['unreadable']/stats['total_images']*100:.1f}%)\n")
    
    out.write(f"\nImage shapes found:\n")
    for shape, count in sorted(stats['shapes'].items()):
        out.write(f"  {shape}: {count} images ({count/stats['total_images']*100:.1f}%)\n")
    
    out.write(f"\nFile formats found:\n")
    for fmt, count in sorted(stats['formats'].items()):
        out.write(f"  {fmt}: {count} images ({count/stats['total_images']*100:.1f}%)\n")
    
    out.write(f"\nRecommendations:\n")
    if stats['grayscale_as_rgb'] > stats['true_rgb']:
        out.write("  - Most images are grayscale stored as 3-channel\n")
        out.write("  - Consider converting to true grayscale for efficiency\n")
        out.write("  - Use cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) in preprocessing\n")
    elif stats['true_rgb'] > 0:
        out.write("  - Dataset contains true RGB images\n")
        out.write("  - Keep RGB format for better feature representation\n")
        out.write("  - Use cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for matplotlib/PIL compatibility\n")
    
    if stats['unreadable'] > 0:
        out.write(f"  - {stats['unreadable']} images could not be read - check file corruption\n")

print(f"Comprehensive analysis completed. Results written to {OUTPUT_FILE}")
print(f"Total images analyzed: {stats['total_images']}")
print(f"Grayscale as RGB: {stats['grayscale_as_rgb']}")
print(f"True RGB: {stats['true_rgb']}")
print(f"Pure Grayscale: {stats['grayscale']}")