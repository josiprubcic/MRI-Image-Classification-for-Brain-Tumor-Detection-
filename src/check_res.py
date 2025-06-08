import os
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_image_sizes(directory_path):
    """Analyze image dimensions and file sizes in a dataset directory."""
    
    results = {
        'dimensions': [],
        'file_sizes': [],
        'widths': [],
        'heights': [],
        'aspect_ratios': [],
        'file_paths': [],
        'categories': []
    }
    
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in allowed_extensions:
                file_path = os.path.join(root, file)
                
                try:
                    # Get file size in KB
                    file_size_kb = os.path.getsize(file_path) / 1024
                    
                    # Read image to get dimensions
                    img = cv2.imread(file_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        aspect_ratio = width / height
                        
                        # Extract category from path
                        category = os.path.relpath(file_path, directory_path)
                        
                        # Store results
                        results['dimensions'].append((width, height))
                        results['file_sizes'].append(file_size_kb)
                        results['widths'].append(width)
                        results['heights'].append(height)
                        results['aspect_ratios'].append(aspect_ratio)
                        results['file_paths'].append(file_path)
                        results['categories'].append(category)
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return results

def print_statistics(results):
    """Print detailed statistics about image sizes."""
    
    if not results['dimensions']:
        print("No images found!")
        return
    
    print(f"\n📊 IMAGE SIZE ANALYSIS")
    print("=" * 50)
    
    # Basic counts
    print(f"Total images analyzed: {len(results['dimensions'])}")
    
    # Dimension statistics
    widths = np.array(results['widths'])
    heights = np.array(results['heights'])
    file_sizes = np.array(results['file_sizes'])
    
    print(f"\n📐 DIMENSIONS:")
    print(f"Width  - Min: {widths.min():4d}, Max: {widths.max():4d}, Mean: {widths.mean():.1f}, Std: {widths.std():.1f}")
    print(f"Height - Min: {heights.min():4d}, Max: {heights.max():4d}, Mean: {heights.mean():.1f}, Std: {heights.std():.1f}")
    
    print(f"\n💾 FILE SIZES (KB):")
    print(f"Min: {file_sizes.min():.1f}, Max: {file_sizes.max():.1f}, Mean: {file_sizes.mean():.1f}, Std: {file_sizes.std():.1f}")
    
    # Most common dimensions
    unique_dims, counts = np.unique(results['dimensions'], axis=0, return_counts=True)
    print(f"\n🔢 MOST COMMON DIMENSIONS:")
    for i in np.argsort(counts)[-5:][::-1]:  # Top 5
        w, h = unique_dims[i]
        print(f"  {w}×{h}: {counts[i]} images")
    
    # Aspect ratios
    aspect_ratios = np.array(results['aspect_ratios'])
    print(f"\n📏 ASPECT RATIOS:")
    print(f"Min: {aspect_ratios.min():.3f}, Max: {aspect_ratios.max():.3f}, Mean: {aspect_ratios.mean():.3f}")
    
    # Category breakdown
    category_stats = defaultdict(list)
    for i, category in enumerate(results['categories']):
        # Extract just the immediate parent directory
        cat = os.path.dirname(category).replace('\\', '/').split('/')[-1] if os.path.dirname(category) else 'root'
        category_stats[cat].append(i)
    
    print(f"\n📁 BY CATEGORY:")
    for cat, indices in category_stats.items():
        cat_widths = [results['widths'][i] for i in indices]
        cat_heights = [results['heights'][i] for i in indices]
        print(f"  {cat}: {len(indices)} images, avg size: {np.mean(cat_widths):.0f}×{np.mean(cat_heights):.0f}")

def create_visualizations(results):
    """Create plots to visualize image size distribution."""
    
    if not results['dimensions']:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Image Size Analysis', fontsize=16)
    
    # Dimension scatter plot
    axes[0,0].scatter(results['widths'], results['heights'], alpha=0.6, s=20)
    axes[0,0].set_xlabel('Width (pixels)')
    axes[0,0].set_ylabel('Height (pixels)')
    axes[0,0].set_title('Width vs Height Distribution')
    axes[0,0].grid(True, alpha=0.3)
    
    # Width histogram
    axes[0,1].hist(results['widths'], bins=30, alpha=0.7, color='blue')
    axes[0,1].set_xlabel('Width (pixels)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Width Distribution')
    axes[0,1].grid(True, alpha=0.3)
    
    # Height histogram
    axes[1,0].hist(results['heights'], bins=30, alpha=0.7, color='green')
    axes[1,0].set_xlabel('Height (pixels)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Height Distribution')
    axes[1,0].grid(True, alpha=0.3)
    
    # File size histogram
    axes[1,1].hist(results['file_sizes'], bins=30, alpha=0.7, color='red')
    axes[1,1].set_xlabel('File Size (KB)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('File Size Distribution')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to analyze both original and preprocessed datasets."""
    
    # Analyze original dataset
    original_dir = "data/BinaryBrainTumorDataset"
    preprocessed_dir = "data/PreprocessedBinaryBrainTumorDataset2"
    
    print("🔍 ANALYZING ORIGINAL DATASET")
    if os.path.exists(original_dir):
        original_results = analyze_image_sizes(original_dir)
        print_statistics(original_results)
        
        # Show visualizations
        print("\nCreating visualizations for original dataset...")
        create_visualizations(original_results)
    else:
        print(f"Original dataset directory not found: {original_dir}")
    
    print("\n" + "="*60)
    
    print("🔍 ANALYZING PREPROCESSED DATASET")
    if os.path.exists(preprocessed_dir):
        preprocessed_results = analyze_image_sizes(preprocessed_dir)
        print_statistics(preprocessed_results)
        
        # Show visualizations
        print("\nCreating visualizations for preprocessed dataset...")
        create_visualizations(preprocessed_results)
    else:
        print(f"Preprocessed dataset directory not found: {preprocessed_dir}")

if __name__ == "__main__":
    main()