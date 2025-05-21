import os
import shutil
from pathlib import Path

def merge_tumor_classes(base_data_path):
    # Define paths
    training_path = os.path.join(base_data_path, 'BrainTumorMRIDataset', 'Training')
    testing_path = os.path.join(base_data_path, 'BrainTumorMRIDataset', 'Testing')
    
    # Create new binary dataset structure
    binary_base = os.path.join(base_data_path, 'BinaryBrainTumorDataset')
    
    # Create necessary directories
    for split in ['Training', 'Testing']:
        for category in ['yes', 'no']:
            Path(os.path.join(binary_base, split, category)).mkdir(parents=True, exist_ok=True)
    
    # Process Training and Testing sets
    for split_path, split_name in [(training_path, 'Training'), (testing_path, 'Testing')]:
        # Skip if the path doesn't exist
        if not os.path.exists(split_path):
            print(f"Path doesn't exist: {split_path}")
            continue
        
        # Get all folders (classes) in the split
        classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            
            # Define target path - 'yes' for tumor classes, 'no' for 'notumor'
            if class_name.lower() == 'notumor':
                target_category = 'no'
            else:
                target_category = 'yes'
            
            target_path = os.path.join(binary_base, split_name, target_category)
            
            # Copy all images from this class to the target category
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Create a filename that preserves the original class
                    new_filename = f"{class_name}_{img_file}"
                    shutil.copy(
                        os.path.join(class_path, img_file),
                        os.path.join(target_path, new_filename)
                    )
                    
            print(f"Processed {len(os.listdir(class_path))} images from {class_name} to {target_category}")

if __name__ == "__main__":
    data_path = "data"  
    merge_tumor_classes(data_path)
    print("Binary dataset created at data/BinaryBrainTumorDataset")
    
    # Print statistics
    binary_dataset_path = os.path.join(data_path, "BinaryBrainTumorDataset")
    for split in ['Training', 'Testing']:
        print(f"\n{split} set:")
        for category in ['yes', 'no']:
            category_path = os.path.join(binary_dataset_path, split, category)
            if os.path.exists(category_path):
                count = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
                print(f"  - {category}: {count} images")