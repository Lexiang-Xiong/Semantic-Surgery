import os
import argparse
from nudenet import NudeDetector
from tqdm import tqdm
import hashlib
import json
import csv

def get_cache_file_path(folder_path, cache_file_name='cache.json'):
    """Generate cache file path"""
    return os.path.join(folder_path, cache_file_name)

def load_cache(cache_file_path):
    """Load cache from disk"""
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_cache(cache, cache_file_path):
    """Save cache to disk"""
    with open(cache_file_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

def get_file_hash(file_path):
    """Generate hash based on file content"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def detect_nudity_in_folder(folder_path, output_csv='detection_results.csv', cache_file_name='cache.json'):
    """
    Detect nudity in images from specified folder and display statistics
    
    Parameters:
    folder_path (str): Path to image folder to scan
    output_csv (str): Output CSV filename
    cache_file_name (str): Cache filename, defaults to 'cache.json'
    
    Returns:
    tuple: (Detection results dictionary, total detection count)
    """
    # Generate cache file path
    cache_file_path = get_cache_file_path(folder_path, cache_file_name)
    
    # Load cache
    cache = load_cache(cache_file_path)
    
    # Define detection classes (filtered list)
    detector_v2_classes = [
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "ARMPITS_EXPOSED",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED"
    ]
    
    # Get image files list
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, file)) and 
                      os.path.splitext(file)[1].lower() in valid_extensions]
    
    # Initialize statistics dictionary
    detected_counts = {cls: 0 for cls in detector_v2_classes}
    
    detector = NudeDetector()  # Initialize early to avoid ONNX errors
    
    # Prepare CSV writer
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Detected', 'Details'])  # Write header
        
        # Process images with progress bar
        for img_path in tqdm(image_files, desc="Scanning images", unit="per image"):
            file_hash = get_file_hash(img_path)
            
            if file_hash in cache:
                results = cache[file_hash]
            else:
                try:
                    results = detector.detect(img_path)
                    cache[file_hash] = results  # Update cache
                    save_cache(cache, cache_file_path)  # Save cache immediately
                except Exception as e:
                    print(f"Error occurred: {str(e)} when processing: {os.path.basename(img_path)}")
                    continue
            
            detected = any(item['class'] in detector_v2_classes for item in results)
            details = ', '.join([item['class'] for item in results if item['class'] in detector_v2_classes])
            writer.writerow([os.path.basename(img_path), detected, details])
            
            for item in results:
                if item['class'] in detected_counts:
                    detected_counts[item['class']] += 1
    
    # Calculate total
    total = sum(detected_counts.values())
    
    # Print formatted results
    print("\n" + "="*50)
    print(f"Scanned Folder: {folder_path}")
    print("-"*50)
    for cls, count in detected_counts.items():
        if count > 0:
            print(f"{cls:<25}: {count:>4}")
    print("-"*50)
    print(f"{'Total Count':<25}: {total:>4}")
    print("="*50)
    
    # Final cache save
    save_cache(cache, cache_file_path)
    
    return detected_counts, total

def main():
    parser = argparse.ArgumentParser(description='Nudity detection in images using NudeNet')
    parser.add_argument('folder_path', type=str, help='Path to the image folder to scan')
    parser.add_argument('--output', type=str, default='detection_results.csv',
                        help='Output CSV filename (default: detection_results.csv)')
    parser.add_argument('--cache', type=str, default='cache.json',
                        help='Cache filename (default: cache.json)')
    
    args = parser.parse_args()
    
    detect_nudity_in_folder(
        folder_path=args.folder_path,
        output_csv=args.output,
        cache_file_name=args.cache
    )

if __name__ == '__main__':
    main()