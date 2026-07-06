import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import shutil
import io
from tqdm import tqdm

# Device එක තෝරාගැනීම (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path = "epoch-82.pth"
use_local_weights = os.path.exists(weights_path)

if use_local_weights:
    try:
        import pyiqa
        model = pyiqa.create_metric('nima', device=device)
        checkpoint = torch.load(weights_path, map_location=device)
        if 'params' in checkpoint:
            model.net.load_state_dict(checkpoint['params'])
        else:
            model.net.load_state_dict(checkpoint)
        model.eval()
        print("Successfully loaded local NIMA Aesthetic Weights! [SUCCESS]")
    except Exception as e:
        print(f"[Warning] Error loading '{weights_path}': {e}. Falling back to pre-trained PyIQA model.")
        use_local_weights = False

if not use_local_weights:
    try:
        import pyiqa
        print("\nInitializing pre-trained NIMA model via pyiqa (this may take a few seconds)...")
        model = pyiqa.create_metric('nima', device=device)
        print("Pre-trained NIMA model loaded successfully! [SUCCESS]")
    except Exception as e:
        print(f"[Error] Local weights file 'epoch-82.pth' is missing and 'pyiqa' could not be initialized: {e}")
        print("Please download 'epoch-82.pth' and place it in the same directory as this script.")
        exit(1)

print(f"Running on: {device}")

# 2. Image Preprocessing Pipeline
nima_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_mean_score(output):
    probs = output.cpu().numpy()[0]
    weights = np.arange(1, 11)
    return np.sum(probs * weights)

def load_image(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    raw_extensions = {'.cr2', '.nef', '.arw', '.dng', '.cr3', '.orf', '.rw2', '.pef'}
    
    if ext in raw_extensions:
        import rawpy
        try:
            with rawpy.imread(img_path) as raw:
                try:
                    # High performance thumbnail extraction
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        return Image.open(io.BytesIO(thumb.data))
                except Exception:
                    pass
                
                # Fast half-size RAW decoding fallback
                try:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                except Exception:
                    rgb = raw.postprocess(half_size=True)
                return Image.fromarray(rgb)
        except Exception as e:
            raise RuntimeError(f"Failed to decode RAW image: {e}")
    else:
        return Image.open(img_path).convert('RGB')

# 3. Input & Output Paths Setup
default_input = r"D:\Club_Photos\Event_Bulk"

input_folder = default_input
if not os.path.exists(input_folder):
    print(f"\nDefault input folder '{default_input}' not found.")
    while True:
        user_input = input("Please enter the path to your input photos folder (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            print("Exiting...")
            exit()
        user_input = user_input.strip('"').strip("'")
        if os.path.exists(user_input) and os.path.isdir(user_input):
            input_folder = user_input
            break
        else:
            print(f"Error: Path '{user_input}' does not exist. Please try again.")

output_folder = os.path.join(input_folder, "Best_Selected")
try:
    os.makedirs(output_folder, exist_ok=True)
except Exception as e:
    print(f"Error creating output folder: {e}")
    exit(1)

# Minimum Score Threshold එක ලබාගැනීම
print("\nEnter minimum quality threshold (1.0 to 10.0) to filter photos.")
print("Photos with scores above this will be selected. (Default: 5.0)")
threshold_input = input("Threshold (or press Enter for default 5.0): ").strip()
try:
    min_score = float(threshold_input) if threshold_input else 5.0
except ValueError:
    min_score = 5.0
    print("Invalid input. Using default threshold: 5.0")

image_scores = []
valid_extensions = {
    '.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp',
    '.cr2', '.nef', '.arw', '.dng', '.cr3', '.orf', '.rw2', '.pef'
}

print("\nScanning input folder...")
try:
    all_files = os.listdir(input_folder)
except Exception as e:
    print(f"Error reading input folder: {e}")
    all_files = []

valid_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions]

if not valid_files:
    print(f"No supported photos found in '{input_folder}'.")
else:
    print(f"Found {len(valid_files)} photos. Processing...")
    
    import concurrent.futures
    max_workers = min(os.cpu_count() or 4, 6)
    preload_limit = max_workers * 2
    
    file_list = list(valid_files)
    futures = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit initial batch to start loading in background
        for i in range(min(len(file_list), preload_limit)):
            img_name = file_list[i]
            img_path = os.path.join(input_folder, img_name)
            futures[executor.submit(load_image, img_path)] = (i, img_name)
            
        next_idx = preload_limit
        
        with tqdm(total=len(file_list), desc="Assessing image quality", leave=True) as pbar:
            while futures:
                # Wait for at least one image loading task to complete
                done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                
                for f in done:
                    idx, img_name = futures.pop(f)
                    try:
                        img = f.result()
                        try:
                            # Run model inference sequentially on main thread
                            with torch.no_grad():
                                score = model(img).item()
                            
                            image_scores.append((img_name, score))
                        finally:
                            # Ensure the PIL Image object is closed immediately to free memory
                            img.close()
                    except Exception as e:
                        # Tqdm progress bar එක කැඩෙන්නේ නැතිවෙන්න print කිරීම
                        tqdm.write(f"Error processing {img_name}: {e}")
                    
                    pbar.update(1)
                    
                    # Submit next photo to keep preload buffer full
                    if next_idx < len(file_list):
                        next_img_name = file_list[next_idx]
                        next_img_path = os.path.join(input_folder, next_img_name)
                        futures[executor.submit(load_image, next_img_path)] = (next_idx, next_img_name)
                        next_idx += 1

if image_scores:
    # Score එක අනුව වැඩිම එකේ සිට අඩුම එකට Sort කිරීම
    image_scores.sort(key=lambda x: x[1], reverse=True)

    # Filter images based on min_score threshold
    selected_photos = [(name, score) for name, score in image_scores if score >= min_score]
    discarded_photos = [(name, score) for name, score in image_scores if score < min_score]

    print(f"\n--- SELECTED PHOTOS (Score >= {min_score:.2f}) ---")
    if selected_photos:
        for i, (img_name, score) in enumerate(selected_photos):
            print(f"{i+1}. {img_name} - Score: {score:.2f}/10")
            
            src = os.path.join(input_folder, img_name)
            dst = os.path.join(output_folder, img_name)
            shutil.copy(src, dst)
        print(f"\nDone! {len(selected_photos)} best photos copied to: {output_folder}")
    else:
        print("No photos met the minimum quality threshold.")

    if discarded_photos:
        print(f"\n--- DISCARDED LOW QUALITY/BLURRY PHOTOS (Score < {min_score:.2f}) ---")
        for i, (img_name, score) in enumerate(discarded_photos[:15]):
            print(f"- {img_name} - Score: {score:.2f}/10")
        if len(discarded_photos) > 15:
            print(f"... and {len(discarded_photos) - 15} more.")
else:
    print("\nNo photos were successfully processed.")