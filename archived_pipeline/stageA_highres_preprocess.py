import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

INPUT_DIR = os.path.join("data", "train_images", "train")
OUTPUT_DIR = os.path.join("data", "train_images_512_ben")
IMG_SIZE = 512

def crop_image_from_gray(img, tol=7):
    """
    Crop the black borders from fundus images based on intensity threshold.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0: 
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def process_and_save(filename):
    try:
        img_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(out_path):
            return True
            
        # Read image (OpenCV uses BGR, we convert to RGB)
        img = cv2.imread(img_path)
        if img is None:
            return False
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop uninformative black borders
        img = crop_image_from_gray(img)
        
        # Resize to 512x512
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Ben's Preprocessing (local average blurring to normalize lighting)
        # cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), IMG_SIZE/10), -4, 128)
        blur_size = int(IMG_SIZE / 10)
        # Blur kernel size must be odd
        if blur_size % 2 == 0:
            blur_size += 1
            
        blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        img = cv2.addWeighted(img, 4, blurred, -4, 128)
        
        # Convert back to BGR for saving
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img)
        return True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpeg")]
    print(f"Found {len(files)} images to process.")
    
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_and_save, f): f for f in files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Ben's Preprocessing (512x512)"):
            if future.result():
                success_count += 1
                
    print(f"Successfully preprocessed {success_count} / {len(files)} images.")
    print(f"Preprocessed dataset saved to {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
