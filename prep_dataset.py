import os
from PIL import Image

# --- Configuration ---
# The folder where you manually dropped your 40 raw melanoma images
INPUT_FOLDER = "raw_melanoma_selection" 
# Where the AI-ready images will go
OUTPUT_FOLDER = "LoRA_Training_Data" 

# Your specific medical prompt
# Note: 'p4th4i' is your unique trigger word
PROMPT_TEXT = "p4th4i style, macroscopic dermoscopy, malignant melanoma, irregular border, polarized light, sharp focus, 8k"

def prep_images():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    count = 0
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        filepath = os.path.join(INPUT_FOLDER, filename)
        
        try:
            with Image.open(filepath) as img:
                # 1. Calculate a perfect 1:1 Center Crop
                width, height = img.size
                new_size = min(width, height)
                
                left = (width - new_size) / 2
                top = (height - new_size) / 2
                right = (width + new_size) / 2
                bottom = (height + new_size) / 2
                
                img_cropped = img.crop((left, top, right, bottom))
                
                # 2. Upscale smoothly to 512x512 using high-quality Lanczos resampling
                img_resized = img_cropped.resize((512, 512), Image.Resampling.LANCZOS)
                
                # 3. Save the new image
                base_name = os.path.splitext(filename)[0]
                new_img_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.jpg")
                
                # Convert to RGB just in case the original was CMYK or RGBA
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                    
                img_resized.save(new_img_path, quality=100)
                
                # 4. Generate the matching .txt caption file
                txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
                with open(txt_path, "w") as txt_file:
                    txt_file.write(PROMPT_TEXT)
                
                count += 1
                print(f"Processed: {filename} -> 512x512 + Caption")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\n✅ Success! {count} images and text files are ready in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    prep_images()