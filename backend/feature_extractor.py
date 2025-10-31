import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

# ============================================================
# GPU CONFIGURATION
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch Version: {torch.__version__}")
else:
    print("⚠️ No GPU found, running on CPU")

# ============================================================
# CONFIGURATION
# ============================================================

IMAGES_FOLDER = os.getenv("IMAGES_FOLDER")
TAGS_FOLDER = os.getenv("TAGS_FOLDER")
OUTPUT_JSON = os.getenv("OUTPUT_JSON")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))





# ============================================================
# LOAD MODEL
# ============================================================

print("🚀 Loading VGG16 model (ImageNet weights)...")

# Load pretrained VGG16
vgg16 = models.vgg16(pretrained=True)

# Remove the classifier (fully connected layers) to get features
# VGG16 features output is 512-dimensional after adaptive pooling
model = nn.Sequential(
    vgg16.features,  # Convolutional layers
    nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
    nn.Flatten()  # Flatten to get 512-dim vector
)

model = model.to(device)
model.eval()  # Set to evaluation mode

print("✅ Model loaded successfully.\n")
print(f"Output feature dimension: 512\n")

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_and_preprocess_image(img_path):
    """Load and preprocess an image for VGG16."""
    try:
        img = Image.open(img_path).convert('RGB')
        return preprocess(img)
    except Exception as e:
        print(f"⚠️ Error loading {img_path}: {e}")
        return None


# ============================================================
# MAIN LOOP WITH BATCHING 
# ============================================================

print("📄 Creating JSON mapping with batching...\n")

image_files = sorted([
    f for f in os.listdir(IMAGES_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
total_images = len(image_files)
print(f"📦 Found {total_images} images.\n")


start_idx = 0
mode = "w"

# Open output file
with open(OUTPUT_JSON, mode=mode, encoding="utf-8") as f_out:
    processed_count = start_idx
    
    # Process in batches
    for start in tqdm(range(start_idx, total_images, BATCH_SIZE), desc="Processing batches"):
        end = min(start + BATCH_SIZE, total_images)
        batch_files = image_files[start:end]
        batch_images = []
        valid_files = []

        # Load batch images
        for img_file in batch_files:
            img_path = os.path.join(IMAGES_FOLDER, img_file)
            img_tensor = load_and_preprocess_image(img_path)
            if img_tensor is not None:
                batch_images.append(img_tensor)
                valid_files.append(img_file)

        if not batch_images:
            continue  # skip empty batch

        # Stack images into a batch tensor
        batch_tensor = torch.stack(batch_images).to(device)

        # Extract features (no gradient computation needed)
        with torch.no_grad():
            batch_features = model(batch_tensor).cpu().numpy()

        # Track problematic images
        missing_tags = []
        empty_tags = []

        # Process each image in the batch
        for i, img_file in enumerate(valid_files):
            img_path = os.path.join(IMAGES_FOLDER, img_file)
            tag_path = os.path.join(TAGS_FOLDER, os.path.splitext(img_file)[0] + ".txt")

            # Read tag
            tag_text = ""
            if not os.path.exists(tag_path):
                missing_tags.append(img_file)
            else:
                with open(tag_path, "r", encoding="utf-8") as t_file:
                    tag_text = t_file.read().strip()
                    if not tag_text:
                        empty_tags.append(img_file)

            # Build record
            record = {
                "id": img_file,
                "path": img_path,
                "tag": tag_text,
                "embedding": batch_features[i].tolist()
            }

            # Write to JSON
            f_out.write(json.dumps(record) + "\n")
            processed_count += 1

        # Print batch summary
        if missing_tags or empty_tags:
            print(f"\n⚠️ Batch [{start+1}-{end}]: Missing tags: {len(missing_tags)}, Empty tags: {len(empty_tags)}")


print(f"\n✅ All done! JSON mapping file created at:\n{OUTPUT_JSON}")
print(f"📊 Total images processed: {processed_count}/{total_images}")