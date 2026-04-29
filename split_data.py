from pathlib import Path
import random
import shutil

# Project paths
BASE_DIR = Path("data")
SOURCE_IMAGES = BASE_DIR / "images"
SOURCE_LABELS = BASE_DIR / "labels"

TRAIN_IMAGES = BASE_DIR / "train" / "images"
TRAIN_LABELS = BASE_DIR / "train" / "labels"
VAL_IMAGES = BASE_DIR / "val" / "images"
VAL_LABELS = BASE_DIR / "val" / "labels"

# Create folders if they do not exist
for folder in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
    folder.mkdir(parents=True, exist_ok=True)

# Image extensions to include
image_extensions = [".jpg", ".jpeg", ".png"]

images = [p for p in SOURCE_IMAGES.iterdir() if p.suffix.lower() in image_extensions]

random.seed(42)
random.shuffle(images)

split_index = int(len(images) * 0.8)
train_images = images[:split_index]
val_images = images[split_index:]

def copy_pair(image_path, image_dest, label_dest):
    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"Warning: missing label for {image_path.name}")
        return

    shutil.copy2(image_path, image_dest / image_path.name)
    shutil.copy2(label_path, label_dest / label_path.name)

for image in train_images:
    copy_pair(image, TRAIN_IMAGES, TRAIN_LABELS)

for image in val_images:
    copy_pair(image, VAL_IMAGES, VAL_LABELS)

print(f"Total images: {len(images)}")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")