"""
Split social media dataset into 80% train / 20% test
"""
import os
import shutil
import random

BASE = "social_media_dataset"
REAL_SRC = os.path.join(BASE, "real")
AI_SRC = os.path.join(BASE, "ai_generated")

TRAIN_REAL = os.path.join(BASE, "train", "REAL")
TEST_REAL = os.path.join(BASE, "test", "REAL")
TRAIN_FAKE = os.path.join(BASE, "train", "FAKE")
TEST_FAKE = os.path.join(BASE, "test", "FAKE")

# Create folders
for folder in [TRAIN_REAL, TEST_REAL, TRAIN_FAKE, TEST_FAKE]:
    os.makedirs(folder, exist_ok=True)

# Get all images
real_images = [f for f in os.listdir(REAL_SRC) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
ai_images = [f for f in os.listdir(AI_SRC) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]

print(f"Found: {len(real_images)} real + {len(ai_images)} AI = {len(real_images) + len(ai_images)} total")

# Shuffle
random.seed(42)
random.shuffle(real_images)
random.shuffle(ai_images)

# Split 80/20
real_split = int(len(real_images) * 0.8)
ai_split = int(len(ai_images) * 0.8)

# Copy to train
for f in real_images[:real_split]:
    shutil.copy(os.path.join(REAL_SRC, f), os.path.join(TRAIN_REAL, f))
for f in ai_images[:ai_split]:
    shutil.copy(os.path.join(AI_SRC, f), os.path.join(TRAIN_FAKE, f))

# Copy to test
for f in real_images[real_split:]:
    shutil.copy(os.path.join(REAL_SRC, f), os.path.join(TEST_REAL, f))
for f in ai_images[ai_split:]:
    shutil.copy(os.path.join(AI_SRC, f), os.path.join(TEST_FAKE, f))

print(f"\nTrain: {real_split} real + {ai_split} AI = {real_split + ai_split}")
print(f"Test: {len(real_images)-real_split} real + {len(ai_images)-ai_split} AI = {len(real_images)+len(ai_images)-real_split-ai_split}")
print("\n✅ Dataset ready!")