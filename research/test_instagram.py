"""
Test CNN and ViT models on real Instagram/social media images
Tests on JPEG compressed images to simulate social media conditions
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CNN_MODEL_PATH = "models/cnn_model.pth"
VIT_MODEL_PATH = "models/vit_model.pth"
TEST_FOLDER = "instagram_test"
RESULTS_FOLDER = "instagram_test/results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================================
# CNN MODEL (must match training architecture)
# ============================================================
class CNNDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNDetector, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def load_models():
    """Load both trained models"""
    print("Loading models...")
    
    # Load CNN
    cnn_model = CNNDetector(num_classes=2).to(DEVICE)
    cnn_checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
    cnn_model.eval()
    print(f"  CNN loaded (trained accuracy: {cnn_checkpoint['accuracy']:.2f}%)")
    
    # Load ViT
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2).to(DEVICE)
    vit_checkpoint = torch.load(VIT_MODEL_PATH, map_location=DEVICE, weights_only=False)
    vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
    vit_model.eval()
    print(f"  ViT loaded (trained accuracy: {vit_checkpoint['accuracy']:.2f}%)")
    
    return cnn_model, vit_model


def predict_image_cnn(model, image_path):
    """Predict using CNN (32x32 input)"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['AI-GENERATED', 'REAL']
    return class_names[predicted.item()], confidence.item()


def predict_image_vit(model, image_path):
    """Predict using ViT (224x224 input)"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['AI-GENERATED', 'REAL']
    return class_names[predicted.item()], confidence.item()


def test_on_compressed_versions(image_path, cnn_model, vit_model):
    """Test how compression affects predictions"""
    original = Image.open(image_path).convert('RGB')
    results = {'original': {}, 'compressed': {}}
    
    # Original
    results['original']['cnn'] = predict_image_cnn(cnn_model, image_path)
    results['original']['vit'] = predict_image_vit(vit_model, image_path)
    
    # Compressed version (simulate Instagram compression)
    compressed_path = os.path.join(RESULTS_FOLDER, "compressed_test.jpg")
    original.save(compressed_path, "JPEG", quality=40)  # Instagram uses ~40-60% quality
    
    results['compressed']['cnn'] = predict_image_cnn(cnn_model, compressed_path)
    results['compressed']['vit'] = predict_image_vit(vit_model, compressed_path)
    
    return results


def test_all_images():
    """Test all images in the instagram_test folder"""
    print("=" * 70)
    print("INSTAGRAM IMAGE TEST - CNN vs ViT")
    print("=" * 70)
    
    # Load models
    cnn_model, vit_model = load_models()
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    test_images = []
    
    for file in os.listdir(TEST_FOLDER):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            test_images.append(os.path.join(TEST_FOLDER, file))
    
    if len(test_images) == 0:
        print("\n⚠️  No images found in 'instagram_test' folder!")
        print("Please add some .jpg or .png images to test.")
        print("\nYou can:")
        print("  1. Save Instagram images manually to the folder")
        print("  2. Or run this command to use test set images:")
        print("     python test_instagram.py --use-test-set")
        return
    
    print(f"\nFound {len(test_images)} images to test")
    print("-" * 70)
    
    all_results = []
    
    for i, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        print(f"\n[{i+1}/{len(test_images)}] Testing: {img_name}")
        
        results = test_on_compressed_versions(img_path, cnn_model, vit_model)
        
        print(f"  CNN: {results['original']['cnn'][0]} ({results['original']['cnn'][1]*100:.1f}%)")
        print(f"  ViT: {results['original']['vit'][0]} ({results['original']['vit'][1]*100:.1f}%)")
        print(f"  After compression:")
        print(f"    CNN: {results['compressed']['cnn'][0]} ({results['compressed']['cnn'][1]*100:.1f}%)")
        print(f"    ViT: {results['compressed']['vit'][0]} ({results['compressed']['vit'][1]*100:.1f}%)")
        
        all_results.append({
            'image': img_name,
            'results': results
        })
    
    # ============================================================
    # GENERATE SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Count agreements and disagreements
    cnn_real = sum(1 for r in all_results if r['results']['original']['cnn'][0] == 'REAL')
    vit_real = sum(1 for r in all_results if r['results']['original']['vit'][0] == 'REAL')
    
    print(f"\nCNN predictions: {cnn_real}/{len(all_results)} REAL ({100*cnn_real/len(all_results):.0f}%)")
    print(f"ViT predictions: {vit_real}/{len(all_results)} REAL ({100*vit_real/len(all_results):.0f}%)")
    
    agreements = sum(1 for r in all_results 
                    if r['results']['original']['cnn'][0] == r['results']['original']['vit'][0])
    print(f"Model agreement: {agreements}/{len(all_results)} ({100*agreements/len(all_results):.0f}%)")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(all_results),
        'cnn_real_count': cnn_real,
        'vit_real_count': vit_real,
        'model_agreement': agreements,
        'detailed_results': all_results
    }
    
    with open(f"{RESULTS_FOLDER}/instagram_test_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {RESULTS_FOLDER}/instagram_test_results.json")


def use_test_set_samples():
    """Use samples from the CIFAKE test set for quick testing"""
    print("Using CIFAKE test set samples for testing...")
    
    # Copy some test images
    import shutil
    import random
    
    fake_dir = "data/test/FAKE"
    real_dir = "data/test/REAL"
    
    # Copy 5 AI and 5 real images
    fake_images = random.sample(os.listdir(fake_dir), 5)
    real_images = random.sample(os.listdir(real_dir), 5)
    
    for i, img in enumerate(fake_images):
        shutil.copy(os.path.join(fake_dir, img), 
                   os.path.join(TEST_FOLDER, f"ai_sample_{i+1}.jpg"))
    
    for i, img in enumerate(real_images):
        shutil.copy(os.path.join(real_dir, img), 
                   os.path.join(TEST_FOLDER, f"real_sample_{i+1}.jpg"))
    
    print("Copied 5 AI + 5 real samples to instagram_test folder.")
    print("Now run: python test_instagram.py")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--use-test-set":
        use_test_set_samples()
    else:
        test_all_images()