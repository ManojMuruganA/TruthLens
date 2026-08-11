"""
Fine-tune CNN on social media dataset for better real-world accuracy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005
IMAGE_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = "social_media_dataset"
MODEL_SAVE_PATH = "models/cnn_social_finetuned.pth"

class CNNDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    print("=" * 60)
    print("Fine-tuning CNN on Social Media Dataset")
    print("=" * 60)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform)
    test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", transform)
    
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    # Load pre-trained CNN and fine-tune
    model = CNNDetector().to(DEVICE)
    cnn_path = "models/cnn_model.pth"
    if os.path.exists(cnn_path):
        checkpoint = torch.load(cnn_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained CNN (accuracy: {checkpoint['accuracy']:.2f}%)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0
    start = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # Test
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
        test_acc = 100 * correct / total
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': best_acc,
            }, MODEL_SAVE_PATH)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | Best: {best_acc:.1f}%")
    
    print(f"\n✅ Done! Best accuracy: {best_acc:.1f}% | Time: {(time.time()-start)/60:.0f} min")
    print(f"Model saved: {MODEL_SAVE_PATH}")