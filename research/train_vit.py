"""
Vision Transformer (ViT) for AI-Generated Image Detection
Fine-tunes pre-trained ViT on CiFAKE dataset
Compares ViT vs CNN performance
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import time
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32        # Smaller batch for 224x224 images on CPU
EPOCHS = 5            # Fine-tuning needs fewer epochs
LEARNING_RATE = 0.0001 # Lower LR for pre-trained model
IMAGE_SIZE = 224       # ViT standard input size
NUM_CLASSES = 2        # REAL vs FAKE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = "data"
MODEL_SAVE_PATH = "models/vit_model.pth"
RESULTS_DIR = "results"

print("=" * 70)
print("VISION TRANSFORMER (ViT) TRAINING")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Model: ViT-Base/16 (pre-trained on ImageNet-21k)")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print("=" * 70)

# ============================================================
# DATA PREPARATION
# ============================================================
print("\n[1/4] Loading dataset...")

# ViT uses ImageNet normalization
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train", transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test", transform=test_transform
)

# Use subset for faster training (comment out for full dataset)
# For full training: use all data (takes 6-12 hours on CPU)
# For quick test: use 20% of data (takes 1-2 hours)
USE_FULL_DATASET = False  # Set to False for quick testing

if not USE_FULL_DATASET:
    from torch.utils.data import Subset
    train_indices = np.random.choice(len(train_dataset), 5000, replace=False)
    test_indices = np.random.choice(len(test_dataset), 1000, replace=False)
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    print("⚠️  Using subset: 5K train, 1K test (for quick testing)")
else:
    print("Using full dataset: 100K train, 20K test")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batches (train): {len(train_loader)}")

# ============================================================
# BUILD ViT MODEL
# ============================================================
print("\n[2/4] Building ViT model...")

model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=NUM_CLASSES
).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ============================================================
# TRAINING
# ============================================================
print("\n[3/4] Starting training...")
print("-" * 70)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
best_accuracy = 0.0

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Show progress every 10% of an epoch
        if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
            progress = 100 * (batch_idx + 1) / len(train_loader)
            print(f"  Epoch {epoch+1}/{EPOCHS} | Progress: {progress:.0f}% | "
                  f"Loss: {loss.item():.4f} | Acc: {100*correct/total:.1f}%")
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Validation phase
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    scheduler.step()
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # Save best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_accuracy,
            'loss': test_loss,
            'model_name': 'vit_base_patch16_224',
        }, MODEL_SAVE_PATH)
    
    epoch_time = time.time() - epoch_start
    print(f"\n  ✅ Epoch {epoch+1}/{EPOCHS} done in {epoch_time/60:.1f} min")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    print(f"  Best Accuracy: {best_accuracy:.2f}%")
    print("-" * 70)

total_time = time.time() - start_time
print(f"\n🎉 Training completed in {total_time/60:.1f} minutes")
print(f"🏆 Best Test Accuracy: {best_accuracy:.2f}%")

# ============================================================
# SAVE RESULTS & COMPARISON
# ============================================================
print("\n[4/4] Saving results and generating comparison...")

# Save ViT history
vit_history = {
    'model': 'ViT-Base/16 (pre-trained ImageNet)',
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'best_accuracy': best_accuracy,
    'total_params': total_params,
    'total_time_minutes': total_time / 60,
    'timestamp': datetime.now().isoformat()
}

with open(f"{RESULTS_DIR}/vit_history.json", 'w') as f:
    json.dump(vit_history, f, indent=2)

# Load CNN history for comparison
cnn_history = None
try:
    with open(f"{RESULTS_DIR}/training_history.json", 'r') as f:
        cnn_history = json.load(f)
except:
    print("⚠️  CNN history not found - comparison will be ViT only")

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ViT Loss
axes[0, 0].plot(train_losses, 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(test_losses, 'r-', label='Test Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('ViT: Loss Curves', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# ViT Accuracy
axes[0, 1].plot(train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
axes[0, 1].plot(test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 1].set_title('ViT: Accuracy Curves', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# CNN vs ViT Accuracy Comparison
if cnn_history:
    axes[1, 0].plot(cnn_history['test_accuracies'], 'b-', label=f'CNN (Best: {cnn_history["best_accuracy"]:.2f}%)', linewidth=2)
    axes[1, 0].plot(test_accuracies, 'purple', label=f'ViT (Best: {best_accuracy:.2f}%)', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('CNN vs ViT: Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

# Bar Chart Comparison
if cnn_history:
    models = ['CNN', 'ViT']
    accuracies = [cnn_history['best_accuracy'], best_accuracy]
    params = [cnn_history.get('total_params', 845602), total_params]
    times = [cnn_history.get('total_time_minutes', 54), total_time/60]
    
    colors = ['#2196F3', '#9C27B0']
    bars = axes[1, 1].bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Best Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Best Accuracy: CNN vs ViT', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylim(80, 100)
    
    for bar, acc, p, t in zip(bars, accuracies, params, times):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                       f'{acc:.2f}%\n{p:,} params\n{t:.0f} min',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/cnn_vs_vit_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR}/cnn_vs_vit_comparison.png")

# Save comparison summary
with open(f"{RESULTS_DIR}/comparison_summary.txt", 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("CNN vs VISION TRANSFORMER - COMPARISON\n")
    f.write("=" * 60 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    if cnn_history:
        f.write(f"{'Metric':<25} {'CNN':<15} {'ViT':<15}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Best Accuracy':<25} {cnn_history['best_accuracy']:.2f}%{'':>9} {best_accuracy:.2f}%\n")
        f.write(f"{'Total Parameters':<25} {cnn_history.get('total_params', 845602):,}{'':>5} {total_params:,}\n")
        f.write(f"{'Training Time':<25} {cnn_history.get('total_time_minutes', 54):.0f} min{'':>8} {total_time/60:.0f} min\n")
        f.write(f"{'Image Size':<25} {'32x32':<15} {'224x224':<15}\n")
        f.write(f"{'Architecture':<25} {'Custom CNN':<15} {'ViT-Base/16':<15}\n")
        f.write(f"{'Pre-trained':<25} {'No':<15} {'Yes (ImageNet)':<15}\n")
        
        winner = "ViT" if best_accuracy > cnn_history['best_accuracy'] else "CNN"
        f.write(f"\n*** Winner (Accuracy): {winner} ***\n")
        f.write(f"   Difference: {abs(best_accuracy - cnn_history['best_accuracy']):.2f}%\n")

print(f"Saved: {RESULTS_DIR}/comparison_summary.txt")
print("\n✅ All done!")
print(f"📊 ViT Best Accuracy: {best_accuracy:.2f}%")