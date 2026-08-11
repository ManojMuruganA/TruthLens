"""
Generate charts for final presentation PPT
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CHART 1: Lab vs Reality Accuracy Gap (Slide 6)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

models = ['CNN\n(Before Fine-tuning)', 'CNN\n(After Fine-tuning)']
accuracies = [65.00, 98.75]
colors = ['#dc2626', '#16a34a']

bars = ax.bar(models, accuracies, color=colors, edgecolor='white', linewidth=2, width=0.5)

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{acc:.2f}%', ha='center', fontsize=18, fontweight='bold')

# Add drop arrow
ax.annotate('+33.75%\nimprovement', 
            xy=(1, 98.75), xytext=(1, 80),
            fontsize=14, ha='center', color='#16a34a', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#16a34a', lw=2))

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('CNN Performance on Instagram Images\nBefore vs After Fine-Tuning', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 110)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/ppt_chart_lab_vs_reality.png', dpi=200, bbox_inches='tight')
print("✅ Saved: results/ppt_chart_lab_vs_reality.png")

# ============================================================
# CHART 2: CNN vs ViT on Instagram Images (Slide 8)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

test_images = ['Real Photo 1', 'Real Photo 2', 'Real Photo 3', 'Real Photo 4']
cnn_scores = [99.99, 96.44, 88.95, 99.92]
vit_scores = [99.59, 99.81, 99.59, 99.17]

x = np.arange(len(test_images))
width = 0.35

bars1 = ax.bar(x - width/2, cnn_scores, width, label='CNN (Fine-tuned)', color='#16a34a', edgecolor='white')
bars2 = ax.bar(x + width/2, vit_scores, width, label='ViT (Overconfident)', color='#dc2626', edgecolor='white')

ax.set_ylabel('Confidence (%)', fontsize=14)
ax.set_title('CNN vs ViT: Confidence on Real Instagram Images\nViT Labels Everything as AI-Generated', 
             fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(test_images, fontsize=11)
ax.legend(fontsize=12)
ax.set_ylim(0, 110)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/ppt_chart_cnn_vs_vit.png', dpi=200, bbox_inches='tight')
print("✅ Saved: results/ppt_chart_cnn_vs_vit.png")

# ============================================================
# CHART 3: Compression Impact on Detection (Slide 9)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

images = ['AI Sample 1', 'AI Sample 2', 'AI Sample 4', 'AI Sample 5']
original = [97.9, 97.1, 99.8, 71.9]
compressed = [86.1, 89.0, 53.0, 71.7]

x = np.arange(len(images))
width = 0.35

bars1 = ax.bar(x - width/2, original, width, label='Original Image', color='#2563eb', edgecolor='white')
bars2 = ax.bar(x + width/2, compressed, width, label='After JPEG 40% (Instagram)', color='#f59e0b', edgecolor='white')

ax.set_ylabel('Confidence (%)', fontsize=14)
ax.set_title('Impact of Instagram Compression on AI Detection\nConfidence Drops Significantly', 
             fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(images, fontsize=11)
ax.legend(fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Add drop annotations
drops = [11.8, 8.1, 46.8, 0.2]
for i, (orig, comp, drop) in enumerate(zip(original, compressed, drops)):
    mid_y = (orig + comp) / 2
    ax.annotate(f'-{drop:.1f}%', xy=(i, mid_y), fontsize=11, ha='center', 
               fontweight='bold', color='#dc2626')

# Highlight AI Sample 5 where it flips
ax.annotate('⚠ AI image\nclassified as REAL\nafter compression!', 
            xy=(3, 71.7), xytext=(3.5, 55),
            fontsize=12, ha='center', color='#dc2626', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2))

plt.tight_layout()
plt.savefig('results/ppt_chart_compression_impact.png', dpi=200, bbox_inches='tight')
print("✅ Saved: results/ppt_chart_compression_impact.png")