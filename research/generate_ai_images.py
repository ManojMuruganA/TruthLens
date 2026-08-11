"""
Generate AI images using Stable Diffusion API (HuggingFace)
Creates 200 AI-generated images mimicking Instagram-style photos
"""
import requests
import os
import time
from PIL import Image
from io import BytesIO

OUTPUT_DIR = "social_media_dataset/ai_generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Instagram-style prompts
PROMPTS = [
    "a beautiful sunset at the beach, realistic photo, natural lighting",
    "a cute cat sitting on a windowsill, professional photography",
    "delicious food on a wooden table, restaurant quality photo",
    "city skyline at night, urban photography",
    "a person smiling in a garden, portrait photography, natural light",
    "coffee cup on a cafe table, aesthetic, warm lighting",
    "mountain landscape during golden hour, nature photography",
    "street fashion portrait, urban style, editorial photography",
    "flower bouquet on white background, product photography",
    "fitness model at gym, athletic photography, motivation",
    "sunset at tropical beach, travel photography",
    "vintage car on country road, classic photography",
    "modern architecture building, architectural photography",
    "group of friends having dinner, lifestyle photography",
    "pet dog playing in park, animal photography",
    "autumn leaves falling, seasonal photography",
    "luxury watch on black background, commercial photography",
    "surfer riding wave, action sports photography",
    "wedding couple in forest, wedding photography",
    "night sky with stars, astrophotography",
    "bicycle on cobblestone street, European travel photo",
    "makeup flatlay on marble surface, beauty photography",
    "concert crowd with stage lights, event photography",
    "baby sleeping peacefully, newborn photography",
    "snowy mountain peak, adventure photography",
    "books and reading glasses on desk, cozy aesthetic",
    "dancer mid-performance, dynamic action shot",
    "tropical fruits on market stall, travel photography",
    "vintage camera collection, still life photography",
    "runner at sunrise, fitness lifestyle photo",
    "ocean waves crashing on rocks, seascape photography",
    "bride getting ready, candid wedding moment",
    "street musician playing guitar, urban culture",
    "campfire under starry sky, outdoor adventure",
    "minimalist interior design, architectural photography",
    "skateboarder doing trick, action photography",
    "tea ceremony setup, zen aesthetic",
    "lightning storm over city, dramatic photography",
    "child blowing birthday candles, family moment",
    "artisan bread on bakery shelf, food photography",
]

print("=" * 60)
print("Generating AI Images for Social Media Dataset")
print("=" * 60)
print(f"Output folder: {OUTPUT_DIR}")
print(f"Prompts available: {len(PROMPTS)}")
print(f"Target: 200 images (each prompt used 5 times)")
print("=" * 60)

# Use HuggingFace free API for Stable Diffusion
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

generated = 0
failed = 0

for i in range(200):
    prompt = PROMPTS[i % len(PROMPTS)]
    
    try:
        print(f"\n[{i+1}/200] Generating: {prompt[:50]}...")
        
        response = requests.post(
            API_URL,
            json={"inputs": prompt, "parameters": {"width": 512, "height": 512}},
            timeout=30
        )
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            filename = f"ai_{i+1:04d}.jpg"
            image.save(os.path.join(OUTPUT_DIR, filename), "JPEG", quality=90)
            generated += 1
            print(f"  ✅ Saved: {filename}")
        else:
            failed += 1
            print(f"  ❌ Failed: {response.status_code}")
        
        # Wait to avoid rate limiting
        time.sleep(2)
        
    except Exception as e:
        failed += 1
        print(f"  ❌ Error: {e}")
        time.sleep(5)

print("\n" + "=" * 60)
print(f"Generation complete!")
print(f"  Generated: {generated}")
print(f"  Failed: {failed}")
print("=" * 60)