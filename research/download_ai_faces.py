"""
Download AI-generated faces from ThisPersonDoesNotExist
These are StyleGAN-generated faces that look like real Instagram portraits
"""
import requests
import os
import time

OUTPUT_DIR = "social_media_dataset/ai_generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Downloading AI-Generated Faces")
print("Source: ThisPersonDoesNotExist.com")
print("=" * 60)

success = 0
failed = 0

for i in range(200):
    try:
        print(f"[{i+1}/200] Downloading...", end=" ")
        response = requests.get("https://thispersondoesnotexist.com/", timeout=15)
        
        if response.status_code == 200:
            filename = f"ai_{i+1:04d}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✅ Saved: {filename}")
            success += 1
        else:
            print(f"❌ Status: {response.status_code}")
            failed += 1
        
        # Wait 2 seconds between requests to be respectful
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        failed += 1
        time.sleep(5)

print("\n" + "=" * 60)
print(f"Complete! Success: {success}, Failed: {failed}")
print(f"Images saved to: {OUTPUT_DIR}")
print("=" * 60)