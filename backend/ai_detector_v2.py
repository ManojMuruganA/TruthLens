"""
TruthLens AI Detector v2
Uses trained CNN and ViT models for AI image detection
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import logging
from io import BytesIO
import os

logger = logging.getLogger(__name__)

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


class AIDetectorV2:
    """AI Content Detector using trained CNN and ViT models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Get the directory where this file is located
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load models
        self.cnn_model = None
        self.vit_model = None
        self._load_models()
    
    def _load_models(self):
        """Load trained CNN and ViT models"""
        try:
            # Load CNN
            cnn_path = os.path.join(self.base_dir, "models", "cnn_social_finetuned.pth")
            if os.path.exists(cnn_path):
                self.cnn_model = CNNDetector(num_classes=2).to(self.device)
                checkpoint = torch.load(cnn_path, map_location=self.device, weights_only=False)
                self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
                self.cnn_model.eval()
                logger.info(f"CNN loaded (accuracy: {checkpoint['accuracy']:.2f}%)")
            else:
                logger.warning(f"CNN model not found at {cnn_path}")
        except Exception as e:
            logger.warning(f"Could not load CNN: {e}")
        
        try:
            # Load ViT
            vit_path = os.path.join(self.base_dir, "models", "vit_model.pth")
            if os.path.exists(vit_path):
                self.vit_model = timm.create_model(
                    'vit_base_patch16_224', pretrained=False, num_classes=2
                ).to(self.device)
                checkpoint = torch.load(vit_path, map_location=self.device, weights_only=False)
                self.vit_model.load_state_dict(checkpoint['model_state_dict'])
                self.vit_model.eval()
                logger.info(f"ViT loaded (accuracy: {checkpoint['accuracy']:.2f}%)")
            else:
                logger.warning(f"ViT model not found at {vit_path}")
        except Exception as e:
            logger.warning(f"Could not load ViT: {e}")
        
        if not self.cnn_model and not self.vit_model:
            logger.error("No models loaded! Detection will fail.")
    
    def detect_image(self, image_content):
        """
        Detect if an image is AI-generated or real
        Uses ensemble of CNN and ViT
        Returns: (verdict, confidence_score, processing_time, details)
        """
        import time
        start_time = time.time()
        
        try:
            image = Image.open(BytesIO(image_content)).convert('RGB')
            
            results = {}
            predictions = []
            confidences = []
            
            # CNN Prediction
            if self.cnn_model:
                cnn_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                
                cnn_tensor = cnn_transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.cnn_model(cnn_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                cnn_verdict = 'AI-Generated' if pred.item() == 0 else 'Real'
                cnn_conf = conf.item()
                results['cnn'] = {'verdict': cnn_verdict, 'confidence': cnn_conf}
                predictions.append(cnn_verdict)
                confidences.append(cnn_conf)
            
            # ViT Prediction
            if self.vit_model:
                vit_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                vit_tensor = vit_transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.vit_model(vit_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                vit_verdict = 'AI-Generated' if pred.item() == 0 else 'Real'
                vit_conf = conf.item()
                results['vit'] = {'verdict': vit_verdict, 'confidence': vit_conf}
                predictions.append(vit_verdict)
                confidences.append(vit_conf)
            
            if not predictions:
                raise ValueError("No models available")
            
            # Ensemble: CNN has more weight (trained on 100K images) unless ViT is very confident
            cnn_result = results.get('cnn', {})
            vit_result = results.get('vit', {})

            cnn_verdict = cnn_result.get('verdict', 'Unknown')
            vit_verdict = vit_result.get('verdict', 'Unknown')
            cnn_conf = cnn_result.get('confidence', 0)
            vit_conf = vit_result.get('confidence', 0)

            # If both agree, use that verdict
            if cnn_verdict == vit_verdict:
                final_verdict = cnn_verdict
            # If they disagree:
            else:
                # CNN is more reliable (trained on 100K vs 5K for ViT)
                # Only override CNN if ViT is extremely confident (>98%)
                if vit_conf > 0.98 and cnn_conf < 0.70:
                    final_verdict = vit_verdict
                    logger.info(f"ViT overrides CNN (ViT confidence: {vit_conf:.2%}, CNN confidence: {cnn_conf:.2%})")
                else:
                    final_verdict = cnn_verdict
                    logger.info(f"CNN trusted over ViT (CNN: {cnn_conf:.2%}, ViT: {vit_conf:.2%})")
            
            avg_confidence = sum(confidences) / len(confidences)
            processing_time = time.time() - start_time
            
            logger.info(f"Detection: {final_verdict} ({avg_confidence:.2%}) | Details: {results}")
            
            return final_verdict, avg_confidence, processing_time
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def detect_video(self, video_content, num_frames=5):
        """Video detection - extracts frames and analyzes"""
        import cv2
        import tempfile
        import os as os_module
        
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(video_content)
                video_path = tmp.name
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                raise ValueError("Video contains no frames")
            
            frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
            frame_predictions = []
            
            import numpy as np
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    img_buffer = BytesIO()
                    pil_image.save(img_buffer, format='JPEG', quality=95)
                    img_bytes = img_buffer.getvalue()
                    
                    try:
                        verdict, confidence, _ = self.detect_image(img_bytes)
                        frame_predictions.append((verdict, confidence))
                    except:
                        continue
            
            cap.release()
            os_module.unlink(video_path)
            
            if not frame_predictions:
                raise ValueError("Could not analyze any frames")
            
            ai_frames = sum(1 for v, _ in frame_predictions if v == 'AI-Generated')
            avg_confidence = np.mean([c for _, c in frame_predictions])
            
            if ai_frames > len(frame_predictions) / 2:
                final_verdict = 'AI-Generated'
            else:
                final_verdict = 'Real'
            
            return final_verdict, avg_confidence, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Video detection failed: {e}")
            raise