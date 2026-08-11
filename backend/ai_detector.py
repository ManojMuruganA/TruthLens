import torch
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import logging
from config import Config

logger = logging.getLogger(__name__)

class AIDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize multiple models for ensemble detection
        self.models = self._load_models()
        logger.info(f"Loaded {len(self.models)} AI detection models")
    
    def _load_models(self):
        """Load multiple detection models for ensemble prediction"""
        models = {}
    
        try:
            # Primary model: AI-generated image detector (this one works)
            models['ai_generated'] = pipeline(
                "image-classification",
                model="umm-maybe/AI-image-detector",
                device=self.device if torch.cuda.is_available() else -1
            )
            logger.info("Loaded AI image detector model")
        except Exception as e:
            logger.warning(f"Could not load AI image detector: {e}")
    
        try:
            # Secondary model: Another working model
            models['deepfake'] = pipeline(
                "image-classification",
                model="dima806/deepfake_vs_real_image_detection",
                device=self.device if torch.cuda.is_available() else -1
            )
            logger.info("Loaded Deepfake detection model")
        except Exception as e:
            logger.warning(f"Could not load Deepfake model: {e}")
    
        # If no models could be loaded, use basic detection
        if not models:
            logger.warning("No HuggingFace models available, using basic detection")
            models['basic'] = self._create_basic_detector()
    
        return models
    
    def _create_basic_detector(self):
        """Create a basic detection heuristic when models aren't available"""
        class BasicDetector:
            def __call__(self, image):
                # Basic noise analysis as fallback
                img_array = np.array(image)
                noise_level = np.std(cv2.Laplacian(img_array, cv2.CV_64F))
                
                # Higher noise might indicate real image, lower noise might indicate AI
                confidence = 1.0 / (1.0 + np.exp(-noise_level / 100))
                
                return [
                    {'label': 'Real', 'score': confidence},
                    {'label': 'Fake', 'score': 1 - confidence}
                ]
        
        return BasicDetector()
    
    def detect_image(self, image_content):
        """
        Detect if an image is AI-generated or real
        Returns: (verdict, confidence_score, processing_time)
        """
        import time
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(BytesIO(image_content)).convert('RGB')
            image = image.resize(Config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Run ensemble detection
            predictions = []
            weights = []
            
            if 'deepfake' in self.models:
                try:
                    pred = self.models['deepfake'](image)
                    # Normalize the prediction
                    fake_score = next((p['score'] for p in pred if p['label'].lower() in ['fake', 'ai', 'generated']), 0)
                    real_score = next((p['score'] for p in pred if p['label'].lower() in ['real', 'authentic', 'human']), 0)
                    
                    if fake_score > real_score:
                        predictions.append(('AI-Generated', fake_score))
                    else:
                        predictions.append(('Real', real_score))
                    weights.append(0.7)  # Higher weight for specialized model
                except Exception as e:
                    logger.warning(f"DeepFake model prediction failed: {e}")
            
            if 'ai_generated' in self.models:
                try:
                    pred = self.models['ai_generated'](image)
                    ai_score = next((p['score'] for p in pred if p['label'].lower() in ['ai', 'artificial', 'generated']), 0)
                    human_score = next((p['score'] for p in pred if p['label'].lower() in ['human', 'real', 'natural']), 0)
                    
                    if ai_score > human_score:
                        predictions.append(('AI-Generated', ai_score))
                    else:
                        predictions.append(('Real', human_score))
                    weights.append(0.3)
                except Exception as e:
                    logger.warning(f"AI detector model prediction failed: {e}")
            
            if not predictions and 'basic' in self.models:
                pred = self.models['basic'](image)
                fake_score = next((p['score'] for p in pred if p['label'].lower() == 'fake'), 0.5)
                real_score = next((p['score'] for p in pred if p['label'].lower() == 'real'), 0.5)
                
                if fake_score > real_score:
                    predictions.append(('AI-Generated', fake_score))
                else:
                    predictions.append(('Real', real_score))
                weights.append(1.0)
            
            if not predictions:
                raise ValueError("No models available for prediction")
            
            # Weighted ensemble
            total_weight = sum(weights)
            weighted_scores = {}
            
            for (verdict, score), weight in zip(predictions, weights):
                if verdict not in weighted_scores:
                    weighted_scores[verdict] = 0
                weighted_scores[verdict] += score * weight
                
            final_verdict = max(weighted_scores, key=weighted_scores.get)
            confidence = weighted_scores[final_verdict] / total_weight
            
            # Apply confidence threshold
            if confidence < Config.CONFIDENCE_THRESHOLD:
                final_verdict = 'Uncertain'
            
            processing_time = time.time() - start_time
            
            return final_verdict, confidence, processing_time
            
        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            raise
    
    def detect_video(self, video_content, num_frames=5):
        """
        Detect if a video contains AI-generated content
        Returns: (verdict, confidence_score, processing_time)
        """
        import time
        start_time = time.time()
        
        try:
            # Save video temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(video_content)
                video_path = tmp.name
            
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                raise ValueError("Video contains no frames")
            
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
            frame_predictions = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Save to bytes
                    img_buffer = BytesIO()
                    pil_image.save(img_buffer, format='JPEG', quality=95)
                    img_bytes = img_buffer.getvalue()
                    
                    try:
                        verdict, confidence, _ = self.detect_image(img_bytes)
                        frame_predictions.append((verdict, confidence))
                    except:
                        continue
            
            cap.release()
            
            # Clean up temp file
            import os
            os.unlink(video_path)
            
            if not frame_predictions:
                raise ValueError("Could not analyze any video frames")
            
            # Majority voting
            ai_count = sum(1 for v, _ in frame_predictions if v == 'AI-Generated')
            avg_confidence = np.mean([c for _, c in frame_predictions])
            
            if ai_count > len(frame_predictions) / 2:
                final_verdict = 'AI-Generated'
                confidence = avg_confidence
            else:
                final_verdict = 'Real'
                confidence = 1 - avg_confidence
            
            processing_time = time.time() - start_time
            
            return final_verdict, confidence, processing_time
            
        except Exception as e:
            logger.error(f"Video detection failed: {e}")
            raise