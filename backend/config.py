import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'truthlens-super-secret-key-2024')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///truthlens.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Celery Configuration
    broker_url = 'memory://'
    result_backend = 'cache+memory://'
    task_always_eager = True
    task_serializer = 'json'
    result_serializer = 'json'
    accept_content = ['json']
    
    # AI Model Configuration
    MODEL_NAME = "Custom CNN + ViT Ensemble (Trained on CiFAKE)"
    MAX_IMAGE_SIZE = (512, 512)
    CONFIDENCE_THRESHOLD = 0.65
    
    # Scraping Configuration
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    REQUEST_TIMEOUT = 30
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB