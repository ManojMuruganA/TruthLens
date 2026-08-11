from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()

def generate_uuid():
    return str(uuid.uuid4())

class DetectionResult(db.Model):
    __tablename__ = 'detection_results'
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    original_url = db.Column(db.String(2048), nullable=False)
    media_url = db.Column(db.String(2048), nullable=True)
    verdict = db.Column(db.String(20), nullable=True, default=None)
    confidence_score = db.Column(db.Float, nullable=True, default=None)
    media_type = db.Column(db.String(10), nullable=True, default=None)
    platform = db.Column(db.String(50), nullable=True, default=None)
    model_used = db.Column(db.String(100), nullable=True, default=None)
    processing_time = db.Column(db.Float, nullable=True, default=None)
    error_message = db.Column(db.Text, nullable=True, default=None)
    status = db.Column(db.String(20), default='PENDING')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45), nullable=True, default=None)
    user_agent = db.Column(db.String(256), nullable=True, default=None)
    
    def to_dict(self):
        return {
            'id': self.id,
            'original_url': self.original_url,
            'media_url': self.media_url,
            'verdict': self.verdict,
            'confidence_score': self.confidence_score,
            'media_type': self.media_type,
            'platform': self.platform,
            'model_used': self.model_used,
            'processing_time': self.processing_time,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'error_message': self.error_message
        }