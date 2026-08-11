from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, DetectionResult
from scraper import SocialMediaScraper
from ai_detector_v2 import AIDetectorV2
from config import Config
from datetime import datetime
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

CORS(app, resources={r"/api/*": {"origins": "*"}})
db.init_app(app)


with app.app_context():
    db.create_all()

# Initialize components
scraper = SocialMediaScraper()
ai_detector = AIDetectorV2()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect_media():
    """Submit a URL for AI detection - processes directly without Celery"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url'].strip()
        
        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Create task record
        task_id = str(uuid.uuid4())
        detection = DetectionResult(
            id=task_id,
            original_url=url,
            status='PROCESSING',
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string if request.user_agent else None
        )
        
        db.session.add(detection)
        db.session.commit()
        
        logger.info(f"Processing detection task: {task_id}")
        
        # Process directly (no Celery)
        try:
            # Extract media
            logger.info(f"Extracting media from: {url}")
            media_info = scraper.extract_media(url)
            
            if not media_info['media_urls']:
                raise ValueError("No media found at the provided URL")
            
            # Download media
            media_url = media_info['media_urls'][0]
            media_type = media_info['media_type']
            media_content = scraper.download_media(media_url)
            
            # Analyze with AI
            logger.info(f"Analyzing {media_type} content...")
            
            if media_type == 'video':
                verdict, confidence, processing_time = ai_detector.detect_video(media_content)
            else:
                verdict, confidence, processing_time = ai_detector.detect_image(media_content)
            
            # Update database
            detection.verdict = verdict
            detection.confidence_score = confidence
            detection.media_type = media_type
            detection.platform = media_info.get('platform', 'unknown')
            detection.media_url = media_url
            detection.model_used = Config.MODEL_NAME
            detection.processing_time = processing_time
            detection.status = 'SUCCESS'
            db.session.commit()
            
            logger.info(f"Analysis complete: {verdict} ({confidence:.2%})")
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            detection.status = 'FAILED'
            detection.error_message = str(e)
            db.session.commit()
        
        return jsonify({
            'task_id': task_id,
            'status': detection.status,
            'message': 'Analysis completed'
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating detection task: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """Get the result of a detection task"""
    try:
        result = db.session.get(DetectionResult, task_id)
        
        if not result:
            return jsonify({'error': 'Task not found'}), 404
        
        if result.status == 'FAILED':
            return jsonify({
                'status': 'FAILED',
                'task_id': task_id,
                'error': result.error_message
            }), 200
        
        if result.status == 'SUCCESS':
            return jsonify({
                'status': 'SUCCESS',
                'result': result.to_dict()
            }), 200
        
        return jsonify({
            'status': result.status,
            'task_id': task_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching result: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get detection history"""
    try:
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        limit = min(limit, 100)
        
        results = DetectionResult.query\
            .order_by(DetectionResult.created_at.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()
        
        return jsonify({
            'results': [r.to_dict() for r in results],
            'count': len(results),
            'limit': limit,
            'offset': offset
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)