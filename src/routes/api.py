import os
import uuid
import threading
import logging
from flask import Blueprint, jsonify, request, send_from_directory, current_app

from config.settings import Config

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    # Get instances from app context
    file_processor = current_app.config['file_processor']
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file_processor.allowed_file(file.filename):
        filename = file.filename
        task_id = str(uuid.uuid4())
        filepath = os.path.join(Config.UPLOAD_FOLDER, f"{task_id}_{filename}")
        file.save(filepath)
        
        # Process file in background thread
        thread = threading.Thread(
            target=file_processor.process_file,
            args=(task_id, filepath, filename)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"task_id": task_id}), 202
    else:
        return jsonify({"error": "Invalid file type"}), 400

@api_bp.route('/progress/<task_id>')
def get_progress(task_id: str):
    # Get instances from app context
    redis_manager = current_app.config['redis_manager']
    
    status = redis_manager.get_task_status(task_id)
    if status is None:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(status)

@api_bp.route('/submit_application', methods=['POST'])
def submit_application():
    # Get instances from app context
    redis_manager = current_app.config['redis_manager']
    
    try:
        application_data = request.get_json()
        if not application_data:
            return jsonify({"error": "No application data provided"}), 400
        
        application_id = redis_manager.save_application(application_data)
        return jsonify({"application_id": application_id}), 201
    except Exception as e:
        logger.error(f"Error submitting application: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@api_bp.route('/results/<filename>')
def serve_result(filename):
    """Serve processed result files"""
    return send_from_directory(Config.RESULTS_FOLDER, filename)

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    # Get instances from app context
    redis_manager = current_app.config['redis_manager']
    
    try:
        redis_manager.connection.ping()
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500