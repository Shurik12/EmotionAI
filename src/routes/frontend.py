import os
import logging
from flask import Blueprint, jsonify, send_from_directory, send_file

from config.settings import Config

logger = logging.getLogger(__name__)

frontend_bp = Blueprint('frontend', __name__)

@frontend_bp.route('/static/<path:filename>')
def serve_static(filename):
    static_path = os.path.join(Config.FRONTEND_BUILD_PATH, 'static')
    if os.path.exists(os.path.join(static_path, filename)):
        return send_from_directory(static_path, filename)
    return jsonify({"error": "File not found"}), 404

@frontend_bp.route('/<path:filename>')
def serve_react_files(filename):
    # Don't interfere with API routes
    if filename.startswith('api/'):
        return jsonify({"error": "Not found"}), 404
    
    file_path = os.path.join(Config.FRONTEND_BUILD_PATH, filename)
    
    # If it's a file that exists, serve it
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(Config.FRONTEND_BUILD_PATH, filename)
    
    # For React Router - serve index.html for all other routes
    return send_from_directory(Config.FRONTEND_BUILD_PATH, 'index.html')

@frontend_bp.route('/')
def serve_root():
    return send_from_directory(Config.FRONTEND_BUILD_PATH, 'index.html')