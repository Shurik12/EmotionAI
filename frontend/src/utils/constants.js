export const EMOTION_COLORS = {
  anger: '#ea4335',
  contempt: '#9E9E9E',
  disgust: '#9C27B0',
  fear: '#4285f4',
  neutral: '#fbbc05',
  happiness: '#34a853',
  surprise: '#673ab7',
  sadness: '#2196F3',
};

export const getEmotionColor = (emotion) => {
  return EMOTION_COLORS[emotion] || '#3f4857';
};

export const BURNOUT_COLORS = {
  low: '#28a745',
  moderate: '#ffc107',
  high: '#fd7e14',
  severe: '#dc3545',
};

export const BURNOUT_LABELS = {
  low: 'Low Risk',
  moderate: 'Moderate Risk',
  high: 'High Risk',
  severe: 'Severe Risk',
};

export const FILE_CONSTANTS = {
  MAX_SIZE: 50 * 1024 * 1024, // 50MB
  VALID_TYPES: [
    // Images
    'image/jpeg',
    'image/png',
    'image/jpg',
    // Videos
    'video/mp4',
    'video/avi',
    'video/webm',
    'video/x-msvideo',
    // Audio
    'audio/mpeg',
    'audio/mp3',
    'audio/wav',
    'audio/x-wav',
    'audio/aac',
    'audio/ogg',
    'audio/flac',
    'audio/m4a',
    'audio/webm',
  ],
  VALID_EXTENSIONS: [
    'jpg', 'jpeg', 'png',
    'mp4', 'avi', 'webm',
    'mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a',
  ],
};

export const PROCESSING_MODES = [
  { value: 'standard', label: 'Standard Analysis' },
  { value: 'burnout', label: 'Burnout Analysis' },
  { value: 'realtime', label: 'Real-time Analysis' },
];