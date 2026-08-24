import { FILE_CONSTANTS } from './constants';

export const validateFile = (file) => {
  if (!file) {
    return { valid: false, error: 'errors.fileNotSelected' };
  }

  // Check by file extension as well as MIME type
  const ext = file.name.split('.').pop().toLowerCase();
  const validExtensions = ['jpg', 'jpeg', 'png', 'mp4', 'avi', 'webm', 'mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a'];
  
  // Check MIME type
  const validMimeTypes = [
    'image/jpeg', 'image/png', 'image/jpg',
    'video/mp4', 'video/avi', 'video/webm', 'video/x-msvideo',
    'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 
    'audio/aac', 'audio/ogg', 'audio/flac', 'audio/m4a',
    'audio/webm'
  ];

  const isValidMime = validMimeTypes.includes(file.type);
  const isValidExt = validExtensions.includes(ext);

  // For WAV files specifically, browsers often report different MIME types
  if (ext === 'wav' && !isValidMime) {
    // WAV files are commonly misidentified, accept them by extension
    return { valid: true };
  }

  if (!isValidMime && !isValidExt) {
    return { valid: false, error: 'errors.unsupportedFormat' };
  }

  if (file.size > FILE_CONSTANTS.MAX_SIZE) {
    return { valid: false, error: 'errors.fileTooLarge' };
  }

  return { valid: true };
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

export const formatTime = (seconds) => {
  if (!seconds || seconds === 0) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};