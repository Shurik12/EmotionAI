import { useState, useCallback } from 'react';
import { apiClient } from '../api/client';
import { validateFile, formatFileSize } from '../utils/validators';

export const useFileUpload = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [fileSize, setFileSize] = useState('');
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  // 0–100 while a large (resumable) upload is in flight.
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileSelect = useCallback((selectedFile) => {
    if (!selectedFile) {
      setError('errors.fileNotSelected');
      return;
    }

    const validation = validateFile(selectedFile);
    if (!validation.valid) {
      setError(validation.error);
      return;
    }

    setFile(selectedFile);
    setFileName(selectedFile.name);
    setFileSize(formatFileSize(selectedFile.size));

    if (selectedFile.type.startsWith('audio/')) {
      setPreview({ type: 'audio', url: URL.createObjectURL(selectedFile) });
    } else if (selectedFile.type.startsWith('image/')) {
      setPreview({ type: 'image', url: URL.createObjectURL(selectedFile) });
    } else {
      setPreview(null);
    }
    
    setUploadProgress(0);
    setError(null);
  }, []);

  const clearFile = useCallback(() => {
    setFile(null);
    setFileName('');
    setFileSize('');
    setPreview(null);
    setUploadProgress(0);
    setError(null);
  }, []);

  const uploadFile = useCallback(async (mode = 'standard') => {
    if (!file) {
      setError('errors.fileNotSelected');
      throw new Error('No file selected');
    }

    setIsProcessing(true);
    setUploadProgress(0);
    setError(null);

    try {
      const data = await apiClient.uploadFile(file, mode, {
        onProgress: setUploadProgress,
      });
      return data;
    } catch (err) {
      setError(err.message || 'errors.uploadFailed');
      throw err;
    } finally {
      setIsProcessing(false);
    }
  }, [file]);

  return {
    file,
    fileName,
    fileSize,
    preview,
    error,
    isProcessing,
    uploadProgress,
    handleFileSelect,
    clearFile,
    uploadFile,
    setError,
  };
};
