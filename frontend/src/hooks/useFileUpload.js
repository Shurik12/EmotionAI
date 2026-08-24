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

  const handleFileSelect = useCallback((selectedFile) => {
    if (!selectedFile) {
      setError('error_file_not_selected');
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
    
    setError(null);
  }, []);

  const clearFile = useCallback(() => {
    setFile(null);
    setFileName('');
    setFileSize('');
    setPreview(null);
    setError(null);
  }, []);

  const uploadFile = useCallback(async (mode = 'standard') => {
    if (!file) {
      setError('error_file_not_selected');
      throw new Error('No file selected');
    }

    setIsProcessing(true);
    setError(null);

    try {
      const data = await apiClient.uploadFile(file, mode);
      return data;
    } catch (err) {
      setError(err.message || 'error_upload_failed');
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
    handleFileSelect,
    clearFile,
    uploadFile,
    setError,
  };
};