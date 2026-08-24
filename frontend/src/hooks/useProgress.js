import { useState, useRef, useCallback } from 'react';
import { apiClient } from '../api/client';

export const useProgress = () => {
  const [progress, setProgress] = useState(null);
  const [isComplete, setIsComplete] = useState(false);
  const intervalRef = useRef(null);

  const startTracking = useCallback((taskId, onComplete) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    setProgress({ text: 'detector.startingProcessing', value: 0 });
    setIsComplete(false);

    intervalRef.current = setInterval(async () => {
      try {
        const data = await apiClient.getProgress(taskId);
        
        let message = data.message || 'detector.processing';
        
        // Handle specific messages from the server
        if (data.message) {
          if (data.message.includes('frame') || data.message.includes('segment')) {
            message = data.message;
          } else if (data.message === 'Burnout analysis complete') {
            message = 'detector.burnoutComplete';
          } else if (data.message === 'Processing audio...' || data.message === 'Processing audio for burnout analysis') {
            message = 'detector.processingAudio';
          } else if (data.message === 'Video processing...') {
            message = 'detector.processingVideo';
          } else if (data.message === 'Image processing...') {
            message = 'detector.processingImage';
          } else if (data.message === 'File uploaded, starting analysis...') {
            message = 'detector.startingProcessing';
          }
        }
        
        setProgress({
          text: message,
          value: data.progress || 0,
        });

        if (data.complete) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
          setIsComplete(true);
          
          setProgress({
            text: 'detector.processingComplete',
            value: 100,
          });
          
          if (onComplete) {
            onComplete(data);
          }
        }
      } catch (error) {
        console.error('Progress tracking error:', error);
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        setProgress(null);
      }
    }, 1000);
  }, []);

  const stopTracking = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setProgress(null);
    setIsComplete(false);
  }, []);

  return { progress, isComplete, startTracking, stopTracking };
};