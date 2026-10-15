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
        
        // Map server progress messages to translation keys. Unknown
        // messages fall back to a generic key so no raw English ever
        // reaches the UI in RU mode.
        const mapMessage = (msg) => {
          if (!msg) return 'detector.processing';
          if (msg.includes('frame') || msg.includes('segment')) {
            // Per-frame / per-segment messages are already localized
            // server-side via the translation system; pass through.
            return msg;
          }
          const lower = msg.toLowerCase();
          if (lower.includes('burnout') && lower.includes('complete')) return 'detector.burnoutComplete';
          if (lower.includes('external influence') && lower.includes('complete')) return 'detector.externalInfluenceComplete';
          if (lower.includes('audio')) return 'detector.processingAudio';
          if (lower.includes('video')) return 'detector.processingVideo';
          if (lower.includes('image')) return 'detector.processingImage';
          if (lower.includes('uploaded') || lower.includes('starting')) return 'detector.startingProcessing';
          return 'detector.fileProcessing';
        };
        const message = mapMessage(data.message);
        
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