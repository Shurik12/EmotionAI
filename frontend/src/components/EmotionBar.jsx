import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { getEmotionColor } from '../utils/constants';

export const EmotionBar = ({ emotion, probability }) => {
  const { t } = useLanguage();
  // Ensure probability is a number
  const prob = typeof probability === 'string' ? parseFloat(probability) : probability;
  const pct = (prob * 100).toFixed(1);

  // Get translated emotion name, fallback to the key if translation not found
  const emotionName = t(`emotions.${emotion}`) || emotion;

  return (
    <div className="emotion-item">
      <div className="emotion-label">
        <span>{emotionName}</span>
        <span>{pct}%</span>
      </div>
      <div className="emotion-bar">
        <div 
          className="emotion-fill" 
          style={{ 
            width: `${pct}%`, 
            backgroundColor: getEmotionColor(emotion) 
          }} 
        />
      </div>
    </div>
  );
};