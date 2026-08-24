import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { getEmotionColor } from '../utils/constants';

export const EmotionTimeline = ({ data, height = 200 }) => {
  const { t } = useLanguage();
  const [selected, setSelected] = React.useState(null);

  if (!data?.length) {
    return (
      <div className="chart-container">
        <h4>Emotion Timeline</h4>
        <div className="no-data">No data available</div>
      </div>
    );
  }

  const lastTs = data[data.length - 1]?.timestamp || 1;

  return (
    <div className="chart-container">
      <h4>Emotion Timeline</h4>
      <div className="timeline-chart" style={{ height: `${height}px` }}>
        {data.map((point, index) => {
          const emotion = point.result?.main_prediction?.label;
          const prob = point.result?.main_prediction?.probability || 0;
          return (
            <div
              key={index}
              className={`timeline-point ${selected === index ? 'selected' : ''}`}
              style={{
                left: `${(point.timestamp / lastTs) * 100}%`,
                backgroundColor: getEmotionColor(emotion),
                opacity: 0.5 + prob * 0.5,
              }}
              onMouseEnter={() => setSelected(index)}
              onMouseLeave={() => setSelected(null)}
            />
          );
        })}
        {selected !== null && (
          <div 
            className="timeline-tooltip"
            style={{ left: `${(data[selected].timestamp / lastTs) * 100}%` }}
          >
            <strong>{data[selected].timestamp.toFixed(1)}s</strong>
            <br />
            {t(`emotions.${data[selected].result?.main_prediction?.label}`)}
          </div>
        )}
      </div>
    </div>
  );
};

export const EmotionDistribution = ({ emotions, height = 200 }) => {
  const { t } = useLanguage();
  
  if (!emotions || !Object.keys(emotions).length) {
    return (
      <div className="chart-container">
        <h4>Emotion Distribution</h4>
        <div className="no-data">No data available</div>
      </div>
    );
  }

  const entries = Object.entries(emotions).sort((a, b) => b[1] - a[1]);

  return (
    <div className="chart-container">
      <h4>Emotion Distribution</h4>
      <div className="distribution-chart" style={{ height: `${height}px` }}>
        {entries.map(([key, value]) => (
          <div key={key} className="distribution-item">
            <div className="distribution-label">
              <span>{t(`emotions.${key}`)}</span>
              <span>{(value * 100).toFixed(1)}%</span>
            </div>
            <div className="distribution-bar">
              <div 
                className="distribution-fill"
                style={{
                  width: `${value * 100}%`,
                  backgroundColor: getEmotionColor(key),
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};