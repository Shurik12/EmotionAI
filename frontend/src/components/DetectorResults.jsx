import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { EmotionBar } from './EmotionBar';
import { BurnoutAnalysis } from './BurnoutAnalysis';
import { getEmotionEntries, getBurnoutAnalysis } from '../utils/helpers';

export const DetectorResults = ({ results }) => {
  const { t } = useLanguage();
  
  if (!results) return null;

  console.log('Results received:', results); // Debug log

  // Determine the actual result data
  const getResultData = () => {
    // If results has a 'result' property with the data
    if (results.result && typeof results.result === 'object') {
      return results.result;
    }
    // If results itself is the data
    return results;
  };

  const resultData = getResultData();
  const burnout = getBurnoutAnalysis(resultData);

  const renderResults = () => {
    switch (results.type) {
      case 'image':
        return <ImageResults results={results} resultData={resultData} />;
      case 'video':
        return <VideoResults results={results} resultData={resultData} />;
      case 'video_realtime':
        return <RealtimeResults results={results} resultData={resultData} />;
      case 'audio':
        return <AudioResults results={results} resultData={resultData} />;
      case 'audio_burnout':
        return <AudioBurnoutResults results={results} resultData={resultData} burnout={burnout} />;
      default:
        // If we have result data but type is not specified, try to render it
        if (results.result) {
          return <AudioResults results={results} resultData={resultData} />;
        }
        return <DefaultResults results={results} />;
    }
  };

  return (
    <div className="results-container">
      {renderResults()}
    </div>
  );
};

const ImageResults = ({ results, resultData }) => {
  const { t } = useLanguage();
  const { emotions } = getEmotionEntries(resultData?.additional_probs);
  const burnout = getBurnoutAnalysis(resultData);

  return (
    <div className="result-card">
      <h3>📷 {t('detector.imageAnalysis') || 'Image Analysis'}</h3>
      {results.image_url && (
        <div className="result-image">
          <img src={results.image_url} alt="Processed" className="processed-image" />
        </div>
      )}
      
      {resultData?.main_prediction && (
        <div className="main-emotion">
          <strong>{t('detector.mainEmotion') || 'Main Emotion'}:</strong> {t(`emotions.${resultData.main_prediction.label}`) || resultData.main_prediction.label} ({(resultData.main_prediction.probability * 100).toFixed(1)}%)
        </div>
      )}
      
      {Object.keys(emotions).length > 0 && (
        <div className="emotion-results">
          <h4>{t('detector.detectedEmotions') || 'Detected Emotions'}</h4>
          {Object.entries(emotions).map(([key, value]) => (
            <EmotionBar key={key} emotion={key} probability={parseFloat(value)} />
          ))}
        </div>
      )}
      
      {burnout && <BurnoutAnalysis data={burnout} />}
    </div>
  );
};

const VideoResults = ({ results, resultData }) => {
  const { t } = useLanguage();

  return (
    <div className="result-card">
      <h3>🎬 {t('detector.videoAnalysisComplete')}</h3>
      <p>{t('detector.framesProcessed', { count: results.frames_processed })}</p>
      
      <div className="video-frames">
        {results.results?.map((frame, index) => (
          <VideoFrame key={index} frame={frame} index={index} />
        ))}
      </div>
    </div>
  );
};

const VideoFrame = ({ frame, index }) => {
  const { t } = useLanguage();
  const { emotions } = getEmotionEntries(frame.result?.additional_probs);

  return (
    <div className="video-frame">
      <h4>{t('detector.frame')} {index + 1}</h4>
      {frame.image_url && (
        <img src={frame.image_url} alt={`Frame ${index + 1}`} className="frame-image" />
      )}
      <div className="frame-emotions">
        {Object.entries(emotions).slice(0, 3).map(([key, value]) => (
          <EmotionBar key={key} emotion={key} probability={parseFloat(value)} />
        ))}
      </div>
    </div>
  );
};

const RealtimeResults = ({ results, resultData }) => {
  const { t } = useLanguage();
  const { emotions } = getEmotionEntries(results.average_emotions);

  return (
    <div className="result-card">
      <h3>📊 {t('detector.realtimeAnalysis') || 'Real-time Video Analysis'}</h3>
      <p>{t('detector.framesProcessed', { count: results.frames_processed })} over {results.duration?.toFixed(1)}s</p>
      
      {Object.keys(emotions).length > 0 && (
        <div className="emotion-results">
          <h4>{t('detector.averageEmotions') || 'Average Emotions'}</h4>
          {Object.entries(emotions).map(([key, value]) => (
            <EmotionBar key={key} emotion={key} probability={parseFloat(value)} />
          ))}
        </div>
      )}
    </div>
  );
};

const AudioResults = ({ results, resultData }) => {
  const { t } = useLanguage();
  const { emotions } = getEmotionEntries(resultData?.additional_probs);
  const burnout = getBurnoutAnalysis(resultData);

  return (
    <div className="result-card">
      <h3>🎵 {t('detector.audioAnalysis') || 'Audio Analysis'}</h3>
      <div className="audio-meta">
        <span>{t('detector.duration') || 'Duration'}: {(results.duration || resultData?.duration_seconds || 0).toFixed(1)}s</span>
        <span>{t('detector.sampleRate') || 'Sample Rate'}: {results.sample_rate || resultData?.sample_rate || 'N/A'} Hz</span>
      </div>
      
      {resultData?.main_prediction && (
        <div className="main-emotion">
          <strong>{t('detector.mainEmotion') || 'Main Emotion'}:</strong> {t(`emotions.${resultData.main_prediction.label}`) || resultData.main_prediction.label} ({(resultData.main_prediction.probability * 100).toFixed(1)}%)
        </div>
      )}
      
      {Object.keys(emotions).length > 0 && (
        <div className="emotion-results">
          <h4>{t('detector.detectedEmotions') || 'Detected Emotions'}</h4>
          {Object.entries(emotions).map(([key, value]) => (
            <EmotionBar key={key} emotion={key} probability={parseFloat(value)} />
          ))}
        </div>
      )}
      
      {burnout && <BurnoutAnalysis data={burnout} />}
    </div>
  );
};

const AudioBurnoutResults = ({ results, resultData, burnout }) => {
  const { t } = useLanguage();
  const { emotions } = getEmotionEntries(resultData?.additional_probs);

  console.log('AudioBurnoutResults:', { results, resultData, burnout, emotions });

  return (
    <>
      <div className="result-card">
        <h3>🎵 {t('detector.audioBurnoutAnalysis') || 'Audio Burnout Analysis'}</h3>
        <div className="audio-meta">
          <span>{t('detector.duration') || 'Duration'}: {(results.duration || resultData?.duration_seconds || 0).toFixed(1)}s</span>
          <span>{t('detector.sampleRate') || 'Sample Rate'}: {results.sample_rate || resultData?.sample_rate || 'N/A'} Hz</span>
        </div>
        
        {resultData?.main_prediction && (
          <div className="main-emotion">
            <strong>{t('detector.mainEmotion') || 'Main Emotion'}:</strong> {t(`emotions.${resultData.main_prediction.label}`) || resultData.main_prediction.label} ({(resultData.main_prediction.probability * 100).toFixed(1)}%)
          </div>
        )}
        
        {Object.keys(emotions).length > 0 && (
          <div className="emotion-results">
            <h4>{t('detector.detectedEmotions') || 'Detected Emotions'}</h4>
            {Object.entries(emotions).map(([key, value]) => (
              <EmotionBar key={key} emotion={key} probability={parseFloat(value)} />
            ))}
          </div>
        )}
        
        {burnout ? (
          <BurnoutAnalysis data={burnout} />
        ) : (
          <div className="no-results">
            <p>{t('detector.noBurnoutData') || 'No burnout analysis data available.'}</p>
            <details>
              <summary>{t('common.showRawData') || 'Show raw data'}</summary>
              <pre className="json-output">{JSON.stringify(results, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
    </>
  );
};

const DefaultResults = ({ results }) => (
  <div className="result-card">
    <h3>{t('detector.analysisResults') || 'Analysis Results'}</h3>
    <pre className="json-output">{JSON.stringify(results, null, 2)}</pre>
  </div>
);