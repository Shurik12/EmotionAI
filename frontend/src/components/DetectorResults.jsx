import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { EmotionBar } from './EmotionBar';
import { EmotionLineChart } from './DetectorCharts';
import { BurnoutAnalysis } from './BurnoutAnalysis';
import { ExternalInfluenceResults } from './ExternalInfluenceResults';
import { getEmotionEntries, getBurnoutAnalysis } from '../utils/helpers';
import { getFeatureColor } from '../utils/constants';

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
      case 'audio':
        return <AudioResults results={results} resultData={resultData} />;
      case 'audio_burnout':
        return <AudioBurnoutResults results={results} resultData={resultData} burnout={burnout} />;
      case 'audio_external_influence':
        return <ExternalInfluenceResults key={results.task_id} results={results} />;
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

export const ValenceArousal = ({ features }) => {
  const { t } = useLanguage();
  if (!features || Object.keys(features).length === 0) return null;
  return (
    <div className="emotion-results">
      <h4>{t('detector.features.title') || 'Valence / Arousal'}</h4>
      {Object.entries(features).map(([key, value]) => {
        const pct = (value * 100).toFixed(1);
        return (
          <div key={key} className="emotion-item">
            <div className="emotion-label">
              <span>{t(`detector.features.${key}`) || key}</span>
              <span>{pct}%</span>
            </div>
            <div className="emotion-bar">
              <div
                className="emotion-fill"
                style={{
                  width: `${pct}%`,
                  backgroundColor: getFeatureColor(key),
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

const ImageResults = ({ results, resultData }) => {
  const { t } = useLanguage();
  const { emotions, features } = getEmotionEntries(resultData?.additional_probs);
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
      
      <ValenceArousal features={features} />
      
      {burnout && <BurnoutAnalysis data={burnout} />}
    </div>
  );
};

const VideoResults = ({ results, resultData }) => {
  const { t } = useLanguage();
  const { emotions } = getEmotionEntries(results.average_emotions);
  const avgMain = results.average_main_emotion;
  const frames = results.results;
  const firstFrame = frames?.[0];
  const lastFrame = frames?.length > 1 ? frames[frames.length - 1] : null;

  return (
    <div className="result-card">
      <h3>🎬 {t('detector.videoAnalysisComplete')}</h3>
      <p>{t('detector.framesProcessed', { count: results.frames_processed })}</p>

      {avgMain && (
        <div className="main-emotion">
          <strong>{t('detector.mainEmotion') || 'Main Emotion'}:</strong>{' '}
          {t(`emotions.${avgMain.label}`) || avgMain.label} ({(avgMain.probability * 100).toFixed(1)}%)
        </div>
      )}

      {Object.keys(emotions).length > 0 && (
        <div className="emotion-results">
          <h4>{t('detector.averageEmotions') || 'Average Emotions'}</h4>
          {Object.entries(emotions).map(([key, value]) => (
            <EmotionBar key={key} emotion={key} probability={parseFloat(value)} />
          ))}
        </div>
      )}
      
      <div className="video-frames">
        {firstFrame && <VideoFrame frame={firstFrame} index={0} />}
        {lastFrame && <VideoFrame frame={lastFrame} index={frames.length - 1} />}
      </div>
    </div>
  );
};

const VideoFrame = ({ frame, index }) => {
  const { t } = useLanguage();
  const { emotions, features } = getEmotionEntries(frame.result?.additional_probs);
  const [imgError, setImgError] = React.useState(false);

  return (
    <div className="video-frame">
      <h4>{t('detector.frame')} {index + 1}</h4>
      {frame.image_url && !imgError && (
        <img src={frame.image_url} alt={`Frame ${index + 1}`} className="frame-image" onError={() => setImgError(true)} />
      )}
      {(!frame.image_url || imgError) && (
        <div className="frame-image-placeholder">{t('detector.noImage') || 'No image'}</div>
      )}
      <div className="frame-emotions">
        {Object.entries(emotions).map(([key, value]) => (
          <EmotionBar key={key} emotion={key} probability={parseFloat(value)} />
        ))}
      </div>
      <ValenceArousal features={features} />
    </div>
  );
};

const AudioResults = ({ results, resultData }) => {
  const { t } = useLanguage();
  const { emotions, features } = getEmotionEntries(resultData?.additional_probs);
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
      
      <ValenceArousal features={features} />
      
      {burnout && <BurnoutAnalysis data={burnout} />}
    </div>
  );
};

const AudioBurnoutResults = ({ results, resultData, burnout }) => {
  const { t } = useLanguage();
  const { emotions, features } = getEmotionEntries(resultData?.additional_probs);

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
        
        <ValenceArousal features={features} />
        
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