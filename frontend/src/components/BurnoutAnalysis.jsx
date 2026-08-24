import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { BURNOUT_COLORS } from '../utils/constants';

export const BurnoutAnalysis = ({ data }) => {
  const { t } = useLanguage();
  
  if (!data) return null;
  
  console.log('BurnoutAnalysis data:', data);

  if (data.error) {
    return (
      <div className="burnout-error">
        <h4>⚠️ {t('burnout.error') || 'Analysis Error'}</h4>
        <p>{data.error}</p>
      </div>
    );
  }

  const { state, level, score, confidence, components, recommendations, top_factor, comment } = data;
  
  // Component labels with translations
  const componentLabels = {
    exhaustion: t('burnout.components.exhaustion') || 'Emotional Exhaustion',
    prosodic_flattening: t('burnout.components.prosodicFlattening') || 'Prosodic Flattening',
    pause_tempo: t('burnout.components.pauseTempo') || 'Pause/Tempo Changes',
    negative_activation: t('burnout.components.negativeActivation') || 'Negative Activation',
    positive_affect_loss: t('burnout.components.positiveAffectLoss') || 'Positive Affect Loss',
  };

  const levelInfo = {
    low: { color: BURNOUT_COLORS.low, label: t('burnout.low') || 'Low Risk' },
    moderate: { color: BURNOUT_COLORS.moderate, label: t('burnout.moderate') || 'Moderate Risk' },
    high: { color: BURNOUT_COLORS.high, label: t('burnout.high') || 'High Risk' },
    severe: { color: BURNOUT_COLORS.severe, label: t('burnout.severe') || 'Severe Risk' },
  };

  const currentLevel = levelInfo[level] || levelInfo.low;

  // Translate state if possible
  const getStateTranslation = (stateValue) => {
    const stateMap = {
      'NORMAL': t('burnout.states.normal') || 'Normal',
      'SHORT_STRESS': t('burnout.states.shortStress') || 'Short-term Stress',
      'SUSTAINED_STRESS': t('burnout.states.sustainedStress') || 'Sustained Stress',
      'BURNOUT_LIKE': t('burnout.states.burnoutLike') || 'Burnout-like State',
      'LOW_AFFECT_UNSPECIFIC': t('burnout.states.lowAffect') || 'Low Affect Unspecified',
      'INSUFFICIENT_DATA': t('burnout.states.insufficientData') || 'Insufficient Data',
    };
    return stateMap[stateValue] || stateValue;
  };

  // Translate a single recommendation
  const translateRecommendation = (rec) => {
    // Try to find a matching translation key
    const key = rec
      .toLowerCase()
      .replace(/[^a-z0-9]/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_|_$/g, '');
    
    // Try different key formats
    const possibleKeys = [
      `burnout.recommendationsList.${key}`,
      `burnout.recommendationsList.${rec.replace(/\s/g, '_').toLowerCase()}`,
    ];
    
    for (const possibleKey of possibleKeys) {
      const translated = t(possibleKey);
      if (translated !== possibleKey) {
        return translated;
      }
    }
    
    // If no translation found, return the original
    return rec;
  };

  return (
    <div className="burnout-analysis">
      <div className="burnout-header">
        <h4>{t('burnout.title') || 'Burnout Risk Analysis'}</h4>
        <span 
          className="burnout-badge"
          style={{ backgroundColor: currentLevel.color }}
        >
          {currentLevel.label}
        </span>
      </div>

      <div className="burnout-metrics">
        <div className="metric">
          <span className="metric-label">{t('burnout.score') || 'Risk Score'}</span>
          <span className="metric-value">{score?.toFixed(1)}%</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t('burnout.confidence') || 'Confidence'}</span>
          <span className="metric-value">{((confidence || 0) * 100).toFixed(1)}%</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t('burnout.topFactor') || 'Top Factor'}</span>
          <span className="metric-value">{t(`burnout.factors.${top_factor}`) || top_factor || state || 'N/A'}</span>
        </div>
      </div>

      {state && (
        <div className="burnout-state">
          <span className="state-label">{t('burnout.state') || 'State'}: </span>
          <span className="state-value">{getStateTranslation(state)}</span>
        </div>
      )}

      {components && Object.keys(components).length > 0 && (
        <div className="burnout-components">
          <h5>{t('burnout.componentAnalysis') || 'Component Analysis'}</h5>
          {Object.entries(components).map(([key, value]) => (
            <div key={key} className="burnout-component">
              <div className="component-header">
                <span>{componentLabels[key] || key}</span>
                <span>{(value * 100).toFixed(1)}%</span>
              </div>
              <div className="component-bar">
                <div 
                  className="component-fill"
                  style={{
                    width: `${Math.min(value * 100, 100)}%`,
                    backgroundColor: value > 0.66 ? '#dc3545' : value > 0.33 ? '#ffc107' : '#28a745',
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {recommendations && recommendations.length > 0 && (
        <div className="burnout-recommendations">
          <h5>{t('burnout.recommendations') || 'Recommendations'}</h5>
          <ul>
            {recommendations.map((rec, index) => (
              <li key={index}>{translateRecommendation(rec)}</li>
            ))}
          </ul>
        </div>
      )}

      {comment && (
        <div className="burnout-comment">
          <em>💡 {t(`burnout.comments.${comment.replace(/\s/g, '_').toLowerCase()}`) || comment}</em>
        </div>
      )}

      <div className="burnout-disclaimer">
        {t('burnout.disclaimer') || 'This analysis is for informational purposes only and does not constitute medical advice.'}
      </div>
    </div>
  );
};