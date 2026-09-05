import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import {
  getBurnoutStateColor,
  getBurnoutStateKey,
  getBurnoutLevelClass,
  getBurnoutLevelColor,
} from '../utils/constants';

export const BurnoutAnalysis = ({ data }) => {
  const { t } = useLanguage();

  if (!data) return null;

  const state = data.state || 'INSUFFICIENT_DATA';
  const level = data.level || 'low';
  const components = data.components || {};
  const recommendations = data.recommendations || [];

  const stateKey = getBurnoutStateKey(state);

  const getTranslatedText = (textKey) => {
    return t(`burnout.${textKey}.${stateKey}`) || t(`burnout.${textKey}.default`);
  };

  const getStateName = () => {
    return t(`burnout.states.${stateKey}`);
  };

  const getLevelLabel = () => {
    const levelMap = {
      low: t('burnout.low'),
      moderate: t('burnout.moderate'),
      high: t('burnout.high'),
      severe: t('burnout.severe'),
    };
    return levelMap[level] || t('burnout.low');
  };

  const getComponentName = (key) => {
    const componentMap = {
      exhaustion: t('burnout.components.exhaustion'),
      emotional_tension: t('burnout.components.emotionalTension'),
      pause_tempo: t('burnout.components.pauseTempo'),
      prosodic_flattening: t('burnout.components.speechColorReduction'),
      negative_activation: t('burnout.components.emotionalTension'),
      positive_affect_loss: t('burnout.components.speechColorReduction'),
      'Positive Affect Loss': t('burnout.components.speechColorReduction'),
      'Prosodic Flattening': t('burnout.components.speechColorReduction'),
      'Pause/Tempo Changes': t('burnout.components.pauseTempo'),
      'Negative Activation': t('burnout.components.emotionalTension'),
      'Emotional Exhaustion': t('burnout.components.exhaustion'),
    };
    return componentMap[key] || key;
  };

  const getComponentValue = (key) => {
    const value = components[key] || 0;
    return Math.round(value * 100);
  };

  const getRecommendationText = (recKey) => {
    return t(`burnout.recommendationsList.${recKey}`) || recKey;
  };

  const isInsufficient = state === 'INSUFFICIENT_DATA';
  const stateColor = getBurnoutStateColor(state);
  const levelClass = getBurnoutLevelClass(level);
  const levelColor = getBurnoutLevelColor(level);

  const getIndicatorColor = (value) => {
    if (value < 30) return '#28a745';
    if (value < 50) return '#ffc107';
    if (value < 70) return '#fd7e14';
    return '#dc3545';
  };

  const sortedComponents = Object.keys(components).sort(
    (a, b) => (components[b] || 0) - (components[a] || 0)
  );

  const highestKey = sortedComponents.length > 0 ? sortedComponents[0] : null;

  const getAttentionLevel = () => {
    if (level === 'severe') return t('burnout.attentionLevel.severe');
    if (level === 'high') return t('burnout.attentionLevel.high');
    if (level === 'moderate') return t('burnout.attentionLevel.moderate');
    return t('burnout.attentionLevel.low');
  };

  const getMainSignal = () => {
    if (highestKey) {
      return getComponentName(highestKey);
    }
    return t('burnout.attentionLevel.noSignals');
  };

  return (
    <div className="burnout-analysis">
      {isInsufficient ? (
        <div className="burnout-insufficient">
          <p>{t('burnout.errors.analysis_failed')}</p>
          <p className="burnout-help-text">{t('burnout.basisText.insufficientData')}</p>
        </div>
      ) : (
        <>
          <div className="burnout-header">
            <div className="burnout-state-badge" style={{ backgroundColor: stateColor }}>
              {getStateName()}
            </div>
          </div>

          {/* REMOVED: Duplicate state description - now only shown in "Basis of Assessment" */}

          <div className="burnout-attention">
            <div className="burnout-attention-grid">
              <div className="burnout-attention-item">
                <span className="burnout-attention-label">{t('burnout.attentionLevel.title')}</span>
                <span className="burnout-attention-value" style={{ color: levelColor }}>
                  {getAttentionLevel()}
                </span>
              </div>
              <div className="burnout-attention-item">
                <span className="burnout-attention-label">{t('burnout.attentionLevel.mainSignal')}</span>
                <span className="burnout-attention-signal">{getMainSignal()}</span>
              </div>
            </div>
          </div>

          <div className="burnout-indicators">
            <h4>{t('burnout.componentAnalysis')}</h4>
            <div className="burnout-indicators-grid">
              {sortedComponents.map((key) => {
                const value = getComponentValue(key);
                return (
                  <div key={key} className="burnout-indicator-item">
                    <div className="burnout-indicator-label">
                      <span>{getComponentName(key)}</span>
                      <span className="burnout-indicator-value">{value}%</span>
                    </div>
                    <div className="burnout-indicator-bar">
                      <div
                        className="burnout-indicator-fill"
                        style={{
                          width: `${value}%`,
                          backgroundColor: getIndicatorColor(value),
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Basis of Assessment - only one instance */}
          <div className="burnout-section">
            <h4>{t('burnout.basis')}</h4>
            <p>{getTranslatedText('basisText')}</p>
          </div>

          {recommendations && recommendations.length > 0 && (
            <div className="burnout-section">
              <h4>{t('burnout.recommendations')}</h4>
              <ul className="burnout-recommendations">
                {recommendations.map((rec, index) => (
                  <li key={index}>{getRecommendationText(rec)}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="burnout-disclaimer">
            <h4>{t('burnout.importantNote')}</h4>
            <p>{t('burnout.disclaimer')}</p>
          </div>
        </>
      )}
    </div>
  );
};