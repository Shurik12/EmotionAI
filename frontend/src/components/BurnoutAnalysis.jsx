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
        <h4>⚠️ {t('burnout.error') || 'Ошибка анализа'}</h4>
        <p>{t(`burnout.errors.${data.error}`) || data.error}</p>
      </div>
    );
  }

  const { 
    state, 
    level, 
    score, 
    confidence, 
    components, 
    recommendations, 
    top_factor, 
    comment
  } = data;
  
  // Component labels with translations
  const componentLabels = {
    exhaustion: t('burnout.components.exhaustion') || 'Эмоциональное истощение',
    prosodic_flattening: t('burnout.components.prosodicFlattening') || 'Просодическое уплощение',
    pause_tempo: t('burnout.components.pauseTempo') || 'Изменения пауз/темпа',
    negative_activation: t('burnout.components.negativeActivation') || 'Негативная активация',
    positive_affect_loss: t('burnout.components.positiveAffectLoss') || 'Потеря положительного аффекта',
  };

  const levelInfo = {
    low: { color: BURNOUT_COLORS.low, label: t('burnout.low') || 'Низкий риск' },
    moderate: { color: BURNOUT_COLORS.moderate, label: t('burnout.moderate') || 'Умеренный риск' },
    high: { color: BURNOUT_COLORS.high, label: t('burnout.high') || 'Высокий риск' },
    severe: { color: BURNOUT_COLORS.severe, label: t('burnout.severe') || 'Очень высокий риск' },
  };

  const currentLevel = levelInfo[level] || levelInfo.low;

  // Translate state using the key
  const getStateTranslation = (stateKey) => {
    return t(`burnout.states.${stateKey}`) || stateKey;
  };

  // Get state display name (for the state badge)
  const getStateKey = (stateValue) => {
    const stateMap = {
      'NORMAL': 'normal',
      'SHORT_STRESS': 'shortStress',
      'SUSTAINED_STRESS': 'sustainedStress',
      'BURNOUT_LIKE': 'burnoutLike',
      'LOW_AFFECT_UNSPECIFIC': 'lowAffect',
      'INSUFFICIENT_DATA': 'insufficientData',
    };
    return stateMap[stateValue] || 'insufficientData';
  };

  // Check if state requires history (sustained or burnout-like)
  const requiresHistory = (stateValue) => {
    return stateValue === 'SUSTAINED_STRESS' || stateValue === 'BURNOUT_LIKE';
  };

  // Translate recommendation using the key
  const translateRecommendation = (key) => {
    return t(`burnout.recommendationsList.${key}`) || key;
  };

  // Translate comment using the key
  const translateComment = (key) => {
    return t(`burnout.comments.${key}`) || key;
  };

  // Translate factor
  const translateFactor = (factor) => {
    return t(`burnout.factors.${factor}`) || factor;
  };

  // Get factor explanation
  const getFactorExplanation = (factor) => {
    const explanationKey = factor
      .toLowerCase()
      .replace(/ /g, '_');
    return t(`burnout.factorExplanations.${explanationKey}`) || '';
  };

  // For insufficient data, show minimal info
  if (state === 'INSUFFICIENT_DATA') {
    return (
      <div className="burnout-analysis">
        <div className="burnout-header">
          <h4>{t('burnout.title') || 'Анализ риска профессионального выгорания'}</h4>
          <span className="burnout-badge" style={{ backgroundColor: '#9E9E9E' }}>
            {t('burnout.states.insufficientData') || 'Недостаточно данных'}
          </span>
        </div>
        <div className="burnout-section">
          <h5 className="section-title">{t('burnout.basis') || 'На основании чего сделан вывод'}</h5>
          <p className="section-text">{t('burnout.basisText.insufficientData') || 'Недостаточно данных для анализа.'}</p>
        </div>
        {recommendations && recommendations.length > 0 && (
          <div className="burnout-recommendations">
            <h5>{t('burnout.recommendations') || 'Действия менеджера'}</h5>
            <ul>
              {recommendations.map((recKey, index) => (
                <li key={index}>{translateRecommendation(recKey)}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="burnout-disclaimer">
          {t('burnout.disclaimer')}
        </div>
      </div>
    );
  }

  // Get the state key for translations
  const stateKey = getStateKey(state);

  return (
    <div className="burnout-analysis">
      <div className="burnout-header">
        <h4>{t('burnout.title') || 'Анализ риска профессионального выгорания'}</h4>
        <span 
          className="burnout-badge"
          style={{ backgroundColor: currentLevel.color }}
        >
          {currentLevel.label}
        </span>
      </div>

      {/* Risk metrics */}
      <div className="burnout-metrics">
        <div className="metric">
          <span className="metric-label">{t('burnout.score') || 'Уровень риска'}</span>
          <span className="metric-value">{score?.toFixed(1)}%</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t('burnout.confidence') || 'Надёжность оценки'}</span>
          <span className="metric-value">{((confidence || 0) * 100).toFixed(1)}%</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t('burnout.topFactor') || 'Основной выявленный сигнал'}</span>
          <span className="metric-value">{translateFactor(top_factor) || top_factor || 'N/A'}</span>
        </div>
      </div>

      {/* 1. На основании чего сделан вывод */}
      <div className="burnout-section">
        <h5 className="section-title">{t('burnout.basis') || 'На основании чего сделан вывод'}</h5>
        <p className="section-text">{t(`burnout.basisText.${stateKey}`) || t('burnout.basisText.default')}</p>
      </div>

      {/* 2. Результат (Вывод для менеджера) */}
      <div className="burnout-section">
        <h5 className="section-title">{t('burnout.conclusion') || 'Вывод для менеджера'}</h5>
        <p className="section-text">{t(`burnout.conclusionText.${stateKey}`) || t('burnout.conclusionText.default')}</p>
      </div>

      {/* 3. Основной выявленный сигнал с пояснением */}
      {top_factor && (
        <div className="burnout-section">
          <h5 className="section-title">{t('burnout.topFactor') || 'Основной выявленный сигнал'}</h5>
          <p className="section-text"><strong>{translateFactor(top_factor)}</strong></p>
          <p className="section-text explanation">{getFactorExplanation(top_factor)}</p>
        </div>
      )}

      {/* 4. Срочность и дальнейшие действия */}
      <div className="burnout-section urgency-section">
        <h5 className="section-title">{t('burnout.urgency') || 'Срочность и дальнейшие действия'}</h5>
        <p className="section-text">{t(`burnout.urgencyText.${level}`) || t('burnout.urgencyText.default')}</p>
      </div>

      {/* 5. Действия менеджера (recommendations) */}
      {recommendations && recommendations.length > 0 && (
        <div className="burnout-section">
          <h5 className="section-title">{t('burnout.recommendations') || 'Действия менеджера'}</h5>
          <ul className="recommendations-list">
            {recommendations.map((recKey, index) => (
              <li key={index}>{translateRecommendation(recKey)}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Component Analysis - "Из чего складывается оценка" */}
      {components && Object.keys(components).length > 0 && (
        <div className="burnout-components">
          <h5>{t('burnout.componentAnalysis') || 'Из чего складывается оценка'}</h5>
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

      {/* State with history note */}
      {state && (
        <div className="burnout-state">
          <span className="state-label">{t('burnout.state') || 'Результат анализа'}: </span>
          <span className="state-value">{getStateTranslation(stateKey)}</span>
          {requiresHistory(state) && (
            <span className="state-note"> ({t('burnout.requiresHistory') || 'требуется несколько записей'})</span>
          )}
        </div>
      )}

      {/* System comment */}
      {comment && (
        <div className="burnout-comment">
          <em>💡 {translateComment(comment)}</em>
        </div>
      )}

      {/* Disclaimer */}
      <div className="burnout-disclaimer">
        {t('burnout.disclaimer')}
      </div>
    </div>
  );
};