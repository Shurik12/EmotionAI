import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import {
  getExternalInfluenceStatusColor,
  getExternalInfluenceStatusKey,
  EXTERNAL_INFLUENCE_STATUS_COLORS,
  EXTERNAL_INFLUENCE_CONFIDENCE_PARTS,
} from '../utils/constants';

const pct = (value) => Math.round((value || 0) * 100);

// Backend insufficientReason codes (ExternalInfluenceAnalyzer) -> translation
// keys in translations.js externalInfluence.reasons.*
const INSUFFICIENT_REASON_KEYS = {
  low_audio_quality: 'lowAudioQuality',
  too_few_fragments: 'tooFewFragments',
  no_valid_fragments: 'noValidFragments',
  low_confidence: 'lowConfidence',
};

// Indicator bars share the status palette: the component ladder mirrors the
// backend score boundaries (0.30 / 0.50 / 0.80) so recalibration keeps the
// UI colors meaningful without editing this component.
const getIndicatorColor = (value) => {
  if (value < 30) return EXTERNAL_INFLUENCE_STATUS_COLORS.LOW;
  if (value < 50) return EXTERNAL_INFLUENCE_STATUS_COLORS.ELEVATED_TENSION;
  if (value < 80) return EXTERNAL_INFLUENCE_STATUS_COLORS.POSSIBLE_EXTERNAL_PRESSURE;
  return EXTERNAL_INFLUENCE_STATUS_COLORS.PROBABLE_EXTERNAL_INFLUENCE;
};

// Read-only result card: analysis runs fully server-side (context flags are
// supplied to the API by integrations, never chosen in the client).
export const ExternalInfluenceResults = ({ results }) => {
  const { t } = useLanguage();

  const envelope = results?.result || {};
  const res = envelope.result || null;
  const diag = envelope.diagnostics || null;
  const taskId = results?.task_id;

  if (!res || !diag) {
    return (
      <div className="ei-result-card">
        <h3>{t('externalInfluence.title')}</h3>
        <div className="no-results">
          <p>{t('externalInfluence.guidance.noResult')}</p>
          <details>
            <summary>{t('common.showRawData')}</summary>
            <pre className="json-output">{JSON.stringify(results, null, 2)}</pre>
          </details>
        </div>
      </div>
    );
  }

  const status = res.status || 'INSUFFICIENT_DATA';
  const statusKey = getExternalInfluenceStatusKey(status);
  const statusColor = getExternalInfluenceStatusColor(status);
  const components = diag.components || {};
  const topFactors = res.topFactors || [];
  const fragmentScores = diag.fragmentScores || [];
  const rollingScores = diag.rollingScores || [];
  const contextFlags = diag.contextFlags || [];
  const isInsufficient = status === 'INSUFFICIENT_DATA';
  const reasonKey =
    (isInsufficient && INSUFFICIENT_REASON_KEYS[diag.insufficientReason]) || null;

  const sortedComponents = Object.keys(components).sort(
    (a, b) => (components[b] || 0) - (components[a] || 0)
  );

  return (
    <div className="ei-result-card">
      <h3>{t('externalInfluence.title')}</h3>
      <div className="ei-meta">
        <span>
          {t('detector.duration')}: {(envelope.duration || 0).toFixed(1)}s
        </span>
        <span>
          {t('detector.sampleRate')}: {envelope.sample_rate || 'N/A'} Hz
        </span>
        <span>
          {t('externalInfluence.quality')}: {pct(diag.audioQuality)}%
        </span>
        <span>
          {t('externalInfluence.fragmentsCount', {
            valid: diag.fragmentCount || 0,
            total: diag.totalFragmentCount || 0,
          })}
        </span>
        {taskId && (
          <span>
            {t('detector.taskId')}: {taskId}
          </span>
        )}
      </div>

      <div className="ei-header">
        <div className="ei-status-badge" style={{ backgroundColor: statusColor }}>
          {t(`externalInfluence.states.${statusKey}`)}
        </div>
        <div className="ei-subtitle">{t('externalInfluence.subtitle')}</div>
      </div>

      {isInsufficient && (
        <div className="ei-insufficient">
          <p className="ei-guidance-text">{t('externalInfluence.guidance.insufficientData')}</p>
          {reasonKey && (
            <p className="ei-reason-text">{t(`externalInfluence.reasons.${reasonKey}`)}</p>
          )}
          <div className="ei-insufficient-detail">
            <span>{t('externalInfluence.confidence')}: {pct(res.confidence)}%</span>
            <span>{t('externalInfluence.quality')}: {pct(diag.audioQuality)}%</span>
            <span>{t('externalInfluence.score')}: {pct(res.score)}%</span>
            <span>
              {t('externalInfluence.fragmentsCount', {
                valid: diag.fragmentCount || 0,
                total: diag.totalFragmentCount || 0,
              })}
            </span>
          </div>
        </div>
      )}

      <div className="ei-metrics-grid">
        <div className="ei-metric-item">
          <span className="ei-metric-label">{t('externalInfluence.score')}</span>
          <div className="ei-score-bar">
            <div
              className="ei-score-fill"
              style={{ width: `${pct(res.score)}%`, backgroundColor: statusColor }}
            />
          </div>
          <span className="ei-metric-value">{pct(res.score)}%</span>
        </div>
        <div className="ei-metric-item">
          <span className="ei-metric-label">{t('externalInfluence.confidence')}</span>
          <div className="ei-metric-value">{pct(res.confidence)}%</div>
        </div>
      </div>

      <div className="ei-confidence-parts">
        {EXTERNAL_INFLUENCE_CONFIDENCE_PARTS.map(({ key, weight }) => {
          const raw = diag[key] || 0;
          return (
            <span key={key} className="ei-confidence-part">
              {weight} × {t(`externalInfluence.confidenceParts.${key}`)} {pct(raw)}%
              {' → '}
              <b>{pct(raw * weight)}%</b>
            </span>
          );
        })}
        <span className="ei-confidence-total">
          = {pct(res.confidence)}%
        </span>
      </div>

      <div className="ei-chip-row">
        <span
          className={`ei-chip ${res.persistentPattern ? 'ei-chip-on' : 'ei-chip-off'}`}
          title={t('externalInfluence.persistentPattern')}
        >
          {t('externalInfluence.persistentPattern')}:{' '}
          {res.persistentPattern ? '✓' : '—'}
        </span>
        <span
          className={`ei-chip ${res.contextConfirmed ? 'ei-chip-on' : 'ei-chip-off'}`}
          title={t('externalInfluence.contextConfirmed')}
        >
          {t('externalInfluence.contextConfirmed')}:{' '}
          {res.contextConfirmed ? '✓' : '—'}
        </span>
      </div>

      {topFactors.length > 0 && (
        <div className="ei-section">
          <h4>{t('externalInfluence.topFactors')}</h4>
          <ul className="ei-factors">
            {topFactors.map((factor) => (
              <li key={factor}>{t(`externalInfluence.factors.${factor}`)}</li>
            ))}
          </ul>
        </div>
      )}

      {sortedComponents.length > 0 && (
        <div className="ei-section">
          <h4>{t('externalInfluence.componentsTitle')}</h4>
          <div className="ei-indicators-grid">
            {sortedComponents.map((key) => {
              const value = pct(components[key]);
              return (
                <div key={key} className="ei-indicator-item">
                  <div className="ei-indicator-label">
                    <span>{t(`externalInfluence.components.${key}`)}</span>
                    <span className="ei-indicator-value">{value}%</span>
                  </div>
                  <div className="ei-indicator-bar">
                    <div
                      className="ei-indicator-fill"
                      style={{ width: `${value}%`, backgroundColor: getIndicatorColor(value) }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {fragmentScores.length > 0 && (
        <div className="ei-section">
          <h4>{t('externalInfluence.fragmentTimeline')}</h4>
          <div className="ei-fragment-timeline">
            {fragmentScores.map((score, index) => (
              <div key={index} className="ei-fragment-cell">
                <div
                  className="ei-fragment-bar"
                  title={`${t('externalInfluence.fragmentLabel', { n: index + 1 })}: ${pct(score)}%`}
                  style={{
                    height: `${Math.max(4, pct(score))}%`,
                    backgroundColor:
                      diag.bestWindowIndex >= 0 &&
                      index >= diag.bestWindowIndex &&
                      index < diag.bestWindowIndex + (diag.windowSize || 1)
                        ? statusColor
                        : getIndicatorColor(pct(score)),
                  }}
                />
                <span className="ei-fragment-num">{index + 1}</span>
              </div>
            ))}
          </div>
          {rollingScores.length > 0 && (
            <div className="ei-rolling-hint">
              {diag.windowSize > 0
                ? `median ×${diag.windowSize}`
                : null}
              {typeof diag.bestWindowIndex === 'number' && diag.bestWindowIndex >= 0
                ? ` · ${t('externalInfluence.fragmentLabel', {
                    n: diag.bestWindowIndex + 1,
                  })}${diag.windowSize > 1 ? '–' + (diag.bestWindowIndex + diag.windowSize) : ''}`
                : null}
            </div>
          )}
        </div>
      )}

      {contextFlags.length > 0 && (
        <div className="ei-section">
          <h4>{t('externalInfluence.contextApplied')}</h4>
          <div className="ei-chip-row">
            {contextFlags.map((flag) => (
              <span key={flag} className="ei-chip ei-chip-on">
                {t(`externalInfluence.context.${flag}`)}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="ei-manager-action" style={{ borderLeftColor: statusColor }}>
        <span className="ei-manager-action-label">{t('externalInfluence.managerAction')}</span>
        <span className="ei-manager-action-value">
          {t(`externalInfluence.actions.${res.managerAction}`)}
        </span>
      </div>

      <div className="ei-disclaimer">
        <h4>{t('externalInfluence.importantNote')}</h4>
        <p>{t('externalInfluence.disclaimer')}</p>
      </div>
    </div>
  );
};
