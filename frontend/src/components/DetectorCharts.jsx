import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { getEmotionColor, getFeatureColor } from '../utils/constants';

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

export const EmotionLineChart = ({ frameResults, height = 280, showFeatures = false, onlyFeatures = false }) => {
  const { t } = useLanguage();
  const [hoverTs, setHoverTs] = React.useState(null);
  const [hoverData, setHoverData] = React.useState(null);
  const chartRef = React.useRef(null);

  const chartTitle = onlyFeatures
    ? (t('detector.features.title') || 'Arousal / Valence')
    : (t('detector.emotionTimeline') || 'Emotion Timeline');

  if (!frameResults?.length) {
    return (
      <div className="chart-container">
        <h4>{chartTitle}</h4>
        <div className="no-data">{t('common.noData') || 'No data'}</div>
      </div>
    );
  }

  // Extract emotion keys from the first frame with data
  const firstFrame = frameResults.find(f => f.result?.additional_probs);
  if (!firstFrame) return null;
  const allKeys = Object.keys(firstFrame.result.additional_probs);
  const emotionKeys = allKeys.filter(k => !['valence', 'arousal'].includes(k));
  const featureKeys = allKeys.filter(k => ['valence', 'arousal'].includes(k));
  let displayKeys;
  if (onlyFeatures) {
    displayKeys = featureKeys;
  } else if (showFeatures) {
    displayKeys = allKeys;
  } else {
    displayKeys = emotionKeys;
  }

  // Build datasets: for each display key, an array of {ts, value}
  const series = {};
  displayKeys.forEach(k => { series[k] = []; });

  const points = frameResults.map(f => {
    const ts = f.timestamp || 0;
    const probs = f.result?.additional_probs || {};
    const entry = { ts, values: {} };
    displayKeys.forEach(k => {
      const raw = probs[k];
      const num = typeof raw === 'string' ? parseFloat(raw) : (raw || 0);
      entry.values[k] = num;
      series[k].push({ ts, value: num });
    });
    return entry;
  });

  // Chart dimensions
  const margin = { top: 16, right: 16, bottom: 32, left: 48 };
  const chartW = 700;  // will be scaled
  const innerW = chartW - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const minTs = points[0]?.ts || 0;
  const maxTs = points[points.length - 1]?.ts || 1;
  const tsRange = Math.max(maxTs - minTs, 0.1);

  const xScale = (ts) => margin.left + ((ts - minTs) / tsRange) * innerW;
  const yScale = (v) => margin.top + (1 - v) * innerH;

  // Find nearest point on hover
  const handleMouseMove = (e) => {
    if (!chartRef.current) return;
    const rect = chartRef.current.getBoundingClientRect();
    // Scale screen-pixel mouse X to SVG viewBox coordinates
    const rectW = rect.width || chartW;  // fallback for jsdom where width=0
    const scaleX = chartW / rectW;
    const mx = (e.clientX - rect.left) * scaleX;
    // find nearest point
    let nearest = null;
    let minDist = Infinity;
    points.forEach(p => {
      const px = xScale(p.ts);
      const d = Math.abs(px - mx);
      if (d < minDist) { minDist = d; nearest = p; }
    });
    if (nearest) {
      setHoverTs(nearest.ts);
      setHoverData(nearest.values);
    }
  };

  const handleMouseLeave = () => {
    setHoverTs(null);
    setHoverData(null);
  };

  // Build SVG path for each emotion
  const buildPath = (data) => {
    return data.map((d, i) => {
      const x = xScale(d.ts);
      const y = yScale(d.value);
      return `${i === 0 ? 'M' : 'L'}${x},${y}`;
    }).join(' ');
  };

  // Y axis ticks (0%, 25%, 50%, 75%, 100%)
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  // X axis ticks (show 5-6 labels)
  const xTickCount = Math.min(points.length, 6);
  const xStep = tsRange / (xTickCount - 1 || 1);

  return (
    <div className="chart-container">
      <h4>{chartTitle}</h4>
      <div className="chart-wrapper">
        <svg
          ref={chartRef}
          viewBox={`0 0 ${chartW} ${height}`}
          className="line-chart-svg"
          preserveAspectRatio="xMidYMid meet"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ width: '100%', height: 'auto', background: 'transparent' }}
        >
        {/* Background grid */}
        <defs>
          {displayKeys.map(k => {
            const color = featureKeys.includes(k) ? getFeatureColor(k) : getEmotionColor(k);
            return (
              <linearGradient key={k} id={`grad-${k}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.2" />
                <stop offset="100%" stopColor={color} stopOpacity="0.02" />
              </linearGradient>
            );
          })}
        </defs>

        {/* Y axis grid lines + labels */}
        {yTicks.map(v => (
          <g key={`y-${v}`}>
            <line
              x1={margin.left} y1={yScale(v)}
              x2={chartW - margin.right} y2={yScale(v)}
              stroke="#e0e0e0" strokeWidth={1}
            />
            <text
              x={margin.left - 8} y={yScale(v) + 4}
              textAnchor="end" fill="#888" fontSize={11}
            >
              {(v * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* X axis labels */}
        {Array.from({ length: xTickCount }, (_, i) => {
          const ts = minTs + i * xStep;
          const x = xScale(ts);
          return (
            <text
              key={`x-${i}`}
              x={x} y={height - 6}
              textAnchor="middle" fill="#888" fontSize={11}
            >
              {ts.toFixed(1)}s
            </text>
          );
        })}

        {/* X axis line */}
        <line
          x1={margin.left} y1={height - margin.bottom}
          x2={chartW - margin.right} y2={height - margin.bottom}
          stroke="#ccc" strokeWidth={1}
        />
        {/* Y axis line */}
        <line
          x1={margin.left} y1={margin.top}
          x2={margin.left} y2={height - margin.bottom}
          stroke="#ccc" strokeWidth={1}
        />

        {/* Area fills under lines */}
        {displayKeys.map(k => (
          <path
            key={`area-${k}`}
            d={`${buildPath(series[k])} L${xScale(maxTs)},${height - margin.bottom} L${xScale(minTs)},${height - margin.bottom} Z`}
            fill={`url(#grad-${k})`}
          />
        ))}

        {/* Emotion/Feature lines */}
        {displayKeys.map(k => {
          const color = featureKeys.includes(k) ? getFeatureColor(k) : getEmotionColor(k);
          return (
            <path
              key={`line-${k}`}
              d={buildPath(series[k])}
              fill="none"
              stroke={color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              className="emotion-line"
            />
          );
        })}

        {/* Hover vertical line */}
        {hoverTs !== null && (
          <line
            x1={xScale(hoverTs)} y1={margin.top}
            x2={xScale(hoverTs)} y2={height - margin.bottom}
            stroke="#666" strokeWidth={1} strokeDasharray="4,3"
          />
        )}

        {/* SVG tooltip inside chart */}
        {hoverData && (() => {
          const tx = xScale(hoverTs);
          const ty = margin.top;
          const rows = Object.entries(hoverData).sort((a, b) => b[1] - a[1]);
          const rowH = 18;
          const pad = 8;
          const tw = 130;
          const th = 22 + rows.length * rowH;
          // Flip to left if too close to right edge, or right if too close to left edge
          const flipLeft = tx - tw - 10 < margin.left;
          const tooltipX = flipLeft ? tx + 10 : (tx + tw + 10 > chartW - margin.right ? tx - tw - 10 : tx + 10);
          return (
            <g>
              <rect
                x={tooltipX}
                y={ty}
                width={tw}
                height={th}
                rx={6}
                fill="white"
                stroke="#ccc"
                strokeWidth={1}
                opacity={0.95}
              />
              <text
                x={tooltipX + pad}
                y={ty + 16}
                fontSize={12}
                fontWeight="bold"
                fill="#333"
              >
                {hoverTs.toFixed(1)}s
              </text>
              {rows.map(([k, v], i) => (
                <g key={k}>
                  <circle
                    cx={tooltipX + pad + 4}
                    cy={ty + 26 + i * rowH}
                    r={4}
                    fill={featureKeys.includes(k) ? getFeatureColor(k) : getEmotionColor(k)}
                  />
                  <text
                    x={tooltipX + pad + 14}
                    y={ty + 30 + i * rowH}
                    fontSize={11}
                    fill="#555"
                  >
                    {featureKeys.includes(k) ? (t(`detector.features.${k}`) || k) : (t(`emotions.${k}`) || k)}
                  </text>
                  <text
                    x={tooltipX + tw - pad}
                    y={ty + 30 + i * rowH}
                    fontSize={11}
                    fill="#333"
                    textAnchor="end"
                    fontWeight="500"
                  >
                    {(v * 100).toFixed(1)}%
                  </text>
                </g>
              ))}
            </g>
          );
        })()}
      </svg>

      {/* Legend (outside SVG) */}
      <div className="chart-legend">
        {displayKeys.map(k => {
          const color = featureKeys.includes(k) ? getFeatureColor(k) : getEmotionColor(k);
          const label = featureKeys.includes(k)
            ? (t(`detector.features.${k}`) || k)
            : (t(`emotions.${k}`) || k);
          return (
            <div key={k} className="chart-legend-item">
              <span
                className="chart-legend-dot"
                style={{ backgroundColor: color }}
              />
              <span>{label}</span>
            </div>
          );
        })}
      </div>
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