import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { getColorForEmotion } from '../utils/emotionColors';
import { useLanguage } from '../hooks/useLanguage';
import { t } from '../utils/translations';
import './Detector.css';

//=============================================================================
// Constants
//=============================================================================
const VALID_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'video/mp4', 'video/avi', 'video/webm',
    'audio/mpeg', 'audio/mp3', 'audio/aac', 'audio/ogg', 'audio/wav'];
const VALID_EXTENSIONS = ['jpg', 'jpeg', 'png', 'mp4', 'avi', 'webm', 'mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a'];
const MAX_FILE_SIZE = 50 * 1024 * 1024;

//=============================================================================
// Utility Functions
//=============================================================================
const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const formatTime = (seconds) => {
    if (!seconds || seconds === 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const getVerdictInfo = (verdict) => ({
    'low': { text: 'verdict_low', class: 'verdict-low', icon: '✅', recommendation: 'recommendation_low' },
    'monitor': { text: 'verdict_monitor', class: 'verdict-monitor', icon: '⚠️', recommendation: 'recommendation_monitor' },
    'high': { text: 'verdict_high', class: 'verdict-high', icon: '🔴', recommendation: 'recommendation_high' },
    'да': { text: 'Ready for task', class: 'verdict-positive', icon: '✅' },
    'нет': { text: 'Not ready for task', class: 'verdict-negative', icon: '❌' }
}[verdict] || { text: verdict, class: '', icon: '📊' });

const getProbabilityColor = (prob) => prob >= 0.7 ? '#dc3545' : prob >= 0.4 ? '#ffc107' : '#28a745';

const getEmotionEntries = (additionalProbs) => {
    const emotions = {};
    const features = {};
    Object.entries(additionalProbs || {}).forEach(([key, value]) => {
        if (['valence', 'arousal'].includes(key)) {
            features[key] = value;
        } else {
            emotions[key] = value;
        }
    });
    return { emotions, features };
};

// Get burnout analysis from nested data
const getBurnoutAnalysis = (data) => {
    if (!data) return null;
    return data.burnout_analysis || (data.result && data.result.burnout_analysis) || null;
};

//=============================================================================
// Burnout Components
//=============================================================================

const BurnoutLevelBadge = ({ level }) => {
    const config = {
        low: { cls: 'low', icon: '✓', label: 'Low Risk' },
        moderate: { cls: 'moderate', icon: '○', label: 'Moderate Risk' },
        high: { cls: 'high', icon: '◐', label: 'High Risk' },
        severe: { cls: 'severe', icon: '●', label: 'Severe Risk' }
    };
    const { cls, icon, label } = config[level] || config.low;
    return <span className={`burnout-level-badge ${cls}`}>{icon} {label}</span>;
};

const BurnoutStateDisplay = ({ state }) => {
    const config = {
        'NORMAL': { color: '#28a745', icon: '✓', label: 'Normal' },
        'SHORT_STRESS': { color: '#ffc107', icon: '○', label: 'Short-term Stress' },
        'SUSTAINED_STRESS': { color: '#fd7e14', icon: '◐', label: 'Sustained Stress' },
        'BURNOUT_LIKE': { color: '#dc3545', icon: '●', label: 'Burnout-like State' },
        'LOW_AFFECT_UNSPECIFIC': { color: '#6c757d', icon: '△', label: 'Low Affect Unspecified' },
        'INSUFFICIENT_DATA': { color: '#adb5bd', icon: '?', label: 'Insufficient Data' }
    };
    const info = config[state] || config['INSUFFICIENT_DATA'];
    
    return (
        <div className="burnout-state">
            <div className="icon" style={{ backgroundColor: info.color }}>{info.icon}</div>
            <div>
                <div className="label">{info.label}</div>
                <div className="sub">State: {state}</div>
            </div>
        </div>
    );
};

const BurnoutComponentBar = ({ label, value }) => {
    const pct = Math.min(value * 100, 100);
    const color = pct > 66 ? '#dc3545' : pct > 33 ? '#ffc107' : '#28a745';
    return (
        <div className="burnout-component-item">
            <div className="header">
                <span>{label}</span>
                <span>{pct.toFixed(1)}%</span>
            </div>
            <div className="bar-track">
                <div className="bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
            </div>
        </div>
    );
};

const BurnoutAnalysis = ({ data }) => {
    if (!data) return null;
    if (data.error) {
        return (
            <div className="burnout-error">
                <h4>⚠️ Burnout Analysis Error</h4>
                <p>{data.error}</p>
            </div>
        );
    }

    const { state, level, score, confidence, components, recommendations, top_factor, comment } = data;
    const labels = {
        exhaustion: 'Emotional Exhaustion',
        prosodic_flattening: 'Prosodic Flattening',
        pause_tempo: 'Pause/Tempo Changes',
        negative_activation: 'Negative Activation',
        positive_affect_loss: 'Positive Affect Loss'
    };

    return (
        <div className="burnout-analysis-card">
            <div className="title">
                <span>Burnout Risk Analysis</span>
                <BurnoutLevelBadge level={level} />
            </div>

            <div className="burnout-metrics">
                {[
                    { label: 'Risk Score', value: `${score?.toFixed(1)}%` },
                    { label: 'Confidence', value: `${((confidence || 0) * 100).toFixed(1)}%` },
                    { label: 'Top Factor', value: top_factor || 'N/A' }
                ].map((m, i) => (
                    <div className="burnout-metric" key={i}>
                        <div className="label">{m.label}</div>
                        <div className="value">{m.value}</div>
                    </div>
                ))}
            </div>

            <BurnoutStateDisplay state={state} />

            <div className="burnout-components">
                <h4>Component Analysis</h4>
                {Object.entries(components || {}).map(([key, value]) => (
                    <BurnoutComponentBar key={key} label={labels[key] || key} value={value} />
                ))}
            </div>

            {recommendations?.length > 0 && (
                <div className="burnout-recommendations">
                    <h4>📋 Recommendations</h4>
                    <ul>{recommendations.map((r, i) => <li key={i}>{r}</li>)}</ul>
                </div>
            )}

            {comment && (
                <div className="burnout-comment"><em>💡 {comment}</em></div>
            )}

            <div className="burnout-disclaimer">
                This analysis is for informational purposes only and does not constitute medical advice.
                Please consult a healthcare professional for clinical decisions.
            </div>
        </div>
    );
};

//=============================================================================
// Chart Components
//=============================================================================
const LineChart = ({ data, labels, title, color, height = 200 }) => {
    if (!data?.length) {
        return (
            <div className="chart-container">
                <h4>{title}</h4>
                <div className="no-data">No data available</div>
            </div>
        );
    }

    const maxVal = Math.max(...data);
    const minVal = Math.min(...data);
    const range = maxVal - minVal || 1;

    return (
        <div className="chart-container">
            <h4>{title}</h4>
            <div className="line-chart" style={{ height: `${height}px` }}>
                {data.map((value, index) => (
                    <div
                        key={index}
                        className="chart-point"
                        style={{
                            left: `${(index / (data.length - 1)) * 100}%`,
                            bottom: `${((value - minVal) / range) * 100}%`,
                            backgroundColor: color
                        }}
                        title={`${labels[index]}: ${value.toFixed(2)}`}
                    />
                ))}
                <div className="chart-labels">
                    <span>{minVal.toFixed(1)}</span>
                    <span>{maxVal.toFixed(1)}</span>
                </div>
            </div>
        </div>
    );
};

const EmotionDistributionChart = ({ emotions, height = 250 }) => {
    if (!emotions || !Object.keys(emotions).length) {
        return (
            <div className="chart-container">
                <h4>Average Emotion Distribution</h4>
                <div className="no-data">No emotion data available</div>
            </div>
        );
    }

    const lang = localStorage.getItem('language');
    const entries = Object.entries(emotions).sort((a, b) => b[1] - a[1]);

    return (
        <div className="chart-container">
            <h4>Average Emotion Distribution</h4>
            <div className="distribution-chart" style={{ height: `${height}px` }}>
                {entries.map(([emotion, value]) => (
                    <div key={emotion} className="distribution-item">
                        <div className="emotion-label">
                            <span>{t(emotion, lang)}</span>
                            <span>{(value * 100).toFixed(1)}%</span>
                        </div>
                        <div className="distribution-bar">
                            <div className="distribution-fill" style={{
                                width: `${value * 100}%`,
                                backgroundColor: getColorForEmotion(emotion)
                            }} />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const TimelineChart = ({ frameResults, height = 200 }) => {
    const [selected, setSelected] = useState(null);
    const lang = localStorage.getItem('language');

    if (!frameResults?.length) {
        return (
            <div className="chart-container">
                <h4>Emotion Timeline</h4>
                <div className="no-data">No timeline data available</div>
            </div>
        );
    }

    const lastTs = frameResults[frameResults.length - 1]?.timestamp || 1;

    return (
        <div className="chart-container">
            <h4>Emotion Timeline</h4>
            <div className="timeline-chart" style={{ height: `${height}px` }}>
                {frameResults.map((frame, index) => {
                    const emotion = frame.result?.main_prediction?.label;
                    const prob = frame.result?.main_prediction?.probability || 0;
                    return (
                        <div
                            key={index}
                            className={`timeline-point ${selected === index ? 'selected' : ''}`}
                            style={{
                                left: `${(frame.timestamp / lastTs) * 100}%`,
                                backgroundColor: getColorForEmotion(emotion),
                                opacity: 0.5 + (prob * 0.5)
                            }}
                            onMouseEnter={() => setSelected(index)}
                            onMouseLeave={() => setSelected(null)}
                            title={`${frame.timestamp.toFixed(1)}s: ${emotion} (${(prob * 100).toFixed(1)}%)`}
                        />
                    );
                })}
                {selected !== null && (
                    <div className="timeline-tooltip" style={{
                        left: `${(frameResults[selected].timestamp / lastTs) * 100}%`
                    }}>
                        <strong>Time: {frameResults[selected].timestamp.toFixed(1)}s</strong><br />
                        Main emotion: {t(frameResults[selected].result?.main_prediction?.label, lang)}<br />
                        Probability: {(frameResults[selected].result?.main_prediction?.probability * 100).toFixed(1)}%
                    </div>
                )}
            </div>
        </div>
    );
};

//=============================================================================
// Emotion Display Components
//=============================================================================
const EmotionBar = ({ emotion, probability }) => {
    const lang = localStorage.getItem('language');
    const pct = (parseFloat(probability) * 100).toFixed(1);
    return (
        <div className="emotion-item">
            <div className="emotion-label">
                <span>{t(emotion, lang)}</span>
                <span>{pct}%</span>
            </div>
            <div className="emotion-bar">
                <div className="emotion-fill" style={{
                    width: `${pct}%`,
                    backgroundColor: getColorForEmotion(emotion)
                }} />
            </div>
        </div>
    );
};

const FeatureBar = ({ label, value }) => {
    const lang = localStorage.getItem('language');
    const displayName = t(label, lang);
    return (
        <div className="feature-item">
            <div className="feature-label">
                <span>{displayName}</span>
                <span>{parseFloat(value).toFixed(2)}</span>
            </div>
            <div className="feature-value">
                <div className="feature-fill" style={{
                    width: `${(parseFloat(value) + 1) * 50}%`,
                    backgroundColor: label === 'valence' ? '#28a745' : '#17a2b8'
                }} />
            </div>
        </div>
    );
};

const MainEmotionDisplay = ({ mainPrediction }) => {
    if (!mainPrediction) return null;
    const lang = localStorage.getItem('language');
    return (
        <div className="main-emotion">
            {t(mainPrediction.label, lang)} ({(mainPrediction.probability * 100).toFixed(1)}%)
        </div>
    );
};

//=============================================================================
// GigaChat Analysis
//=============================================================================
const GigaChatAnalysis = ({ gigachatData }) => {
    const lang = localStorage.getItem('language');
    if (!gigachatData) return null;

    let data = gigachatData;
    if (typeof data === 'string') {
        try { data = JSON.parse(data); } catch { return <div className="gigachat-analysis">{data}</div>; }
    }

    const verdict = getVerdictInfo(data.verdict);

    return (
        <div className="gigachat-analysis">
            <h4>
                <span>🤖 AI Clinical Assessment</span>
                {data.scores && <span className="total-score-badge">Score: {data.scores.total_score}</span>}
            </h4>

            <div className={`gigachat-verdict ${verdict.class}`}>
                <div className="verdict-icon">{verdict.icon}</div>
                <div className="verdict-content">
                    <strong>{t(verdict.text, lang)}</strong>
                    {verdict.recommendation && (
                        <div className="verdict-recommendation">{t(verdict.recommendation, lang)}</div>
                    )}
                </div>
            </div>

            <div className="gigachat-probability">
                <span>Probability:</span>
                <div className="probability-bar-container">
                    <div className="probability-bar" style={{
                        width: `${(data.probability || 0) * 100}%`,
                        backgroundColor: getProbabilityColor(data.probability || 0)
                    }} />
                    <span className="probability-value">{((data.probability || 0) * 100).toFixed(1)}%</span>
                </div>
            </div>

            {data.scores && (
                <div className="gigachat-scores">
                    <div className="scores-header">
                        <span>Depression Pattern Scores</span>
                        <span className="total-score">Total: {data.scores.total_score}</span>
                    </div>
                    <div className="scores-grid">
                        {Object.entries(data.scores).filter(([k]) => k !== 'total_score').map(([key, value]) => (
                            <div key={key} className="score-item">
                                <span>{key.replace('_score', '').toUpperCase()}</span>
                                <span className={`score-value score-${value}`}>{value}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {data.reasoning && (
                <div className="gigachat-reasoning">
                    <strong>Clinical Reasoning:</strong>
                    <p>{data.reasoning}</p>
                </div>
            )}
        </div>
    );
};

//=============================================================================
// Audio Components
//=============================================================================
const AudioVisualizer = ({ audioData, currentTime }) => {
    const canvasRef = useRef(null);
    const [waveformData, setWaveformData] = useState([]);

    useEffect(() => {
        if (!audioData?.length) return;
        const sampleCount = 200;
        const step = Math.max(1, Math.floor(audioData.length / sampleCount));
        const sampled = [];
        for (let i = 0; i < audioData.length && sampled.length < sampleCount; i += step) {
            const chunk = audioData.slice(i, Math.min(i + step, audioData.length));
            const avg = chunk.reduce((a, b) => a + Math.abs(b), 0) / chunk.length;
            sampled.push(avg);
        }
        setWaveformData(sampled);
    }, [audioData]);

    useEffect(() => {
        if (!canvasRef.current || !waveformData.length) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;
        ctx.clearRect(0, 0, width, height);

        const barWidth = width / waveformData.length;
        const centerY = height / 2;
        const duration = 30;

        waveformData.forEach((value, index) => {
            const x = index * barWidth;
            const barHeight = value * height * 0.9;
            const isCurrent = currentTime > 0 && index / waveformData.length > (currentTime % duration) / duration;

            const gradient = ctx.createLinearGradient(0, centerY - barHeight / 2, 0, centerY + barHeight / 2);
            if (isCurrent) {
                gradient.addColorStop(0, '#667eea');
                gradient.addColorStop(1, '#764ba2');
            } else {
                gradient.addColorStop(0, '#4a90d9');
                gradient.addColorStop(1, '#357abd');
            }
            ctx.fillStyle = gradient;
            ctx.fillRect(x, centerY - barHeight / 2, Math.max(1, barWidth - 1), barHeight);
        });

        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();
        ctx.setLineDash([]);
    }, [waveformData, currentTime]);

    return (
        <div className="audio-visualizer">
            <canvas ref={canvasRef} width={600} height={120} className="waveform-canvas" />
        </div>
    );
};

//=============================================================================
// Emotion Results
//=============================================================================
const EmotionResults = ({ result }) => {
    if (!result) return null;

    const { emotions, features } = getEmotionEntries(result.additional_probs);
    const burnout = getBurnoutAnalysis(result);

    return (
        <div className="result-card">
            <MainEmotionDisplay mainPrediction={result.main_prediction} />
            <div className="emotion-display">
                {Object.entries(emotions).map(([key, value]) => (
                    <EmotionBar key={key} emotion={key} probability={value} />
                ))}
                {result.gigachat && <GigaChatAnalysis gigachatData={result.gigachat} />}
            </div>

            {Object.keys(features).length > 0 && (
                <div className="additional-features">
                    <h4>{t('additional_features', localStorage.getItem('language'))}</h4>
                    {Object.entries(features).map(([key, value]) => (
                        <FeatureBar key={key} label={key} value={value} />
                    ))}
                </div>
            )}

            {burnout && (
                <div className="burnout-section">
                    <BurnoutAnalysis data={burnout} />
                </div>
            )}

            {result.burnout_error && (
                <div className="burnout-error" style={{ marginTop: '15px' }}>
                    <strong>⚠️ Burnout Analysis Note:</strong> {result.burnout_error}
                </div>
            )}
        </div>
    );
};

const AudioEmotionDisplay = ({ audioResult }) => {
    if (!audioResult) return null;
    const resultData = audioResult.result || audioResult;
    return (
        <div className="audio-result-container">
            <div className="audio-emotion-results">
                <EmotionResults result={resultData} />
            </div>
        </div>
    );
};

//=============================================================================
// Result Renderers
//=============================================================================
const ImageResults = ({ results }) => (
    <>
        {results.image_url && (
            <div className="preview-container processed-image">
                <img src={results.image_url} loading="lazy" className="processed-image" alt="Processed image" />
            </div>
        )}
        <EmotionResults result={results.result} />
    </>
);

const VideoResults = ({ results }) => (
    <>
        <div className="result-card">
            <h3>{t('video_analysis_complete', localStorage.getItem('language'))}</h3>
            <p>{t('processed_frames', localStorage.getItem('language'))}: {results.frames_processed}</p>
        </div>
        <div className="results-container video-results">
            {results.results?.map((frame, index) => (
                <div key={index} className="frame-result">
                    <h4>{t('frame', localStorage.getItem('language'))} {frame.frame + 1}</h4>
                    {frame.image_url && (
                        <img src={frame.image_url} alt={`Frame ${frame.frame + 1}`} loading="lazy" className="processed-image" />
                    )}
                    <EmotionResults result={frame.result} />
                </div>
            ))}
        </div>
    </>
);

const RealtimeResults = ({ results }) => {
    const [timeRange, setTimeRange] = useState([0, 1]);
    const lang = localStorage.getItem('language');

    const getFilteredData = useCallback(() => {
        if (!results?.frame_results) return { valence: [], arousal: [], labels: [], frames: [] };
        const totalDuration = results.duration || 1;
        const start = timeRange[0] * totalDuration;
        const end = timeRange[1] * totalDuration;
        const filtered = results.frame_results.filter(f => f.timestamp >= start && f.timestamp <= end);
        return {
            valence: filtered.map(f => f.valence || 0),
            arousal: filtered.map(f => f.arousal || 0),
            labels: filtered.map(f => `${f.timestamp.toFixed(1)}s`),
            frames: filtered
        };
    }, [results, timeRange]);

    const data = getFilteredData();

    return (
        <>
            <div className="result-card">
                <h3>Real-time Video Analysis Complete</h3>
                <p>Analyzed {results.frames_processed} frames over {results.duration?.toFixed(1) || 0} seconds</p>
                {results.statistics && (
                    <div className="statistics-summary">
                        <h4>Summary Statistics</h4>
                        <div className="stats-grid">
                            {['valence', 'arousal'].map(metric => (
                                <React.Fragment key={metric}>
                                    <div className="stat-item">
                                        <label>{metric.charAt(0).toUpperCase() + metric.slice(1)} Range:</label>
                                        <span>{results.statistics[`${metric}_min`]?.toFixed(2)} to {results.statistics[`${metric}_max`]?.toFixed(2)}</span>
                                    </div>
                                    <div className="stat-item">
                                        <label>Average {metric}:</label>
                                        <span>{results.statistics[`${metric}_avg`]?.toFixed(2)}</span>
                                    </div>
                                </React.Fragment>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {results.duration > 5 && (
                <div className="time-range-selector">
                    <h4>Select Time Range</h4>
                    <div className="range-inputs">
                        {[0, 1].map((idx) => (
                            <input
                                key={idx}
                                type="range"
                                min="0"
                                max="1"
                                step="0.01"
                                value={timeRange[idx]}
                                onChange={(e) => {
                                    const newRange = [...timeRange];
                                    newRange[idx] = parseFloat(e.target.value);
                                    if (newRange[0] > newRange[1]) newRange[0] = newRange[1];
                                    setTimeRange(newRange);
                                }}
                            />
                        ))}
                    </div>
                    <div className="time-labels">
                        <span>0s</span>
                        <span>{(results.duration * timeRange[0]).toFixed(1)}s - {(results.duration * timeRange[1]).toFixed(1)}s</span>
                        <span>{results.duration.toFixed(1)}s</span>
                    </div>
                </div>
            )}

            <div className="charts-grid-realtime">
                {[
                    { data: data.valence, title: 'Valence Trend', color: '#28a745' },
                    { data: data.arousal, title: 'Arousal Trend', color: '#17a2b8' }
                ].map((chart, idx) => (
                    <LineChart key={idx} data={chart.data} labels={data.labels} title={chart.title} color={chart.color} height={180} />
                ))}
                <EmotionDistributionChart emotions={results.average_emotions} height={250} />
                <TimelineChart frameResults={data.frames} height={180} />
            </div>

            <div className="sample-frames">
                <h4>Sample Frame Analysis</h4>
                <div className="frames-grid">
                    {results.frame_results?.slice(0, 4).map((frame, index) => (
                        <SampleFrame key={index} frame={frame} />
                    ))}
                </div>
            </div>
        </>
    );
};

const SampleFrame = ({ frame }) => {
    const lang = localStorage.getItem('language');

    return (
        <div className="sample-frame">
            <div className="frame-header"><strong>Time: {frame.timestamp.toFixed(1)}s</strong></div>
            {frame.image_url && (
                <div className="frame-image-container">
                    <img src={frame.image_url} alt={`Frame at ${frame.timestamp.toFixed(1)} seconds`} loading="lazy" className="frame-image" />
                </div>
            )}
            <div className="frame-results">
                {frame.result && (
                    <>
                        <MainEmotionDisplay mainPrediction={frame.result.main_prediction} />
                        <div className="emotion-display">
                            {Object.entries(getEmotionEntries(frame.result.additional_probs).emotions).map(([key, value]) => (
                                <EmotionBar key={key} emotion={key} probability={value} />
                            ))}
                        </div>
                        {frame.result.burnout_analysis && (
                            <div style={{ marginTop: '12px' }}>
                                <BurnoutLevelBadge level={frame.result.burnout_analysis.level} />
                            </div>
                        )}
                        {(frame.valence !== undefined || frame.arousal !== undefined) && (
                            <div className="valence-arousal">
                                {frame.valence !== undefined && (
                                    <div className="va-item"><span>Valence: </span><span>{frame.valence.toFixed(2)}</span></div>
                                )}
                                {frame.arousal !== undefined && (
                                    <div className="va-item"><span>Arousal: </span><span>{frame.arousal.toFixed(2)}</span></div>
                                )}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

const AudioResults = ({ results }) => (
    <>
        <div className="result-card">
            <h3>🎵 Audio Emotion Analysis</h3>
            <p>Duration: {(results.duration || 0).toFixed(1)} seconds | Sample Rate: {results.sample_rate || 'N/A'} Hz</p>
        </div>
        <AudioEmotionDisplay audioResult={results} />
    </>
);

const AudioBurnoutResults = ({ results }) => {
    const burnout = getBurnoutAnalysis(results);
    return (
        <>
            <div className="result-card audio-burnout">
                <h3>🎵 Audio Burnout Analysis</h3>
                <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', fontSize: '0.9rem', color: '#6c757d' }}>
                    <span>Duration: {(results.duration || 0).toFixed(1)} seconds</span>
                    <span>Sample Rate: {results.sample_rate || 'N/A'} Hz</span>
                    {burnout && <span style={{ backgroundColor: '#e9ecef', padding: '2px 10px', borderRadius: '12px' }}>{burnout.state || 'N/A'}</span>}
                </div>
            </div>
            <AudioEmotionDisplay audioResult={results} />
        </>
    );
};

const DefaultResults = ({ results }) => (
    <div className="result-card">
        <h3>Analysis Results</h3>
        <pre>{JSON.stringify(results, null, 2)}</pre>
    </div>
);

//=============================================================================
// Main Detector Component
//=============================================================================
const Detector = () => {
    const [state, setState] = useState({
        file: null,
        fileName: '',
        fileSize: '',
        preview: null,
        results: null,
        consentGiven: false,
        errorKey: '',
        isProcessing: false,
        showProgress: false,
        progressText: '',
        progressComplete: false,
        currentTaskId: null,
        processingMode: 'standard'
    });

    const fileInputRef = useRef(null);
    const progressIntervalRef = useRef(null);
    const { language, updateTexts } = useLanguage();

    useEffect(() => {
        updateTexts();
        return () => {
            if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
        };
    }, [language]);

    const setField = (key, value) => setState(prev => ({ ...prev, [key]: value }));
    const getField = (key) => state[key];

    const showError = (key) => {
        setField('errorKey', key);
        setTimeout(() => setField('errorKey', ''), 3000);
    };

    const hideProgress = () => {
        setField('showProgress', false);
        setField('currentTaskId', null);
        setField('isProcessing', false);
        setField('progressComplete', false);
    };

    const validateFile = (file) => {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!VALID_TYPES.includes(file.type) && !VALID_EXTENSIONS.includes(ext)) {
            showError('error_unsupported_format');
            return false;
        }
        if (file.size > MAX_FILE_SIZE) {
            showError('error_file_too_large');
            return false;
        }
        return true;
    };

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (!file || !validateFile(file)) return;
        setState(prev => ({
            ...prev,
            file,
            fileName: file.name,
            fileSize: formatFileSize(file.size),
            preview: file.type.startsWith('audio/') ? { type: 'audio', url: URL.createObjectURL(file) } : null,
            results: null
        }));
    };

    const clearFile = () => {
        setState(prev => ({ ...prev, file: null, fileName: '', fileSize: '', preview: null, results: null }));
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('highlight');
        const file = e.dataTransfer.files[0];
        if (file && validateFile(file)) {
            setState(prev => ({
                ...prev,
                file,
                fileName: file.name,
                fileSize: formatFileSize(file.size),
                preview: file.type.startsWith('audio/') ? { type: 'audio', url: URL.createObjectURL(file) } : null,
                results: null
            }));
        }
    };

    const uploadFile = async () => {
        if (!getField('file')) { showError('error_file_not_selected'); return; }
        if (!getField('consentGiven')) { showError('error_consent_required'); return; }

        setState(prev => ({
            ...prev,
            results: null,
            showProgress: true,
            progressText: t('starting_processing', localStorage.getItem('language')),
            isProcessing: true,
            errorKey: '',
            progressComplete: false
        }));

        const formData = new FormData();
        formData.append('file', getField('file'));
        formData.append('model', 'emotieff');

        const mode = getField('processingMode');
        const endpoint = mode === 'burnout' ? '/api/upload_burnout' :
                         mode === 'realtime' ? '/api/upload_realtime' : '/api/upload';

        try {
            const response = await fetch(endpoint, { method: 'POST', body: formData });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'error_network_response');
            }
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            setField('currentTaskId', data.task_id);
            setField('progressText', t('file_processing', localStorage.getItem('language')));
            checkProgress(data.task_id);
        } catch (error) {
            console.error('Upload error:', error);
            showError(error.message);
            hideProgress();
        }
    };

    const checkProgress = (taskId) => {
        if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);

        progressIntervalRef.current = setInterval(async () => {
            try {
                const response = await fetch(`/api/progress/${taskId}`);
                if (!response.ok) throw new Error('error_checking_progress');
                const data = await response.json();
                if (data.error) throw new Error(data.error);

                if (data.message) {
                    let text = data.message;
                    const match = data.message.match(/(frame|segment) (\d+) of (\d+)/);
                    if (match) {
                        const unit = match[1] === 'frame' ? 'processing_frame' : 'processing_segment';
                        text = t(unit, localStorage.getItem('language'))
                            .replace('{current}', match[2])
                            .replace('{total}', match[3]);
                    } else if (data.message.includes('audio')) {
                        text = t('processing_audio', localStorage.getItem('language')) || 'Processing audio...';
                    } else {
                        text = t(data.message, localStorage.getItem('language')) || data.message;
                    }
                    setField('progressText', text);
                }

                if (data.complete) {
                    clearInterval(progressIntervalRef.current);
                    setField('progressText', t('processing_complete', localStorage.getItem('language')));
                    setField('progressComplete', true);
                    setTimeout(() => {
                        hideProgress();
                        setField('results', data);
                    }, 1000);
                }
            } catch (error) {
                console.error('Progress error:', error);
                clearInterval(progressIntervalRef.current);
                showError(error.message);
                hideProgress();
            }
        }, 1000);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('highlight');
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('highlight');
    };

    const renderResults = useMemo(() => {
        const results = getField('results');
        if (!results) return null;

        if (results.type === 'audio_burnout') return <AudioBurnoutResults results={results} />;
        if (results.result && getBurnoutAnalysis(results.result)) {
            return (
                <>
                    <div className="result-card">
                        <h3>🎵 Audio Burnout Analysis</h3>
                        <p>Duration: {(results.duration || 0).toFixed(1)} seconds | Sample Rate: {results.sample_rate || 'N/A'} Hz</p>
                    </div>
                    <AudioEmotionDisplay audioResult={results} />
                </>
            );
        }

        switch (results.type) {
            case 'image': return <ImageResults results={results} />;
            case 'video': return <VideoResults results={results} />;
            case 'video_realtime': return <RealtimeResults results={results} />;
            case 'audio': return <AudioResults results={results} />;
            default: return <DefaultResults results={results} />;
        }
    }, [state.results]);

    return (
        <div className="detector-container">
            <div style={{ textAlign: 'center' }}>
                <h1>{t('detector_title', localStorage.getItem('language'))}</h1>
            </div>

            <div className="upload-section">
                <div className="upload-container" onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
                    <div className="upload-icon">📁</div>
                    <div className="upload-text">
                        <h3>{t('drag_file', localStorage.getItem('language'))}</h3>
                        <p>{t('or', localStorage.getItem('language'))}</p>
                    </div>
                    <input type="file" ref={fileInputRef} className="file-input" accept="image/*,video/*,audio/*" onChange={handleFileSelect} disabled={getField('isProcessing')} />
                    <button className="btn primary" onClick={() => fileInputRef.current?.click()} disabled={getField('isProcessing')}>
                        {t('choose_file', localStorage.getItem('language'))}
                    </button>
                    <p className="supported-formats">{t('supported_formats', localStorage.getItem('language'))}</p>
                </div>

                {getField('errorKey') && (
                    <div className="error-message">{t(getField('errorKey'), localStorage.getItem('language'))}</div>
                )}

                {getField('preview')?.type === 'audio' && (
                    <div className="preview-container audio-preview">
                        <audio controls src={getField('preview').url} className="audio-player" />
                    </div>
                )}

                {getField('fileName') && (
                    <div className="file-info">
                        <div>
                            <span>{t('selected_file', localStorage.getItem('language'))}</span>
                            <strong>{getField('fileName')}</strong>
                            <br />({getField('fileSize')})
                        </div>
                        <button className="btn" style={{ marginTop: '8px', backgroundColor: '#dc3545' }} onClick={clearFile} disabled={getField('isProcessing')}>
                            {t('clear_button', localStorage.getItem('language'))}
                        </button>
                    </div>
                )}

                <div style={{ margin: '15px 0', textAlign: 'center' }}>
                    <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <span style={{ marginRight: '10px' }}>{t('processing_mode', localStorage.getItem('language'))}</span>
                        <select
                            value={getField('processingMode')}
                            onChange={(e) => setField('processingMode', e.target.value)}
                            disabled={getField('isProcessing')}
                            style={{ padding: '5px', borderRadius: '4px', border: '1px solid #ced4da' }}
                        >
                            <option value="standard">Standard Analysis</option>
                            <option value="burnout">Burnout Analysis</option>
                            <option value="realtime">Real-time Analysis</option>
                        </select>
                    </label>
                </div>

                <div style={{ margin: '15px 0', textAlign: 'center' }}>
                    <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <input type="checkbox" checked={getField('consentGiven')} onChange={(e) => setField('consentGiven', e.target.checked)} disabled={getField('isProcessing')} style={{ marginRight: '8px' }} />
                        <span>
                            {t('consent_text', localStorage.getItem('language'))}
                            <a href="#privacy" className="nav-link" style={{ color: '#495057' }}>{t('privacy_policy', localStorage.getItem('language'))}</a>
                        </span>
                    </label>
                </div>

                <button
                    className="btn primary"
                    onClick={uploadFile}
                    disabled={getField('isProcessing')}
                >
                    {getField('processingMode') === 'burnout' ? 'Analyze Burnout' : t('analyze_emotions', localStorage.getItem('language'))}
                </button>
            </div>

            {getField('showProgress') && (
                <div className="progress-container">
                    <div className="progress-header"><h3>{t('processing', localStorage.getItem('language'))}</h3></div>
                    <div className={`progress-wheel ${getField('progressComplete') ? 'complete' : ''}`} />
                    <div className="progress-text">{getField('progressText')}</div>
                </div>
            )}

            {getField('results') && <div className="results-container">{renderResults}</div>}
        </div>
    );
};

export default Detector;