export const EMOTION_COLORS = {
  anger: '#ea4335',
  contempt: '#9E9E9E',
  disgust: '#9C27B0',
  fear: '#4285f4',
  neutral: '#fbbc05',
  happiness: '#34a853',
  surprise: '#673ab7',
  sadness: '#2196F3',
};

export const getEmotionColor = (emotion) => {
  return EMOTION_COLORS[emotion] || '#3f4857';
};

export const BURNOUT_COLORS = {
  low: '#28a745',
  moderate: '#ffc107',
  high: '#fd7e14',
  severe: '#dc3545',
};

export const BURNOUT_STATE_COLORS = {
  NORMAL: '#28a745',
  SHORT_STRESS: '#ffc107',
  SUSTAINED_STRESS: '#fd7e14',
  BURNOUT_LIKE: '#dc3545',
  LOW_AFFECT_UNSPECIFIC: '#6c757d',
  INSUFFICIENT_DATA: '#6c757d',
};

export const BURNOUT_STATE_MAP = {
  NORMAL: 'normal',
  SHORT_STRESS: 'shortStress',
  SUSTAINED_STRESS: 'sustainedStress',
  BURNOUT_LIKE: 'burnoutLike',
  LOW_AFFECT_UNSPECIFIC: 'lowAffect',
  INSUFFICIENT_DATA: 'insufficientData',
};

export const BURNOUT_LEVEL_CLASSES = {
  low: 'level-low',
  moderate: 'level-moderate',
  high: 'level-high',
  severe: 'level-severe',
};

export const BURNOUT_LABELS = {
  low: 'Low Risk',
  moderate: 'Moderate Risk',
  high: 'High Risk',
  severe: 'Severe Risk',
};

export const FILE_CONSTANTS = {
  MAX_SIZE: 50 * 1024 * 1024,
  VALID_TYPES: [
    'image/jpeg',
    'image/png',
    'image/jpg',
    'video/mp4',
    'video/avi',
    'video/webm',
    'video/x-msvideo',
    'audio/mpeg',
    'audio/mp3',
    'audio/wav',
    'audio/x-wav',
    'audio/aac',
    'audio/ogg',
    'audio/flac',
    'audio/m4a',
    'audio/webm',
  ],
  VALID_EXTENSIONS: [
    'jpg', 'jpeg', 'png',
    'mp4', 'avi', 'webm',
    'mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a',
  ],
};

export const PROCESSING_MODES = [
  { value: 'external_influence', labelKey: 'detector.modes.externalInfluence' },
  { value: 'burnout', labelKey: 'detector.modes.burnout' },
  { value: 'standard', labelKey: 'detector.modes.standard' },
  { value: 'realtime', labelKey: 'detector.modes.realtime' },
];

export const getProcessingModeLabel = (value, t) => {
  const mode = PROCESSING_MODES.find((m) => m.value === value);
  return mode ? t(mode.labelKey) : value;
};

export const getBurnoutStateColor = (state) => {
  return BURNOUT_STATE_COLORS[state] || '#6c757d';
};

export const getBurnoutStateKey = (state) => {
  return BURNOUT_STATE_MAP[state] || 'insufficientData';
};

export const getBurnoutLevelClass = (level) => {
  return BURNOUT_LEVEL_CLASSES[level] || 'level-low';
};

export const getBurnoutLevelColor = (level) => {
  const colorMap = {
    low: '#28a745',
    moderate: '#ffc107',
    high: '#fd7e14',
    severe: '#dc3545',
  };
  return colorMap[level] || '#28a745';
};

// External influence (scam signal) status ladder - codes must match the
// backend ExternalInfluenceStatus enum exactly.
export const EXTERNAL_INFLUENCE_STATUS_COLORS = {
  INSUFFICIENT_DATA: '#6c757d',
  LOW: '#28a745',
  ELEVATED_TENSION: '#ffc107',
  POSSIBLE_EXTERNAL_PRESSURE: '#fd7e14',
  PROBABLE_EXTERNAL_INFLUENCE: '#dc3545',
  HIGH_EXTERNAL_INFLUENCE_RISK: '#8b0000',
};

export const EXTERNAL_INFLUENCE_STATE_MAP = {
  INSUFFICIENT_DATA: 'insufficientData',
  LOW: 'low',
  ELEVATED_TENSION: 'elevatedTension',
  POSSIBLE_EXTERNAL_PRESSURE: 'possiblePressure',
  PROBABLE_EXTERNAL_INFLUENCE: 'probableInfluence',
  HIGH_EXTERNAL_INFLUENCE_RISK: 'highRisk',
};

export const getExternalInfluenceStatusColor = (status) => {
  return EXTERNAL_INFLUENCE_STATUS_COLORS[status] || '#6c757d';
};

export const getExternalInfluenceStatusKey = (status) => {
  return EXTERNAL_INFLUENCE_STATE_MAP[status] || 'insufficientData';
};