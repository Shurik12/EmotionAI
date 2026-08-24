export const getEmotionEntries = (additionalProbs) => {
  if (!additionalProbs) return { emotions: {}, features: {} };
  
  const emotions = {};
  const features = {};
  
  Object.entries(additionalProbs).forEach(([key, value]) => {
    // Handle string values that might be numbers
    const numValue = parseFloat(value);
    if (['valence', 'arousal'].includes(key)) {
      features[key] = numValue;
    } else {
      emotions[key] = numValue;
    }
  });
  
  return { emotions, features };
};

export const getBurnoutAnalysis = (data) => {
  if (!data) return null;
  
  // Check multiple possible locations for burnout data
  if (data.burnout_analysis) {
    return data.burnout_analysis;
  }
  if (data.result && data.result.burnout_analysis) {
    return data.result.burnout_analysis;
  }
  if (data.burnout) {
    return data.burnout;
  }
  
  return null;
};

export const getVerdictInfo = (verdict) => {
  const map = {
    low: { text: 'Low Risk', class: 'verdict-low', icon: '✅' },
    monitor: { text: 'Monitor', class: 'verdict-monitor', icon: '⚠️' },
    high: { text: 'High Risk', class: 'verdict-high', icon: '🔴' },
  };
  return map[verdict] || { text: verdict, class: '', icon: '📊' };
};

export const debounce = (fn, delay) => {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
};

export const generateId = () => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};