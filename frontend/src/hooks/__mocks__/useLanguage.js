const mockTranslations = {
  'emotions.anger': 'Anger',
  'emotions.contempt': 'Contempt',
  'emotions.disgust': 'Disgust',
  'emotions.fear': 'Fear',
  'emotions.happiness': 'Happiness',
  'emotions.neutral': 'Neutral',
  'emotions.sadness': 'Sadness',
  'emotions.surprise': 'Surprise',
  'detector.emotionTimeline': 'Emotion Timeline',
  'detector.frame': 'Frame',
  'detector.frames': 'Frames',
  'detector.averageEmotions': 'Average Emotions',
  'detector.detectedEmotions': 'Detected Emotions',
  'detector.imageAnalysis': 'Image Analysis',
  'detector.audioAnalysis': 'Audio Analysis',
  'detector.mainEmotion': 'Main Emotion',
  'detector.framesProcessed': 'Processed frames: {count}',
  'detector.features.title': 'Valence / Arousal',
  'detector.features.valence': 'Valence',
  'detector.features.arousal': 'Arousal',
  'detector.noImage': 'No image',
  'common.noData': 'No data',
};

export const useLanguage = () => {
  const t = (key, params) => {
    let str = mockTranslations[key] || key;
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        str = str.replace(`{${k}}`, v);
      });
    }
    return str;
  };
  return { t, language: 'en', setLanguage: () => {} };
};

export default useLanguage;
