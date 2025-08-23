export const getColorForEmotion = (emotionKey) => {
	const colors = {
		happiness: '#34a853',
		neutral: '#fbbc05',
		sadness: '#2196F3',
		disgust: '#9C27B0',
		fear: '#4285f4',
		surprise: '#673ab7',
		anger: '#ea4335'
	};
	return colors[emotionKey] || '#3f4857ff';
};