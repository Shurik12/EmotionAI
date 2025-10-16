import React, { useState, useRef, useEffect } from 'react';
import { getColorForEmotion } from '../utils/emotionColors';
import { useLanguage } from '../hooks/useLanguage';
import { t } from '../utils/translations';
import './Detector.css';

// Chart components
const LineChart = ({ data, labels, title, color, height = 200 }) => {
	if (!data || data.length === 0) {
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
				<div className="chart-line" style={{ borderColor: color }}></div>
				<div className="chart-labels">
					<span>{minVal.toFixed(1)}</span>
					<span>{maxVal.toFixed(1)}</span>
				</div>
			</div>
		</div>
	);
};

const EmotionDistributionChart = ({ emotions, height = 250 }) => {
	if (!emotions || Object.keys(emotions).length === 0) {
		return (
			<div className="chart-container">
				<h4>Average Emotion Distribution</h4>
				<div className="no-data">No emotion data available</div>
			</div>
		);
	}

	const emotionEntries = Object.entries(emotions).sort((a, b) => b[1] - a[1]);

	return (
		<div className="chart-container">
			<h4>Average Emotion Distribution</h4>
			<div className="distribution-chart" style={{ height: `${height}px` }}>
				{emotionEntries.map(([emotion, value]) => (
					<div key={emotion} className="distribution-item">
						<div className="emotion-label">
							<span>{t(emotion, localStorage.getItem('language'))}</span>
							<span>{(value * 100).toFixed(1)}%</span>
						</div>
						<div className="distribution-bar">
							<div
								className="distribution-fill"
								style={{
									width: `${value * 100}%`,
									backgroundColor: getColorForEmotion(emotion)
								}}
							/>
						</div>
					</div>
				))}
			</div>
		</div>
	);
};

const TimelineChart = ({ frameResults, height = 200 }) => {
	if (!frameResults || frameResults.length === 0) {
		return (
			<div className="chart-container">
				<h4>Emotion Timeline</h4>
				<div className="no-data">No timeline data available</div>
			</div>
		);
	}

	const [selectedFrame, setSelectedFrame] = useState(null);

	return (
		<div className="chart-container">
			<h4>Emotion Timeline</h4>
			<div className="timeline-chart" style={{ height: `${height}px` }}>
				{frameResults.map((frame, index) => {
					const mainEmotion = frame.result?.main_prediction?.label;
					const probability = frame.result?.main_prediction?.probability || 0;

					return (
						<div
							key={index}
							className={`timeline-point ${selectedFrame === index ? 'selected' : ''}`}
							style={{
								left: `${(frame.timestamp / (frameResults[frameResults.length - 1]?.timestamp || 1)) * 100}%`,
								backgroundColor: getColorForEmotion(mainEmotion),
								opacity: 0.5 + (probability * 0.5)
							}}
							onMouseEnter={() => setSelectedFrame(index)}
							onMouseLeave={() => setSelectedFrame(null)}
							title={`${frame.timestamp.toFixed(1)}s: ${mainEmotion} (${(probability * 100).toFixed(1)}%)`}
						/>
					);
				})}
				{selectedFrame !== null && (
					<div
						className="timeline-tooltip"
						style={{
							left: `${(frameResults[selectedFrame].timestamp / (frameResults[frameResults.length - 1]?.timestamp || 1)) * 100}%`
						}}
					>
						<strong>Time: {frameResults[selectedFrame].timestamp.toFixed(1)}s</strong>
						<br />
						Main emotion: {t(frameResults[selectedFrame].result?.main_prediction?.label, localStorage.getItem('language'))}
						<br />
						Confidence: {(frameResults[selectedFrame].result?.main_prediction?.probability * 100).toFixed(1)}%
					</div>
				)}
			</div>
		</div>
	);
};

const SampleFrame = ({ frame, index }) => {
	return (
		<div key={index} className="sample-frame">
			<div className="frame-header">
				<strong>Time: {frame.timestamp.toFixed(1)}s</strong>
			</div>
			{frame.image_url && (
				<div className="frame-image-container">
					<img
						src={frame.image_url}
						alt={`Frame at ${frame.timestamp.toFixed(1)} seconds`}
						loading="lazy"
						className="frame-image"
					/>
				</div>
			)}
			<div className="frame-results">
				{frame.result && (
					<>
						<div className="main-emotion">
							<span>{t(frame.result.main_prediction?.label, localStorage.getItem('language'))}</span>
							({(frame.result.main_prediction?.probability * 100).toFixed(1)}%)
						</div>
						<div className="emotion-display">
							{frame.result.additional_probs && Object.entries(frame.result.additional_probs)
								.filter(([key]) => key !== 'valence' && key !== 'arousal')
								.map(([emotionKey, prob]) => {
									const percentage = (parseFloat(prob) * 100).toFixed(1);
									return (
										<div key={emotionKey} className="emotion-item">
											<div className="emotion-label">
												<span>{t(emotionKey, localStorage.getItem('language'))}</span>
												<span>{percentage}%</span>
											</div>
											<div className="emotion-bar">
												<div
													className="emotion-fill"
													style={{
														width: `${percentage}%`,
														backgroundColor: getColorForEmotion(emotionKey)
													}}
												></div>
											</div>
										</div>
									);
								})}
						</div>
						{(frame.valence !== undefined || frame.arousal !== undefined) && (
							<div className="valence-arousal">
								{frame.valence !== undefined && (
									<div className="va-item">
										<span>Valence: </span>
										<span>{frame.valence.toFixed(2)}</span>
									</div>
								)}
								{frame.arousal !== undefined && (
									<div className="va-item">
										<span>Arousal: </span>
										<span>{frame.arousal.toFixed(2)}</span>
									</div>
								)}
							</div>
						)}
					</>
				)}
			</div>
		</div>
	);
};

const Detector = () => {
	const [currentFile, setCurrentFile] = useState(null);
	const [fileInfo, setFileInfo] = useState('');
	const [showProgress, setShowProgress] = useState(false);
	const [progressText, setProgressText] = useState('');
	const [preview, setPreview] = useState(null);
	const [results, setResults] = useState(null);
	const [consentGiven, setConsentGiven] = useState(false);
	const [errorKey, setErrorKey] = useState('');
	const [isProcessing, setIsProcessing] = useState(false);
	const [progressComplete, setProgressComplete] = useState(false);
	const [currentTaskId, setCurrentTaskId] = useState(null);
	const [processingMode, setProcessingMode] = useState('standard'); // 'standard' or 'realtime'
	const [selectedTimeRange, setSelectedTimeRange] = useState([0, 1]);

	const fileInputRef = useRef(null);
	const progressIntervalRef = useRef(null);
	const audioRef = useRef(null);

	const { language, updateTexts } = useLanguage();

	useEffect(() => {
		updateTexts();
	}, [language]);

	useEffect(() => {
		return () => {
			// Clean up interval on component unmount
			if (progressIntervalRef.current) {
				clearInterval(progressIntervalRef.current);
			}
		};
	}, []);

	const showError = (translationKey) => {
		setErrorKey(translationKey);
		setTimeout(() => {
			setErrorKey('');
		}, 3000);
	};

	const [fileName, setFileName] = useState('');
	const [fileSize, setFileSize] = useState('');

	const handleFileSelect = (e) => {
		const file = e.target.files[0];
		if (!file) return;

		if (!validateFile(file)) {
			return;
		}

		setCurrentFile(file);
		setFileName(file.name);
		setFileSize(formatFileSize(file.size));
		setPreview(null);
		setResults(null);
		setSelectedTimeRange([0, 1]);

		// Create preview for audio files
		if (file.type.startsWith('audio/')) {
			const audioUrl = URL.createObjectURL(file);
			setPreview({
				type: 'audio',
				url: audioUrl
			});
		}
	};

	const validateFile = (file) => {
		const validTypes = [
			'image/jpeg', 'image/png', 'image/jpg',
			'video/mp4', 'video/avi', 'video/webm',
			'audio/mpeg', 'audio/mp3', 'audio/aac', 'audio/ogg', 'audio/wav'
		];
		const maxSize = 50 * 1024 * 1024;

		if (!validTypes.includes(file.type)) {
			showError('error_unsupported_format');
			return false;
		}

		if (file.size > maxSize) {
			showError('error_file_too_large');
			return false;
		}

		return true;
	};

	const formatFileSize = (bytes) => {
		if (bytes === 0) return '0 Bytes';

		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));

		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	};

	const clearFile = () => {
		setCurrentFile(null);
		setFileName('');
		setFileSize('');
		setPreview(null);
		setResults(null);
		setSelectedTimeRange([0, 1]);
		if (fileInputRef.current) {
			fileInputRef.current.value = '';
		}
	};

	const uploadFile = async () => {
		console.log('uploadFile called, currentFile:', currentFile);
		console.log('consentGiven:', consentGiven);
		console.log('processingMode:', processingMode);

		if (!currentFile) {
			showError('error_file_not_selected');
			return;
		}

		if (!consentGiven) {
			showError('error_consent_required');
			return;
		}

		console.log('Starting upload process...');

		// Reset UI state
		setResults(null);
		setShowProgress(true);
		setProgressText(t('starting_processing', localStorage.getItem('language')));
		setIsProcessing(true);
		setErrorKey('');
		setProgressComplete(false);

		console.log('Progress state set to visible');

		const formData = new FormData();
		formData.append('file', currentFile);
		formData.append('model', "emotieff");

		const endpoint = processingMode === 'realtime' ? '/api/upload_realtime' : '/api/upload';

		try {
			const response = await fetch(endpoint, {
				method: 'POST',
				body: formData
			});

			if (!response.ok) {
				const err = await response.json();
				throw new Error(err.error || 'error_network_response');
			}

			const data = await response.json();

			if (data.error) {
				throw new Error(data.error);
			}

			// Store for language switching
			window.lastAnalysisResult = data.result;

			setCurrentTaskId(data.task_id);
			setProgressText(t('file_processing', localStorage.getItem('language')));

			checkProgress(data.task_id);
		} catch (error) {
			console.error('Error checking progress:', error);
			showError(error.message);
			hideProgress();
		}
	};

	const checkProgress = (taskId) => {
		// Clear any existing interval
		if (progressIntervalRef.current) {
			clearInterval(progressIntervalRef.current);
		}

		progressIntervalRef.current = setInterval(async () => {
			try {
				const response = await fetch(`/api/progress/${taskId}`);

				if (!response.ok) {
					throw new Error('error_checking_progress');
				}

				const data = await response.json();

				if (data.error) {
					throw new Error(data.error);
				}

				if (data.message) {
					// Handle frame/segment processing messages
					let newProgressText = '';
					if (data.message.includes("frame") || data.message.includes("segment")) {
						const frameMatch = data.message.match(/(frame|segment) (\d+) of (\d+)/);
						if (frameMatch) {
							const unit = frameMatch[1] === 'frame' ? 'processing_frame' : 'processing_segment';
							newProgressText = t(unit, localStorage.getItem('language'))
								.replace('{current}', frameMatch[2])
								.replace('{total}', frameMatch[3]);
						}
					} else {
						newProgressText = t(data.message, localStorage.getItem('language')) || data.message;
					}

					setProgressText(newProgressText);
				}

				if (data.complete) {
					clearInterval(progressIntervalRef.current);
					setProgressText(t('processing_complete', localStorage.getItem('language')));
					setProgressComplete(true);

					setTimeout(() => {
						hideProgress();
						displayResults(data);
					}, 1000);
				}
			} catch (error) {
				console.error('Error checking progress:', error);
				clearInterval(progressIntervalRef.current);
				showError(error.message);
				hideProgress();
			}
		}, 1000);
	};

	const hideProgress = () => {
		setShowProgress(false);
		setCurrentTaskId(null);
		setIsProcessing(false);
		setProgressComplete(false);
	};

	const displayResults = (data) => {
		console.log('API Response:', data);
		setResults(data);
		setPreview(null);
	};

	const displayEmotionResult = (result) => {
		if (!result) return null;

		const mainPred = result.main_prediction || {};
		const additional = result.additional_probs || {};

		// Separate emotions from valence/arousal
		const emotions = {};
		const additionalFeatures = {};

		Object.entries(additional).forEach(([key, value]) => {
			if (key === 'valence' || key === 'arousal') {
				additionalFeatures[key] = value;
			} else {
				emotions[key] = value;
			}
		});

		return (
			<div className="result-card">
				<div className="main-emotion">
					<span data-i18n={mainPred.label}>{t(mainPred.label, localStorage.getItem('language'))}</span>
					({(mainPred.probability * 100).toFixed(1)}%)
				</div>
				<div className="emotion-display">
					{Object.entries(emotions).map(([emotionKey, prob]) => {
						const percentage = (parseFloat(prob) * 100).toFixed(1);
						return (
							<div key={emotionKey} className="emotion-item">
								<div className="emotion-label">
									<span data-i18n={emotionKey}>{t(emotionKey, localStorage.getItem('language'))}</span>
									<span>{percentage}%</span>
								</div>
								<div className="emotion-bar">
									<div
										className="emotion-fill"
										style={{
											width: `${percentage}%`,
											backgroundColor: getColorForEmotion(emotionKey)
										}}
									></div>
								</div>
							</div>
						);
					})}
				</div>

				{/* Display valence and arousal if they exist */}
				{Object.keys(additionalFeatures).length > 0 && (
					<div className="additional-features">
						<h4 data-i18n="additional_features">{t('additional_features', localStorage.getItem('language'))}</h4>
						{Object.entries(additionalFeatures).map(([key, value]) => (
							<div key={key} className="feature-item">
								<div className="feature-label">
									<span data-i18n={key}>{t(key, localStorage.getItem('language'))}</span>
									<span>{parseFloat(value).toFixed(2)}</span>
								</div>
								{/* Optional: Display as a different kind of bar or numerical value */}
								<div className="feature-value">
									<div
										className="feature-fill"
										style={{
											width: `${(parseFloat(value) + 1) * 50}%`, // Scale from -1 to +1 to 0-100%
											backgroundColor: key === 'valence' ? '#4CAF50' : '#2196F3'
										}}
									></div>
								</div>
							</div>
						))}
					</div>
				)}
			</div>
		);
	};

	// Filter data based on selected time range
	const getFilteredData = () => {
		if (!results?.frame_results) return { valence: [], arousal: [], labels: [], frames: [] };

		const frameResults = results.frame_results;
		const totalDuration = results.duration || 1;
		const startTime = selectedTimeRange[0] * totalDuration;
		const endTime = selectedTimeRange[1] * totalDuration;

		const filtered = frameResults.filter(frame =>
			frame.timestamp >= startTime && frame.timestamp <= endTime
		);

		return {
			valence: filtered.map(f => f.valence || 0),
			arousal: filtered.map(f => f.arousal || 0),
			labels: filtered.map(f => `${f.timestamp.toFixed(1)}s`),
			frames: filtered
		};
	};

	// Drag and drop handlers
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

	const handleDrop = (e) => {
		e.preventDefault();
		e.stopPropagation();
		e.currentTarget.classList.remove('highlight');

		const files = e.dataTransfer.files;
		if (files.length) {
			const fileInput = fileInputRef.current;
			if (fileInput) {
				fileInput.files = files;
				handleFileSelect({ target: { files } });
			}
		}
	};

	return (
		<div className="detector-container">
			<div style={{ textAlign: 'center' }} data-i18n="detected_title">
				<h1 data-i18n="detector_title"></h1>
			</div>

			<div className="upload-section">
				<div
					className="upload-container"
					id="dropArea"
					onDragOver={handleDragOver}
					onDragLeave={handleDragLeave}
					onDrop={handleDrop}
				>
					<div className="upload-icon">📁</div>
					<div className="upload-text">
						<h3 data-i18n="drag_file"></h3>
						<p data-i18n="or"></p>
					</div>
					<input
						type="file"
						id="fileInput"
						ref={fileInputRef}
						className="file-input"
						accept="image/*,video/*,audio/*"
						onChange={handleFileSelect}
						disabled={isProcessing}
					/>
					<button
						className="btn primary"
						id="selectFileBtn"
						onClick={() => fileInputRef.current?.click()}
						disabled={isProcessing}
						data-i18n="choose_file"
					>
					</button>
					<p className="supported-formats" data-i18n="supported_formats"></p>
				</div>

				{errorKey && (
					<div className="error-message">
						{t(errorKey, localStorage.getItem('language'))}
					</div>
				)}

				{preview && preview.type === 'audio' && (
					<div className="preview-container audio-preview">
						<audio ref={audioRef} controls src={preview.url} className="audio-player">
							Your browser does not support the audio element.
						</audio>
					</div>
				)}

				{fileName && (
					<div className="file-info">
						<div style={{ wordBreak: 'break-all' }}>
							<span data-i18n="selected_file"></span> <strong>{fileName}</strong>
							<br />({fileSize})
						</div>
						<button
							className="btn"
							style={{ marginTop: '8px', backgroundColor: 'var(--error-color)' }}
							onClick={clearFile}
							disabled={isProcessing}
							data-i18n="clear_button"
						>{t('clear_button', localStorage.getItem('language'))}</button>
					</div>
				)}

				{/* Processing Mode Selection */}
				<div style={{ margin: '15px 0', textAlign: 'center' }}>
					<label style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
						<span style={{ marginRight: '10px' }} data-i18n="processing_mode">Processing Mode:</span>
						<select
							value={processingMode}
							onChange={(e) => setProcessingMode(e.target.value)}
							disabled={isProcessing}
							style={{ padding: '5px', borderRadius: '4px', border: '1px solid #ced4da' }}
						>
							<option value="standard">Standard Analysis</option>
							<option value="realtime">Real-time Analysis</option>
						</select>
					</label>
				</div>

				<div style={{ margin: '15px 0', textAlign: 'center' }}>
					<label style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
						<input
							type="checkbox"
							id="dataConsent"
							required
							style={{ marginRight: '8px' }}
							checked={consentGiven}
							onChange={(e) => setConsentGiven(e.target.checked)}
							disabled={isProcessing}
						/>
						<span data-i18n="consent_text">
							<a
								href="#privacy"
								className="nav-link"
								style={{ color: 'var(--primary-color)' }}
								onClick={(e) => e.preventDefault()}
							>
								{t('privacy_policy', localStorage.getItem('language'))}
							</a>
						</span>
					</label>
				</div>

				<button
					className="btn primary"
					id="uploadBtn"
					onClick={uploadFile}
					disabled={isProcessing}
					data-i18n="analyze_emotions"
				>
					{t('analyze_emotions', localStorage.getItem('language'))}
				</button>
			</div>

			{showProgress && (
				<div className="progress-container" id="progressContainer">
					<div className="progress-header">
						<h3 data-i18n="processing"></h3>
					</div>
					<div className={`progress-wheel ${progressComplete ? 'complete' : ''}`} id="progressWheel"></div>
					<div className="progress-text" id="progressText">
						{progressText}
					</div>
				</div>
			)}

			{results && (
				<div className="results-container" id="resultsContainer">
					{results.type === 'image' ? (
						<>
							{results.image_url && (
								<div className="preview-container processed-image">
									<img
										src={results.image_url}
										loading="lazy"
										className="processed-image"
										data-i18n="processed_image_alt"
									/>
								</div>
							)}
							{displayEmotionResult(results.result)}
						</>
					) : results.type === 'video' ? (
						<>
							<div className="result-card">
								<h3 data-i18n="video_analysis_complete"></h3>
								<p data-i18n="processed_frames">{results.frames_processed}</p>
							</div>
							<div className="results-container video-results">
								{results.results && results.results.map((frame, index) => (
									<div key={index} className="frame-result">
										<h4>{t('frame', localStorage.getItem('language'))} {frame.frame + 1}</h4>
										{frame.image_url && (
											<img
												src={frame.image_url}
												alt={`${t('video_frame', localStorage.getItem('language'))} ${frame.frame + 1}`}
												loading="lazy"
												className="processed-image"
											/>
										)}
										{displayEmotionResult(frame.result)}
									</div>
								))}
							</div>
						</>
					) : results.type === 'video_realtime' ? (
						<>
							<div className="result-card">
								<h3>Real-time Video Analysis Complete</h3>
								<p>Analyzed {results.frames_processed} frames over {results.duration?.toFixed(1) || 0} seconds</p>

								{results.statistics && (
									<div className="statistics-summary">
										<h4>Summary Statistics</h4>
										<div className="stats-grid">
											<div className="stat-item">
												<label>Valence Range:</label>
												<span>{results.statistics.valence_min?.toFixed(2)} to {results.statistics.valence_max?.toFixed(2)}</span>
											</div>
											<div className="stat-item">
												<label>Arousal Range:</label>
												<span>{results.statistics.arousal_min?.toFixed(2)} to {results.statistics.arousal_max?.toFixed(2)}</span>
											</div>
											<div className="stat-item">
												<label>Average Valence:</label>
												<span>{results.statistics.valence_avg?.toFixed(2)}</span>
											</div>
											<div className="stat-item">
												<label>Average Arousal:</label>
												<span>{results.statistics.arousal_avg?.toFixed(2)}</span>
											</div>
										</div>
									</div>
								)}
							</div>

							{/* Time Range Selector */}
							{results.duration > 5 && (
								<div className="time-range-selector">
									<h4>Select Time Range</h4>
									<div className="range-inputs">
										<input
											type="range"
											min="0"
											max="1"
											step="0.01"
											value={selectedTimeRange[0]}
											onChange={(e) => setSelectedTimeRange([parseFloat(e.target.value), selectedTimeRange[1]])}
										/>
										<input
											type="range"
											min="0"
											max="1"
											step="0.01"
											value={selectedTimeRange[1]}
											onChange={(e) => setSelectedTimeRange([selectedTimeRange[0], parseFloat(e.target.value)])}
										/>
									</div>
									<div className="time-labels">
										<span>0s</span>
										<span>{(results.duration * selectedTimeRange[0]).toFixed(1)}s - {(results.duration * selectedTimeRange[1]).toFixed(1)}s</span>
										<span>{results.duration.toFixed(1)}s</span>
									</div>
								</div>
							)}

							{/* Reorganized Charts Grid - 2x2 layout */}
							<div className="charts-grid-realtime">
								<div className="chart-row">
									<LineChart
										data={getFilteredData().valence}
										labels={getFilteredData().labels}
										title="Valence Trend"
										color="#4CAF50"
										height={180}
									/>
									<LineChart
										data={getFilteredData().arousal}
										labels={getFilteredData().labels}
										title="Arousal Trend"
										color="#2196F3"
										height={180}
									/>
								</div>

								<div className="chart-row">
									<EmotionDistributionChart
										emotions={results.average_emotions}
										height={250}
									/>
									<TimelineChart
										frameResults={getFilteredData().frames}
										height={180}
									/>
								</div>
							</div>

							{/* Sample Frame Results with Images */}
							<div className="sample-frames">
								<h4>Sample Frame Analysis</h4>
								<div className="frames-grid">
									{results.frame_results?.slice(0, 4).map((frame, index) => (
										<SampleFrame key={index} frame={frame} index={index} />
									))}
								</div>
							</div>
						</>
					) : results.type === 'audio' ? (
						<>
							{results.audio_url && (
								<div className="preview-container processed-image">
									<audio
										controls
										src={results.audio_url}
										className="audio-segment"
										data-i18n="processed_audio_alt"
									/>
								</div>
							)}
							{displayEmotionResult(results.result)}
						</>
					) : (
						// Fallback for unknown response format
						<div className="result-card">
							<h3>Analysis Results</h3>
							<pre>{JSON.stringify(results, null, 2)}</pre>
						</div>
					)}
				</div>
			)}
		</div>
	);
};

export default Detector;