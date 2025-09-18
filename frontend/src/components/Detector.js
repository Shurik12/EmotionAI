import React, { useState, useRef, useEffect } from 'react';
import { getColorForEmotion } from '../utils/emotionColors';
import { useLanguage } from '../hooks/useLanguage';
import { t } from '../utils/translations';
import './Detector.css';

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
		if (fileInputRef.current) {
			fileInputRef.current.value = '';
		}
	};

	const uploadFile = async () => {

		console.log('uploadFile called, currentFile:', currentFile);
		console.log('consentGiven:', consentGiven);

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

		try {
			const response = await fetch('/api/upload', {
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

		return (
			<div className="result-card">
				<div className="main-emotion">
					<span data-i18n={mainPred.label}>{t(mainPred.label, localStorage.getItem('language'))}</span>
					({(mainPred.probability * 100).toFixed(1)}%)
				</div>
				<div className="emotion-display">
					{Object.entries(additional).map(([emotionKey, prob]) => {
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
			</div>
		);
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
						className="btn primary" // Added 'primary' class
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
					disabled={false}
					data-i18n="analyze_emotions"
				>
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