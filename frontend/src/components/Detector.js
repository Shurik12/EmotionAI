import React, { useState, useRef, useEffect } from 'react';
import { getColorForEmotion } from '../utils/emotionColors';
import { useLanguage } from '../hooks/useLanguage';
import { t as translate } from '../utils/translations';
import './Detector.css';

const Detector = () => {
	const [currentFile, setCurrentFile] = useState(null);
	const [fileInfo, setFileInfo] = useState('');
	const [showProgress, setShowProgress] = useState(false);
	const [progressText, setProgressText] = useState('');
	const [preview, setPreview] = useState(null);
	const [results, setResults] = useState(null);
	const [consentGiven, setConsentGiven] = useState(false);
	const [error, setError] = useState('');
	const [isProcessing, setIsProcessing] = useState(false);
	const [progressComplete, setProgressComplete] = useState(false);
	const [currentTaskId, setCurrentTaskId] = useState(null);

	const fileInputRef = useRef(null);
	const progressIntervalRef = useRef(null);
	const { language, updateTexts } = useLanguage();

	useEffect(() => {
		return () => {
			// Clean up interval on component unmount
			if (progressIntervalRef.current) {
				clearInterval(progressIntervalRef.current);
			}
		};
	}, []);

	const t = (key, replacements = {}) => {
		return translate(key, language, replacements);
	};

	const showError = (message) => {
		setError(message);
		setTimeout(() => {
			setError('');
		}, 3000);
	};

	const handleFileSelect = (e) => {
		const file = e.target.files[0];
		if (!file) return;

		if (!validateFile(file)) {
			return;
		}

		setCurrentFile(file);
		setFileInfo(`
      <div style="word-break: break-all;">
        <span>${t('selected_file')}:</span> <strong>${file.name}</strong><br>
        (${formatFileSize(file.size)})
      </div>
      <button class="btn" id="clearFileBtn" 
              style="margin-top: 8px; background-color: var(--error-color);">
        ${t('clear_button')}
      </button>
    `);
		setPreview(null);
		setResults(null);
	};

	const validateFile = (file) => {
		const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'video/mp4', 'video/avi', 'video/webm'];
		const maxSize = 50 * 1024 * 1024;

		if (!validTypes.includes(file.type)) {
			showError(t('error_unsupported_format'));
			return false;
		}

		if (file.size > maxSize) {
			showError(t('error_file_too_large'));
			return false;
		}

		setError('');
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
		setFileInfo('');
		setPreview(null);
		if (fileInputRef.current) {
			fileInputRef.current.value = '';
		}
	};

	useEffect(() => {
		// Add event listener for clear button after fileInfo is rendered
		if (fileInfo) {
			const clearBtn = document.getElementById('clearFileBtn');
			if (clearBtn) {
				clearBtn.addEventListener('click', clearFile);
				return () => {
					clearBtn.removeEventListener('click', clearFile);
				};
			}
		}
	}, [fileInfo]);

	const uploadFile = async () => {
		if (!currentFile) {
			showError(t('error_file_not_selected'));
			return;
		}

		if (!consentGiven) {
			showError(t('error_consent_required'));
			return;
		}

		console.log('Starting upload process...');

		// Reset UI state
		setResults(null);
		setShowProgress(true);
		setProgressText(t('starting_processing'));
		setIsProcessing(true);
		setError(null);
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
				throw new Error(t(err.error) || t('error_network_response'));
			}

			const data = await response.json();

			if (data.error) {
				throw new Error(t(data.error));
			}

			// Store for language switching
			window.lastAnalysisResult = data.result;

			setCurrentTaskId(data.task_id);
			setProgressText(t('file_processing'));

			checkProgress(data.task_id);
		} catch (error) {
			console.error(t('error_checking_progress'), error);
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
					throw new Error(t('error_checking_progress'));
				}

				const data = await response.json();

				if (data.error) {
					throw new Error(t(data.error));
				}

				if (data.message) {
					// Handle frame processing messages
					let newProgressText = '';
					if (data.message.includes("frame")) {
						const frameMatch = data.message.match(/frame (\d+) of (\d+)/);
						if (frameMatch) {
							newProgressText = t('processing_frame')
								.replace('{current}', frameMatch[1])
								.replace('{total}', frameMatch[2]);
						}
					} else {
						newProgressText = t(data.message) || data.message;
					}

					setProgressText(newProgressText);
				}

				if (data.complete) {
					clearInterval(progressIntervalRef.current);
					setProgressText(t('processing_complete'));
					setProgressComplete(true);

					setTimeout(() => {
						hideProgress();
						displayResults(data);
					}, 1000);
				}
			} catch (error) {
				console.error(t('error_checking_progress'), error);
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
					{t('detected_emotion')}:
					<span>{mainPred.label || 'Unknown'}</span>
					({(mainPred.probability * 100).toFixed(1)}%)
				</div>
				<div className="emotion-display">
					{Object.entries(additional).map(([emotionKey, prob]) => {
						const percentage = (parseFloat(prob) * 100).toFixed(1);
						return (
							<div key={emotionKey} className="emotion-item">
								<div className="emotion-label">
									<span>{emotionKey}</span>
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
			<div style={{ textAlign: 'center' }}>
				<h1>{t('detector_title')}</h1>
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
						<h3>{t('drag_file')}</h3>
						<p>{t('or')}</p>
					</div>
					<input
						type="file"
						id="fileInput"
						ref={fileInputRef}
						className="file-input"
						accept="image/*,video/*"
						onChange={handleFileSelect}
						disabled={isProcessing}
					/>
					<button
						className="btn primary" // Added 'primary' class
						id="selectFileBtn"
						onClick={() => fileInputRef.current?.click()}
						disabled={isProcessing}
					>
						{t('choose_file')}
					</button>
					<p className="supported-formats">
						{t('supported_formats')}
					</p>
				</div>

				{error && (
					<div className="error-message">
						{error}
					</div>
				)}

				{fileInfo && (
					<div
						className="file-info"
						id="fileInfo"
						dangerouslySetInnerHTML={{ __html: fileInfo }}
					/>
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
						<span>
							{t('consent_text')}{' '}
							<a
								href="#privacy"
								className="nav-link"
								style={{ color: 'var(--primary-color)' }}
								onClick={(e) => e.preventDefault()}
							>
								{t('privacy_policy')}
							</a>
						</span>
					</label>
				</div>

				<button
					className="btn primary" // Added 'primary' class
					id="uploadBtn"
					onClick={uploadFile}
					disabled={!currentFile || !consentGiven || isProcessing}
				>
					{t('analyze_emotions')}
				</button>
			</div>

			{showProgress && (
				<div className="progress-container" id="progressContainer">
					<div className="progress-header">
						<h3>{t('processing')}</h3>
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
										alt={t('processed_image_alt')} 
										loading="lazy"
										className="processed-image"
									/>
								</div>
							)}
							{displayEmotionResult(results.result)}
						</>
					) : results.type === 'video' ? (
						<>
							<div className="result-card">
								<h3>{t('video_analysis_complete')}</h3>
								<p>{t('processed_frames').replace('{count}', results.frames_processed)}</p>
							</div>
							{results.results && results.results.map((frame, index) => (
								<div key={index} className="frame-result">
									<h4>{t('frame')} {frame.frame + 1}</h4>
									{frame.image_url && (
										<img 
											src={frame.image_url} 
											alt={`${t('video_frame')} ${frame.frame + 1}`} 
											loading="lazy"
											className="processed-image"
										/>
									)}
									{displayEmotionResult(frame.result)}
								</div>
							))}
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