import React, { useState, useRef, useEffect } from 'react';
import { getColorForEmotion } from '../utils/emotionColors';
import { useLanguage } from '../hooks/useLanguage';
import { t as translate } from '../utils/translations'; // Import the translation function

const Detector = () => {
	const [currentFile, setCurrentFile] = useState(null);
	const [fileInfo, setFileInfo] = useState('');
	const [showProgress, setShowProgress] = useState(false);
	const [progressText, setProgressText] = useState('');
	const [preview, setPreview] = useState(null);
	const [results, setResults] = useState(null);
	const [consentGiven, setConsentGiven] = useState(false);
	const [error, setError] = useState('');
	const fileInputRef = useRef(null);
	const { language, updateTexts } = useLanguage();

	useEffect(() => {
		updateTexts();
	}, [language, updateTexts]);

	// Use the centralized translation function
	const t = (key, replacements = {}) => {
		return translate(key, language, replacements);
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

		// Clear previous results
		setPreview(null);
		setResults(null);
	};

	const validateFile = (file) => {
		const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'video/mp4', 'video/avi', 'video/webm'];
		const maxSize = 50 * 1024 * 1024;

		if (!validTypes.includes(file.type)) {
			setError(t('error_unsupported_format'));
			return false;
		}

		if (file.size > maxSize) {
			setError(t('error_file_too_large'));
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
		if (fileInputRef.current) {
			fileInputRef.current.value = '';
		}
	};

	const uploadFile = () => {
		if (!currentFile) {
			setError(t('error_file_not_selected'));
			return;
		}

		if (!consentGiven) {
			setError(t('error_consent_required'));
			return;
		}

		setShowProgress(true);
		setProgressText(t('starting_processing'));

		// Simulate file upload and processing
		// In a real app, you would use fetch API here
		setTimeout(() => {
			setShowProgress(false);

			// Mock results for demonstration
			if (currentFile.type.includes('image')) {
				setPreview(URL.createObjectURL(currentFile));
				setResults({
					type: 'image',
					result: {
						main_prediction: { label: 'happiness', probability: 0.85 },
						additional_probs: {
							happiness: 0.85,
							neutral: 0.10,
							sadness: 0.03,
							surprise: 0.02
						}
					}
				});
			} else {
				// Mock video results
				setResults({
					type: 'video',
					frames_processed: 24,
					results: [
						{
							frame: 0,
							image_url: URL.createObjectURL(currentFile),
							result: {
								main_prediction: { label: 'neutral', probability: 0.75 },
								additional_probs: {
									neutral: 0.75,
									happiness: 0.15,
									sadness: 0.07,
									surprise: 0.03
								}
							}
						}
					]
				});
			}
		}, 3000);
	};

	const displayEmotionResult = (result) => {
		const mainPred = result.main_prediction;
		const additional = result.additional_probs;

		return (
			<div className="result-card">
				<div className="main-emotion">
					{t('detected_emotion')}:
					<span data-i18n={mainPred.label}>{mainPred.label}</span>
					({(mainPred.probability * 100).toFixed(1)}%)
				</div>
				<div className="emotion-display">
					{Object.entries(additional).map(([emotionKey, prob]) => {
						const percentage = (parseFloat(prob) * 100).toFixed(1);
						return (
							<div key={emotionKey} className="emotion-item">
								<div className="emotion-label">
									<span data-i18n={emotionKey}>{emotionKey}</span>
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
				<h1 data-i18n="detector_title">{t('detector_title')}</h1>
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
						<h3 data-i18n="drag_file">{t('drag_file')}</h3>
						<p data-i18n="or">{t('or')}</p>
					</div>
					<input
						type="file"
						id="fileInput"
						ref={fileInputRef}
						className="file-input"
						accept="image/*,video/*"
						onChange={handleFileSelect}
					/>
					<button
						className="btn"
						id="selectFileBtn"
						onClick={() => fileInputRef.current?.click()}
						data-i18n="choose_file"
					>
						{t('choose_file')}
					</button>
					<p className="supported-formats" data-i18n="supported_formats">
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
						/>
						<span data-i18n="consent_text">
							{t('consent_text')}{' '}
							<a
								href="#privacy"
								className="nav-link"
								style={{ color: 'var(--primary-color)' }}
								onClick={(e) => {
									e.preventDefault();
									// You would navigate to privacy page here
								}}
								data-i18n="privacy_policy"
							>
								{t('privacy_policy')}
							</a>
						</span>
					</label>
				</div>

				<button
					className="btn"
					id="uploadBtn"
					onClick={uploadFile}
					disabled={!currentFile || !consentGiven}
					data-i18n="analyze_emotions"
				>
					{t('analyze_emotions')}
				</button>
			</div>

			{showProgress && (
				<div className="progress-container" id="progressContainer">
					<div className="progress-header">
						<h3 data-i18n="processing">{t('processing')}</h3>
					</div>
					<div className="progress-wheel" id="progressWheel"></div>
					<div className="progress-text" id="progressText">
						{progressText}
					</div>
				</div>
			)}

			{preview && (
				<div className="preview-container" id="previewContainer">
					<img src={preview} alt={t('processed_image_alt')} loading="lazy" />
				</div>
			)}

			{results && (
				<div className="results-container" id="resultsContainer">
					{results.type === 'image' ? (
						displayEmotionResult(results.result)
					) : (
						<>
							<div className="result-card">
								<h3>{t('video_analysis_complete')}</h3>
								<p>{t('processed_frames').replace('{count}', results.frames_processed)}</p>
							</div>
							{results.results.map((frame, index) => (
								<div key={index} className="frame-result">
									<h4>{t('frame')} {frame.frame + 1}</h4>
									<img src={frame.image_url} alt={`${t('video_frame')} ${frame.frame + 1}`} loading="lazy" />
									{displayEmotionResult(frame.result)}
								</div>
							))}
						</>
					)}
				</div>
			)}
		</div>
	);
};

export default Detector;