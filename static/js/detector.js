let lastAnalysisResult = null; // Store the last result for re-translation

function initializeDetector() {
    // DOM elements
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const selectFileBtn = document.getElementById('selectFileBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInfo = document.getElementById('fileInfo');
    const progressContainer = document.getElementById('progressContainer');
    const progressWheel = document.getElementById('progressWheel');
    const progressText = document.getElementById('progressText');
    const previewContainer = document.getElementById('previewContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    
    // State variables
    let currentFile = null;
    let currentTaskId = null;
    let progressInterval = null;
    
    // Helper function to get translated text
    function t(key) {
        const lang = getCurrentLanguage();
        return translations[lang][key] || key;
    }

    // Event listeners
    selectFileBtn.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', handleFileSelect);
    
    uploadBtn.addEventListener('click', function() {
        // clearErrors();
        
        if (!currentFile) {
            showError(t('error_file_not_selected'));
            return;
        }

        if (!document.getElementById('dataConsent').checked) {
            showError(t('error_consent_required'));
            return;
        }
        
        uploadFile();
    });
    
    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    // Touch events for mobile
    dropArea.addEventListener('touchstart', handleTouchStart, false);
    dropArea.addEventListener('touchend', handleTouchEnd, false);
    
    function handleTouchStart(e) {
        e.preventDefault();
        dropArea.classList.add('highlight');
    }
    
    function handleTouchEnd(e) {
        e.preventDefault();
        dropArea.classList.remove('highlight');
        fileInput.click();
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            fileInput.files = files;
            handleFileSelect();
        }
    }
    
    function handleFileSelect() {
        const file = fileInput.files[0];
        
        if (!file) return;
        
        // Validate file
        if (!validateFile(file)) {
            return;
        }
        
        currentFile = file;
        
        // Display file info - mobile optimized
        fileInfo.innerHTML = `
            <div style="word-break: break-all;">
                <span data-i18n="selected_file"></span>: <strong>${file.name}</strong><br>
                (${formatFileSize(file.size)})
            </div>
            <button class="btn" id="clearFileBtn" 
                    style="margin-top: 8px; background-color: var(--error-color);"
                    data-i18n="clear_button">
            </button>
        `;

        updateTexts();
        
        // Add event listener to clear button
        document.getElementById('clearFileBtn').addEventListener('click', clearFile);
        
        // Enable upload button
        uploadBtn.disabled = false;
        
        // Clear previous results
        previewContainer.innerHTML = '';
        resultsContainer.innerHTML = '';
    }
    
    function validateFile(file) {
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
        
        return true;
    }

    function clearErrors() {
        const errorMessages = document.querySelectorAll('.error-message');
        errorMessages.forEach(msg => msg.remove());
        dropArea.classList.remove('error');
    }
    
    function showError(message) {
        clearErrors();
        
        // Create error element
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        // Add error to drop area
        dropArea.classList.add('error');
        dropArea.appendChild(errorDiv);
        
        setTimeout(() => {
            dropArea.classList.remove('error');
            errorDiv.remove();
        }, 3000);
    }
    
    function clearFile() {
        fileInput.value = '';
        currentFile = null;
        fileInfo.innerHTML = '';
        uploadBtn.disabled = false;
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function uploadFile() {
        if (!currentFile) {
            showError('Сначала выберите файл');
            return;
        }

        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-state';
        loadingDiv.innerHTML = '<div class="loading-spinner"></div>';
        document.body.appendChild(loadingDiv);
        loadingDiv.style.display = 'flex';
        
        // Reset UI
        previewContainer.innerHTML = '';
        resultsContainer.innerHTML = '';
        
        // Show progress container
        progressContainer.style.display = 'block';
        progressText.textContent = t('starting_processing');
        
        // Disable buttons during upload
        uploadBtn.disabled = true;
        selectFileBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('model', "emotieff");

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            if (!response.ok) {
                const err = await response.json();
                throw new Error(t(err.error) || t('error_network_response'));
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(t(data.error));
            }
            window.lastAnalysisResult = data.result; // Store for language switching
            
            currentTaskId = data.task_id;
            progressText.textContent = t('file_processing');
            checkProgress();
        })
        .catch(error => {
            console.error(t('error_checking_progress'), error);
            showError(error.message);
            hideProgress();
            uploadBtn.disabled = false;
            selectFileBtn.disabled = false;
        })
        .finally(() => {
            loadingDiv.style.display = 'none';
            loadingDiv.remove();
        });
    }

    function checkProgress() {
        if (!currentTaskId) return;
        
        progressInterval = setInterval(() => {
            fetch(`/api/progress/${currentTaskId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(t('error_checking_progress'));
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        throw new Error(t(data.error));
                    }
                    
                    if (data.message) {
                        // Handle frame processing messages
                        if (data.message.includes("frame")) {
                            const frameMatch = data.message.match(/frame (\d+) of (\d+)/);
                            if (frameMatch) {
                                progressText.textContent = t('processing_frame')
                                    .replace('{current}', frameMatch[1])
                                    .replace('{total}', frameMatch[2]);
                            }
                        } else {
                            progressText.textContent = t(data.message) || data.message;
                        }
                    }
                    
                    if (data.complete) {
                        clearInterval(progressInterval);
                        progressWheel.style.animation = 'none';
                        progressWheel.style.border = '6px solid var(--secondary-color)';
                        progressText.textContent = t('processing_complete');
                        
                        setTimeout(() => {
                            hideProgress();
                            displayResults(data);
                            uploadBtn.disabled = false;
                            selectFileBtn.disabled = false;
                        }, 1000);
                    }
                })
                .catch(error => {
                    console.error(t('error_checking_progress'), error);
                    clearInterval(progressInterval);
                    showError(error.message);
                    hideProgress();
                    uploadBtn.disabled = false;
                    selectFileBtn.disabled = false;
                });
        }, 1000);
    }
    
    function hideProgress() {
        progressContainer.style.display = 'none';
        currentTaskId = null;
        progressWheel.style.animation = 'spin 1s linear infinite';
        progressWheel.style.border = '6px solid var(--medium-gray)';
        progressWheel.style.borderTop = '6px solid var(--primary-color)';
    }
    
    function displayResults(data) {
        previewContainer.innerHTML = '';
        resultsContainer.innerHTML = '';

        // Add model info display
        const modelInfo = document.createElement('div');
        modelInfo.className = 'result-card';
        modelInfo.innerHTML = `
            <h3>${t('analysis_complete')}</h3>
        `;
        resultsContainer.appendChild(modelInfo);
        
        if (data.type === 'image') {
            // Display image preview
            const img = document.createElement('img');
            img.src = data.image_url;
            img.alt = t('processed_image_alt');
            img.loading = 'lazy';
            previewContainer.appendChild(img);
            
            // Display results
            displayEmotionResult(data.result, resultsContainer);
        } 
        else if (data.type === 'video') {
            resultsContainer.innerHTML = `
                <div class="result-card">
                    <h3>${t('video_analysis_complete')}</h3>
                    <p>${t('processed_frames').replace('{count}', data.frames_processed)}</p>
                </div>
            `;
            
            data.results.forEach(frame => {
                const frameDiv = document.createElement('div');
                frameDiv.className = 'frame-result';
                
                const frameTitle = document.createElement('h4');
                frameTitle.textContent = `${t('frame')} ${frame.frame + 1}`;
                frameDiv.appendChild(frameTitle);
                
                const img = document.createElement('img');
                img.src = frame.image_url;
                img.alt = `${t('video_frame')} ${frame.frame + 1}`;
                img.loading = 'lazy';
                frameDiv.appendChild(img);
                
                displayEmotionResult(frame.result, frameDiv);
                
                resultsContainer.appendChild(frameDiv);
            });
        }
    }

    function displayEmotionResult(result, container) {
        lastAnalysisResult = result; // Store for language switching
        
        const mainPred = result.main_prediction;
        const additional = result.additional_probs;
        
        // Create HTML with data-i18n attributes
        const resultHTML = `
            <div class="result-card">
                <div class="main-emotion">
                    <span data-i18n="detected_emotion"></span>: 
                    <span data-i18n="${mainPred.label}"></span>
                    (${(mainPred.probability * 100).toFixed(1)}%)
                </div>
                <div class="emotion-display">
                    ${Object.entries(additional)
                        .map(([emotionKey, prob]) => {
                            const percentage = (parseFloat(prob) * 100).toFixed(1);
                            return `
                                <div class="emotion-item">
                                    <div class="emotion-label">
                                        <span data-i18n="${emotionKey}"></span>
                                        <span>${percentage}%</span>
                                    </div>
                                    <div class="emotion-bar">
                                        <div class="emotion-fill" 
                                            style="width: ${percentage}%; 
                                            background-color: ${getColorForEmotion(emotionKey)};">
                                        </div>
                                    </div>
                                </div>
                            `;
                        })
                        .join('')}
                </div>
            </div>
        `;
        
        container.innerHTML = resultHTML;
        updateTexts(); // This will apply translations to the new elements
    }

    // Helper function to get color
    function getColorForEmotion(emotionKey) {
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
    }
}