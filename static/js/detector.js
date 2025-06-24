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
    
    // Event listeners
    selectFileBtn.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', handleFileSelect);
    
    uploadBtn.addEventListener('click', function() {
        if (!document.getElementById('dataConsent').checked) {
            showError('Необходимо дать согласие на обработку персональных данных');
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
                Выбран файл: <strong>${file.name}</strong><br>
                (${formatFileSize(file.size)})
            </div>
            <button class="btn" id="clearFileBtn" 
                    style="margin-top: 8px; background-color: var(--error-color);">
                Очистить
            </button>
        `;
        
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
        const maxSize = 16 * 1024 * 1024; // 16MB
        
        if (!validTypes.includes(file.type)) {
            showError('Неподдерживаемый формат файла. Загрузите изображение (JPG, PNG) или видео (MP4, AVI, WEBM).');
            return false;
        }
        
        if (file.size > maxSize) {
            showError('Файл слишком большой. Максимальный размер 16MB.');
            return false;
        }
        
        return true;
    }
    
    function showError(message) {
        dropArea.classList.add('error');
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        fileInfo.innerHTML = '';
        fileInfo.appendChild(errorDiv);
        
        setTimeout(() => {
            dropArea.classList.remove('error');
        }, 3000);
    }
    
    function clearFile() {
        fileInput.value = '';
        currentFile = null;
        fileInfo.innerHTML = '';
        uploadBtn.disabled = true;
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
        
        // Show loading state
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-state';
        loadingDiv.innerHTML = '<div class="loading-spinner"></div>';
        document.body.appendChild(loadingDiv);
        loadingDiv.style.display = 'flex';
        
        // Всегда используем emotieff модель
        const selectedModel = "emotieff";
        
        // Reset UI
        previewContainer.innerHTML = '';
        resultsContainer.innerHTML = '';
        
        // Show progress container
        progressContainer.style.display = 'block';
        progressText.textContent = 'Загрузка файла...';
        
        // Disable buttons during upload
        uploadBtn.disabled = true;
        selectFileBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('model', selectedModel);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // First check for rate limiting
            if (response.status === 429) {
                return response.json().then(errorData => {
                    window.rateLimitHandler.showRateLimitError(errorData);
                    throw new Error('rate_limit_exceeded');
                }).catch(() => {
                    window.rateLimitHandler.showRateLimitError();
                    throw new Error('rate_limit_exceeded');
                });
            }
            
            if (!response.ok) {
                throw new Error('Ошибка отклика сети');
            }
            return response.json();
        })
        .then(data => {
            if (!data) {
                throw new Error('Пустой ответ от сервера');
            }
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            currentTaskId = data.task_id;
            progressText.textContent = 'Обработка файла...';
            checkProgress();
        })
        .catch(error => {
            console.error('Error:', error);
            
            // Don't show default error for rate limits (already handled)
            if (error.message !== 'rate_limit_exceeded') {
                showError(error.message || 'Во время загрузки произошла ошибка');
            }
            
            hideProgress();
            
            // Re-enable buttons
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
            fetch(`/progress/${currentTaskId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Не удалось проверить прогресс');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    if (data.message) {
                        progressText.textContent = data.message;
                    }
                    
                    if (data.complete) {
                        clearInterval(progressInterval);
                        progressWheel.style.animation = 'none';
                        progressWheel.style.border = '6px solid var(--secondary-color)';
                        progressText.textContent = 'Обработка завершена!';
                        
                        setTimeout(() => {
                            hideProgress();
                            displayResults(data);
                            
                            // Re-enable buttons
                            uploadBtn.disabled = false;
                            selectFileBtn.disabled = false;
                        }, 1000);
                    }
                })
                .catch(error => {
                    console.error('Ошибка при проверке прогресса:', error);
                    clearInterval(progressInterval);
                    showError(error.message || 'Не удалось проверить прогресс');
                    hideProgress();
                    
                    // Re-enable buttons
                    uploadBtn.disabled = false;
                    selectFileBtn.disabled = false;
                });
        }, 1000);
    }
    
    function hideProgress() {
        progressContainer.style.display = 'none';
        currentTaskId = null;
        // Reset wheel for next use
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
            <h3>Анализ завершен</h3>
        `;
        resultsContainer.appendChild(modelInfo);
        
        if (data.type === 'image') {
            // Display image preview
            const img = document.createElement('img');
            img.src = data.image_url;
            img.alt = 'Обработанное изображение с распознаванием эмоций';
            img.loading = 'lazy';
            previewContainer.appendChild(img);
            
            // Display results
            displayEmotionResult(data.result, resultsContainer);
        } 
        else if (data.type === 'video') {
            resultsContainer.innerHTML = `
                <div class="result-card">
                    <h3>Видеоанализ завершен</h3>
                    <p>Обработанные ${data.frames_processed} ключевые кадры</p>
                </div>
            `;
            
            data.results.forEach(frame => {
                const frameDiv = document.createElement('div');
                frameDiv.className = 'frame-result';
                
                const frameTitle = document.createElement('h4');
                frameTitle.textContent = `Frame ${frame.frame + 1}`;
                frameDiv.appendChild(frameTitle);
                
                const img = document.createElement('img');
                img.src = frame.image_url;
                img.alt = `Video frame ${frame.frame + 1}`;
                img.loading = 'lazy';
                frameDiv.appendChild(img);
                
                displayEmotionResult(frame.result, frameDiv);
                
                resultsContainer.appendChild(frameDiv);
            });
        }
    }
    
    function displayEmotionResult(result, container) {
        const mainPred = result.main_prediction;
        const additional = result.additional_probs;
        
        // Get color based on emotion - updated for both models
        const getColorForEmotion = (emotion) => {
            const emotionLower = emotion.toLowerCase();
            if (emotionLower.includes('счастье') || emotionLower.includes('happiness')) return '#34a853'; // Green
            if (emotionLower.includes('злость') || emotionLower.includes('anger')) return '#ea4335'; // Red
            if (emotionLower.includes('страх') || emotionLower.includes('fear')) return '#4285f4'; // Blue
            if (emotionLower.includes('грусть') || emotionLower.includes('sadness')) return '#2196F3'; // Light Blue
            if (emotionLower.includes('нейтральное') || emotionLower.includes('neutral')) return '#fbbc05'; // Yellow
            if (emotionLower.includes('удивление') || emotionLower.includes('surprise')) return '#673ab7'; // Purple
            if (emotionLower.includes('отвращение') || emotionLower.includes('disgust')) return '#9C27B0'; // Violet
            return '#4285f4'; // Default blue
        };
        
        // Sort emotions by probability (highest first)
        const sortedEmotions = Object.entries(additional)
            .map(([emotion, prob]) => ({
                emotion,
                prob: parseFloat(prob),
                percentage: (parseFloat(prob) * 100).toFixed(1)
            }))
            .sort((a, b) => b.prob - a.prob);
        
        const resultHTML = `
            <div class="result-card">
                <div class="main-emotion">
                    Detected Emotion: ${mainPred.label} (${(mainPred.probability * 100).toFixed(1)}%)
                </div>
                <div class="emotion-display">
                    ${sortedEmotions.map(item => {
                        const color = getColorForEmotion(item.emotion);
                        return `
                            <div class="emotion-item">
                                <div class="emotion-label">
                                    <span>${item.emotion}</span>
                                    <span>${item.percentage}%</span>
                                </div>
                                <div class="emotion-bar">
                                    <div class="emotion-fill" style="width: ${item.percentage}%; background-color: ${color};"></div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', resultHTML);
    }
}