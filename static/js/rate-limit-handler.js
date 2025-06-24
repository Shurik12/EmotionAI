class RateLimitHandler {
    constructor() {
        this.errorModal = this.createErrorModal();
        document.body.appendChild(this.errorModal);
    }

    createErrorModal() {
        const modal = document.createElement('div');
        modal.id = 'rateLimitModal';
        modal.className = 'modal';
        modal.style.display = 'none';
        
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <h2>Превышен лимит запросов</h2>
                <div class="rate-limit-message" id="rateLimitMessage"></div>
                <button class="btn" id="rateLimitCloseBtn">Понятно</button>
            </div>
        `;
        
        return modal;
    }

    showRateLimitError(errorData) {
        const modal = document.getElementById('rateLimitModal');
        const messageElement = document.getElementById('rateLimitMessage');
        const closeBtn = document.getElementById('rateLimitCloseBtn');
        
        let message = 'Превышен лимит запросов. Пожалуйста, попробуйте позже.';
        
        if (errorData && errorData.message) {
            message = errorData.message;
        }
        
        if (errorData && errorData.reset_at) {
            const resetTime = new Date(errorData.reset_at);
            const timeString = resetTime.toLocaleTimeString('ru-RU');
            message += ` Полное восстановление лимита в ${timeString}.`;
        }
        
        messageElement.textContent = message;
        modal.style.display = 'block';
        
        // Close handlers
        modal.querySelector('.close-modal').onclick = () => {
            modal.style.display = 'none';
        };
        
        closeBtn.onclick = () => {
            modal.style.display = 'none';
        };
        
        window.onclick = (event) => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        };
    }
}

// Initialize globally
window.rateLimitHandler = new RateLimitHandler();

// Handle fetch responses for rate limits
function checkRateLimit(response) {
    if (response.status === 429) {
        return response.json().then(errorData => {
            window.rateLimitHandler.showRateLimitError(errorData);
            throw new Error('Rate limit exceeded');
        }).catch(() => {
            window.rateLimitHandler.showRateLimitError();
            throw new Error('Rate limit exceeded');
        });
    }
    return response;
}