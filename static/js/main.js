// Modal handling functions
function openApplicationModal(plan) {
    document.getElementById('selectedPlan').value = plan;
    document.getElementById('applicationModal').style.display = 'block';
}

function closeApplicationModal() {
    document.getElementById('applicationModal').style.display = 'none';
}

// SPA Router with History API
document.addEventListener('DOMContentLoaded', function() {
    // Check cookies consent
    if (!localStorage.getItem('cookiesAccepted')) {
        document.getElementById('cookieConsent').style.display = 'block';
    }

    document.getElementById('acceptCookies').addEventListener('click', function() {
        localStorage.setItem('cookiesAccepted', 'true');
        document.getElementById('cookieConsent').style.display = 'none';
    });

    // Modal functionality
    document.querySelector('.close-modal').addEventListener('click', closeApplicationModal);
    
    window.addEventListener('click', function(event) {
        if (event.target === document.getElementById('applicationModal')) {
            closeApplicationModal();
        }
    });

    // Mobile menu functionality
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mainNav = document.getElementById('mainNav');

    mobileMenuBtn.addEventListener('click', function() {
        mainNav.classList.toggle('show');
    });

    // Close mobile menu when clicking a link or outside
    document.addEventListener('click', function(event) {
        const isClickInsideNav = mainNav.contains(event.target) || 
                                mobileMenuBtn.contains(event.target);
        
        if (!isClickInsideNav && mainNav.classList.contains('show')) {
            mainNav.classList.remove('show');
        }
    });

    // Form submission handler
    document.getElementById('planApplicationForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = {
            plan: document.getElementById('selectedPlan').value,
            name: document.getElementById('name').value,
            email: document.getElementById('email').value,
            phone: document.getElementById('phone').value,
            company: document.getElementById('company').value || ''
        };
        
        fetch('/submit_application', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Спасибо за вашу заявку! Мы свяжемся с вами в ближайшее время.');
                closeApplicationModal();
                document.getElementById('planApplicationForm').reset();
            } else {
                alert('Произошла ошибка при отправке заявки. Пожалуйста, попробуйте позже.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Произошла ошибка при отправке заявки. Пожалуйста, попробуйте позже.');
        });
    });

    // Handle navigation with History API
    function navigateTo(path) {
        // Remove leading slash if present
        path = path.replace(/^\//, '');
        
        // Update content
        loadContent(path);
        
        // Update browser history
        if (path === 'home') {
            history.pushState(null, '', '/');
        } else {
            history.pushState(null, '', '/' + path);
        }
        
        // Update document title
        document.title = getPageTitle(path);
    }

    function getPageTitle(path) {
        const titles = {
            'home': 'Razuma | Распознавание эмоций',
            'features': 'Razuma | Возможности',
            'detector': 'Razuma | Демо',
            'privacy': 'Razuma | Конфиденциальность',
            'contact': 'Razuma | Контакты'
        };
        return titles[path] || 'Razuma';
    }

    function loadContent(path) {
        const contentDiv = document.getElementById('app-content');
        
        // Add fade-out effect
        contentDiv.classList.add('fade-out');
        
        // Wait for fade-out to complete before loading new content
        setTimeout(() => {
            contentDiv.innerHTML = '';
            contentDiv.classList.remove('fade-out');
            
            switch(path) {
                case 'home':
                    loadHomeContent();
                    break;
                case 'features':
                    loadFeaturesContent();
                    break;
                case 'detector':
                    loadDetectorContent();
                    break;
                case 'privacy':
                    loadPrivacyContent();
                    break;
                case 'contact':
                    loadContactContent();
                    break;
                default:
                    loadHomeContent();
            }
        }, 300); // Match this with your CSS transition duration
    }

    // Navigation event listeners
    document.querySelectorAll('.nav-link, .footer-links a').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const path = this.getAttribute('href').replace('#', '');
            navigateTo(path);
        });
    });

    document.getElementById('home-link').addEventListener('click', function(e) {
        e.preventDefault();
        navigateTo('home');
    });

    // Handle initial load and popstate events
    function handleInitialLoad() {
        // Get the path from the URL
        let path = window.location.pathname.replace(/^\//, '');
        if (path === '') path = 'home';
        
        loadContent(path);
        document.title = getPageTitle(path);
    }

    // Handle browser back/forward
    window.addEventListener('popstate', function() {
        let path = window.location.pathname.replace(/^\//, '');
        if (path === '') path = 'home';
        loadContent(path);
    });

    // Initialize the app
    handleInitialLoad();

    // Update all links to use the new navigation system
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        const href = link.getAttribute('href').replace('#', '');
        link.addEventListener('click', function(e) {
            e.preventDefault();
            navigateTo(href);
        });
    });
});