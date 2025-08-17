// Content loading functions
function loadHomeContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <section class="hero">
            <h1 data-i18n="main_title">Razuma – флагманская ИИ-платформа для комплексного анализа эмоций по фото и видео</h1>
            <p data-i18n="main_description">Razuma – это ИИ модель анализа эмоций потребителей: быстро, точно, без личного присутствия. Она разработана для анализа эффективности маркетинговых исследований, поведения потребителей, проверки гипотез, A/B-тестирования продуктов и услуг, когда особенно важно выводить новые продукты, повышать удовлетворенность клиентов и снижать затраты.</p>
            
            <h3 style="text-align: center; margin: 30px 0 20px; font-size: 18px; font-weight: 500;" data-i18n="platform_usage">
                Платформа Razuma применяется в различных сферах, где важно понимать эмоциональное восприятие продукта или услуги потребителем:
            </h3>
            
            <div class="features-grid" style="max-width: 800px; margin: 0 auto 40px; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));">
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;" data-i18n="feature1">Анализ промо-роликов и A/B-тестирование креативов</p>
                </div>
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;" data-i18n="feature2">Продуктовое и UX тестирование</p>
                </div>
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;" data-i18n="feature3">Глубокая оценка эмоционального восприятия клиентского опыта</p>
                </div>
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;" data-i18n="feature4">Оценка эффективности выступлений и презентаций</p>
                </div>
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;" data-i18n="feature5">Редактирование сценариев для повышения эмоционального отклика</p>
                </div>
            </div>
            
            <div class="btn-container">
                <a href="#detector" class="btn" data-i18n="analyze_now">Анализировать сейчас</a>
                <a href="#features" class="btn btn-outline" data-i18n="learn_more">Узнать больше</a>
            </div>
        </section>

        <section class="pricing">
            <h2 class="section-title" data-i18n="pricing_title">Тарифы</h2>
            <div class="pricing-grid">
                <div class="pricing-card">
                    <h3 data-i18n="free_plan">Free</h3>
                    <div class="price" data-i18n="free_price">Бесплатно</div>
                    <ul class="features-list">
                        <li data-i18n="free_feature1">Анализ 3-х фото в месяц</li>
                        <li data-i18n="free_feature2">До 30 секунд видео или аудио в месяц</li>
                    </ul>
                    <button class="btn use-plan-btn" data-plan="free" data-i18n="use_plan">Использовать</button>
                </div>
                
                <div class="pricing-card">
                    <h3 data-i18n="light_plan">Light</h3>
                    <div class="price">9 990 ₽/мес</div>
                    <ul class="features-list">
                        <li data-i18n="light_feature1">Ведение до 10 проектов</li>
                        <li data-i18n="light_feature2">Неограниченный анализ фото</li>
                        <li data-i18n="light_feature3">До 2 часов видео и аудио</li>
                        <li data-i18n="light_feature4">Проведение A/B-анализа</li>
                        <li data-i18n="light_feature5">Полный API-доступ</li>
                    </ul>
                    <button class="btn apply-plan-btn" data-plan="pro" data-i18n="apply_plan">Отправить заявку</button>
                </div>
                
                <div class="pricing-card">
                    <h3 data-i18n="pro_plan">Pro</h3>
                    <div class="price">29 990 ₽/мес</div>
                    <ul class="features-list">
                        <li data-i18n="pro_feature1">Неограниченное количество проектов</li>
                        <li data-i18n="pro_feature2">Анализ любого количества фото, видео и аудио</li>
                        <li data-i18n="pro_feature3">Проведение A/B-анализа</li>
                        <li data-i18n="pro_feature4">Полный API-доступ</li>
                        <li data-i18n="pro_feature5">Расширенная аналитика и техподдержка</li>
                        <li data-i18n="pro_feature6">Персональные экспертные отчеты</li>
                    </ul>
                    <button class="btn apply-plan-btn" data-plan="business" data-i18n="apply_plan">Отправить заявку</button>
                </div>

                <div class="pricing-card">
                    <h3 data-i18n="business_plan">Business</h3>
                    <div class="price" data-i18n="business_price">По запросу</div>
                    <ul class="features-list">
                        <li data-i18n="business_feature1">Неограниченное количество проектов</li>
                        <li data-i18n="business_feature2">Анализ любого количества фото, видео и аудио</li>
                        <li data-i18n="business_feature3">Проведение A/B-анализа</li>
                        <li data-i18n="business_feature4">Полный API-доступ</li>
                        <li data-i18n="business_feature5">Расширенная аналитика и техподдержка</li>
                        <li data-i18n="business_feature6">Персональные экспертные отчеты</li>
                        <li><strong data-i18n="business_feature7">Внедрение в CRM</strong></li>
                    </ul>
                    <button class="btn apply-plan-btn" data-plan="business" data-i18n="apply_plan">Отправить заявку</button>
                </div>
            </div>
        </section>
    `;

    // Add event listeners for plan buttons
    document.querySelectorAll('.use-plan-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            window.location.hash = '#detector';
            loadContent('#detector');
        });
    });

    document.querySelectorAll('.apply-plan-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const plan = this.getAttribute('data-plan');
            openApplicationModal(plan);
        });
    });

    updateTexts();
}

function loadFeaturesContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <section class="features">
            <div class="container">
                <h1 data-i18n="features_title">Razuma – это экосистема ИИ-инструментов, которая сочетает методы нейромаркетинга, emotion AI и API-интеграции для бизнеса любого масштаба</h1>
                <h2 class="section-title" data-i18n="features_main">Основные возможности</h2>
                <div class="features-grid">
                    <div class="feature-card">
                        <p data-i18n="feature_card1">Распознавание эмоций в реальном времени по фото, видео и аудио</p>
                    </div>
                    <div class="feature-card">
                        <p data-i18n="feature_card2">Создание плагинов для CRM (Bitrix24, amoCRM и др.)</p>
                    </div>
                    <div class="feature-card">
                        <p data-i18n="feature_card3">Анализ A/B-тестирования рекламных креативов или упаковки продуктов</p>
                    </div>
                    <div class="feature-card">
                        <p data-i18n="feature_card4">Анализ UX интерфейса</p>
                    </div>
                    <div class="feature-card">
                        <p data-i18n="feature_card5">Автоматическая оценка параметров эмоционального состояния клиента и рекомендации на ее основе</p>
                    </div>
                    <div class="feature-card">
                        <p data-i18n="feature_card6">Интеграции через API и SDK</p>
                    </div>
                    <div class="feature-card">
                        <p data-i18n="feature_card7">Адаптация продукта и разработка white label по техническому заданию</p>
                    </div>
                </div>
                <div class="clients-block">
                    <p data-i18n="clients_title">Кто уже использует Razuma:</p>
                    <p><strong>B2B:</strong> <span data-i18n="clients_b2b">агентства, бренды, разработчики продуктов, обучающие платформы</span></p>
                    <p><strong>B2C / МСП:</strong> <span data-i18n="clients_b2c">специалисты, малый бизнес, консультанты</span></p>
                </div>
            </div>
        </section>
    `;
    updateTexts();
}

function loadDetectorContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <div class="detector-container">
            <div style="text-align: center;">
                <h1 data-i18n="detector_title">Распознавание эмоций с помощью ИИ</h1>
            </div>
            
            <div class="upload-section">
                <div class="upload-container" id="dropArea">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">
                        <h3 data-i18n="drag_file">Перетащите файл сюда</h3>
                        <p data-i18n="or">или</p>
                    </div>
                    <input type="file" id="fileInput" class="file-input" accept="image/*,video/*">
                    <button class="btn" id="selectFileBtn" data-i18n="choose_file">Выбрать файл</button>
                    <p class="supported-formats" data-i18n="supported_formats">Поддерживаемые форматы: JPG, PNG, MP4, AVI, WEBM (макс. 16MB)</p>
                </div>
                
                <div class="file-info" id="fileInfo"></div>
                <div style="margin: 15px 0; text-align: center;">
                    <label style="display: flex; align-items: center; justify-content: center;">
                        <input type="checkbox" id="dataConsent" required style="margin-right: 8px;">
                        <span data-i18n="consent_text">Я даю согласие на обработку моих персональных данных в соответствии с 
                        <a href="#privacy" class="nav-link" style="color: var(--primary-color);" onclick="event.preventDefault(); window.dispatchEvent(new CustomEvent('navigate', {detail: 'privacy'}));" data-i18n="privacy_policy">Политикой обработки персональных данных</a></span>
                    </label>
                </div>
                <button class="btn" id="uploadBtn" data-i18n="analyze_emotions">Анализировать эмоции</button>
            </div>

            <div class="progress-container" id="progressContainer">
                <div class="progress-header">
                    <h3 data-i18n="processing">Обработка</h3>
                </div>
                <div class="progress-wheel" id="progressWheel"></div>
                <div class="progress-text" id="progressText" data-i18n="starting_processing">Начало обработки...</div>
            </div>
            
            <div class="preview-container" id="previewContainer"></div>
            
            <div class="results-container" id="resultsContainer"></div>
        </div>
    `;

    initializeDetector();
    updateTexts();
}

function loadPrivacyContent() {
    const lang = getCurrentLanguage();
    const contentDiv = document.getElementById('app-content');
    const privacyPolicyTitle = translations[lang]?.['privacy_policy'] || 'Privacy Policy';
    
    // Create a container with a unique ID
    contentDiv.innerHTML = `
        <div class="privacy-policy-container">
            <h1 data-i18n="privacy_policy">${privacyPolicyTitle}</h1>
            <div class="iframe-container">
                <iframe 
                    id="privacyIframe"
                    src="/static/privacy-policy.html?lang=${lang}&t=${Date.now()}" 
                    width="100%" 
                    height="500px" 
                    frameborder="0"
                    title="${privacyPolicyTitle}"
                    loading="lazy"
                ></iframe>
            </div>
            <p class="fallback-text" style="display: none;" data-i18n="privacy_error"></p>
        </div>
    `;

    // Listen for language change messages from iframe
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'languageChange') {
            setLanguage(e.data.language);
        }
    });

    updateTexts();
}

function loadContactContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <section class="hero">
            <h1 data-i18n="contact_title">Контактная информация</h1>
        </section>

        <section style="max-width: 800px; margin: 40px auto; padding: 30px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h2 data-i18n="legal_info">Юридическая информация</h2>
            <p><strong data-i18n="company_name">Наименование компании:</strong> ИП Тесаков Роман Валентинович</p>
            <p><strong data-i18n="legal_address">Юридический адрес:</strong>121087 Москва Новозаводская ул., д.2, корп.8, кв. 19</p>
            <p><strong data-i18n="inn">ИНН:</strong> 500402730770</p>
            <p><strong data-i18n="ogrnip">ОГРНИП:</strong> 324774600219962</p>
            
            <h2 style="margin-top: 30px;" data-i18n="contact_details">Контактные данные</h2>
            <p><strong data-i18n="email">Email:</strong> inbox@razuma.pro</p>
            <p><strong data-i18n="phone">Телефон:</strong> +7 (903) 295-89-71</p>
            <p><strong data-i18n="working_hours">Режим работы:</strong> Пн-Пт, 9:00-18:00</p>
        </section>
    `;
    updateTexts();
}