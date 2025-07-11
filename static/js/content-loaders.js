// Content loading functions
function loadHomeContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <section class="hero">
            <h1>Razuma – флагманская ИИ-платформа для комплексного анализа эмоций по фото и видео</h1>
            <p>Razuma – это ИИ модель анализа эмоций потребителей: быстро, точно, без личного присутствия. Она разработана для анализа эффективности маркетинговых исследований, поведения потребителей, проверки гипотез, A/B-тестирования продуктов и услуг, когда особенно важно выводить новые продукты, повышать удовлетворенность клиентов и снижать затраты.</p>
            
            <h3 style="text-align: center; margin: 30px 0 20px; font-size: 18px; font-weight: 500;">
                Платформа Razuma применяется в различных сферах, где важно понимать эмоциональное восприятие продукта или услуги потребителем:
            </h3>
            
            <div class="features-grid" style="max-width: 800px; margin: 0 auto 40px; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));">
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;">Анализ промо-роликов и A/B-тестирование креативов</p>
                </div>
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;">Продуктовое и UX тестирование</p>
                </div>
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;">Глубокая оценка эмоционального восприятия клиентского опыта</p>
                </div>
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;">Оценка эффективности выступлений и презентаций</p>
                </div>
                <div class="feature-card" style="padding: 15px; min-height: auto;">
                    <p style="font-size: 14px; font-weight: 500;">Редактирование сценариев для повышения эмоционального отклика</p>
                </div>
            </div>
            
            <div class="btn-container">
                <a href="#detector" class="btn">Анализировать сейчас</a>
                <a href="#features" class="btn btn-outline">Узнать больше</a>
            </div>
        </section>

        <section class="pricing">
            <h2 class="section-title">Тарифы</h2>
            <div class="pricing-grid">
                <div class="pricing-card">
                    <h3>Free</h3>
                    <div class="price">Бесплатно</div>
                    <ul class="features-list">
                        <li>Анализ 3-х фото в месяц</li>
                        <li>До 30 секунд видео или аудио в месяц</li>
                    </ul>
                    <button class="btn use-plan-btn" data-plan="free">Использовать</button>
                </div>
                
                <div class="pricing-card">
                    <h3>Light</h3>
                    <div class="price">9 990 ₽/мес</div>
                    <ul class="features-list">
                        <li>Ведение до 10 проектов</li>
                        <li>Неограниченный анализ фото</li>
                        <li>До 2 часов видео и аудио</li>
                        <li>Проведение A/B-анализа</li>
                        <li>Полный API-доступ</li>
                    </ul>
                    <button class="btn apply-plan-btn" data-plan="pro">Отправить заявку</button>
                </div>
                
                <div class="pricing-card">
                    <h3>Pro</h3>
                    <div class="price">29 990 ₽/мес</div>
                    <ul class="features-list">
                        <li>Неограниченное количество проектов</li>
                        <li>Анализ любого количества фото, видео и аудио</li>
                        <li>Проведение A/B-анализа</li>
                        <li>Полный API-доступ</li>
                        <li>Расширенная аналитика и техподдержка</li>
                        <li>Персональные экспертные отчеты</li>
                    </ul>
                    <button class="btn apply-plan-btn" data-plan="business">Отправить заявку</button>
                </div>

                <div class="pricing-card">
                    <h3>Business</h3>
                    <div class="price">По запросу</div>
                    <ul class="features-list">
                        <li>Неограниченное количество проектов</li>
                        <li>Анализ любого количества фото, видео и аудио</li>
                        <li>Проведение A/B-анализа</li>
                        <li>Полный API-доступ</li>
                        <li>Расширенная аналитика и техподдержка</li>
                        <li>Персональные экспертные отчеты</li>
                        <li><strong>Внедрение в CRM</strong></li>
                    </ul>
                    <button class="btn apply-plan-btn" data-plan="business">Отправить заявку</button>
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
}

function loadFeaturesContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <section class="features">
            <div class="container">
                <h1>Razuma – это экосистема ИИ-инструментов, которая сочетает методы нейромаркетинга, emotion AI и API-интеграции для бизнеса любого масштаба</h1>
                <h2 class="section-title">Основные возможности</h2>
                <div class="features-grid">
                    <div class="feature-card">
                        <p>Распознавание эмоций в реальном времени по фото, видео и аудио</p>
                    </div>
                    <div class="feature-card">
                        <p>Создание плагинов для CRM (Bitrix24, amoCRM и др.)</p>
                    </div>
                    <div class="feature-card">
                        <p>Анализ A/B-тестирования рекламных креативов или упаковки продуктов</p>
                    </div>
                    <div class="feature-card">
                        <p>Анализ UX интерфейса</p>
                    </div>
                    <div class="feature-card">
                        <p>Автоматическая оценка параметров эмоционального состояния клиента и рекомендации на ее основе</p>
                    </div>
                    <div class="feature-card">
                        <p>Интеграции через API и SDK</p>
                    </div>
                    <div class="feature-card">
                        <p>Адаптация продукта и разработка white label по техническому заданию</p>
                    </div>
                </div>
                <div class="clients-block">
                    <p>Кто уже использует Razuma:</p>
                    <p><strong>B2B:</strong> агентства, бренды, разработчики продуктов, обучающие платформы</p>
                    <p><strong>B2C / МСП:</strong> специалисты, малый бизнес, консультанты</p>
                </div>
            </div>
        </section>
    `;
}

function loadDetectorContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <div class="detector-container">
            <div style="text-align: center;">
                <h1>Распознавание эмоций с помощью ИИ</h1>
            </div>
            
            <div class="upload-section">
                <div class="upload-container" id="dropArea">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">
                        <h3>Перетащите файл сюда</h3>
                        <p>или</p>
                    </div>
                    <input type="file" id="fileInput" class="file-input" accept="image/*,video/*">
                    <button class="btn" id="selectFileBtn">Выбрать файл</button>
                    <p class="supported-formats">Поддерживаемые форматы: JPG, PNG, MP4, AVI, WEBM (макс. 16MB)</p>
                </div>
                
                <div class="file-info" id="fileInfo"></div>
                <div style="margin: 15px 0; text-align: center;">
                    <label style="display: flex; align-items: center; justify-content: center;">
                        <input type="checkbox" id="dataConsent" required style="margin-right: 8px;">
                        <span>Я даю согласие на обработку моих персональных данных в соответствии с 
                        <a href="#privacy" class="nav-link" style="color: var(--primary-color);" onclick="event.preventDefault(); window.dispatchEvent(new CustomEvent('navigate', {detail: 'privacy'}));">Политикой обработки персональных данных</a></span>
                    </label>
                </div>
                <button class="btn" id="uploadBtn" disabled>Анализировать эмоции</button>
            </div>

            <div class="progress-container" id="progressContainer">
                <div class="progress-header">
                    <h3>Обработка</h3>
                </div>
                <div class="progress-wheel" id="progressWheel"></div>
                <div class="progress-text" id="progressText">Начало обработки...</div>
            </div>
            
            <div class="preview-container" id="previewContainer"></div>
            
            <div class="results-container" id="resultsContainer"></div>
        </div>
    `;

    initializeDetector();
}

function loadPrivacyContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <div class="privacy-policy-container">
            <iframe 
                src="/static/privacy-policy.html" 
                width="100%" 
                height="500px" 
                frameborder="0"
                title="Privacy Policy"
                loading="lazy"
            ></iframe>
            <p class="fallback-text" style="display: none;">Privacy policy could not be loaded.</p>
        </div>
    `;

    const iframe = contentDiv.querySelector('iframe');
    iframe.onerror = function() {
        contentDiv.querySelector('.fallback-text').style.display = 'block';
        iframe.style.display = 'none';
    };
}

function loadContactContent() {
    const contentDiv = document.getElementById('app-content');
    contentDiv.innerHTML = `
        <section class="hero">
            <h1>Контактная информация</h1>
        </section>

        <section style="max-width: 800px; margin: 40px auto; padding: 30px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h2>Юридическая информация</h2>
            <p><strong>Наименование компании:</strong> ИП Тесаков Роман Валентинович</p>
            <p><strong>Юридический адрес:</strong>121087 Москва Новозаводская ул., д.2, корп.8, кв. 19</p>
            <p><strong>ИНН:</strong> 500402730770</p>
            <p><strong>ОГРНИП:</strong> 324774600219962</p>
            
            <h2 style="margin-top: 30px;">Контактные данные</h2>
            <p><strong>Email:</strong> inbox@razuma.pro</p>
            <p><strong>Телефон:</strong> +7 (903) 295-89-71</p>
            <p><strong>Режим работы:</strong> Пн-Пт, 9:00-18:00</p>
        </section>
    `;
}