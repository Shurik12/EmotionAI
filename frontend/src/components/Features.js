import React, { useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { t } from '../utils/translations';

const Features = () => {
	const { language, updateTexts } = useLanguage();

	useEffect(() => {
		updateTexts();
	}, [language, updateTexts]);
	
	return (
		<section className="features">
			<div className="container">
				<h1 data-i18n="features_title">{t('features_title')}</h1>
				<h2 className="section-title" data-i18n="features_main">{t('features_main')}</h2>

				<div className="features-grid">
					<div className="feature-card">
						<p data-i18n="feature_card1">{t('feature_card1')}</p>
					</div>

					<div className="feature-card">
						<p data-i18n="feature_card2">{t('feature_card2')}</p>
					</div>

					<div className="feature-card">
						<p data-i18n="feature_card3">{t('feature_card3')}</p>
					</div>

					<div className="feature-card">
						<p data-i18n="feature_card4">{t('feature_card4')}</p>
					</div>

					<div className="feature-card">
						<p data-i18n="feature_card5">{t('feature_card5')}</p>
					</div>

					<div className="feature-card">
						<p data-i18n="feature_card6">{t('feature_card6')}</p>
					</div>

					<div className="feature-card">
						<p data-i18n="feature_card7">{t('feature_card7')}</p>
					</div>
				</div>

				<div className="clients-block">
					<p data-i18n="clients_title">{t('clients_title')}</p>
					<p><strong>B2B:</strong> <span data-i18n="clients_b2b">{t('clients_b2b')}</span></p>
					<p><strong>B2C / МСП:</strong> <span data-i18n="clients_b2c">{t('clients_b2c')}</span></p>
				</div>
			</div>
		</section>
	);
};

export default Features;