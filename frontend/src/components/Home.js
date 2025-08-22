import React, { useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { t as translate } from '../utils/translations';

const Home = ({ openApplicationModal, navigateTo }) => {
	const { language, updateTexts } = useLanguage();

	useEffect(() => {
		updateTexts();

		// Add event listeners for plan buttons
		const usePlanButtons = document.querySelectorAll('.use-plan-btn');
		usePlanButtons.forEach(btn => {
			btn.addEventListener('click', () => {
				navigateTo('detector');
			});
		});

		const applyPlanButtons = document.querySelectorAll('.apply-plan-btn');
		applyPlanButtons.forEach(btn => {
			btn.addEventListener('click', function () {
				const plan = this.getAttribute('data-plan');
				openApplicationModal(plan);
			});
		});

		return () => {
			// Clean up event listeners
			usePlanButtons.forEach(btn => btn.removeEventListener('click', () => { }));
			applyPlanButtons.forEach(btn => btn.removeEventListener('click', () => { }));
		};
	}, [language, updateTexts, navigateTo, openApplicationModal]);

	const t = (key, replacements = {}) => {
		return translate(key, language, replacements);
	};

	return (
		<>
			<section className="hero">
				<h1 data-i18n="main_title">{t('main_title')}</h1>
				<p data-i18n="main_description">{t('main_description')}</p>

				<h3 style={{ textAlign: 'center', margin: '30px 0 20px', fontSize: '18px', fontWeight: '500' }} data-i18n="platform_usage">
					{t('platform_usage')}
				</h3>

				<div className="features-grid" style={{ maxWidth: '800px', margin: '0 auto 40px', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))' }}>
					<div className="feature-card" style={{ padding: '15px', minHeight: 'auto' }}>
						<p style={{ fontSize: '14px', fontWeight: '500' }} data-i18n="feature1">{t('feature1')}</p>
					</div>
					<div className="feature-card" style={{ padding: '15px', minHeight: 'auto' }}>
						<p style={{ fontSize: '14px', fontWeight: '500' }} data-i18n="feature2">{t('feature2')}</p>
					</div>
					<div className="feature-card" style={{ padding: '15px', minHeight: 'auto' }}>
						<p style={{ fontSize: '14px', fontWeight: '500' }} data-i18n="feature3">{t('feature3')}</p>
					</div>
					<div className="feature-card" style={{ padding: '15px', minHeight: 'auto' }}>
						<p style={{ fontSize: '14px', fontWeight: '500' }} data-i18n="feature4">{t('feature4')}</p>
					</div>
					<div className="feature-card" style={{ padding: '15px', minHeight: 'auto' }}>
						<p style={{ fontSize: '14px', fontWeight: '500' }} data-i18n="feature5">{t('feature5')}</p>
					</div>
				</div>

				<div className="btn-container">
					<a href="#detector" className="btn" data-i18n="analyze_now" onClick={(e) => {
						e.preventDefault();
						navigateTo('detector');
					}}>{t('analyze_now')}</a>
					<a href="#features" className="btn btn-outline" data-i18n="learn_more" onClick={(e) => {
						e.preventDefault();
						navigateTo('features');
					}}>{t('learn_more')}</a>
				</div>
			</section>

			<section className="pricing">
				<h2 className="section-title" data-i18n="pricing_title">{t('pricing_title')}</h2>
				<div className="pricing-grid">
					<div className="pricing-card">
						<h3 data-i18n="free_plan">{t('free_plan')}</h3>
						<div className="price" data-i18n="free_price">{t('free_price')}</div>
						<ul className="features-list">
							<li data-i18n="free_feature1">{t('free_feature1')}</li>
							<li data-i18n="free_feature2">{t('free_feature2')}</li>
						</ul>
						<button className="btn use-plan-btn" data-plan="free" data-i18n="use_plan">
							{t('use_plan')}
						</button>
					</div>

					<div className="pricing-card">
						<h3 data-i18n="light_plan">{t('light_plan')}</h3>
						<div className="price">9 990 ₽/мес</div>
						<ul className="features-list">
							<li data-i18n="light_feature1">{t('light_feature1')}</li>
							<li data-i18n="light_feature2">{t('light_feature2')}</li>
							<li data-i18n="light_feature3">{t('light_feature3')}</li>
							<li data-i18n="light_feature4">{t('light_feature4')}</li>
							<li data-i18n="light_feature5">{t('light_feature5')}</li>
						</ul>
						<button className="btn apply-plan-btn" data-plan="pro" data-i18n="apply_plan">
							{t('apply_plan')}
						</button>
					</div>

					<div className="pricing-card">
						<h3 data-i18n="pro_plan">{t('pro_plan')}</h3>
						<div className="price">29 990 ₽/мес</div>
						<ul className="features-list">
							<li data-i18n="pro_feature1">{t('pro_feature1')}</li>
							<li data-i18n="pro_feature2">{t('pro_feature2')}</li>
							<li data-i18n="pro_feature3">{t('pro_feature3')}</li>
							<li data-i18n="pro_feature4">{t('pro_feature4')}</li>
							<li data-i18n="pro_feature5">{t('pro_feature5')}</li>
							<li data-i18n="pro_feature6">{t('pro_feature6')}</li>
						</ul>
						<button className="btn apply-plan-btn" data-plan="business" data-i18n="apply_plan">
							{t('apply_plan')}
						</button>
					</div>

					<div className="pricing-card">
						<h3 data-i18n="business_plan">{t('business_plan')}</h3>
						<div className="price" data-i18n="business_price">{t('business_price')}</div>
						<ul className="features-list">
							<li data-i18n="business_feature1">{t('business_feature1')}</li>
							<li data-i18n="business_feature2">{t('business_feature2')}</li>
							<li data-i18n="business_feature3">{t('business_feature3')}</li>
							<li data-i18n="business_feature4">{t('business_feature4')}</li>
							<li data-i18n="business_feature5">{t('business_feature5')}</li>
							<li data-i18n="business_feature6">{t('business_feature6')}</li>
							<li><strong data-i18n="business_feature7">{t('business_feature7')}</strong></li>
						</ul>
						<button className="btn apply-plan-btn" data-plan="business" data-i18n="apply_plan">
							{t('apply_plan')}
						</button>
					</div>
				</div>
			</section>
		</>
	);
};

export default Home;