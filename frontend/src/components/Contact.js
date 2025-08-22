import React, { useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { t as translate } from '../utils/translations';

const Contact = () => {
	const { language, updateTexts } = useLanguage();

	useEffect(() => {
		updateTexts();
	}, [language, updateTexts]);

	const t = (key, replacements = {}) => {
		return translate(key, language, replacements);
	};

	return (
		<>
			<section className="hero">
				<h1 data-i18n="contact_title">{t('contact_title')}</h1>
			</section>

			<section style={{
				maxWidth: '800px',
				margin: '40px auto',
				padding: '30px',
				background: 'white',
				boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
				borderRadius: '8px'
			}}>
				<h2 data-i18n="legal_info">{t('legal_info')}</h2>
				<p><strong data-i18n="company_name">{t('company_name')}:</strong> ИП Тесаков Роман Валентинович</p>
				<p><strong data-i18n="legal_address">{t('legal_address')}:</strong> 121087 Москва Новозаводская ул., д.2, корп.8, кв. 19</p>
				<p><strong data-i18n="inn">{t('inn')}:</strong> 500402730770</p>
				<p><strong data-i18n="ogrnip">{t('ogrnip')}:</strong> 324774600219962</p>

				<h2 style={{ marginTop: '30px' }} data-i18n="contact_details">{t('contact_details')}</h2>
				<p><strong data-i18n="email">{t('email')}:</strong> inbox@razuma.pro</p>
				<p><strong data-i18n="phone">{t('phone')}:</strong> +7 (903) 295-89-71</p>
				<p><strong data-i18n="working_hours">{t('working_hours')}:</strong> Пн-Пт, 9:00-18:00</p>
			</section>
		</>
	);
};

export default Contact;