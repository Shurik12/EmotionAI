import { useState, useEffect } from 'react';
import { translations } from '../utils/translations';

export const useLanguage = () => {
	const [language, setLanguage] = useState(
		localStorage.getItem('language') || 'ru'
	);

	useEffect(() => {
		document.documentElement.lang = language;
		localStorage.setItem('language', language);
		updateTexts();
	}, [language]);

	const updateTexts = () => {
		const elements = document.querySelectorAll('[data-i18n]');
		elements.forEach(el => {
			const key = el.getAttribute('data-i18n');
			if (translations[language] && translations[language][key]) {
				el.textContent = translations[language][key];
			}
		});
	};

	return { language, setLanguage, updateTexts };
};