// src/components/Footer.js
import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { t as translate } from '../utils/translations';

const Footer = ({ navigateTo }) => {
	const { language } = useLanguage();

	const t = (key, replacements = {}) => {
		return translate(key, language, replacements);
	};

	const handleFooterClick = (path) => {
		navigateTo(path);
	};

	return (
		<footer>
			<div className="footer-logo">Razuma</div>
			<div className="footer-links">
				<a
					href="#detector"
					onClick={(e) => {
						e.preventDefault();
						handleFooterClick('detector');
					}}
					data-i18n="footer_demo"
				>
					{t('footer_demo')}
				</a>
				<a
					href="#privacy"
					onClick={(e) => {
						e.preventDefault();
						handleFooterClick('privacy');
					}}
					data-i18n="footer_privacy"
				>
					{t('footer_privacy')}
				</a>
				<a
					href="#contact"
					onClick={(e) => {
						e.preventDefault();
						handleFooterClick('contact');
					}}
					data-i18n="footer_contacts"
				>
					{t('footer_contacts')}
				</a>
			</div>
			<p className="copyright" data-i18n="copyright">
				{t('copyright')}
			</p>
		</footer>
	);
};

export default Footer;