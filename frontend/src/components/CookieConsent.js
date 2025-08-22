import React, { useState, useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { t as translate } from '../utils/translations';

const CookieConsent = () => {
	const [showConsent, setShowConsent] = useState(false);
	const { language } = useLanguage();

	useEffect(() => {
		// Check if user has already accepted cookies
		const cookiesAccepted = localStorage.getItem('cookiesAccepted');
		if (!cookiesAccepted) {
			setShowConsent(true);
		}
	}, []);

	const acceptCookies = () => {
		localStorage.setItem('cookiesAccepted', 'true');
		setShowConsent(false);
	};

	const t = (key, replacements = {}) => {
		return translate(key, language, replacements);
	};

	if (!showConsent) return null;

	return (
		<div id="cookieConsent" style={{
			position: 'fixed',
			bottom: '0',
			left: '0',
			right: '0',
			backgroundColor: 'rgba(0, 0, 0, 0.8)',
			color: 'white',
			padding: '15px',
			textAlign: 'center',
			zIndex: '1000'
		}}>
			<p style={{ margin: '0 0 10px 0' }}>
				{t('cookie_text')}{' '}
				<a
					href="#privacy"
					style={{ color: 'var(--primary-color)' }}
					onClick={(e) => {
						e.preventDefault();
						// You would navigate to privacy page here
					}}
				>
					{t('cookie_more')}
				</a>
			</p>
			<button
				id="acceptCookies"
				onClick={acceptCookies}
				style={{
					backgroundColor: 'var(--primary-color)',
					color: 'white',
					border: 'none',
					padding: '8px 16px',
					borderRadius: '4px',
					cursor: 'pointer'
				}}
			>
				{t('cookie_accept')}
			</button>
		</div>
	);
};

export default CookieConsent;