import React, { useEffect, useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { t } from '../utils/translations';

const Privacy = () => {
	const { language } = useLanguage();
	const [iframeError, setIframeError] = useState(false);

	const handleIframeError = () => {
		setIframeError(true);
	};

	return (
		<div className="privacy-policy-container">
			<h1 data-i18n="privacy_policy">{t('privacy_policy')}</h1>

			{!iframeError ? (
				<div className="iframe-container">
					<iframe
						id="privacyIframe"
						src={`/static/privacy-policy.html?lang=${language}&t=${Date.now()}`}
						width="100%"
						height="500px"
						frameBorder="0"
						title={t('privacy_policy')}
						loading="lazy"
						onError={handleIframeError}
					></iframe>
				</div>
			) : (
				<p className="fallback-text" style={{ textAlign: 'center', color: '#666', marginTop: '2rem' }}>
					{t('privacy_error')}
				</p>
			)}
		</div>
	);
};

export default Privacy;