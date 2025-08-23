import { useState, useEffect } from 'react';

export const useNavigation = () => {
	const [currentPage, setCurrentPage] = useState('home');

	useEffect(() => {
		// Get initial page from URL
		const path = window.location.pathname.replace(/^\//, '');
		const initialPage = path === '' ? 'home' : path;
		setCurrentPage(initialPage);

		// Handle browser back/forward
		const handlePopState = () => {
			const newPath = window.location.pathname.replace(/^\//, '');
			const newPage = newPath === '' ? 'home' : newPath;
			setCurrentPage(newPage);
		};

		window.addEventListener('popstate', handlePopState);
		return () => window.removeEventListener('popstate', handlePopState);
	}, []);

	const navigateTo = (path) => {
		// Remove leading slash if present
		path = path.replace(/^\//, '');

		// Update state
		setCurrentPage(path);

		// Update browser history
		if (path === 'home') {
			window.history.pushState(null, '', '/');
		} else {
			window.history.pushState(null, '', `/${path}`);
		}

		// Update document title
		const titles = {
			'home': 'Razuma | Распознавание эмоций',
			'features': 'Razuma | Возможности',
			'detector': 'Razuma | Демо',
			'privacy': 'Razuma | Конфиденциальность',
			'contact': 'Razuma | Контакты'
		};
		document.title = titles[path] || 'Razuma';
	};

	return { currentPage, navigateTo };
};