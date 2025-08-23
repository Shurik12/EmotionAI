import React from 'react';

const Header = ({ navigateTo, language, setLanguage }) => {
	const [showMobileMenu, setShowMobileMenu] = React.useState(false);

	const handleLanguageChange = (e) => {
		setLanguage(e.target.value);
	};

	const handleMobileMenuClick = () => {
		setShowMobileMenu(!showMobileMenu);
	};

	const handleNavClick = (path) => {
		navigateTo(path);
		setShowMobileMenu(false);
	};

	return (
		<header>
			<button
				id="mobileMenuBtn"
				className="mobile-menu-btn"
				onClick={handleMobileMenuClick}
			>
				☰
			</button>

			<img
				style={{ height: '7%', width: '7%' }}
				src="/static/media/Razuma_Black.svg"
				alt="Razuma Logo"
			/>

			<a href="/" className="logo" onClick={(e) => {
				e.preventDefault();
				handleNavClick('home');
			}}>
				Razuma
			</a>

			<nav className={`main-nav ${showMobileMenu ? 'show' : ''}`} id="mainNav">
				<a
					href="#features"
					className="nav-link"
					data-i18n="nav_features"
					onClick={(e) => {
						e.preventDefault();
						handleNavClick('features');
					}}
				>
					Возможности
				</a>
				<a
					href="#detector"
					className="nav-link"
					data-i18n="nav_demo"
					onClick={(e) => {
						e.preventDefault();
						handleNavClick('detector');
					}}
				>
					Демо
				</a>
				<a
					href="#privacy"
					className="nav-link"
					data-i18n="nav_privacy"
					onClick={(e) => {
						e.preventDefault();
						handleNavClick('privacy');
					}}
				>
					Конфиденциальность
				</a>
				<a
					href="#contact"
					className="nav-link"
					data-i18n="nav_contacts"
					onClick={(e) => {
						e.preventDefault();
						handleNavClick('contact');
					}}
				>
					Контакты
				</a>
			</nav>

			<div className="language-selector">
				<select
					id="languageSelect"
					value={language}
					onChange={handleLanguageChange}
				>
					<option value="ru">Русский</option>
					<option value="en">English</option>
				</select>
			</div>

			<a
				href="#detector"
				className="btn"
				data-i18n="try_demo"
				onClick={(e) => {
					e.preventDefault();
					handleNavClick('detector');
				}}
			>
				Попробовать демо
			</a>
		</header>
	);
};

export default Header;