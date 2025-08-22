// src/App.js
import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import CookieConsent from './components/CookieConsent';
import ApplicationModal from './components/ApplicationModal';
import Home from './components/Home';
import Features from './components/Features';
import Detector from './components/Detector';
import Privacy from './components/Privacy';
import Contact from './components/Contact';
import { useNavigation } from './hooks/useNavigation';
import { useLanguage } from './hooks/useLanguage';
import './styles/styles.css';

function App() {
	const { currentPage, navigateTo } = useNavigation();
	const { language, setLanguage } = useLanguage();
	const [showApplicationModal, setShowApplicationModal] = useState(false);
	const [selectedPlan, setSelectedPlan] = useState('');

	const openApplicationModal = (plan) => {
		setSelectedPlan(plan);
		setShowApplicationModal(true);
	};

	const closeApplicationModal = () => {
		setShowApplicationModal(false);
	};

	const renderContent = () => {
		switch (currentPage) {
			case 'home':
				return <Home openApplicationModal={openApplicationModal} navigateTo={navigateTo} />;
			case 'features':
				return <Features />;
			case 'detector':
				return <Detector />;
			case 'privacy':
				return <Privacy />;
			case 'contact':
				return <Contact />;
			default:
				return <Home openApplicationModal={openApplicationModal} navigateTo={navigateTo} />;
		}
	};

	return (
		<div className="container">
			<Header
				navigateTo={navigateTo}
				language={language}
				setLanguage={setLanguage}
			/>

			<main id="app-content">
				{renderContent()}
			</main>

			<Footer navigateTo={navigateTo} />

			<CookieConsent />

			{showApplicationModal && (
				<ApplicationModal
					selectedPlan={selectedPlan}
					closeModal={closeApplicationModal}
				/>
			)}
		</div>
	);
}

export default App;